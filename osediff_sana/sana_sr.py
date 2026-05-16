import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class SanaSRModel(nn.Module):
    """冻结SANA，加LoRA"""

    def __init__(self, vae, transformer, scheduler, cfg, steps=1000):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        self.scale = getattr(cfg, "scaling_factor", 1.0)
        self.shift = getattr(cfg, "shift_factor", 0.0)
        self.steps = steps
        self.timestep_scale = float(getattr(transformer.config, "timestep_scale", 1.0))

        self.scheduler.set_timesteps(self.steps)
        self.register_buffer("train_timesteps", self.scheduler.timesteps.clone().float(), persistent=False)
        self.register_buffer("train_sigmas", self.scheduler.sigmas[:-1].clone().float(), persistent=False)

    @torch.no_grad()
    def encode(self, images):
        """编码图像到SANA压缩潜空间。"""
        kind = next(self.vae.parameters()).dtype
        images = images.to(dtype=kind)
        out = self.vae.encode(images)
        if hasattr(out, "latent"):
            z = out.latent
        elif hasattr(out, "latent_dist"):
            z = out.latent_dist.sample()
        else:
            if isinstance(out, (tuple, list)):
                z = out[0]
            else:
                z = out
        return (z - self.shift) * self.scale

    def decode(self, latents):
        z = latents / self.scale + self.shift
        kind = next(self.vae.parameters()).dtype
        z = z.to(dtype=kind)
        out = self.vae.decode(z)
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def get_timestep_sigma_by_index(self, index, batch_size, device, dtype):
        """读训练时间步和噪声系数"""
        if isinstance(index, int):
            i = torch.full((batch_size,), index, device=device, dtype=torch.long)
        else:
            i = index.to(device=device, dtype=torch.long)

        last = self.train_timesteps.shape[0] - 1
        i = i.clamp(0, last)

        t = self.train_timesteps[i].to(device=device)
        sigma = self.train_sigmas[i].to(device=device, dtype=dtype)
        return t, sigma

    def add_noise(self, z, noise, sigma):
        sigma = sigma.view(-1, 1, 1, 1).to(z.dtype)
        return (1 - sigma) * z + sigma * noise

    def one_step_denoise(self, z, sigma):
        """一步更新"""
        sigma = sigma.view(-1, 1, 1, 1).to(z.dtype)
        return z - sigma * self.last

    def _get_model_dtype(self):
        kind = None
        for p in self.transformer.parameters():
            if kind is None:
                kind = p.dtype
            if not p.requires_grad:
                return p.dtype
        if kind is not None:
            return kind
        return torch.float32

    def _dit_forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        kind = self._get_model_dtype()
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        timestep = timestep * self.timestep_scale
        out = self.transformer(
            hidden_states=hidden_states.to(kind),
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(kind),
            encoder_attention_mask=encoder_attention_mask,
        )
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, tuple):
            return out[0]
        return out

    def generate_fake_latent(self, z_lq, gen_timestep, text_embeds, text_mask):
        """根据文本条件生成单步恢复潜变量。"""
        if hasattr(self.transformer, "enable_adapter_layers"):
            self.transformer.enable_adapter_layers()
        n = z_lq.shape[0]
        t, sigma = self.get_timestep_sigma_by_index(
            index=gen_timestep, batch_size=n, device=z_lq.device, dtype=z_lq.dtype
        )

        pred = self._dit_forward(z_lq, t, text_embeds, text_mask)
        self.last = pred
        return self.one_step_denoise(z_lq, sigma)

    def predict_noise_pair(self, z_fake, z_hq, timesteps, sigma, noise, text_embeds, text_mask):
        """计算适配器关闭时的恢复/目标响应对。"""
        x = self.add_noise(z_fake, noise, sigma)
        y = self.add_noise(z_hq, noise, sigma)

        if hasattr(self.transformer, "disable_adapter_layers"):
            self.transformer.disable_adapter_layers()
        try:
            a = self._dit_forward(x, timesteps, text_embeds, text_mask)
            with torch.no_grad():
                b = self._dit_forward(y, timesteps, text_embeds, text_mask)
        finally:
            if hasattr(self.transformer, "enable_adapter_layers"):
                self.transformer.enable_adapter_layers()

        return a, b


def setup_lora(transformer, config):
    """注入LoRA训练参数"""
    for p in transformer.parameters():
        p.requires_grad_(False)

    cfg = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["lora_target_modules"],
        lora_dropout=config["model"]["lora_dropout"],
    )
    transformer = get_peft_model(transformer, cfg)

    for name, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    transformer.print_trainable_parameters()
    return transformer
