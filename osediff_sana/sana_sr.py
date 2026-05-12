"""
SANA-SR Model: Frozen SANA + LoRA for one-step image super-resolution.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class SanaSRModel(nn.Module):
    """Wrapper that manages SANA components for SR training."""

    def __init__(self, vae, transformer, scheduler, vae_config, num_train_timesteps=1000):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler

        self.scaling_factor = getattr(vae_config, "scaling_factor", 1.0)
        self.shift_factor = getattr(vae_config, "shift_factor", 0.0)
        self.num_train_timesteps = num_train_timesteps
        self.timestep_scale = float(getattr(transformer.config, "timestep_scale", 1.0))

        self.scheduler.set_timesteps(self.num_train_timesteps)
        self.register_buffer("train_timesteps", self.scheduler.timesteps.clone().float(), persistent=False)
        self.register_buffer("train_sigmas", self.scheduler.sigmas[:-1].clone().float(), persistent=False)

    @torch.no_grad()
    def encode(self, images):
        vae_dtype = next(self.vae.parameters()).dtype
        images = images.to(dtype=vae_dtype)
        encoded = self.vae.encode(images)
        if hasattr(encoded, "latent"):
            raw = encoded.latent
        elif hasattr(encoded, "latent_dist"):
            raw = encoded.latent_dist.sample()
        else:
            raw = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
        latents = (raw - self.shift_factor) * self.scaling_factor
        return latents

    def decode(self, latents):
        vae_latents = latents / self.scaling_factor + self.shift_factor
        vae_dtype = next(self.vae.parameters()).dtype
        vae_latents = vae_latents.to(dtype=vae_dtype)
        decoded = self.vae.decode(vae_latents)
        if hasattr(decoded, "sample"):
            return decoded.sample
        return decoded[0] if isinstance(decoded, (tuple, list)) else decoded

    def get_timestep_sigma_by_index(self, index, batch_size, device, dtype):
        if isinstance(index, int):
            idx = torch.full((batch_size,), index, device=device, dtype=torch.long)
        else:
            idx = index.to(device=device, dtype=torch.long)

        max_idx = self.train_timesteps.shape[0] - 1
        idx = idx.clamp(0, max_idx)

        t = self.train_timesteps[idx].to(device=device)
        sigma = self.train_sigmas[idx].to(device=device, dtype=dtype)
        return t, sigma

    def add_noise(self, z_0, noise, sigma):
        sigma = sigma.view(-1, 1, 1, 1).to(z_0.dtype)
        return (1 - sigma) * z_0 + sigma * noise

    def one_step_denoise(self, z_t, sigma):
        sigma = sigma.view(-1, 1, 1, 1).to(z_t.dtype)
        return z_t - sigma * self._last_vpred

    def _get_model_dtype(self):
        # LoRA trainable params may be promoted to fp32 for optimizer stability,
        # but the frozen SANA backbone should still run in its original low-precision
        # dtype (typically fp16/bf16). Prefer the first frozen parameter dtype.
        first_dtype = None
        for p in self.transformer.parameters():
            if first_dtype is None:
                first_dtype = p.dtype
            if not p.requires_grad:
                return p.dtype
        if first_dtype is not None:
            return first_dtype
        return torch.float32

    def _dit_forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask):
        model_dtype = self._get_model_dtype()
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        timestep = timestep * self.timestep_scale
        out = self.transformer(
            hidden_states=hidden_states.to(model_dtype),
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states.to(model_dtype),
            encoder_attention_mask=encoder_attention_mask,
        )
        return out.sample if hasattr(out, "sample") else (out[0] if isinstance(out, tuple) else out)

    def generate_fake_latent(self, z_lq, gen_timestep, text_embeds, text_mask):
        if hasattr(self.transformer, "enable_adapter_layers"):
            self.transformer.enable_adapter_layers()
        B = z_lq.shape[0]
        t, sigma_gen = self.get_timestep_sigma_by_index(
            index=gen_timestep, batch_size=B, device=z_lq.device, dtype=z_lq.dtype
        )

        v_pred = self._dit_forward(z_lq, t, text_embeds, text_mask)
        self._last_vpred = v_pred
        z_fake = self.one_step_denoise(z_lq, sigma_gen)
        return z_fake

    def predict_noise_pair(self, z_fake, z_hq, timesteps, sigma, noise, text_embeds, text_mask):
        z_fake_noisy = self.add_noise(z_fake, noise, sigma)
        z_hq_noisy = self.add_noise(z_hq, noise, sigma)

        if hasattr(self.transformer, "disable_adapter_layers"):
            self.transformer.disable_adapter_layers()
        try:
            n_fake = self._dit_forward(z_fake_noisy, timesteps, text_embeds, text_mask)
            with torch.no_grad():
                n_ref = self._dit_forward(z_hq_noisy, timesteps, text_embeds, text_mask)
        finally:
            if hasattr(self.transformer, "enable_adapter_layers"):
                self.transformer.enable_adapter_layers()

        return n_fake, n_ref


def setup_lora(transformer, config):
    for p in transformer.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=config["model"]["lora_rank"],
        lora_alpha=config["model"]["lora_alpha"],
        target_modules=config["model"]["lora_target_modules"],
        lora_dropout=config["model"]["lora_dropout"],
    )
    transformer = get_peft_model(transformer, lora_cfg)

    for _name, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    transformer.print_trainable_parameters()
    return transformer
