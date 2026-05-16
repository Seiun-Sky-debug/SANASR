import argparse
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import yaml
from osediff_sana.hf_compat import import_sana_pipeline, patch_hf_cache_home

patch_hf_cache_home()
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

SanaPipeline = import_sana_pipeline()

try:
    import lpips
except ImportError as e:
    raise ImportError("Please install lpips first: pip install lpips") from e

from osediff_sana.dataset import SRDataset
from osediff_sana.perceptual_losses import DinoPerceptualLoss
from osediff_sana.sana_sr import SanaSRModel, setup_lora
from osediff_sana.prompt_utils import (
    TargetedPromptExtractor,
    build_primary_secondary_prompts,
    build_suffix_only_prompts,
    encode_sana_prompts,
    format_prompts_for_logging,
    select_prompt_images,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_torch_dtype(name: str | None, default: torch.dtype = torch.float32) -> torch.dtype:
    if not name:
        return default
    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(name).strip().lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype_map[key]


def build_train_dataset(dc_data):
    main_train_dataset = SRDataset(
        hq_dir=dc_data["hq_dir"],
        lq_dir=dc_data.get("lq_dir") or None,
        image_size=dc_data["image_size"],
        scale_factor=dc_data["scale_factor"],
        is_train=True,
        use_osediff_degradation=dc_data.get("use_osediff_degradation", False),
    )
    realsr_hq = dc_data.get("realsr_train_hq_dir", "")
    if realsr_hq:
        realsr_train_dataset = SRDataset(
            hq_dir=realsr_hq,
            lq_dir=None,
            image_size=dc_data["image_size"],
            scale_factor=dc_data["scale_factor"],
            is_train=True,
            use_osediff_degradation=dc_data.get(
                "realsr_use_osediff_degradation", dc_data.get("use_osediff_degradation", True)
            ),
        )
        return ConcatDataset([main_train_dataset, realsr_train_dataset])
    return main_train_dataset


def encode_prompt_batch(pipe: SanaPipeline, prompts, device: torch.device):
    return encode_sana_prompts(pipe.tokenizer, pipe.text_encoder, prompts, device)


def resolve_lora_dir(path_str: str):
    """解析LoRA目录或权重文件路径。"""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_file():
        if path.name == "adapter_model.safetensors":
            return str(path.parent)
        return None

    candidates = [path, path / "lora", path / "generator_lora"]
    for cand in candidates:
        if cand.exists() and (cand / "adapter_model.safetensors").exists():
            return str(cand)
    return None


def _get_param_root(model):
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def _iter_trainable_lora_params(model):
    param_root = _get_param_root(model)
    for name, p in param_root.named_parameters():
        if p.requires_grad:
            yield name, p


def cast_trainable_lora_params_to_fp32(model):
    for _name, p in _iter_trainable_lora_params(model):
        if p.dtype != torch.float32:
            p.data = p.data.float()


def snapshot_trainable_lora_params(model):
    return {name: p.detach().float().clone() for name, p in _iter_trainable_lora_params(model)}


def restore_trainable_lora_params(model, snapshot):
    if snapshot is None:
        return
    param_root = _get_param_root(model)
    named_params = dict(param_root.named_parameters())
    with torch.no_grad():
        for name, old_p in snapshot.items():
            if name in named_params:
                p = named_params[name]
                p.copy_(old_p.to(device=p.device, dtype=p.dtype))


def trainable_lora_params_are_finite(model) -> bool:
    for _name, p in _iter_trainable_lora_params(model):
        if not torch.isfinite(p.detach()).all():
            return False
    return True


def safe_value(x):
    if torch.isfinite(x):
        return float(x.item())
    return float("nan")


def init_ema_state(model):
    ema_state = {}
    for name, p in _iter_trainable_lora_params(model):
        ema_state[name] = p.detach().float().clone()
    return ema_state


def update_ema_state(ema_state, model, decay):
    if ema_state is None:
        return
    with torch.no_grad():
        one_minus = 1.0 - float(decay)
        for name, p in _iter_trainable_lora_params(model):
            ema_state[name].mul_(float(decay)).add_(p.detach().float(), alpha=one_minus)


@contextmanager
def maybe_ema_scope(model, ema_state):
    if ema_state is None:
        yield
        return

    backup = {}
    named_params = dict(model.transformer.named_parameters())
    with torch.no_grad():
        for name, ema_p in ema_state.items():
            if name not in named_params:
                continue
            p = named_params[name]
            backup[name] = p.detach().clone()
            p.copy_(ema_p.to(device=p.device, dtype=p.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, old_p in backup.items():
                named_params[name].copy_(old_p)


def save_checkpoint(model, optimizer_g, ema_state, global_step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_{global_step:07d}")
    os.makedirs(path, exist_ok=True)
    model.transformer.save_pretrained(os.path.join(path, "lora"))
    torch.save(
        {
            "optimizer_g": optimizer_g.state_dict(),
            "ema_state": ema_state,
            "global_step": global_step,
        },
        os.path.join(path, "train_state.pt"),
    )
    print(f"[SAVE] Checkpoint saved to {path}")


def save_best_checkpoint(model, optimizer_g, ema_state, global_step, best_g_loss, save_dir):
    path = os.path.join(save_dir, "best_g")
    os.makedirs(path, exist_ok=True)
    model.transformer.save_pretrained(os.path.join(path, "lora"))
    torch.save(
        {
            "optimizer_g": optimizer_g.state_dict(),
            "ema_state": ema_state,
            "global_step": global_step,
            "best_g_loss": float(best_g_loss),
        },
        os.path.join(path, "train_state.pt"),
    )
    print(f"[SAVE] New best-G checkpoint: step={global_step}, G={best_g_loss:.6f} -> {path}")


def load_checkpoint(path, model, optimizer_g, device):
    lora_path = os.path.join(path, "lora")
    if os.path.exists(lora_path) and not hasattr(model.transformer, "peft_config"):
        from peft import PeftModel

        model.transformer = PeftModel.from_pretrained(
            model.transformer, lora_path, is_trainable=True
        ).to(device)
        print(f"[LOAD] LoRA weights loaded from {lora_path}")

    state_path = os.path.join(path, "train_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        if "optimizer_g" in state:
            optimizer_g.load_state_dict(state["optimizer_g"])
        return state["global_step"], state.get("ema_state", None)
    return 0, None


def train_one_step(model, lpips_model, dino_model, batch, te, tm, optimizer_g, config, device, te_aux=None, tm_aux=None):
    tc = config["training"]
    B = batch["lq"].shape[0]

    vae_dtype = next(model.vae.parameters()).dtype
    lq_img = batch["lq"].to(device, dtype=vae_dtype)
    hq_img = batch["hq"].to(device, dtype=vae_dtype)

    z_lq = model.encode(lq_img)
    z_hq = model.encode(hq_img)
    if not (torch.isfinite(z_lq).all() and torch.isfinite(z_hq).all()):
        optimizer_g.zero_grad(set_to_none=True)
        return {
            "loss_g": float("nan"),
            "loss_l2": float("nan"),
            "loss_lpips": float("nan"),
            "loss_dino": float("nan"),
            "loss_vsd": float("nan"),
            "loss_vsd_lora": float("nan"),
            "loss_prompt_consistency": float("nan"),
            "img_loss_bs": float(tc.get("img_loss_batch_size", B)),
            "lora_grad_norm": 0.0,
            "skip_non_finite": 1.0,
        }

    z_fake = model.generate_fake_latent(
        z_lq, gen_timestep=tc["gen_timestep"], text_embeds=te, text_mask=tm
    )
    if not torch.isfinite(z_fake).all():
        optimizer_g.zero_grad(set_to_none=True)
        return {
            "loss_g": float("nan"),
            "loss_l2": float("nan"),
            "loss_lpips": float("nan"),
            "loss_dino": float("nan"),
            "loss_vsd": float("nan"),
            "loss_vsd_lora": float("nan"),
            "loss_prompt_consistency": float("nan"),
            "img_loss_bs": float(tc.get("img_loss_batch_size", B)),
            "lora_grad_norm": 0.0,
            "skip_non_finite": 1.0,
        }
    loss_prompt_consistency = z_fake.new_tensor(0.0)
    if te_aux is not None and tm_aux is not None and float(tc.get("lambda_prompt_consistency", 0.0)) > 0:
        z_fake_aux = model.generate_fake_latent(
            z_lq, gen_timestep=tc["gen_timestep"], text_embeds=te_aux, text_mask=tm_aux
        )
        if not torch.isfinite(z_fake_aux).all():
            optimizer_g.zero_grad(set_to_none=True)
            return {
                "loss_g": float("nan"),
                "loss_l2": float("nan"),
                "loss_lpips": float("nan"),
                "loss_dino": float("nan"),
                "loss_vsd": float("nan"),
                "loss_vsd_lora": float("nan"),
                "loss_prompt_consistency": float("nan"),
                "img_loss_bs": float(tc.get("img_loss_batch_size", B)),
                "lora_grad_norm": 0.0,
                "skip_non_finite": 1.0,
            }
        loss_prompt_consistency = F.mse_loss(z_fake.float(), z_fake_aux.float())

    t_idx = torch.randint(tc["noise_t_min"], tc["noise_t_max"], (B,), device=device)
    t, sigma = model.get_timestep_sigma_by_index(index=t_idx, batch_size=B, device=device, dtype=z_fake.dtype)
    noise = torch.randn_like(z_fake)
    n_fake, n_ref = model.predict_noise_pair(
        z_fake, z_hq, timesteps=t, sigma=sigma, noise=noise, text_embeds=te, text_mask=tm
    )
    if not (torch.isfinite(n_fake).all() and torch.isfinite(n_ref).all()):
        optimizer_g.zero_grad(set_to_none=True)
        return {
            "loss_g": float("nan"),
            "loss_l2": float("nan"),
            "loss_lpips": float("nan"),
            "loss_dino": float("nan"),
            "loss_vsd": float("nan"),
            "loss_vsd_lora": float("nan"),
            "loss_prompt_consistency": float("nan"),
            "img_loss_bs": float(tc.get("img_loss_batch_size", B)),
            "lora_grad_norm": 0.0,
            "skip_non_finite": 1.0,
        }

    img_loss_bs = int(tc.get("img_loss_batch_size", B))
    img_loss_bs = max(1, min(B, img_loss_bs))
    if img_loss_bs < B:
        img_idx = torch.randperm(B, device=device)[:img_loss_bs]
        z_fake_img = z_fake[img_idx]
        hq_img_img = hq_img[img_idx]
        t_img = t[img_idx]
        sigma_img = sigma[img_idx]
        noise_img = noise[img_idx]
        te_img = te[img_idx]
        tm_img = tm[img_idx]
    else:
        z_fake_img = z_fake
        hq_img_img = hq_img
        t_img = t
        sigma_img = sigma
        noise_img = noise
        te_img = te
        tm_img = tm

    sr_img = model.decode(z_fake_img).clamp(-1, 1)
    sr_f = sr_img.float()
    hq_f = hq_img_img.float()
    sr_01 = sr_f * 0.5 + 0.5
    hq_01 = hq_f * 0.5 + 0.5

    loss_l2 = F.mse_loss(sr_f, hq_f)
    loss_lpips = (
        lpips_model(sr_f, hq_f).mean()
        if lpips_model is not None and float(tc.get("lambda_lpips", 0.0)) > 0
        else sr_f.new_tensor(0.0)
    )
    loss_dino = (
        dino_model(sr_01, hq_01)
        if dino_model is not None and float(tc.get("lambda_dino", 0.0)) > 0
        else sr_f.new_tensor(0.0)
    )

    n_fake_f = n_fake.float()
    n_ref_f = n_ref.float()
    eps = 1e-6
    fake_mean = n_fake_f.mean(dim=(2, 3))
    ref_mean = n_ref_f.mean(dim=(2, 3))
    fake_var = n_fake_f.var(dim=(2, 3), unbiased=False).clamp_min(eps)
    ref_var = n_ref_f.var(dim=(2, 3), unbiased=False).clamp_min(eps)
    loss_vsd = 0.5 * (
        torch.log(ref_var / fake_var)
        + (fake_var + (fake_mean - ref_mean).pow(2)) / ref_var
        - 1.0
    )
    loss_vsd = loss_vsd.mean()

    z_fake_noisy_img = model.add_noise(z_fake_img, noise_img, sigma_img)
    if hasattr(model.transformer, "enable_adapter_layers"):
        model.transformer.enable_adapter_layers()
    n_fake_lora = model._dit_forward(z_fake_noisy_img, t_img, te_img, tm_img)
    if hasattr(model.transformer, "disable_adapter_layers"):
        model.transformer.disable_adapter_layers()
    with torch.no_grad():
        n_fake_base = model._dit_forward(z_fake_noisy_img, t_img, te_img, tm_img)
    if hasattr(model.transformer, "enable_adapter_layers"):
        model.transformer.enable_adapter_layers()
    loss_vsd_lora = F.mse_loss(n_fake_lora.float(), n_fake_base.float())

    loss_g = (
        float(tc.get("lambda_l2", 1.0)) * loss_l2
        + float(tc.get("lambda_lpips", 2.0)) * loss_lpips
        + float(tc.get("lambda_dino", 0.0)) * loss_dino
        + float(tc.get("lambda_vsd", 1.0)) * loss_vsd
        + float(tc.get("lambda_vsd_lora", 1.0)) * loss_vsd_lora
        + float(tc.get("lambda_prompt_consistency", 0.0)) * loss_prompt_consistency
    )

    if not torch.isfinite(loss_g):
        optimizer_g.zero_grad(set_to_none=True)
        return {
            "loss_g": float("nan"),
            "loss_l2": safe_value(loss_l2),
            "loss_lpips": safe_value(loss_lpips),
            "loss_dino": safe_value(loss_dino),
            "loss_vsd": safe_value(loss_vsd),
            "loss_vsd_lora": safe_value(loss_vsd_lora),
            "loss_prompt_consistency": float(loss_prompt_consistency.item())
            if torch.isfinite(loss_prompt_consistency)
            else float("nan"),
            "img_loss_bs": float(img_loss_bs),
            "lora_grad_norm": 0.0,
            "skip_non_finite": 1.0,
        }

    param_snapshot = snapshot_trainable_lora_params(model)
    optimizer_g.zero_grad(set_to_none=True)
    loss_g.backward()
    if tc.get("max_grad_norm", 0) > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.transformer.parameters() if p.requires_grad],
            tc["max_grad_norm"],
        )
    optimizer_g.step()
    if not trainable_lora_params_are_finite(model):
        restore_trainable_lora_params(model, param_snapshot)
        optimizer_g.zero_grad(set_to_none=True)
        return {
            "loss_g": float("nan"),
            "loss_l2": safe_value(loss_l2),
            "loss_lpips": safe_value(loss_lpips),
            "loss_dino": safe_value(loss_dino),
            "loss_vsd": safe_value(loss_vsd),
            "loss_vsd_lora": safe_value(loss_vsd_lora),
            "loss_prompt_consistency": float(loss_prompt_consistency.item())
            if torch.isfinite(loss_prompt_consistency)
            else float("nan"),
            "img_loss_bs": float(img_loss_bs),
            "lora_grad_norm": 0.0,
            "skip_non_finite": 1.0,
        }

    grad_norm_sq = torch.tensor(0.0, device=device)
    for p in model.transformer.parameters():
        if p.requires_grad and p.grad is not None:
            grad_norm_sq = grad_norm_sq + p.grad.detach().float().pow(2).sum()
    if grad_norm_sq.item() > 0:
        lora_grad_norm = grad_norm_sq.sqrt()
    else:
        lora_grad_norm = torch.tensor(0.0, device=device)

    return {
        "loss_g": float(loss_g.item()),
        "loss_l2": float(loss_l2.item()),
        "loss_lpips": float(loss_lpips.item()),
        "loss_dino": float(loss_dino.item()),
        "loss_vsd": float(loss_vsd.item()),
        "loss_vsd_lora": float(loss_vsd_lora.item()),
        "loss_prompt_consistency": float(loss_prompt_consistency.item()),
        "img_loss_bs": float(img_loss_bs),
        "lora_grad_norm": float(lora_grad_norm.item()),
        "skip_non_finite": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="OSEDiff-style SANA training with targeted prompts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init_lora_path",
        type=str,
        default=None,
        help="Initialize trainable LoRA from an existing adapter path/checkpoint without restoring optimizer state.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    model_path = config["model"]["sana_path"]
    print(f"[INFO] Loading SANA pipeline from: {model_path}")
    model_dtype = parse_torch_dtype(config.get("model", {}).get("weight_dtype", "fp16"), default=torch.float16)
    vae_dtype = parse_torch_dtype(config.get("model", {}).get("vae_dtype", "fp32"), default=torch.float32)
    pipe = SanaPipeline.from_pretrained(model_path, torch_dtype=model_dtype)
    vae = pipe.vae.eval().to(device=device, dtype=vae_dtype)
    transformer = pipe.transformer.to(device)
    pipe.text_encoder = pipe.text_encoder.to(device).eval()
    scheduler = pipe.scheduler

    prompt_cfg = config["prompting"]
    prompt_rng = random.Random(int(prompt_cfg.get("prompt_seed", 1234)))
    ram_prompt_start_step = int(prompt_cfg.get("ram_prompt_start_step", 0) or 0)
    prompt_extractor = None
    if ram_prompt_start_step <= 0:
        prompt_extractor = TargetedPromptExtractor(
            ram_path=prompt_cfg["ram_path"],
            ram_ft_path=prompt_cfg.get("ram_ft_path", ""),
            device=str(device),
            dtype=torch.float16,
            prompt_format=prompt_cfg.get("prompt_format", "tags_suffix"),
            instruction_template_mode=prompt_cfg.get("instruction_template_mode", "fixed"),
            instruction_template_id=int(prompt_cfg.get("instruction_template_id", 0)),
            preserve_terms=prompt_cfg.get("preserve_terms", []),
            negative_terms=prompt_cfg.get("negative_terms", []),
        )

    resume_lora_path = None
    if args.resume:
        resume_lora_path = resolve_lora_dir(args.resume)
    init_lora_path = None
    if args.init_lora_path:
        init_lora_path = resolve_lora_dir(args.init_lora_path)
    if resume_lora_path:
        from peft import PeftModel

        print(f"[RESUME] Loading LoRA adapter from: {resume_lora_path}")
        transformer = PeftModel.from_pretrained(
            transformer, resume_lora_path, is_trainable=True
        ).to(device)
        cast_trainable_lora_params_to_fp32(transformer)
    elif init_lora_path:
        from peft import PeftModel

        print(f"[INIT ] Loading initial LoRA adapter from: {init_lora_path}")
        transformer = PeftModel.from_pretrained(
            transformer, init_lora_path, is_trainable=True
        ).to(device)
        cast_trainable_lora_params_to_fp32(transformer)
    else:
        transformer = setup_lora(transformer, config)

    model = SanaSRModel(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        vae_config=vae.config,
        num_train_timesteps=config["training"]["num_train_timesteps"],
    ).to(device)

    tc = config["training"]
    lora_params = [p for p in model.transformer.parameters() if p.requires_grad]
    optimizer_g = torch.optim.AdamW(
        lora_params,
        lr=tc["learning_rate_g"],
        weight_decay=tc["weight_decay"],
        betas=(tc.get("adam_beta1", 0.9), tc.get("adam_beta2", 0.999)),
        eps=tc.get("adam_epsilon", 1e-8),
    )
    ema_decay = float(tc.get("ema_decay", 0.0))
    ema_state = None
    if ema_decay > 0:
        ema_state = init_ema_state(model)

    lambda_lpips = float(tc.get("lambda_lpips", 2.0))
    lambda_dino = float(tc.get("lambda_dino", 0.0))

    lpips_model = None
    if lambda_lpips > 0:
        print("[INFO] Loading LPIPS (vgg)")
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)

    dino_model = None
    if lambda_dino > 0:
        dino_model_name = tc.get("dino_model_name_or_path", "facebook/dinov2-large")
        dino_dtype = parse_torch_dtype(tc.get("dino_dtype", "fp32"), default=torch.float32)
        dino_layers = tc.get("dino_layer_indices", [-4, -1])
        print(f"[INFO] Loading DINO perceptual backbone: {dino_model_name}")
        dino_model = DinoPerceptualLoss(
            model_name_or_path=dino_model_name,
            layer_indices=dino_layers,
            use_cls_token=bool(tc.get("dino_use_cls_token", False)),
            normalize_features=bool(tc.get("dino_normalize_features", True)),
            dtype=dino_dtype,
        ).to(device).eval()

    train_dataset = build_train_dataset(config["data"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    print(f"[DATA] Training samples: {len(train_dataset)}")

    log_cfg = config["logging"]
    output_dir = log_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    writer = None
    if log_cfg["use_tensorboard"]:
        writer = SummaryWriter(os.path.join(output_dir, "tb_logs"))

    global_step = 0
    if args.resume:
        global_step, loaded_ema = load_checkpoint(args.resume, model, optimizer_g, device)
        if ema_state is not None:
            if loaded_ema is not None:
                ema_state = loaded_ema
            else:
                ema_state = init_ema_state(model)
        print(f"[RESUME] From step {global_step}")

    phase2_lr = tc.get("learning_rate_g_phase2")
    if ram_prompt_start_step > 0 and args.resume and global_step > ram_prompt_start_step:
        prompt_extractor = TargetedPromptExtractor(
            ram_path=prompt_cfg["ram_path"],
            ram_ft_path=prompt_cfg.get("ram_ft_path", ""),
            device=str(device),
            dtype=torch.float16,
            prompt_format=prompt_cfg.get("prompt_format", "tags_suffix"),
            instruction_template_mode=prompt_cfg.get("instruction_template_mode", "fixed"),
            instruction_template_id=int(prompt_cfg.get("instruction_template_id", 0)),
            preserve_terms=prompt_cfg.get("preserve_terms", []),
            negative_terms=prompt_cfg.get("negative_terms", []),
        )
        if phase2_lr is not None:
            for pg in optimizer_g.param_groups:
                pg["lr"] = float(phase2_lr)
            print(f"[PHASE2] Resumed inside HQ-prompt phase; LR set to {float(phase2_lr)}")

    print(f"\n{'='*60}")
    print("  OSEDiff-SANA Training")
    print(f"  LoRA params:     {sum(p.numel() for p in lora_params):,}")
    print(f"  Model dtype:     {model_dtype}")
    print(f"  VAE dtype:       {vae_dtype}")
    print(f"  Batch size:      {tc['batch_size']}")
    print(f"  LR (G):          {tc['learning_rate_g']}")
    print(f"  Lambda L2:       {tc.get('lambda_l2', 1.0)}")
    print(f"  Lambda LPIPS:    {tc.get('lambda_lpips', 2.0)}")
    print(f"  Lambda DINO:     {tc.get('lambda_dino', 0.0)}")
    print(f"  Lambda VSD:      {tc.get('lambda_vsd', 1.0)}")
    print(f"  Lambda VSDLoRA:  {tc.get('lambda_vsd_lora', 1.0)}")
    if lambda_dino > 0:
        print(f"  DINO backbone:   {tc.get('dino_model_name_or_path', 'facebook/dinov2-large')}")
        print(f"  DINO layers:     {tc.get('dino_layer_indices', [-4, -1])}")
    print(f"  EMA decay:       {tc.get('ema_decay', 0.0)}")
    print(f"  Img loss BS:     {tc.get('img_loss_batch_size', tc['batch_size'])}")
    print(f"  Gen timestep:    {tc['gen_timestep']}")
    print(f"  Noise idx range: [{tc['noise_t_min']}, {tc['noise_t_max']})")
    print(f"  Prompt source:   {prompt_cfg['prompt_source']}")
    print(f"  Prompt suffix:   {prompt_cfg.get('prompt_suffix', '')}")
    print(f"  Lambda PCons:    {tc.get('lambda_prompt_consistency', 0.0)}")
    if ram_prompt_start_step > 0:
        print(f"  Two-phase prompt: suffix-only for steps 1..{ram_prompt_start_step}, then HQ+RAM")
        if phase2_lr is not None:
            print(f"  Phase-2 LR:      {float(phase2_lr)} (after step {ram_prompt_start_step})")
    print(f"{'='*60}\n")

    max_steps = tc.get("max_steps", float("inf"))
    epoch = 0
    best_g_loss = float("inf")

    while global_step < max_steps:
        epoch += 1
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            global_step += 1
            if global_step > max_steps:
                break

            hq_01 = batch["hq"].float() * 0.5 + 0.5
            lq_01 = batch["lq"].float() * 0.5 + 0.5
            B = batch["lq"].shape[0]
            aux_prompts = None
            if ram_prompt_start_step > 0 and global_step <= ram_prompt_start_step:
                prompts = build_suffix_only_prompts(B, prompt_cfg.get("prompt_suffix", ""))
            else:
                if ram_prompt_start_step > 0 and prompt_extractor is None:
                    print(
                        f"[PHASE2] Step {global_step}: loading RAM/DAPE for HQ-targeted prompts "
                        f"(ram_prompt_start_step={ram_prompt_start_step})"
                    )
                    prompt_extractor = TargetedPromptExtractor(
                        ram_path=prompt_cfg["ram_path"],
                        ram_ft_path=prompt_cfg.get("ram_ft_path", ""),
                        device=str(device),
                        dtype=torch.float16,
                        prompt_format=prompt_cfg.get("prompt_format", "tags_suffix"),
                        instruction_template_mode=prompt_cfg.get("instruction_template_mode", "fixed"),
                        instruction_template_id=int(prompt_cfg.get("instruction_template_id", 0)),
                        preserve_terms=prompt_cfg.get("preserve_terms", []),
                        negative_terms=prompt_cfg.get("negative_terms", []),
                    )
                    if phase2_lr is not None:
                        for pg in optimizer_g.param_groups:
                            pg["lr"] = float(phase2_lr)
                        print(f"[PHASE2] learning_rate_g -> {float(phase2_lr)}")
                prompt_imgs = select_prompt_images(hq_01, lq_01, prompt_cfg["prompt_source"])
                captions = prompt_extractor.extract_tags(prompt_imgs)
                prompts, aux_prompts = build_primary_secondary_prompts(
                    captions=captions,
                    prompt_cfg=prompt_cfg,
                    rng=prompt_rng,
                )
            te, tm = encode_prompt_batch(pipe, prompts, device)
            te_aux = tm_aux = None
            if aux_prompts is not None:
                te_aux, tm_aux = encode_prompt_batch(pipe, aux_prompts, device)

            losses = train_one_step(
                model,
                lpips_model,
                dino_model,
                batch,
                te,
                tm,
                optimizer_g,
                config,
                device,
                te_aux=te_aux,
                tm_aux=tm_aux,
            )
            if losses.get("skip_non_finite", 0.0) > 0:
                msg = (
                    f"[Step {global_step}] "
                    f"SKIP_NON_FINITE "
                    f"prompt={format_prompts_for_logging(prompts)}"
                )
                pbar.set_postfix_str(msg)
                if writer:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                continue
            if ema_state is not None:
                update_ema_state(ema_state, model, ema_decay)

            if losses["loss_g"] < best_g_loss:
                best_g_loss = losses["loss_g"]
                with maybe_ema_scope(model, ema_state):
                    save_best_checkpoint(model, optimizer_g, ema_state, global_step, best_g_loss, output_dir)

            if global_step % log_cfg["log_interval"] == 0:
                msg = (
                    f"[Step {global_step}] "
                    f"G={losses['loss_g']:.4f} "
                    f"L2={losses['loss_l2']:.4f} "
                    f"LPIPS={losses['loss_lpips']:.4f} "
                    f"DINO={losses['loss_dino']:.4f} "
                    f"VSD={losses['loss_vsd']:.4f} "
                    f"VSD_LoRA={losses['loss_vsd_lora']:.4f} "
                    f"gN={losses['lora_grad_norm']:.2e} "
                    f"imgBS={int(losses['img_loss_bs'])} "
                    f"prompt={format_prompts_for_logging(prompts)}"
                )
                pbar.set_postfix_str(msg)
                if writer:
                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", v, global_step)
                    if ram_prompt_start_step > 0:
                        use = 0.0
                        if global_step > ram_prompt_start_step:
                            use = 1.0
                        writer.add_scalar(
                            "train/use_ram_prompt",
                            use,
                            global_step,
                        )

            if global_step % log_cfg["save_interval"] == 0:
                with maybe_ema_scope(model, ema_state):
                    save_checkpoint(model, optimizer_g, ema_state, global_step, output_dir)

    with maybe_ema_scope(model, ema_state):
        save_checkpoint(model, optimizer_g, ema_state, global_step, output_dir)
    if writer:
        writer.close()
    print("[DONE] OSEDiff-SANA training finished.")


if __name__ == "__main__":
    main()

