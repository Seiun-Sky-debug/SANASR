from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn
from PIL import Image
from torchvision import transforms


def _build_ram_transforms():
    return transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _iter_ram_search_roots() -> List[Path]:
    roots: List[Path] = []

    seesr_root = os.environ.get("SEESR_ROOT", "").strip()
    if seesr_root:
        roots.append(Path(seesr_root).expanduser())

    project_root = Path(__file__).resolve().parent.parent
    cwd = Path.cwd()
    roots.extend(
        [
            project_root / "third_party" / "SeeSR",
            project_root / "external" / "SeeSR",
            project_root / "SeeSR",
            cwd / "third_party" / "SeeSR",
            cwd / "external" / "SeeSR",
            cwd / "SeeSR",
        ]
    )

    deduped: List[Path] = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str not in seen:
            deduped.append(root)
            seen.add(root_str)
    return deduped


def _ensure_local_ram_on_path():
    for root in _iter_ram_search_roots():
        try:
            root = root.resolve()
        except Exception:
            continue
        if not root.exists():
            continue
        if not ((root / "ram").exists() or (root / "basicsr").exists() or (root / "model").exists()):
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            break


def _to_pil_list(images: torch.Tensor) -> List[Image.Image]:
    to_pil = transforms.ToPILImage()
    return [to_pil(img.cpu()) for img in images]


DEFAULT_QUALITY_SUFFIX = "clean, natural, sharp, high-quality, high-resolution"
DEFAULT_PRESERVE_TERMS = [
    "faithful structure",
    "realistic texture",
    "natural color",
    "clear local details",
]
DEFAULT_NEGATIVE_TERMS = [
    "blur",
    "noise",
    "compression artifacts",
    "oversmoothing",
    "fake texture",
    "hallucinated details",
    "unnatural distortion",
]


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
        return [v for v in parts if v]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if text:
        return [text]
    return []


def _normalize_caption(caption: str) -> str:
    text = str(caption or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def _caption_to_tags(caption: str) -> List[str]:
    text = _normalize_caption(caption)
    if not text:
        return []
    parts = re.split(r"\s*,\s*", text)
    cleaned = []
    seen = set()
    for part in parts:
        part = re.sub(r"\s+", " ", part.strip(" ,.;"))
        if not part:
            continue
        lowered = part.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(part)
    return cleaned


def _tags_to_content_phrase(tags: Sequence[str], max_tags: int = 8) -> str:
    use_tags = [str(t).strip() for t in tags if str(t).strip()][:max_tags]
    if not use_tags:
        return "the input image content"
    if len(use_tags) == 1:
        return use_tags[0]
    if len(use_tags) == 2:
        return f"{use_tags[0]} and {use_tags[1]}"
    return ", ".join(use_tags[:-1]) + f", and {use_tags[-1]}"


def _compose_quality_phrase(prompt_suffix: str) -> str:
    suffix = _normalize_caption(prompt_suffix)
    if suffix:
        return suffix
    return DEFAULT_QUALITY_SUFFIX


def build_prompt_from_caption(
    caption: str,
    prompt_suffix: str = "",
    prompt_format: str = "tags_suffix",
    template_id: int = 0,
    preserve_terms: Sequence[str] | None = None,
    negative_terms: Sequence[str] | None = None,
) -> str:
    prompt_format = str(prompt_format or "tags_suffix").strip().lower()
    caption = _normalize_caption(caption)
    suffix = _compose_quality_phrase(prompt_suffix)

    if prompt_format == "tags_only":
        return caption or suffix
    if prompt_format == "tags_suffix":
        if caption:
            return f"{caption}, {suffix}"
        return suffix
    if prompt_format != "restoration_instruction":
        raise ValueError(
            f"Unsupported prompt_format={prompt_format}. "
            "Expected one of: tags_only, tags_suffix, restoration_instruction."
        )

    tags = _caption_to_tags(caption)
    content = _tags_to_content_phrase(tags)
    preserve = ", ".join(_ensure_list(preserve_terms) or DEFAULT_PRESERVE_TERMS)
    negative = ", ".join(_ensure_list(negative_terms) or DEFAULT_NEGATIVE_TERMS)

    templates = [
        (
            f"Task: restore this image into a {suffix} photo. "
            f"Content: {content}. "
            f"Preserve: {preserve}. "
            f"Avoid: {negative}."
        ),
        (
            f"Restore a {suffix} realistic photo of {content}. "
            f"Keep {preserve}. "
            f"Remove {negative}."
        ),
        (
            f"This is an image restoration task. "
            f"Subject and scene: {content}. "
            f"Make it {suffix}. "
            f"Preserve {preserve}. "
            f"Do not introduce {negative}."
        ),
    ]
    idx = int(template_id) % len(templates)
    return templates[idx]


def build_prompts_from_captions(
    captions: Sequence[str],
    prompt_suffix: str = "",
    prompt_format: str = "tags_suffix",
    template_mode: str = "fixed",
    template_id: int = 0,
    preserve_terms: Sequence[str] | None = None,
    negative_terms: Sequence[str] | None = None,
    rng: random.Random | None = None,
) -> List[str]:
    rng = rng or random
    prompts: List[str] = []
    for caption in captions:
        cur_template_id = int(template_id)
        if str(template_mode).strip().lower() == "random":
            cur_template_id = rng.randint(0, 2)
        prompts.append(
            build_prompt_from_caption(
                caption=caption,
                prompt_suffix=prompt_suffix,
                prompt_format=prompt_format,
                template_id=cur_template_id,
                preserve_terms=preserve_terms,
                negative_terms=negative_terms,
            )
        )
    return prompts


def build_primary_secondary_prompts(
    captions: Sequence[str],
    prompt_cfg: dict,
    rng: random.Random | None = None,
) -> tuple[List[str], List[str] | None]:
    rng = rng or random
    suffix = prompt_cfg.get("prompt_suffix", "")
    preserve_terms = prompt_cfg.get("preserve_terms", [])
    negative_terms = prompt_cfg.get("negative_terms", [])

    mix_cfg = prompt_cfg.get("prompt_mix", {}) or {}
    mix_enabled = bool(mix_cfg.get("enabled", False))
    probs = {
        "restoration_instruction": float(mix_cfg.get("instruction_prob", 0.5)),
        "tags_suffix": float(mix_cfg.get("tags_suffix_prob", 0.3)),
        "tags_only": float(mix_cfg.get("tags_only_prob", 0.2)),
    }
    total_prob = sum(max(v, 0.0) for v in probs.values())
    if not mix_enabled or total_prob <= 0:
        probs = {
            str(prompt_cfg.get("prompt_format", "tags_suffix")).strip().lower(): 1.0,
        }
        total_prob = 1.0

    lambda_consistency = float(prompt_cfg.get("lambda_prompt_consistency", 0.0))
    primary_prompts: List[str] = []
    secondary_prompts = None
    if lambda_consistency > 0:
        secondary_prompts = []
    instruction_mode = prompt_cfg.get("instruction_template_mode", "fixed")
    instruction_template_id = int(prompt_cfg.get("instruction_template_id", 0))

    modes = list(probs.keys())
    weights = [max(probs[m], 0.0) / total_prob for m in modes]
    for caption in captions:
        primary_mode = rng.choices(modes, weights=weights, k=1)[0]
        primary_prompts.append(
            build_prompts_from_captions(
                [caption],
                prompt_suffix=suffix,
                prompt_format=primary_mode,
                template_mode=instruction_mode,
                template_id=instruction_template_id,
                preserve_terms=preserve_terms,
                negative_terms=negative_terms,
                rng=rng,
            )[0]
        )
        if secondary_prompts is not None:
            if primary_mode == "restoration_instruction":
                secondary_mode = "tags_suffix"
            else:
                secondary_mode = "restoration_instruction"
            if secondary_mode == "restoration_instruction":
                mode = instruction_mode
            else:
                mode = "fixed"
            secondary_prompts.append(
                build_prompts_from_captions(
                    [caption],
                    prompt_suffix=suffix,
                    prompt_format=secondary_mode,
                    template_mode=mode,
                    template_id=instruction_template_id,
                    preserve_terms=preserve_terms,
                    negative_terms=negative_terms,
                    rng=rng,
                )[0]
            )
    return primary_prompts, secondary_prompts


def encode_sana_prompts(
    tokenizer,
    text_encoder: nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    max_sequence_length: int = 300,
):
    """编码SANA文本条件"""
    tokenizer.padding_side = "right"
    text_inputs = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=prompt_attention_mask,
        )
        if isinstance(prompt_embeds, (tuple, list)):
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = prompt_embeds.last_hidden_state

    select_index = [0] + list(range(-max_sequence_length + 1, 0))
    prompt_embeds = prompt_embeds[:, select_index]
    prompt_attention_mask = prompt_attention_mask[:, select_index]
    return prompt_embeds, prompt_attention_mask


class TargetedPromptExtractor:
    """RAM/DAPE提取。"""

    def __init__(
        self,
        ram_path: str,
        ram_ft_path: str = "",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prompt_format: str = "tags_suffix",
        instruction_template_mode: str = "fixed",
        instruction_template_id: int = 0,
        preserve_terms: Sequence[str] | None = None,
        negative_terms: Sequence[str] | None = None,
    ):
        _ensure_local_ram_on_path()
        self.device = device
        self.dtype = dtype
        self.transforms = _build_ram_transforms()
        self.prompt_format = prompt_format
        self.instruction_template_mode = instruction_template_mode
        self.instruction_template_id = instruction_template_id
        self.preserve_terms = _ensure_list(preserve_terms)
        self.negative_terms = _ensure_list(negative_terms)
        self.model = self._load_model(ram_path=ram_path, ram_ft_path=ram_ft_path, device=device, dtype=dtype)

    def _load_model(self, ram_path: str, ram_ft_path: str, device: str, dtype: torch.dtype):
        if ram_ft_path:
            try:
                from ram.models.ram_lora import ram
            except ImportError as e:
                raise ImportError(
                    "Failed to import `ram.models.ram_lora`. "
                    "Install a compatible SeeSR/RAM package or set SEESR_ROOT to "
                    "an external SeeSR checkout before using targeted prompts."
                ) from e
            model = ram(
                pretrained=ram_path,
                pretrained_condition=ram_ft_path,
                image_size=384,
                vit="swin_l",
            )
        else:
            try:
                from ram.models.ram_lora import ram
            except ImportError:
                try:
                    from ram.models.ram import ram
                except ImportError as e:
                    raise ImportError(
                        "Failed to import `ram` package. Install a compatible "
                        "SeeSR/RAM package or set SEESR_ROOT first."
                    ) from e
            if "ram_lora" in ram.__module__:
                model = ram(
                    pretrained=ram_path,
                    pretrained_condition=None,
                    image_size=384,
                    vit="swin_l",
                )
            else:
                model = ram(
                    pretrained=ram_path,
                    image_size=384,
                    vit="swin_l",
                )
        model.eval()
        model.to(device=device, dtype=dtype)
        return model

    def extract_tags(self, images: torch.Tensor) -> List[str]:
        pil_list = _to_pil_list(images)
        batch = torch.stack([self.transforms(transforms.ToTensor()(img)) for img in pil_list], dim=0)
        batch = batch.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            captions, _ = self.model.generate_tag(batch)
        return [_normalize_caption(str(cap)) for cap in captions]

    def __call__(self, images: torch.Tensor, prompt_suffix: str = "") -> List[str]:
        captions = self.extract_tags(images)
        return build_prompts_from_captions(
            captions=captions,
            prompt_suffix=prompt_suffix,
            prompt_format=self.prompt_format,
            template_mode=self.instruction_template_mode,
            template_id=self.instruction_template_id,
            preserve_terms=self.preserve_terms,
            negative_terms=self.negative_terms,
        )


def select_prompt_images(
    hq_img_01: torch.Tensor,
    lq_img_01: torch.Tensor,
    prompt_source: str,
) -> torch.Tensor:
    src = prompt_source.lower()
    if src == "hq":
        return hq_img_01
    if src == "lq":
        return lq_img_01
    raise ValueError(f"Unsupported prompt_source={prompt_source}. Expected 'hq' or 'lq'.")


def build_suffix_only_prompts(batch_size: int, prompt_suffix: str) -> List[str]:
    """生成提示词。"""
    suffix = (prompt_suffix or "").strip()
    if not suffix:
        suffix = "high quality, detailed"
    return [suffix] * int(batch_size)


def format_prompts_for_logging(
    prompts: Sequence[str],
    limit: int = 2,
) -> str:
    items = [p.strip() for p in prompts[:limit]]
    if not items:
        return ""
    if len(prompts) > limit:
        items.append("...")
    return " | ".join(items)

