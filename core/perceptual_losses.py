from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError as e:
    raise ImportError("Please install transformers with DINOv2 support first.") from e


def _resolve_hw(processor) -> tuple[int, int]:
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, dict):
        if "height" in crop_size and "width" in crop_size:
            return int(crop_size["height"]), int(crop_size["width"])
        if "shortest_edge" in crop_size:
            edge = int(crop_size["shortest_edge"])
            return edge, edge

    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge

    return 224, 224


class DinoPerceptualLoss(nn.Module):
    """用冻结DINO特征计算感知损失。"""

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov2-large",
        layer_indices: Iterable[int] = (-4, -1),
        use_cls_token: bool = False,
        normalize_features: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.layer_indices = list(layer_indices)
        self.use_cls_token = bool(use_cls_token)
        self.normalize_features = bool(normalize_features)

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        height, width = _resolve_hw(self.processor)
        self.target_height = int(height)
        self.target_width = int(width)

        mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def _prepare(self, images_01: torch.Tensor) -> torch.Tensor:
        x = images_01.float()
        x = F.interpolate(
            x,
            size=(self.target_height, self.target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.image_mean) / self.image_std
        kind = next(self.backbone.parameters()).dtype
        x = x.to(dtype=kind)
        return x

    def _select_hidden_states(self, outputs) -> List[torch.Tensor]:
        states = outputs.hidden_states
        if states is None:
            states = [outputs.last_hidden_state]

        out = []
        n = len(states)
        for idx in self.layer_indices:
            if idx >= 0:
                i = idx
            else:
                i = n + idx
            i = max(0, min(n - 1, i))
            x = states[i]
            if x.ndim == 3 and x.shape[1] > 1 and not self.use_cls_token:
                x = x[:, 1:, :]
            if self.normalize_features:
                x = F.normalize(x.float(), dim=-1)
            else:
                x = x.float()
            out.append(x)
        return out

    def forward(self, pred_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        x = self._prepare(pred_01)
        y = self._prepare(target_01)

        a = self.backbone(x, output_hidden_states=True)
        afeat = self._select_hidden_states(a)

        with torch.no_grad():
            b = self.backbone(y, output_hidden_states=True)
            bfeat = [z.detach() for z in self._select_hidden_states(b)]

        loss = afeat[0].new_tensor(0.0)
        for a, b in zip(afeat, bfeat):
            loss = loss + F.l1_loss(a, b)
        return loss / max(1, len(afeat))
