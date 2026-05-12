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

        image_mean = getattr(self.processor, "image_mean", [0.485, 0.456, 0.406])
        image_std = getattr(self.processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

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
        model_dtype = next(self.backbone.parameters()).dtype
        x = x.to(dtype=model_dtype)
        return x

    def _select_hidden_states(self, outputs) -> List[torch.Tensor]:
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            hidden_states = [outputs.last_hidden_state]

        picked = []
        total = len(hidden_states)
        for idx in self.layer_indices:
            real_idx = idx if idx >= 0 else total + idx
            real_idx = max(0, min(total - 1, real_idx))
            feat = hidden_states[real_idx]
            if feat.ndim == 3 and feat.shape[1] > 1 and not self.use_cls_token:
                feat = feat[:, 1:, :]
            if self.normalize_features:
                feat = F.normalize(feat.float(), dim=-1)
            else:
                feat = feat.float()
            picked.append(feat)
        return picked

    def forward(self, pred_01: torch.Tensor, target_01: torch.Tensor) -> torch.Tensor:
        pred_in = self._prepare(pred_01)
        target_in = self._prepare(target_01)

        pred_out = self.backbone(pred_in, output_hidden_states=True)
        pred_feats = self._select_hidden_states(pred_out)

        with torch.no_grad():
            target_out = self.backbone(target_in, output_hidden_states=True)
            target_feats = [feat.detach() for feat in self._select_hidden_states(target_out)]

        loss = pred_feats[0].new_tensor(0.0)
        for pred_feat, target_feat in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pred_feat, target_feat)
        return loss / max(1, len(pred_feats))
