"""LoRA adapters for SAM3, shared by scripts/finetune_sam3_lora.py and the backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        lora_output = F.linear(self.dropout(inputs), self.lora_A)
        lora_output = F.linear(lora_output, self.lora_B)
        return base_output + lora_output * self.scale


def should_wrap_module(
    module_name: str,
    prefixes: Sequence[str],
    suffixes: Sequence[str],
) -> bool:
    return any(module_name.startswith(prefix) for prefix in prefixes) and any(
        module_name.endswith(suffix) for suffix in suffixes
    )


def inject_lora(
    model: nn.Module,
    prefixes: Sequence[str],
    suffixes: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    replaced: list[str] = []
    for parent_name, parent in model.named_modules():
        # nn.MultiheadAttention reads out_proj.weight directly instead of calling it,
        # so its projection cannot be swapped for a LoRA wrapper.
        if isinstance(parent, nn.MultiheadAttention):
            continue
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if not should_wrap_module(full_name, prefixes, suffixes):
                continue
            setattr(
                parent,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced.append(full_name)

    if not replaced:
        raise RuntimeError(
            "No linear modules matched the requested LoRA prefixes and suffixes."
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A" in name or ".lora_B" in name
    }


def build_lora_sam3_model(
    lora_checkpoint: Path,
    *,
    checkpoint_path: str | None,
    bpe_path: str | None,
    load_from_hf: bool,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Builds the base SAM3 image model and applies a saved LoRA checkpoint for inference."""
    from sam3 import build_sam3_image_model

    payload = torch.load(lora_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        load_from_HF=load_from_hf,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    inject_lora(
        model,
        prefixes=config["lora_prefixes"],
        suffixes=config["lora_suffixes"],
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
        dropout=0.0,
    )
    weights = payload["lora_state_dict"]
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing_lora = [name for name in missing if "lora_" in name]
    if missing_lora or unexpected:
        raise RuntimeError(
            f"LoRA checkpoint {lora_checkpoint} does not match the model: "
            f"missing={missing_lora[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device)
    model.eval()
    return model, {"epoch": payload.get("epoch"), "config": config}
