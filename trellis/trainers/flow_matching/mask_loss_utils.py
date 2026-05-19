"""Keep-region (part) mask utilities for flow-matching velocity supervision."""

from __future__ import annotations

import torch


def masked_mse_velocity(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Masked MSE for dense grids (Stage-1 SS).

    pred/target: [B, C, D, H, W]. mask broadcastable to pred
    (e.g. [B, 1, D, H, W] or expanded).
    """
    if mask.shape != pred.shape:
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(0)
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)
    else:
        mask = mask.to(dtype=pred.dtype, device=pred.device)
    diff = (pred - target) ** 2 * mask
    denom = mask.sum() * pred.shape[1] + eps
    return diff.sum() / denom


def masked_mse_velocity_sparse(
    pred_feats: torch.Tensor,
    target_feats: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked MSE for sparse SLat feats [N, C]; mask [N] or [N, 1]."""
    if mask.dim() == pred_feats.dim() - 1:
        m = mask.unsqueeze(-1).to(dtype=pred_feats.dtype, device=pred_feats.device)
    else:
        m = mask.to(dtype=pred_feats.dtype, device=pred_feats.device)
    diff = (pred_feats - target_feats) ** 2 * m
    denom = m.sum() * pred_feats.shape[-1] + eps
    return diff.sum() / denom


def prepare_ss_keep_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Normalize mask to [B, 1, D, H, W] for ref [B, C, D, H, W].
    Accepts [D,H,W], [1,D,H,W], [B,D,H,W], [B,1,D,H,W].
    """
    m = mask.to(dtype=ref.dtype, device=ref.device)
    if m.dim() == 3:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 4:
        m = m.unsqueeze(1)
    elif m.dim() == 5:
        if m.shape[1] != 1:
            m = m[:, :1]
    else:
        raise ValueError(f"mask_keep_ss: unexpected rank {m.dim()} shape={tuple(m.shape)}")
    if m.shape[0] == 1 and ref.shape[0] != 1:
        m = m.expand(ref.shape[0], *m.shape[1:])
    elif m.shape[0] != ref.shape[0]:
        raise ValueError(
            f"mask_keep_ss batch {m.shape[0]} != ref batch {ref.shape[0]}"
        )
    return m


def should_disable_mask_for_global_style(disable: bool, edit_type) -> bool:
    if not disable or edit_type is None:
        return False
    if isinstance(edit_type, str):
        return edit_type == "global"
    if isinstance(edit_type, (list, tuple)):
        return any(x == "global" for x in edit_type)
    return False
