from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from ..modules.spatial import patchify, unpatchify
from .sparse_structure_flow import SparseStructureFlowModel
from .structured_latent_flow import SLatFlowModel


class InputConcatSparseStructureFlowModel(SparseStructureFlowModel):
    """
    Sparse-structure flow model that injects a 3D condition by concatenating
    target/source tokens along the sequence axis.

    x_t and ori_voxel keep the same feature width C. They are patchified and
    tokenized separately as [B, N_t, C_h] / [B, N_o, C_h], concatenated as
    [B, N_t + N_o, C_h], and only target tokens are decoded back to the edited
    sparse-structure latent.
    """

    def __init__(
        self,
        *args,
        token_concat_condition: Optional[dict] = None,
        concat_condition: Optional[dict] = None,
        **kwargs,
    ):
        token_concat_condition = token_concat_condition or concat_condition or {}
        self.use_type_embedding = bool(token_concat_condition.get("use_type_embedding", True))
        self.target_type_id = int(token_concat_condition.get("target_type_id", 0))
        self.source_type_id = int(token_concat_condition.get("source_type_id", 1))

        super().__init__(*args, **kwargs)

        self.type_embedding = nn.Embedding(2, self.model_channels)
        nn.init.zeros_(self.type_embedding.weight)
        self.debug_last_shapes = {}

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        return self.input_layer(h)

    def _add_position_and_type(self, h: torch.Tensor, type_id: int) -> torch.Tensor:
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None].to(dtype=h.dtype, device=h.device)
        if self.use_type_embedding:
            type_emb = self.type_embedding.weight[type_id].to(dtype=h.dtype, device=h.device)
            h = h + type_emb.view(1, 1, -1)
        return h

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        ori_voxel: Optional[torch.Tensor] = None,
        input_concat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        concat = input_concat if input_concat is not None else ori_voxel
        if concat is None:
            raise ValueError("InputConcatSparseStructureFlowModel requires ori_voxel or input_concat.")
        expected_x_shape = [x.shape[0], self.in_channels, *[self.resolution] * 3]
        if [*x.shape] != expected_x_shape:
            raise ValueError(f"Input shape mismatch, got {x.shape}, expected {expected_x_shape}.")
        if concat.shape[0] != x.shape[0] or concat.shape[1] != x.shape[1] or concat.shape[2:] != x.shape[2:]:
            raise ValueError(
                f"concat shape {concat.shape} must match x_t batch/channel/spatial shape {x.shape} "
                "for token/sequence concat."
            )

        target_h = self._tokenize(x)
        source_h = self._tokenize(concat.to(dtype=x.dtype, device=x.device))
        target_h = self._add_position_and_type(target_h, self.target_type_id)
        source_h = self._add_position_and_type(source_h, self.source_type_id)

        h = torch.cat([target_h, source_h], dim=1)
        self.debug_last_shapes = {
            "target_tokens": int(target_h.shape[1]),
            "source_tokens": int(source_h.shape[1]),
            "concat_tokens": int(h.shape[1]),
            "feature_dim": int(h.shape[-1]),
        }

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = h[:, : target_h.shape[1], :].type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        return h


class InputConcatSLatFlowModel(SLatFlowModel):
    """
    SLat flow model that injects an aligned sparse latent condition by
    concatenating it to x_t.feats before the main trunk.
    """

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        ori_slat: Optional[sp.SparseTensor] = None,
        input_concat: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        concat = input_concat if input_concat is not None else ori_slat
        if concat is None:
            raise ValueError("InputConcatSLatFlowModel requires ori_slat or input_concat.")
        if concat.feats.shape[0] != x.feats.shape[0]:
            raise ValueError(
                f"concat feat count {concat.feats.shape[0]} does not match x_t feat count {x.feats.shape[0]}."
            )

        feats = torch.cat([x.feats, concat.feats.to(dtype=x.feats.dtype, device=x.feats.device)], dim=-1)
        x = x.replace(feats)
        return super().forward(x, t, cond, **kwargs)
