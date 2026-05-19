from typing import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import sparse as sp
from .structured_latent_flow import SLatFlowModel


class LoRALinear(nn.Module):
    """Standard LoRA wrapper for dense Linear and TRELLIS SparseLinear."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        for param in self.base_layer.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _delta(self, feats: torch.Tensor) -> torch.Tensor:
        lora_dtype = self.lora_A.weight.dtype
        delta = self.lora_B(self.lora_A(self.dropout(feats.to(lora_dtype)))) * self.scaling
        return delta.to(feats.dtype)

    def forward(self, x):
        if isinstance(x, sp.SparseTensor):
            out = self.base_layer(x)
            return out.replace(out.feats + self._delta(x.feats).to(out.feats.dtype))
        return self.base_layer(x) + self._delta(x)


class TokenConcatLoRASLatFlowModel(SLatFlowModel):
    """
    Token-concat SLat Flow baseline.

    x_t and ori_slat keep the same feature width C. They pass through the shared
    SLat sparse input/tokenization path separately, receive target/source type
    embeddings, are concatenated along the sparse token axis per batch, and then
    go through the original SLat Flow Transformer blocks. Only target tokens are
    selected for the output decoder and flow-matching loss.
    """

    def __init__(
        self,
        *args,
        token_concat_condition: Optional[dict] = None,
        lora: Optional[dict] = None,
        concat_condition: Optional[dict] = None,
        **kwargs,
    ):
        token_concat_condition = token_concat_condition or concat_condition or {}
        self.use_type_embedding = bool(token_concat_condition.get("use_type_embedding", True))
        self.target_type_id = int(token_concat_condition.get("target_type_id", 0))
        self.source_type_id = int(token_concat_condition.get("source_type_id", 1))
        self.lora_config = lora or {}

        super().__init__(*args, **kwargs)

        self.type_embedding = nn.Embedding(2, self.model_channels)
        nn.init.zeros_(self.type_embedding.weight)

        self.debug_last_shapes = {}

    def _encode_tokens(self, x: sp.SparseTensor, t_emb: torch.Tensor) -> Tuple[sp.SparseTensor, List[torch.Tensor]]:
        h = self.input_layer(x).type(self.dtype)
        skips = []
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        return h, skips

    def _add_position_and_type(self, h: sp.SparseTensor, type_id: int) -> sp.SparseTensor:
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        if self.use_type_embedding:
            type_emb = self.type_embedding.weight[type_id].to(dtype=h.feats.dtype, device=h.feats.device)
            h = h.replace(h.feats + type_emb.unsqueeze(0))
        return h

    @staticmethod
    def _batch_size(x: sp.SparseTensor, y: sp.SparseTensor) -> int:
        return max(int(x.shape[0]), int(y.shape[0]))

    def _concat_token_sequence(
        self,
        target: sp.SparseTensor,
        source: sp.SparseTensor,
    ) -> Tuple[sp.SparseTensor, List[int], List[slice]]:
        if target.feats.shape[-1] != source.feats.shape[-1]:
            raise ValueError(
                f"target/source token feature dim must match, got "
                f"{target.feats.shape[-1]} and {source.feats.shape[-1]}."
            )

        batch_size = self._batch_size(target, source)
        coords_chunks = []
        feats_chunks = []
        target_counts = []
        combined_layout = []
        start = 0
        for batch_idx in range(batch_size):
            target_slice = target.layout[batch_idx]
            source_slice = source.layout[batch_idx]
            target_counts.append(target_slice.stop - target_slice.start)

            batch_coords = torch.cat(
                [target.coords[target_slice], source.coords[source_slice].to(device=target.coords.device)],
                dim=0,
            )
            batch_feats = torch.cat(
                [target.feats[target_slice], source.feats[source_slice].to(dtype=target.feats.dtype, device=target.feats.device)],
                dim=0,
            )
            coords_chunks.append(batch_coords)
            feats_chunks.append(batch_feats)
            stop = start + batch_feats.shape[0]
            combined_layout.append(slice(start, stop))
            start = stop

        combined = sp.SparseTensor(
            coords=torch.cat(coords_chunks, dim=0),
            feats=torch.cat(feats_chunks, dim=0),
            shape=torch.Size([batch_size, target.feats.shape[-1]]),
            layout=combined_layout,
        )
        return combined, target_counts, combined_layout

    @staticmethod
    def _select_target_tokens(
        combined: sp.SparseTensor,
        target_template: sp.SparseTensor,
        target_counts: Sequence[int],
        combined_layout: Sequence[slice],
    ) -> sp.SparseTensor:
        target_feats = []
        for batch_idx, target_count in enumerate(target_counts):
            combined_slice = combined_layout[batch_idx]
            target_feats.append(combined.feats[combined_slice.start: combined_slice.start + target_count])
        return target_template.replace(torch.cat(target_feats, dim=0))

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        ori_slat: Optional[sp.SparseTensor] = None,
        input_concat: Optional[sp.SparseTensor] = None,
        **kwargs,
    ) -> sp.SparseTensor:
        source = input_concat if input_concat is not None else ori_slat
        if source is None:
            raise ValueError("TokenConcatLoRASLatFlowModel requires ori_slat or input_concat source tokens.")
        if x.feats.shape[-1] != source.feats.shape[-1]:
            raise ValueError(
                f"x_t and ori_slat feature dim must match before tokenization, got "
                f"{x.feats.shape[-1]} and {source.feats.shape[-1]}."
            )
        if x.shape[0] != source.shape[0]:
            raise ValueError(f"x_t and ori_slat batch size must match, got {x.shape[0]} and {source.shape[0]}.")

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        target_h, target_skips = self._encode_tokens(x, t_emb)
        source_h, _ = self._encode_tokens(source, t_emb)
        target_h = self._add_position_and_type(target_h, self.target_type_id)
        source_h = self._add_position_and_type(source_h, self.source_type_id)

        tokens, target_counts, combined_layout = self._concat_token_sequence(target_h, source_h)
        self.debug_last_shapes = {
            "target_tokens": int(target_h.feats.shape[0]),
            "source_tokens": int(source_h.feats.shape[0]),
            "concat_tokens": int(tokens.feats.shape[0]),
            "feature_dim": int(tokens.feats.shape[-1]),
        }

        h = tokens
        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = self._select_target_tokens(h, target_h, target_counts, combined_layout)

        for block, skip in zip(self.out_blocks, reversed(target_skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h

    def _is_lora_candidate(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, LoRALinear) or not isinstance(module, nn.Linear):
            return False
        if not name.startswith("blocks."):
            return False
        return ".self_attn." in name or ".cross_attn." in name or ".mlp.mlp." in name

    @staticmethod
    def _target_aliases(target: str) -> Tuple[str, ...]:
        aliases = {
            "q_proj": ("to_q",),
            "k_proj": ("to_kv",),
            "v_proj": ("to_kv",),
            "out_proj": ("to_out",),
            "proj": ("to_out",),
            "qkv": ("to_qkv",),
            "fc1": ("mlp.mlp.0",),
            "fc2": ("mlp.mlp.2",),
        }
        return aliases.get(target, (target,))

    def _matches_lora_targets(self, name: str, target_modules: Optional[Sequence[str]]) -> bool:
        if not target_modules:
            return True
        leaf = name.rsplit(".", 1)[-1]
        for target in target_modules:
            for alias in self._target_aliases(str(target)):
                if leaf == alias or name.endswith(alias) or name.endswith(f".{alias}"):
                    return True
        return False

    def apply_lora(
        self,
        rank: int = 64,
        alpha: float = 64,
        dropout: float = 0.05,
        target_modules: Optional[Sequence[str]] = None,
    ) -> List[str]:
        replacements = []
        for name, module in list(self.named_modules()):
            if self._is_lora_candidate(name, module) and self._matches_lora_targets(name, target_modules):
                replacements.append((name, module))
        if not replacements and target_modules:
            for name, module in list(self.named_modules()):
                if self._is_lora_candidate(name, module):
                    replacements.append((name, module))

        replaced_names = []
        for name, module in replacements:
            parent_name, child_name = name.rsplit(".", 1)
            parent = self.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=float(alpha), dropout=float(dropout)))
            replaced_names.append(name)
        return replaced_names


ConcatLoRASLatFlowModel = TokenConcatLoRASLatFlowModel
