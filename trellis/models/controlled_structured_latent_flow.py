"""
Text-conditioned ControlNet for 3D editing — Stage 2 (Structured Latent).
Main trunk (text-cond TRELLIS SLat) is frozen; ControlNet branch injects
original SLat features via additive residuals.
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .structured_latent_flow import SparseResBlock3d


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class SLatControlNet(nn.Module):
    """
    ControlNet side branch for SLat (Structured Latent) editing.
    Mirrors the main trunk's input path + transformer blocks.
    Receives noisy x_t plus clean ori_slat (mapped to edited coords),
    produces per-layer residuals injected into the main trunk.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size)
            assert np.log2(patch_size) == len(io_block_channels)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        first_ch = io_block_channels[0] if io_block_channels else model_channels
        self.input_layer = sp.SparseLinear(in_channels, first_ch)
        self.before_proj = zero_module(sp.SparseLinear(in_channels, first_ch))

        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(
                io_block_channels, io_block_channels[1:] + [model_channels]
            ):
                self.input_blocks.extend([
                    SparseResBlock3d(chs, model_channels, out_channels=chs)
                    for _ in range(num_io_res_blocks - 1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs, model_channels, out_channels=next_chs, downsample=True
                    )
                )

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.after_proj_list = nn.ModuleList([
            zero_module(sp.SparseLinear(model_channels, model_channels))
            for _ in range(num_blocks)
        ])

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.after_proj_list.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.after_proj_list.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.before_proj.weight, 0)
        nn.init.constant_(self.before_proj.bias, 0)
        for proj in self.after_proj_list:
            nn.init.constant_(proj.weight, 0)
            nn.init.constant_(proj.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        ori_slat: sp.SparseTensor,
    ) -> List[sp.SparseTensor]:
        h = self.input_layer(x)
        h = h + self.before_proj(ori_slat)
        h = h.type(self.dtype)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.input_blocks:
            h = block(h, t_emb)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)

        controls = []
        for i, block in enumerate(self.blocks):
            h = block(h, t_emb, cond)
            controls.append(self.after_proj_list[i](h))

        return controls


class ControlledSLatFlowModel(nn.Module):
    """
    Main trunk (frozen text-cond TRELLIS SLat) + ControlNet side branch.
    forward() takes ori_slat as additional input for ControlNet conditioning.
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.controlnet = SLatControlNet(
            resolution, in_channels, model_channels, cond_channels,
            out_channels, num_blocks, num_heads, num_head_channels,
            mlp_ratio, patch_size, num_io_res_blocks, io_block_channels,
            pe_mode, use_fp16, use_checkpoint, share_mod,
            qk_rms_norm, qk_rms_norm_cross,
        )

        if self.io_block_channels is not None:
            assert int(np.log2(patch_size)) == np.log2(patch_size)
            assert np.log2(patch_size) == len(io_block_channels)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True),
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(
            in_channels,
            model_channels if io_block_channels is None else io_block_channels[0],
        )

        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(
                io_block_channels, io_block_channels[1:] + [model_channels]
            ):
                self.input_blocks.extend([
                    SparseResBlock3d(chs, model_channels, out_channels=chs)
                    for _ in range(num_io_res_blocks - 1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs, model_channels, out_channels=next_chs, downsample=True
                    )
                )

        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(
                reversed(io_block_channels),
                [model_channels] + list(reversed(io_block_channels[1:])),
            ):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels, out_channels=chs, upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels, out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks - 1)
                ])

        self.out_layer = sp.SparseLinear(
            model_channels if io_block_channels is None else io_block_channels[0],
            out_channels,
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        ori_slat: sp.SparseTensor = None,
        **kwargs,
    ) -> sp.SparseTensor:
        controls = self.controlnet(x, t, cond, ori_slat)

        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)

        for block in self.blocks:
            h = h.replace(h.feats + controls.pop(0).feats)
            h = block(h, t_emb, cond)

        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
