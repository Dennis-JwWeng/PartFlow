from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
from .render_loss import (
    RenderLossConfig,
    schedule_lambda_render,
    compute_single_view_train_render_loss,
    load_target_image,
    load_render_camera,
    read_edit_type,
)
from .mask_loss_utils import masked_mse_velocity_sparse, should_disable_mask_for_global_style


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            drop_last=True,
            persistent_workers=(self.dataloader_persistent_workers and self.dataloader_num_workers > 0),
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)
        
    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if not isinstance(v, list) else v[:batch] for k, v in data.items()}
            noise = data['x_0'].replace(torch.randn_like(data['x_0'].feats))
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, cfg_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = sp.sparse_cat(sample_gt)
        sample = sp.sparse_cat(sample)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class TokenConcatLoRASLatFlowTrainingMixin:
    """
    Loads the pretrained SLat Flow trunk, freezes it, and trains only LoRA A/B
    plus target/source type embeddings for token-concat SLat Flow.
    """

    def __init__(
        self,
        *args,
        slat_flow_pretrain: Optional[str] = None,
        token_concat_lora_pretrain: Optional[str] = None,
        lora: Optional[dict] = None,
        **kwargs,
    ):
        self.slat_flow_pretrain_path = slat_flow_pretrain or token_concat_lora_pretrain
        self.token_concat_lora_config = lora or {}
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        denoiser = self.models.get('denoiser')
        if denoiser is None:
            return

        if self.slat_flow_pretrain_path is not None:
            from safetensors.torch import load_file
            if self.is_master:
                print(f'\n[TokenConcatLoRA/SLat] Loading SLat Flow pretrain from: {self.slat_flow_pretrain_path}')
            pretrain = load_file(self.slat_flow_pretrain_path)
            incompatible = denoiser.load_state_dict(pretrain, strict=False)
            if self.is_master:
                print(
                    f'  missing_keys: {len(getattr(incompatible, "missing_keys", []))}, '
                    f'unexpected_keys: {len(getattr(incompatible, "unexpected_keys", []))}'
                )

        for param in denoiser.parameters():
            param.requires_grad = False

        cfg = self.token_concat_lora_config
        enabled = bool(cfg.get('enabled', cfg.get('enable', True)))
        rank = int(cfg.get('rank', cfg.get('r', 64)))
        alpha = float(cfg.get('alpha', cfg.get('lora_alpha', 64)))
        dropout = float(cfg.get('dropout', cfg.get('lora_dropout', 0.05)))
        target_modules = cfg.get('target_modules', None)

        if enabled:
            if not hasattr(denoiser, 'apply_lora'):
                raise TypeError('TokenConcatLoRASLatFlowTrainingMixin requires a denoiser with apply_lora().')
            replaced = denoiser.apply_lora(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
        else:
            replaced = []

        train_type_embedding = bool(getattr(denoiser, 'use_type_embedding', True))
        if train_type_embedding:
            if hasattr(denoiser, 'type_embedding'):
                for param in denoiser.type_embedding.parameters():
                    param.requires_grad = True
            else:
                raise TypeError('TokenConcatLoRASLatFlowTrainingMixin requires denoiser.type_embedding.')

        trainable = [(name, param) for name, param in denoiser.named_parameters() if param.requires_grad]
        invalid_trainable = [
            name for name, _ in trainable
            if '.lora_A.' not in name
            and '.lora_B.' not in name
            and not (train_type_embedding and name.startswith('type_embedding.'))
        ]
        if invalid_trainable:
            raise RuntimeError(f'Unexpected trainable base parameters: {invalid_trainable[:8]}')

        n_total = sum(param.numel() for param in denoiser.parameters())
        n_train = sum(param.numel() for _, param in trainable)
        if self.is_master:
            print(
                f'[TokenConcatLoRA/SLat] LoRA enabled={enabled}, rank={rank}, alpha={alpha}, dropout={dropout}, '
                f'targeted_linear_layers={len(replaced)}'
            )
            for name in replaced[:16]:
                print(f'  lora: {name}')
            if len(replaced) > 16:
                print(f'  ... ({len(replaced) - 16} more LoRA layers)')
            trainable_desc = 'type_embedding + LoRA' if train_type_embedding else 'LoRA only'
            print(f'  Total params: {n_total:,}, Trainable ({trainable_desc}): {n_train:,}')
            summary_path = os.path.join(self.output_dir, 'token_concat_lora_slat_trainable_summary.txt')
            with open(summary_path, 'w') as fp:
                fp.write(f'total_params: {n_total}\n')
                fp.write(f'trainable_params: {n_train}\n')
                fp.write(f'lora_rank: {rank}\n')
                fp.write(f'lora_layers: {len(replaced)}\n')
                fp.write(f'train_type_embedding: {train_type_embedding}\n')
                fp.write('\n'.join(replaced))
                fp.write('\n\ntrainable_tensors:\n')
                for name, param in trainable:
                    fp.write(f'{name}\t{tuple(param.shape)}\t{param.dtype}\n')

    def get_inference_cond(self, cond, ori_slat=None, **kwargs):
        result = super().get_inference_cond(cond, **kwargs)
        if ori_slat is not None:
            result['ori_slat'] = ori_slat
        return result

    def vis_cond(self, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size=4, verbose=False):
        return {}


class EditingTokenConcatLoRAImageSparseFlowMatchingCFGTrainer(
    TokenConcatLoRASLatFlowTrainingMixin,
    ImageConditionedMixin,
    SparseFlowMatchingCFGTrainer,
):
    """Image-conditioned SLat Flow trainer for token-concat + LoRA editing."""
    pass


# Backward-compatible trainer name; implementation is token/sequence concat, not feature concat.
ConcatLoRASLatFlowTrainingMixin = TokenConcatLoRASLatFlowTrainingMixin
EditingConcatLoRAImageSparseFlowMatchingCFGTrainer = EditingTokenConcatLoRAImageSparseFlowMatchingCFGTrainer


class _RenderLossSetupMixin:
    """
    Helper mixin that owns a frozen SLat decoder + GaussianRenderer and the
    Stage2 single-view target render-loss configuration. Intentionally NEVER
    wraps any compute in torch.no_grad — gradients must flow back to v_pred.
    Decoder/renderer parameters are frozen via requires_grad=False so they
    are never updated by the optimizer.
    """

    def _init_render_loss(
        self,
        render_loss: Optional[Dict] = None,
        render_loss_decoder: Optional[Dict] = None,
        slat_normalization: Optional[Dict] = None,
    ):
        self._render_loss_cfg = RenderLossConfig.from_dict(render_loss)
        self._render_loss_dec_cfg = render_loss_decoder or {}
        norm = slat_normalization
        if norm is not None:
            mean = norm['mean']
            std = norm['std']
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(list(mean), dtype=torch.float32).reshape(1, -1)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(list(std), dtype=torch.float32).reshape(1, -1)
            norm = {'mean': mean, 'std': std}
        self._render_loss_norm = norm
        self._render_loss_decoder = None
        self._render_loss_renderer = None
        self._render_loss_pose_cache: Dict[str, tuple] = {}
        self._render_loss_image_cache: Dict[str, torch.Tensor] = {}
        self._render_loss_edit_type_cache: Dict[str, Optional[str]] = {}
        if getattr(self, 'is_master', True) and self._render_loss_cfg.enabled:
            print(f'[render_loss] enabled with config: {self._render_loss_cfg}')

    def _ensure_render_loss_modules(self, device):
        if not self._render_loss_cfg.enabled:
            return
        if self._render_loss_decoder is None:
            from ... import models as _models
            from ...renderers import GaussianRenderer
            dec_cfg = self._render_loss_dec_cfg or {}
            pretrained = dec_cfg.get(
                'pretrained',
                'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
            )
            slat_dec_path = dec_cfg.get('path')
            slat_dec_ckpt = dec_cfg.get('ckpt')
            if slat_dec_path is not None:
                cfg = json.load(open(os.path.join(slat_dec_path, 'config.json'), 'r'))
                decoder = getattr(_models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
                ckpt_path = os.path.join(slat_dec_path, 'ckpts', f'decoder_{slat_dec_ckpt}.pt')
                decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            else:
                decoder = _models.from_pretrained(pretrained)
            decoder = decoder.to(device).eval()
            for p in decoder.parameters():
                p.requires_grad = False
            self._render_loss_decoder = decoder

            renderer = GaussianRenderer()
            renderer.rendering_options.resolution = self._render_loss_cfg.resolution
            # near/far must be loose enough to fit cameras at distance ~2 used by
            # H3D view.meta.json (default 0.8/1.6 clips the object away). White bg
            # matches the H3D after.png convention so MSE/DreamSim aren't dominated
            # by a background mismatch.
            renderer.rendering_options.near = 0.1
            renderer.rendering_options.far = 10.0
            renderer.rendering_options.bg_color = (1, 1, 1)
            renderer.rendering_options.ssaa = 1
            renderer.pipe.kernel_size = 0.1
            renderer.pipe.use_mip_gaussian = True
            self._render_loss_renderer = renderer

    def _edit_type_mask(self, case_dirs):
        """List[bool] aligned with case_dirs: True iff edit_type is allowed.
        If cfg.allowed_edit_types is None, every entry is True."""
        allow = self._render_loss_cfg.allowed_edit_types
        if allow is None:
            return [True] * len(case_dirs)
        allow_set = set(allow)
        out = []
        for cd in case_dirs:
            if cd not in self._render_loss_edit_type_cache:
                self._render_loss_edit_type_cache[cd] = read_edit_type(cd)
            out.append(self._render_loss_edit_type_cache[cd] in allow_set)
        return out

    def _resolve_render_targets(self, case_dirs, device, mask=None):
        """Build per-batch tensors. Cases with mask[i]=False (or unallowed
        edit type) are filled with zeros — they will be filtered out by
        sample_mask inside compute_single_view_train_render_loss, so the
        zero entries never contribute to the loss."""
        H = self._render_loss_cfg.resolution
        imgs, exts, ints = [], [], []
        for i, cd in enumerate(case_dirs):
            ok = True if mask is None else bool(mask[i])
            if ok:
                if cd not in self._render_loss_image_cache:
                    self._render_loss_image_cache[cd] = load_target_image(
                        cd, self._render_loss_cfg, H,
                    )
                imgs.append(self._render_loss_image_cache[cd])
                if cd not in self._render_loss_pose_cache:
                    self._render_loss_pose_cache[cd] = load_render_camera(
                        cd, self._render_loss_cfg,
                    )
                extr, intr = self._render_loss_pose_cache[cd]
                exts.append(extr)
                ints.append(intr)
            else:
                imgs.append(torch.zeros(3, H, H))
                exts.append(torch.eye(4, dtype=torch.float32))
                ints.append(torch.eye(3, dtype=torch.float32))
        imgs_t = torch.stack(imgs, dim=0).to(device)         # [B,3,H,W]
        exts_t = torch.stack(exts, dim=0).to(device)         # [B,4,4]
        ints_t = torch.stack(ints, dim=0).to(device)         # [B,3,3]
        return imgs_t, exts_t, ints_t


class SparseControlNetTrainingMixin(_RenderLossSetupMixin):
    """
    Mixin that loads pretrained weights into both main trunk and ControlNet,
    then freezes the main trunk so only ControlNet is trainable.
    Sparse (SLat) version — handles SparseTensor data.
    Must appear before SparseFlowMatchingTrainer in MRO.
    """
    def __init__(
        self,
        *args,
        controlnet_pretrain: str = None,
        render_loss: Optional[Dict] = None,
        render_loss_decoder: Optional[Dict] = None,
        slat_normalization: Optional[Dict] = None,
        # ---- mask-keep velocity loss (ported from Ori3DEdit_wjw) ----
        use_mask_loss: bool = False,
        use_slat_mask_loss: bool = True,
        lambda_slat_mask: float = 0.3,
        disable_mask_loss_for_global_style: bool = True,
        mask_loss_on_keep_region_only: bool = True,
        **kwargs,
    ):
        self.controlnet_pretrain_path = controlnet_pretrain
        self.use_mask_loss = bool(use_mask_loss)
        self.use_slat_mask_loss = bool(use_slat_mask_loss)
        self.lambda_slat_mask = float(lambda_slat_mask)
        self.disable_mask_loss_for_global_style = bool(disable_mask_loss_for_global_style)
        self.mask_loss_on_keep_region_only = bool(mask_loss_on_keep_region_only)
        super().__init__(*args, **kwargs)
        self._init_render_loss(render_loss, render_loss_decoder, slat_normalization)
        if self.use_mask_loss and getattr(self, 'is_master', True):
            print(f'[mask_loss] enabled  lambda_slat_mask={self.lambda_slat_mask}  '
                  f'disable_for_global_style={self.disable_mask_loss_for_global_style}')

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        if self.controlnet_pretrain_path is None:
            return

        from safetensors.torch import load_file
        if self.is_master:
            print(f'\nLoading SLat ControlNet pretrain from: {self.controlnet_pretrain_path}')

        pretrain = load_file(self.controlnet_pretrain_path)
        denoiser = self.models['denoiser']

        denoiser.load_state_dict(pretrain, strict=False)
        denoiser.controlnet.load_state_dict(pretrain, strict=False)
        denoiser.controlnet.initialize_weights()

        for param in denoiser.parameters():
            param.requires_grad = False
        for param in denoiser.controlnet.parameters():
            param.requires_grad = True

        n_total = sum(p.numel() for p in denoiser.parameters())
        n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        if self.is_master:
            print(f'  Total params: {n_total:,}, Trainable (ControlNet): {n_train:,}')

    def training_losses(self, x_0, cond=None, ori_slat=None, case_dirs=None, **kwargs):
        # Pop fields that the denoiser doesn't accept (avoid TypeError on forward)
        mask_keep_slat = kwargs.pop("mask_keep_slat", None)
        edit_type = kwargs.pop("edit_type", None)
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, ori_slat=ori_slat, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss_fm"] = terms["mse"]
        terms["loss"] = terms["mse"]

        # ---- optional mask-keep velocity loss (Stage2 SLat) ---------------- #
        use_keep = (
            self.use_mask_loss
            and self.use_slat_mask_loss
            and ori_slat is not None
            and mask_keep_slat is not None
            and mask_keep_slat.numel() == pred.feats.shape[0]
            and not should_disable_mask_for_global_style(
                self.disable_mask_loss_for_global_style, edit_type
            )
        )
        if use_keep:
            v_ori = self.get_v(ori_slat, noise, t)
            mk = mask_keep_slat.to(device=pred.feats.device, dtype=pred.feats.dtype)
            terms["loss_mask_keep"] = masked_mse_velocity_sparse(pred.feats, v_ori.feats, mk)
            terms["mask_keep_ratio"] = mk.mean().detach()
            terms["loss"] = terms["loss"] + self.lambda_slat_mask * terms["loss_mask_keep"]
        elif self.use_mask_loss:
            terms["loss_mask_keep"] = pred.feats.new_tensor(0.0)
            terms["mask_keep_ratio"] = pred.feats.new_tensor(0.0)

        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": float(mse_per_instance[time_bin == i].mean())}

        # ---- optional Stage2 single-view target render loss ---------------- #
        rcfg = getattr(self, '_render_loss_cfg', None)
        if rcfg is not None and rcfg.enabled:
            # NOTE: self.step += 1 happens AFTER run_step in base.py — i.e. the
            # current call corresponds to the WILL-BE step counter (self.step + 1)
            # that gets printed in stdout / written to log. Using (step + 1) here
            # makes the every_n_steps gate align with the i_print boundary so the
            # render-loss diagnostics are actually visible in stdout when
            # every_n_steps divides i_print.
            step = int(getattr(self, 'step', 0) or 0) + 1
            gate = (step % max(int(rcfg.every_n_steps), 1) == 0)
            if gate and (t < rcfg.t_max).any().item():
                if case_dirs is None:
                    raise RuntimeError(
                        '[render_loss] enable_train_render_loss=True but the dataset '
                        'did not provide `case_dirs`. Set provide_case_dir=True on the '
                        'dataset or disable render loss.'
                    )
                self._ensure_render_loss_modules(x_0.feats.device)
                edit_mask = self._edit_type_mask(case_dirs)
                if not any(edit_mask):
                    # no batch sample has an allowed edit type — skip render loss entirely
                    return terms, {}
                target_imgs, extr, intr = self._resolve_render_targets(
                    case_dirs, x_0.feats.device, mask=edit_mask,
                )
                rd = compute_single_view_train_render_loss(
                    x_t=x_t,
                    v_pred=pred,
                    t=t,
                    decoder=self._render_loss_decoder,
                    renderer=self._render_loss_renderer,
                    target_images=target_imgs,
                    extrinsics=extr,
                    intrinsics=intr,
                    cfg=rcfg,
                    decoder_normalization=self._render_loss_norm,
                    sample_mask=edit_mask,
                )
                lam = schedule_lambda_render(step, int(getattr(self, 'max_steps', 0)), rcfg)
                added = 0.0
                if rd['n_used'] > 0 and lam > 0:
                    terms['loss'] = terms['loss'] + lam * rd['loss']
                    added = float((lam * rd['loss']).detach().item())
                # Diagnostics: separate FM vs render-loss contributions and the
                # ratio of the WEIGHTED render loss to the FM loss (i.e. how
                # much of the total gradient is coming from render loss).
                fm_val = float(terms['mse'].detach().item()) if hasattr(terms['mse'], 'detach') else float(terms['mse'])
                terms['fm_loss'] = fm_val
                terms['render_loss'] = rd['loss'].detach()
                terms['render_loss_weighted'] = added
                terms['render_loss_mse'] = rd['loss_mse']
                terms['render_loss_dreamsim'] = rd['loss_dreamsim']
                terms['render_lambda'] = float(lam)
                terms['render_n_used'] = int(rd['n_used'])
                terms['render_elapsed'] = float(rd['elapsed'])
                terms['render_to_fm_ratio'] = (added / max(fm_val, 1e-12)) if fm_val > 0 else 0.0

        return terms, {}

    def get_inference_cond(self, cond, ori_slat=None, **kwargs):
        result = super().get_inference_cond(cond, **kwargs)
        if ori_slat is not None:
            result['ori_slat'] = ori_slat
        return result

    def vis_cond(self, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def run_snapshot(self, num_samples, batch_size=4, verbose=False):
        return {}


class EditingControlNetTextSparseFlowMatchingCFGTrainer(
    SparseControlNetTrainingMixin,
    TextConditionedMixin,
    SparseFlowMatchingCFGTrainer,
):
    """
    Text-conditioned SLat ControlNet trainer for 3D editing (Stage 2).
    Loads text-cond TRELLIS SLat pretrained weights, freezes main trunk,
    trains ControlNet to inject original SLat conditioning.
    """
    pass


class EditingControlNetImageSparseFlowMatchingCFGTrainer(
    SparseControlNetTrainingMixin,
    ImageConditionedMixin,
    SparseFlowMatchingCFGTrainer,
):
    """
    Image-conditioned SLat ControlNet trainer for 3D editing (Stage 2).
    Loads image-cond TRELLIS SLat pretrained weights, freezes main trunk,
    trains ControlNet to inject original SLat conditioning. The cross-attn
    cond is the DINOv2 patch tokens of a target/reference image.
    """
    pass
