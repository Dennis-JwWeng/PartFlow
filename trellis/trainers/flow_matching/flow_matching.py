from typing import *
import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
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
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": float(mse_per_instance[time_bin == i].mean())}

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
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
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

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
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


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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





class Stage1SparseStructureLoraFinetuneMixin:
    """
    Stage-1 (sparse structure) denoiser: optional TRELLIS pretrain + **standard HuggingFace PEFT LoRA**.

    Uses `LoraConfig` + `get_peft_model` (no MoE). Scope is only `models["denoiser"]` — SLat / stage 2
    is untouched.
    """

    def __init__(
        self,
        *args,
        stage1_pretrain: Optional[str] = None,
        stage1_lora: Optional[dict] = None,
        **kwargs,
    ):
        self.stage1_pretrain_path = stage1_pretrain
        self.stage1_lora_config = stage1_lora if stage1_lora is not None else {}
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)

        cfg = self.stage1_lora_config
        want_lora = bool(cfg.get("enable", False))
        if self.stage1_pretrain_path is None and not want_lora:
            return

        denoiser = self.models.get("denoiser")
        if denoiser is None:
            if self.is_master:
                print('[Stage1Lora] No model key "denoiser" — skipping stage-1 LoRA setup.', flush=True)
            return

        if self.stage1_pretrain_path:
            from safetensors.torch import load_file

            if self.is_master:
                print(f"\n[Stage1Lora] Loading stage-1 pretrain: {self.stage1_pretrain_path}", flush=True)
            pretrain = load_file(self.stage1_pretrain_path)
            model_state = denoiser.state_dict()
            skipped = []
            filtered_pretrain = {}
            for key, value in pretrain.items():
                if key in model_state and value.shape != model_state[key].shape:
                    skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                    continue
                filtered_pretrain[key] = value
            incompatible = denoiser.load_state_dict(filtered_pretrain, strict=False)
            if self.is_master:
                miss = getattr(incompatible, "missing_keys", [])
                unex = getattr(incompatible, "unexpected_keys", [])
                print(f"  missing_keys: {len(miss)}, unexpected_keys: {len(unex)}, shape_skipped: {len(skipped)}", flush=True)
                for key, src_shape, dst_shape in skipped[:8]:
                    print(f"  shape_skipped: {key}: {src_shape} -> {dst_shape}", flush=True)
                if len(skipped) > 8:
                    print(f"  ... ({len(skipped) - 8} more shape-skipped keys)", flush=True)

        if not want_lora:
            return

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                'Stage-1 LoRA requires the `peft` package. Install e.g. `pip install "peft>=0.7"`.' 
            ) from e

        r = int(cfg.get("r", 16))
        lora_alpha = cfg.get("alpha", cfg.get("lora_alpha", r))
        lora_alpha = float(lora_alpha)
        lora_dropout = float(cfg.get("lora_dropout", 0.0))
        use_rslora = bool(cfg.get("use_rslora", True))
        target_modules = cfg.get(
            "target_modules",
            ["to_qkv", "to_q", "to_kv", "to_out"],
        )
        modules_to_save = cfg.get("modules_to_save", ["input_layer"])
        if isinstance(modules_to_save, list) and len(modules_to_save) == 0:
            modules_to_save = None

        if self.is_master:
            print(
                f"\n[Stage1Lora/PEFT] LoraConfig: r={r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}, "
                f"use_rslora={use_rslora}, target_modules={list(target_modules)}, modules_to_save={modules_to_save}"
            )

        # Crown SFT uses RSLoRA; keep configurable for older `peft` builds.
        try:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=int(lora_alpha) if lora_alpha == int(lora_alpha) else lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_rslora=use_rslora,
                target_modules=list(target_modules),
                modules_to_save=list(modules_to_save) if modules_to_save else None,
            )
        except TypeError:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=int(lora_alpha) if lora_alpha == int(lora_alpha) else lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(target_modules),
                modules_to_save=list(modules_to_save) if modules_to_save else None,
            )
            if self.is_master and use_rslora:
                print("  [Stage1Lora/PEFT] Note: `use_rslora` not supported by this `peft` version; disabled.")

        denoiser_peft = get_peft_model(denoiser, lora_config)
        self.models["denoiser"] = denoiser_peft
        if self.is_master and hasattr(denoiser_peft, "print_trainable_parameters"):
            denoiser_peft.print_trainable_parameters()

        if self.is_master:
            trainable = [(name, param) for name, param in denoiser_peft.named_parameters() if param.requires_grad]
            total_params = sum(param.numel() for param in denoiser_peft.parameters())
            trainable_params = sum(param.numel() for _, param in trainable)
            print(
                f"[Stage1Lora/PEFT] trainable tensors={len(trainable)}, "
                f"trainable_params={trainable_params:,}, total_params={total_params:,}",
                flush=True,
            )
            summary_path = os.path.join(self.output_dir, "denoiser_lora_trainable_summary.txt")
            with open(summary_path, "w") as fp:
                fp.write(f"total_params: {total_params}\n")
                fp.write(f"trainable_params: {trainable_params}\n")
                fp.write(f"trainable_tensors: {len(trainable)}\n")
                fp.write("\n")
                for name, param in trainable:
                    fp.write(f"{name}\t{tuple(param.shape)}\t{param.dtype}\n")
            print(f"[Stage1Lora/PEFT] wrote trainable summary: {summary_path}", flush=True)


class ImageConditionedFlowMatchingCFGStage1LoRATrainer(
    Stage1SparseStructureLoraFinetuneMixin,
    ImageConditionedFlowMatchingCFGTrainer,
):
    """
    Image-conditioned flow matching with CFG, plus optional **stage-1** pretrain + **HuggingFace PEFT LoRA**
    on the SS denoiser (no MoE).
    """

    pass

class ControlNetTrainingMixin:
    """
    Mixin that loads pretrained weights into both main trunk and ControlNet,
    then freezes the main trunk so only ControlNet is trainable.
    Must appear before FlowMatchingTrainer in MRO.
    """
    def __init__(self, *args, controlnet_pretrain: str = None, **kwargs):
        self.controlnet_pretrain_path = controlnet_pretrain
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        """Load pretrained weights and freeze main trunk before optimizer creation."""
        super()._pre_init_models(**kwargs)
        if self.controlnet_pretrain_path is None:
            return

        from safetensors.torch import load_file
        if self.is_master:
            print(f'\nLoading ControlNet pretrain from: {self.controlnet_pretrain_path}')

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

    def training_losses(self, x_0, cond=None, ori_voxel=None, **kwargs):
        """Override to pass ori_voxel to the denoiser."""
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, ori_voxel=ori_voxel, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": float(mse_per_instance[time_bin == i].mean())}

        return terms, {}

    def get_inference_cond(self, cond, ori_voxel=None, **kwargs):
        """Pass ori_voxel through to the sampler."""
        result = super().get_inference_cond(cond, **kwargs)
        if ori_voxel is not None:
            result['ori_voxel'] = ori_voxel
        return result

    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass

    def vis_cond(self, **kwargs):
        return {}


class EditingControlNetTextFlowMatchingCFGTrainer(
    ControlNetTrainingMixin,
    TextConditionedMixin,
    FlowMatchingCFGTrainer,
):
    """
    Text-conditioned ControlNet trainer for 3D editing.
    Loads text-cond TRELLIS pretrained weights, freezes main trunk,
    trains ControlNet to inject original voxel conditioning.
    """
    pass


class EditingControlNetImageFlowMatchingCFGTrainer(
    ControlNetTrainingMixin,
    ImageConditionedMixin,
    FlowMatchingCFGTrainer,
):
    """
    Image-conditioned ControlNet trainer for 3D editing — Stage 1 (SS).
    Loads image-cond TRELLIS pretrained weights, freezes main trunk,
    trains ControlNet to inject original voxel conditioning. The cross-attn
    cond is the DINOv2 patch tokens of a target/reference image.
    """
    pass
