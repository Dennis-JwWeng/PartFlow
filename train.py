import os
import sys
# Cap BLAS thread pools BEFORE numpy/torch import. Two reasons:
#  1) DataLoader fork workers otherwise deadlock re-entering scipy/OpenBLAS whose
#     thread-pool locks are inherited across fork (seen as a hang in `import scipy.linalg._fblas`).
#  2) Avoids CPU oversubscription when many workers each spawn BLAS threads.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import glob
import argparse
from easydict import EasyDict as edict

import torch
import torch.multiprocessing as mp
import numpy as np
import random
from pathlib import Path

# Pre-import scipy BLAS in the parent so forked DataLoader workers inherit it from
# sys.modules and never re-run the module-level BLAS init that deadlocks after fork.
try:
    import scipy.linalg  # noqa: F401
    import scipy.linalg.cython_blas  # noqa: F401
    import scipy.linalg._fblas  # noqa: F401
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))

from trellis import models, datasets, trainers
from trellis.utils.dist_utils import setup_dist


def find_ckpt(cfg):
    # Load checkpoint
    cfg['load_ckpt'] = None
    if cfg.load_dir != '':
        if cfg.ckpt == 'latest':
            files = glob.glob(os.path.join(cfg.load_dir, 'ckpts', 'misc_*.pt'))
            if len(files) != 0:
                cfg.load_ckpt = max([
                    int(os.path.basename(f).split('step')[-1].split('.')[0])
                    for f in files
                ])
        elif cfg.ckpt == 'none':
            cfg.load_ckpt = None
        else:
            cfg.load_ckpt = int(cfg.ckpt)
    return cfg


def setup_rng(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    np.random.seed(rank)
    random.seed(rank)


def get_model_summary(model):
    model_summary = 'Parameters:\n'
    model_summary += '=' * 128 + '\n'
    model_summary += f'{"Name":<{72}}{"Shape":<{32}}{"Type":<{16}}{"Grad"}\n'
    num_params = 0
    num_trainable_params = 0
    for name, param in model.named_parameters():
        model_summary += f'{name:<{72}}{str(param.shape):<{32}}{str(param.dtype):<{16}}{param.requires_grad}\n'
        num_params += param.numel()
        if param.requires_grad:
            num_trainable_params += param.numel()
    model_summary += '\n'
    model_summary += f'Number of parameters: {num_params}\n'
    model_summary += f'Number of trainable parameters: {num_trainable_params}\n'
    return model_summary


def main(local_rank, cfg):
    # Set up distributed training
    rank = cfg.node_rank * cfg.num_gpus + local_rank
    world_size = cfg.num_nodes * cfg.num_gpus
    if world_size > 1:
        setup_dist(rank, local_rank, world_size, cfg.master_addr, cfg.master_port)

    # Seed rngs
    setup_rng(rank)

    # Load data
    dataset = getattr(datasets, cfg.dataset.name)(cfg.data_dir, **cfg.dataset.args)

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    if rank == 0:
        for name, backbone in model_dict.items():
            model_summary = get_model_summary(backbone)
            print(f'\nBackbone: {name}, parameters: {sum(p.numel() for p in backbone.parameters()):,}')
            with open(os.path.join(cfg.output_dir, f'{name}_model_summary.txt'), 'w') as fp:
                print(model_summary, file=fp)

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(model_dict, dataset, **cfg.trainer.args, output_dir=cfg.output_dir, load_dir=cfg.load_dir, step=cfg.load_ckpt)
    if cfg.smoke_steps:
        trainer.smoke_verify = True
        # A conservative fixed initial scale makes a short witness meaningful;
        # full runs retain the trainer's dynamic-scale default.
        if trainer.fp16_mode == 'inflat_all':
            trainer.log_scale = 0.0
        original_run_step = trainer.run_step
        observations = []
        gate = model_dict['denoiser'].controlnet.after_proj_list[0].weight

        def audited_step(data_list):
            before = gate.detach().float().clone()
            result = original_run_step(data_list)
            delta = (gate.detach().float() - before).norm().item()
            losses = result['loss']
            for key in ('loss', 'loss_fm', 'loss_mask_keep', 'render_loss'):
                if not np.isfinite(losses[key]):
                    raise RuntimeError(f'Smoke: nonfinite {key}')
            if losses['render_n_used'] < 1 or losses['mask_keep_ratio'] <= 0:
                raise RuntimeError('Smoke did not exercise render and nonempty keep mask')
            if delta <= 0:
                bad = [n for n, p in model_dict['denoiser'].named_parameters()
                       if p.requires_grad and p.grad is not None and not p.grad.isfinite().all()]
                raise RuntimeError(f'Smoke optimizer did not update the zero gate; losses={losses}, '
                                   f'nonfinite_gradients={bad[:8]}, scale={getattr(trainer, "log_scale", None)}')
            row = dict(step=trainer.step+1, gate_update_norm=delta, **losses)
            observations.append(row)
            print('SMOKE_STEP ' + json.dumps(row), flush=True)
            return result

        trainer.run_step = audited_step

    # Train
    if not cfg.tryrun:
        if cfg.profile:
            trainer.profile()
        else:
            trainer.run()
            if cfg.smoke_steps:
                report = dict(passed=True, steps=observations,
                              trainable_params=sum(p.numel() for p in model_dict['denoiser'].parameters() if p.requires_grad),
                              frozen_params=sum(p.numel() for p in model_dict['denoiser'].parameters() if not p.requires_grad),
                              peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                              config=str(cfg.config), data_dir=str(cfg.data_dir))
                Path(cfg.output_dir, 'smoke_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    ## io and resume
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default=os.path.join(_ROOT, 'data', 'Pxform_v1', 'training'), help='Expanded Pxform training directory')
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help='Local TRELLIS-image-large directory containing ckpts/')
    parser.add_argument('--mask_sidecar_root', type=str, default=None,
                        help='Override dataset mask_sidecar_root with real part-mask sidecars')
    parser.add_argument('--smoke_steps', type=int, default=0,
                        help='Run this many real optimizer steps with mask/render loss, batch=1')
    parser.add_argument('--auto_retry', type=int, default=3, help='Number of retries on error')
    ## debug
    parser.add_argument('--tryrun', action='store_true', help='Try run without training')
    parser.add_argument('--profile', action='store_true', help='Profile training')
    ## multi-node and multi-gpu
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs per node, default to all')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir
    opt.num_gpus = torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    ## Load config
    config = json.load(open(opt.config, 'r'))
    ## Combine arguments and config
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    if cfg.mask_sidecar_root:
        cfg.dataset.args['mask_sidecar_root'] = cfg.mask_sidecar_root
    if cfg.pretrained_dir:
        ckpts = Path(cfg.pretrained_dir).resolve() / 'ckpts'
        for key in ('controlnet_pretrain',):
            if key in cfg.trainer.args:
                cfg.trainer.args[key] = str(ckpts / Path(cfg.trainer.args[key]).name)
        decoder = cfg.trainer.args.get('render_loss_decoder')
        if decoder and decoder.get('pretrained'):
            decoder['pretrained'] = str(ckpts / Path(decoder['pretrained']).name)
    else:
        value = cfg.trainer.args.get('controlnet_pretrain')
        if value and not Path(value).is_file():
            from huggingface_hub import hf_hub_download
            parts = value.split('/')
            if len(parts) < 3:
                raise FileNotFoundError(value)
            cfg.trainer.args['controlnet_pretrain'] = hf_hub_download('/'.join(parts[:2]), '/'.join(parts[2:]))
    if cfg.smoke_steps:
        if cfg.num_gpus != 1 or cfg.num_nodes != 1:
            raise ValueError('Smoke uses one GPU; pass --num_gpus 1')
        a = cfg.trainer.args
        a.update(max_steps=cfg.smoke_steps, batch_size_per_gpu=1, batch_split=1,
                 i_print=1, i_log=1, i_save=cfg.smoke_steps, i_sample=999999,
                 dataloader_num_workers=0, prefetch_data=False, p_uncond=0.0,
                 t_schedule={'name': 'logitNormal', 'args': {'mean': -2.0, 'std': 0.1}})
        a['render_loss'].update(enabled=True, start_ratio=0.0, end_ratio=0.0,
                                every_n_steps=1, t_max=1.0, resolution=64,
                                allowed_edit_types=None, first_only=True)
        cfg.dataset.args.update(max_samples=2)
    print('\n\nConfig:')
    print('=' * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    # Prepare output directory
    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.smoke_steps:
            Path(cfg.output_dir, 'smoke_report.json').unlink(missing_ok=True)
        ## Save command and config
        with open(os.path.join(cfg.output_dir, 'command.txt'), 'w') as fp:
            print(' '.join(['python'] + sys.argv), file=fp)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as fp:
            json.dump({k: cfg[k] for k in ('models', 'dataset', 'trainer')}, fp, indent=4)

    # Run
    if cfg.auto_retry == 0:
        cfg = find_ckpt(cfg)
        if cfg.num_gpus > 1:
            mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
        else:
            main(0, cfg)
    else:
        for rty in range(cfg.auto_retry):
            try:
                cfg = find_ckpt(cfg)
                if cfg.num_gpus > 1:
                    mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
                else:
                    main(0, cfg)
                break
            except Exception as e:
                print(f'Error: {e}')
                print(f'Retrying ({rty + 1}/{cfg.auto_retry})...')
        else:
            print('ERROR: Training failed after all retries.', file=sys.stderr)
            sys.exit(1)
