"""
Datasets for text-conditioned 3D editing ControlNet — Stage 2 (Structured Latent).
Provides ori_slat (original SLat mapped to edited coords), x_0 (edited SLat),
and text prompt.
"""
import os
import json
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from ..modules.sparse.basic import SparseTensor
from .structured_latent import SLatVisMixin
from ..utils.data_utils import load_balanced_group_indices


def _slat_coords_np_to_xyz(coords: np.ndarray) -> np.ndarray:
    """
    TRELLIS training npz uses coords (N, 3). Some exports (e.g. 3DEditVerse) store
    (N, 4) with batch index in column 0; collate_fn prepends batch again, so strip it.
    """
    if coords.ndim != 2:
        raise ValueError(f"SLat coords must be 2D, got {coords.shape}")
    if coords.shape[1] == 3:
        return coords
    if coords.shape[1] == 4:
        return np.ascontiguousarray(coords[:, 1:])
    raise ValueError(f"SLat coords expected 3 or 4 columns, got {coords.shape}")


def map_ori_to_edit_coords(ori_coords, ori_feats, edit_coords, return_occ=False):
    """
    Map original SLat features onto the edited coordinate system.
    For positions present in both, copy original features.
    For new positions (only in edited), fill with zeros.

    Args:
        ori_coords: [N_ori, 3] int
        ori_feats:  [N_ori, C] float
        edit_coords: [N_edit, 3] int
        return_occ: return a [N_edit, 1] overlap flag when True.
    Returns:
        mapped_feats: [N_edit, C] float on edit_coords
    """
    C = ori_feats.shape[1]
    N_edit = edit_coords.shape[0]
    mapped = np.zeros((N_edit, C), dtype=np.float32)
    occ = np.zeros((N_edit, 1), dtype=np.float32)

    ori_set = {}
    for i in range(ori_coords.shape[0]):
        key = (ori_coords[i, 0], ori_coords[i, 1], ori_coords[i, 2])
        ori_set[key] = i

    for j in range(N_edit):
        key = (edit_coords[j, 0], edit_coords[j, 1], edit_coords[j, 2])
        if key in ori_set:
            mapped[j] = ori_feats[ori_set[key]]
            occ[j] = 1

    return (mapped, occ) if return_occ else mapped


class EditingSLatTextOverfit(SLatVisMixin, Dataset):
    """
    Single-sample overfit dataset for SLat ControlNet pipeline validation.
    Loads one (ori, edit) SLat pair and maps ori features onto edited coords.
    """
    def __init__(
        self,
        roots: str,
        *,
        edit_prompt: str,
        synthetic_length: int = 65536,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.value_range = (0, 1)
        super().__init__(
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        data_dir = roots
        ori_data = np.load(os.path.join(data_dir, 'ori_latents.npz'))
        edit_data = np.load(os.path.join(data_dir, 'edit_latents.npz'))

        ori_coords = _slat_coords_np_to_xyz(ori_data['coords'])
        ori_feats_np = ori_data['feats'].astype(np.float32)
        edit_coords = _slat_coords_np_to_xyz(edit_data['coords'])
        edit_feats_np = edit_data['feats'].astype(np.float32)

        if self.normalization is not None:
            mean_np = np.array(self.normalization['mean'], dtype=np.float32).reshape(1, -1)
            std_np = np.array(self.normalization['std'], dtype=np.float32).reshape(1, -1)
            ori_feats_np = (ori_feats_np - mean_np) / std_np
            edit_feats_np = (edit_feats_np - mean_np) / std_np

        mapped_ori_feats = map_ori_to_edit_coords(ori_coords, ori_feats_np, edit_coords)

        self.edit_coords = torch.tensor(edit_coords).int()
        self.edit_feats = torch.tensor(edit_feats_np).float()
        self.ori_feats_on_edit = torch.tensor(mapped_ori_feats).float()

        self.edit_prompt = edit_prompt
        self.synthetic_length = synthetic_length

        self.loads = [self.edit_coords.shape[0]] * synthetic_length

        n_overlap = (self.ori_feats_on_edit.abs().sum(-1) > 0).sum().item()
        print(f'EditingSLatTextOverfit: 1 sample, prompt="{edit_prompt}"')
        print(f'  edit: {self.edit_coords.shape[0]} voxels, '
              f'ori_mapped: {n_overlap}/{self.edit_coords.shape[0]} overlap')

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, index):
        return {
            'coords': self.edit_coords.clone(),
            'feats': self.edit_feats.clone(),
            'ori_feats': self.ori_feats_on_edit.clone(),
            'cond': self.edit_prompt,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list = []
            feats_list = []
            ori_feats_list = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                n = b['coords'].shape[0]
                coords_list.append(
                    torch.cat([
                        torch.full((n, 1), i, dtype=torch.int32),
                        b['coords'],
                    ], dim=-1)
                )
                feats_list.append(b['feats'])
                ori_feats_list.append(b['ori_feats'])
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            ori_feats = torch.cat(ori_feats_list)

            x_0 = SparseTensor(coords=coords, feats=feats)
            x_0._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            x_0.register_spatial_cache('layout', layout)

            ori_slat = SparseTensor(coords=coords.clone(), feats=ori_feats)
            ori_slat._shape = x_0._shape
            ori_slat.register_spatial_cache('layout', layout)

            pack = {
                'x_0': x_0,
                'ori_slat': ori_slat,
                'cond': [b['cond'] for b in sub_batch],
            }
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    def __str__(self):
        return f'{self.__class__.__name__} (1 sample, len={self.synthetic_length})'


class EditingSLatText(SLatVisMixin, Dataset):
    """
    Full dataset for text-conditioned SLat editing.
    Scans data_dir for categories listed in edit_prompts.json.
    """
    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        super().__init__(
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        data_dir = roots
        cache_file = os.path.join(data_dir, 'valid_slat_instances.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.instances = json.load(f)
            print(f'EditingSLatText: {len(self.instances)} instances loaded (from cache)')
        else:
            with open(os.path.join(data_dir, prompts_file), 'r') as f:
                all_prompts = json.load(f)
            self.instances = []
            for category, items in all_prompts.items():
                for instance_id, prompt in items.items():
                    instance_dir = os.path.join(data_dir, category, str(instance_id))
                    ori_path = os.path.join(instance_dir, 'ori_latents.npz')
                    edit_path = os.path.join(instance_dir, 'edit_latents.npz')
                    if os.path.exists(ori_path) and os.path.exists(edit_path):
                        self.instances.append({
                            'dir': instance_dir,
                            'prompt': prompt,
                        })
            print(f'EditingSLatText: {len(self.instances)} instances loaded (scanned)')
        self.loads = [1000] * len(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        try:
            info = self.instances[index]
            ori_data = np.load(os.path.join(info['dir'], 'ori_latents.npz'))
            edit_data = np.load(os.path.join(info['dir'], 'edit_latents.npz'))

            ori_coords = _slat_coords_np_to_xyz(ori_data['coords'])
            ori_feats_np = ori_data['feats'].astype(np.float32)
            edit_coords = _slat_coords_np_to_xyz(edit_data['coords'])
            edit_feats_np = edit_data['feats'].astype(np.float32)

            if self.normalization is not None:
                mean_np = np.array(self.normalization['mean'], dtype=np.float32).reshape(1, -1)
                std_np = np.array(self.normalization['std'], dtype=np.float32).reshape(1, -1)
                ori_feats_np = (ori_feats_np - mean_np) / std_np
                edit_feats_np = (edit_feats_np - mean_np) / std_np

            mapped_ori_feats = map_ori_to_edit_coords(
                ori_coords, ori_feats_np, edit_coords
            )

            coords = torch.tensor(edit_coords).int()
            feats = torch.tensor(edit_feats_np).float()
            ori_feats_t = torch.tensor(mapped_ori_feats).float()

            return {
                'coords': coords,
                'feats': feats,
                'ori_feats': ori_feats_t,
                'cond': info['prompt'],
            }
        except Exception as e:
            print(f'Error loading instance {index}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    collate_fn = staticmethod(EditingSLatTextOverfit.collate_fn)

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'
