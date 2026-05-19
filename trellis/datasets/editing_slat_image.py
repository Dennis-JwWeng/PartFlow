"""
Datasets for image-conditioned 3D editing ControlNet — Stage 2 (Structured Latent).
Provides ori_slat (original SLat mapped to edited coords), x_0 (edited SLat),
and a target edit image (PIL → tensor) that the trainer's ImageConditionedMixin
will encode through DINOv2 to produce the cross-attention conditioning.

Mirrors `editing_slat_text.py` but swaps the text prompt for an image
condition. Instance directory layout (created by the prepare script):

    <instance_dir>/
        ori_latents.npz
        edit_latents.npz
        ori_image.png
        edit_image.png   # used as the conditioning image
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
from .editing_slat_text import _slat_coords_np_to_xyz, map_ori_to_edit_coords
from .editing_image_latent import _find_image_path, _load_image_as_tensor


_DEFAULT_IMAGE_SIZE = 518


class EditingSLatImageOverfit(SLatVisMixin, Dataset):
    """
    Single-sample overfit dataset for image-conditioned SLat ControlNet
    pipeline validation. Loads one (ori, edit) SLat pair, maps ori features
    onto edited coords, and uses a fixed edit image as conditioning.
    """
    def __init__(
        self,
        roots: str,
        *,
        edit_image: Optional[str] = None,
        synthetic_length: int = 65536,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        provide_case_dir: bool = False,
    ):
        self.normalization = normalization
        self.image_size = image_size
        self.value_range = (0, 1)
        self.provide_case_dir = provide_case_dir
        self._case_dir = roots
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

        mapped_ori_feats, ori_occ = map_ori_to_edit_coords(
            ori_coords, ori_feats_np, edit_coords, return_occ=True
        )

        self.edit_coords = torch.tensor(edit_coords).int()
        self.edit_feats = torch.tensor(edit_feats_np).float()
        self.ori_feats_on_edit = torch.tensor(mapped_ori_feats).float()
        self.ori_occ_on_edit = torch.tensor(ori_occ).float()

        if edit_image is None:
            edit_image = _find_image_path(data_dir, 'edit')
        if edit_image is None or not os.path.isfile(edit_image):
            raise FileNotFoundError(f'edit_image not found under {data_dir}')
        self.cond_img = _load_image_as_tensor(edit_image, self.image_size)

        self.synthetic_length = synthetic_length
        self.loads = [self.edit_coords.shape[0]] * synthetic_length

        n_overlap = (self.ori_feats_on_edit.abs().sum(-1) > 0).sum().item()
        print(f'EditingSLatImageOverfit: 1 sample, edit_image="{edit_image}"')
        print(f'  edit: {self.edit_coords.shape[0]} voxels, '
              f'ori_mapped: {n_overlap}/{self.edit_coords.shape[0]} overlap')

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, index):
        out = {
            'coords': self.edit_coords.clone(),
            'feats': self.edit_feats.clone(),
            'ori_feats': self.ori_feats_on_edit.clone(),
            'ori_occ': self.ori_occ_on_edit.clone(),
            'cond': self.cond_img.clone(),
        }
        if self.provide_case_dir:
            out['case_dir'] = self._case_dir
        return out

    @staticmethod
    def collate_fn(batch, split_size=None):  # noqa: C901
        # Local helper preserved as @staticmethod to keep the existing call site
        # working unchanged for the baseline (no case_dirs in batch).
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
            ori_occ_list = []
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
                ori_occ_list.append(b.get('ori_occ', torch.ones((n, 1), dtype=b['feats'].dtype)))
                layout.append(slice(start, start + n))
                start += n

            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            ori_feats = torch.cat(ori_feats_list)
            ori_occ_feats = torch.cat(ori_occ_list)
            cond = torch.stack([b['cond'] for b in sub_batch], dim=0)

            x_0 = SparseTensor(coords=coords, feats=feats)
            x_0._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            x_0.register_spatial_cache('layout', layout)

            ori_slat = SparseTensor(coords=coords.clone(), feats=ori_feats)
            ori_slat._shape = x_0._shape
            ori_slat.register_spatial_cache('layout', layout)

            ori_occ = SparseTensor(coords=coords.clone(), feats=ori_occ_feats)
            ori_occ._shape = torch.Size([len(group), 1])
            ori_occ.register_spatial_cache('layout', layout)

            pack = {
                'x_0': x_0,
                'ori_slat': ori_slat,
                'ori_occ': ori_occ,
                'cond': cond,
            }
            if 'case_dir' in sub_batch[0]:
                pack['case_dirs'] = [b['case_dir'] for b in sub_batch]
            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs

    def __str__(self):
        return f'{self.__class__.__name__} (1 sample, len={self.synthetic_length})'


class EditingSLatImage(SLatVisMixin, Dataset):
    """
    Full dataset for image-conditioned SLat editing.
    Scans data_dir for instances listed in edit_prompts.json (the prompt is
    ignored — only used to enumerate ids); each instance dir must contain
    ori_latents.npz, edit_latents.npz, and an edit_image.png.
    """
    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        max_num_voxels: int = 32768,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        provide_case_dir: bool = False,
    ):
        self.normalization = normalization
        self.max_num_voxels = max_num_voxels
        self.image_size = image_size
        self.value_range = (0, 1)
        self.provide_case_dir = provide_case_dir
        super().__init__(
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        data_dir = roots
        cache_file = os.path.join(data_dir, 'valid_slat_image_instances.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.instances = json.load(f)
            print(f'EditingSLatImage: {len(self.instances)} instances loaded (from cache)')
        else:
            with open(os.path.join(data_dir, prompts_file), 'r') as f:
                all_prompts = json.load(f)
            self.instances = []
            for category, items in all_prompts.items():
                for instance_id, _ in items.items():
                    instance_dir = os.path.join(data_dir, category, str(instance_id))
                    ori_path = os.path.join(instance_dir, 'ori_latents.npz')
                    edit_path = os.path.join(instance_dir, 'edit_latents.npz')
                    edit_img = _find_image_path(instance_dir, 'edit')
                    if (
                        os.path.exists(ori_path)
                        and os.path.exists(edit_path)
                        and edit_img is not None
                    ):
                        self.instances.append({'dir': instance_dir})
            print(f'EditingSLatImage: {len(self.instances)} instances loaded (scanned)')
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

            mapped_ori_feats, ori_occ = map_ori_to_edit_coords(
                ori_coords, ori_feats_np, edit_coords, return_occ=True
            )

            coords = torch.tensor(edit_coords).int()
            feats = torch.tensor(edit_feats_np).float()
            ori_feats_t = torch.tensor(mapped_ori_feats).float()
            ori_occ_t = torch.tensor(ori_occ).float()

            edit_img_path = _find_image_path(info['dir'], 'edit')
            if edit_img_path is None:
                raise FileNotFoundError(f"No edit image in {info['dir']}")
            cond_img = _load_image_as_tensor(edit_img_path, self.image_size)

            out = {
                'coords': coords,
                'feats': feats,
                'ori_feats': ori_feats_t,
                'ori_occ': ori_occ_t,
                'cond': cond_img,
            }
            if self.provide_case_dir:
                out['case_dir'] = info['dir']
            return out
        except Exception as e:
            print(f'Error loading instance {index}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    collate_fn = staticmethod(EditingSLatImageOverfit.collate_fn)

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'
