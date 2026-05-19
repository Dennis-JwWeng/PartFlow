"""
Datasets for text-conditioned 3D editing with ControlNet.
Provides ori_voxel (original ss latent), x_0 (edited ss latent), and text prompt.
"""
import os
import json
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from .sparse_structure_latent import SparseStructureLatentVisMixin

def _load_ss_latent_array(npz_path: str) -> np.ndarray:
    """Load dense SS latent from npz. TRELLIS uses key 'mean'; 3DEditVerse uses 'ss'."""
    data = np.load(npz_path)
    if 'mean' in data.files:
        return data['mean']
    if 'ss' in data.files:
        return data['ss']
    raise KeyError(
        f"{npz_path}: expected 'mean' or 'ss' in npz, got {list(data.files)}"
    )


class EditingTextSparseStructureLatent(SparseStructureLatentVisMixin, Dataset):
    """
    Full dataset for text-conditioned editing.
    Scans data_dir for categories listed in edit_prompts.json,
    loads ori/edit ss latents and the corresponding text prompt.
    """
    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.value_range = (0, 1)
        super().__init__(
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

        data_dir = roots
        cache_file = os.path.join(data_dir, 'valid_ss_instances.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.instances = json.load(f)
            print(f'EditingTextSparseStructureLatent: {len(self.instances)} instances loaded (from cache)')
        else:
            with open(os.path.join(data_dir, prompts_file), 'r') as f:
                all_prompts = json.load(f)
            self.instances = []
            for category, items in all_prompts.items():
                for instance_id, prompt in items.items():
                    instance_dir = os.path.join(data_dir, category, str(instance_id))
                    ori_path = os.path.join(instance_dir, 'ori_ss_latents.npz')
                    edit_path = os.path.join(instance_dir, 'edit_ss_latents.npz')
                    if os.path.exists(ori_path) and os.path.exists(edit_path):
                        self.instances.append({
                            'dir': instance_dir,
                            'prompt': prompt,
                        })
            print(f'EditingTextSparseStructureLatent: {len(self.instances)} instances loaded (scanned)')

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        try:
            info = self.instances[index]
            ori_z = torch.tensor(
                _load_ss_latent_array(os.path.join(info['dir'], 'ori_ss_latents.npz'))
            ).float()
            edit_z = torch.tensor(
                _load_ss_latent_array(os.path.join(info['dir'], 'edit_ss_latents.npz'))
            ).float()

            if self.normalization is not None:
                ori_z = (ori_z - self.mean) / self.std
                edit_z = (edit_z - self.mean) / self.std

            return {
                'x_0': edit_z,
                'ori_voxel': ori_z,
                'cond': info['prompt'],
            }
        except Exception as e:
            print(f'Error loading instance {index}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'


class EditingOverfitTextSparseStructureLatent(SparseStructureLatentVisMixin, Dataset):
    """
    Single-sample overfit dataset for pipeline validation.
    """
    def __init__(
        self,
        roots: str,
        *,
        edit_prompt: str,
        synthetic_length: int = 65536,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.value_range = (0, 1)
        super().__init__(
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

        data_dir = roots
        self.ori_z = torch.tensor(
            _load_ss_latent_array(os.path.join(data_dir, 'ori_ss_latents.npz'))
        ).float()
        self.edit_z = torch.tensor(
            _load_ss_latent_array(os.path.join(data_dir, 'edit_ss_latents.npz'))
        ).float()

        if self.normalization is not None:
            self.ori_z = (self.ori_z - self.mean) / self.std
            self.edit_z = (self.edit_z - self.mean) / self.std

        self.edit_prompt = edit_prompt
        self.synthetic_length = synthetic_length

        print(f'EditingOverfitTextSparseStructureLatent: 1 sample, prompt="{edit_prompt}"')

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, index):
        return {
            'x_0': self.edit_z.clone(),
            'ori_voxel': self.ori_z.clone(),
            'cond': self.edit_prompt,
        }

    def __str__(self):
        return f'{self.__class__.__name__} (1 sample, len={self.synthetic_length})'
