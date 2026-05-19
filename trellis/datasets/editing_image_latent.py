"""
Datasets for image-conditioned 3D editing with ControlNet — Stage 1 (Sparse Structure).
Provides ori_voxel (original ss latent), x_0 (edited ss latent), and a target
edit image (PIL → tensor) that the trainer's ImageConditionedMixin will encode
through DINOv2 to produce the cross-attention conditioning.

Mirrors `editing_text_latent.py` but swaps the text prompt for an image
condition. Instance directory layout (created by the prepare script):

    <instance_dir>/
        ori_ss_latents.npz
        edit_ss_latents.npz
        ori_image.png      # rendered image of the source 3D
        edit_image.png     # rendered image of the target/edited 3D (cond)
"""
import os
import json
from typing import *
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .sparse_structure_latent import SparseStructureLatentVisMixin


_DEFAULT_IMAGE_SIZE = 518


def _match_prefixed_image(instance_dir: str, prefixes: Tuple[str, ...]) -> Optional[str]:
    """Return the first image file whose basename starts with one of the prefixes."""
    try:
        filenames = sorted(os.listdir(instance_dir))
    except FileNotFoundError:
        return None

    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    for fn in filenames:
        lower = fn.lower()
        if not lower.endswith(valid_exts):
            continue
        if any(fn.startswith(prefix) for prefix in prefixes):
            return os.path.join(instance_dir, fn)
    return None


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


def _find_image_path(instance_dir: str, name: str) -> Optional[str]:
    """Find an image inside the instance dir, tolerant to common naming variants."""
    if name == 'edit':
        candidates = (
            'edit_image.png',
            'edit_img.png',
            'after_edited_Flux.png',
        )
        prefix_fallbacks = ('edited_',)
    elif name == 'ori':
        candidates = (
            'ori_image.png',
            'ori.png',
            'original.png',
        )
        prefix_fallbacks = ('ori_',)
    else:
        raise ValueError(f"Unknown image kind: {name}")

    for cand in candidates:
        p = os.path.join(instance_dir, cand)
        if os.path.isfile(p):
            return p

    return _match_prefixed_image(instance_dir, prefix_fallbacks)


def _load_image_as_tensor(path: str, image_size: int = _DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """
    Load an RGBA/RGB image, tight-crop on alpha when present, resize to (image_size,
    image_size), and return a [3, H, W] float tensor in [0, 1]. Alpha (if any) is
    pre-multiplied so the background is black, matching the convention used by
    `ImageConditionedMixin` for image-to-3D training.
    """
    image = Image.open(path)
    if image.mode == 'RGBA':
        alpha = np.array(image.getchannel(3))
        nz = alpha.nonzero()
        if nz[0].size > 0:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize * 1.2
            aug_bbox = [
                int(cx - aug_hsize), int(cy - aug_hsize),
                int(cx + aug_hsize), int(cy + aug_hsize),
            ]
            image = image.crop(aug_bbox)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        rgb = image.convert('RGB')
        rgb_t = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return rgb_t * alpha_t.unsqueeze(0)
    else:
        rgb = image.convert('RGB').resize((image_size, image_size), Image.Resampling.LANCZOS)
        return torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0


class EditingImageSparseStructureLatent(SparseStructureLatentVisMixin, Dataset):
    """
    Full dataset for image-conditioned editing.
    Scans data_dir for instances listed in edit_prompts.json (the prompt is ignored,
    only used to enumerate instance ids — see `prepare_3deditverse_for_ori3dedit.py`),
    loads ori/edit ss latents and the corresponding edit image as conditioning.
    """
    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str = 'edit_prompts.json',
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.image_size = image_size
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
        cache_file = os.path.join(data_dir, 'valid_ss_image_instances.json')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.instances = json.load(f)
            print(f'EditingImageSparseStructureLatent: {len(self.instances)} instances loaded (from cache)')
        else:
            with open(os.path.join(data_dir, prompts_file), 'r') as f:
                all_prompts = json.load(f)
            self.instances = []
            for category, items in all_prompts.items():
                for instance_id, _ in items.items():
                    instance_dir = os.path.join(data_dir, category, str(instance_id))
                    ori_path = os.path.join(instance_dir, 'ori_ss_latents.npz')
                    edit_path = os.path.join(instance_dir, 'edit_ss_latents.npz')
                    edit_img = _find_image_path(instance_dir, 'edit')
                    if (
                        os.path.exists(ori_path)
                        and os.path.exists(edit_path)
                        and edit_img is not None
                    ):
                        self.instances.append({'dir': instance_dir})
            print(f'EditingImageSparseStructureLatent: {len(self.instances)} instances loaded (scanned)')

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

            edit_img_path = _find_image_path(info['dir'], 'edit')
            if edit_img_path is None:
                raise FileNotFoundError(f"No edit image in {info['dir']}")
            cond_img = _load_image_as_tensor(edit_img_path, self.image_size)

            return {
                'x_0': edit_z,
                'ori_voxel': ori_z,
                'cond': cond_img,
            }
        except Exception as e:
            print(f'Error loading instance {index}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        return f'{self.__class__.__name__} ({len(self.instances)} instances)'


class EditingOverfitImageSparseStructureLatent(SparseStructureLatentVisMixin, Dataset):
    """
    Single-sample overfit dataset for image-conditioned SS ControlNet validation.

    Loads one (ori_ss_latents, edit_ss_latents) pair plus a fixed conditioning
    image from `roots` and serves it `synthetic_length` times.
    """
    def __init__(
        self,
        roots: str,
        *,
        edit_image: Optional[str] = None,
        synthetic_length: int = 65536,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.normalization = normalization
        self.image_size = image_size
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

        if edit_image is None:
            edit_image = _find_image_path(data_dir, 'edit')
        if edit_image is None or not os.path.isfile(edit_image):
            raise FileNotFoundError(f'edit_image not found under {data_dir}')
        self.cond_img = _load_image_as_tensor(edit_image, self.image_size)

        self.synthetic_length = synthetic_length
        print(f'EditingOverfitImageSparseStructureLatent: 1 sample, edit_image="{edit_image}"')

    def __len__(self):
        return self.synthetic_length

    def __getitem__(self, index):
        return {
            'x_0': self.edit_z.clone(),
            'ori_voxel': self.ori_z.clone(),
            'cond': self.cond_img.clone(),
        }

    def __str__(self):
        return f'{self.__class__.__name__} (1 sample, len={self.synthetic_length})'
