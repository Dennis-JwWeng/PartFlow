"""H3D/3DEditVerse adapters for Ori3DEdit image-conditioned ControlNet training.

These classes read the H3D dataset from the per-case extracted layout
(`<h3d_root_path>/<edit_type>/<shard>/<obj_id>/<edit_id>/...`) and return
the batch contracts expected by Ori3DEdit trainers:

Stage 1 SS trainer:
    {"x_0": edited_ss, "ori_voxel": original_ss, "cond": edited_image}

Stage 2 SLat trainer:
    {"x_0": edited_slat SparseTensor, "ori_slat": original_slat SparseTensor, "cond": edited_image}
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .editing_3deditformer_image import (
    Editing3DEditFormerImageSparseStructureLatent,
    Editing3DEditFormerSLatImage,
)
from .editing_image_latent import _load_image_as_tensor


H3D_EDIT_TYPES = (
    "deletion",
    "addition",
    "modification",
    "scale",
    "material",
    "color",
    "global",
)


def _pil_to_tensor(image: Image.Image, image_size: int = 518) -> torch.Tensor:
    """Match Ori3DEdit image conditioning preprocessing for in-memory PIL images."""
    if image.mode == "RGBA":
        alpha = np.array(image.getchannel(3))
        nz = alpha.nonzero()
        if nz[0].size > 0:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize * 1.2
            image = image.crop([
                int(cx - aug_hsize), int(cy - aug_hsize),
                int(cx + aug_hsize), int(cy + aug_hsize),
            ])
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        rgb = image.convert("RGB")
        rgb_t = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return rgb_t * alpha_t.unsqueeze(0)

    rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    return torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0


def _slat_arrays(npz_dict: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = npz_dict["slat_coords"]
    if coords.ndim == 2 and coords.shape[1] == 4:
        coords = coords[:, 1:]
    feats = npz_dict["slat_feats"]
    return torch.tensor(coords).int(), torch.tensor(feats).float()


def _slat_keep_mask_tensor(arr: np.ndarray, n_tokens: int) -> torch.Tensor:
    """Return a [N] float tensor for slat_keep_mask aligned with slat_coords on
    this NPZ side. Broadcast a scalar (numel == 1) to all tokens."""
    t = torch.tensor(np.asarray(arr)).float().reshape(-1)
    if t.numel() == n_tokens:
        return t
    if t.numel() == 1:
        return t.expand(n_tokens).contiguous()
    raise ValueError(f"mask_keep_slat: expected {n_tokens} entries, got {t.numel()}")


def _align_feats_to_coords(
    source_coords: torch.Tensor,
    source_feats: torch.Tensor,
    target_coords: torch.Tensor,
    return_occ: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Project source sparse features onto target coords.

    Missing source coords are filled with zeros on the target layout. When
    requested, also return a [N, 1] occupancy flag showing which target coords
    existed in the source. This helper is called per sample before batch indices
    are prepended, so coordinates never match across batch items.
    """
    if source_coords.shape == target_coords.shape and torch.equal(source_coords, target_coords):
        occ = source_feats.new_ones((target_coords.shape[0], 1))
        return (source_feats, occ) if return_occ else source_feats

    aligned_feats = source_feats.new_zeros((target_coords.shape[0], *source_feats.shape[1:]))
    occ = source_feats.new_zeros((target_coords.shape[0], 1))
    if source_coords.numel() == 0 or target_coords.numel() == 0:
        return (aligned_feats, occ) if return_occ else aligned_feats

    max_coord = torch.cat([source_coords.reshape(-1), target_coords.reshape(-1)]).max().item()
    base = int(max_coord) + 1

    def coord_keys(coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(torch.long)
        return coords[:, 0] * base * base + coords[:, 1] * base + coords[:, 2]

    source_keys = coord_keys(source_coords)
    target_keys = coord_keys(target_coords)
    order = torch.argsort(source_keys)
    sorted_keys = source_keys[order]
    pos = torch.searchsorted(sorted_keys, target_keys)

    valid_pos = pos < sorted_keys.numel()
    valid = valid_pos.clone()
    if valid_pos.any():
        valid[valid_pos] = sorted_keys[pos[valid_pos]] == target_keys[valid_pos]

    if valid.any():
        aligned_feats[valid] = source_feats[order[pos[valid]]]
        occ[valid] = 1
    return (aligned_feats, occ) if return_occ else aligned_feats


class H3DEditingDatasetBase(Dataset):
    def __init__(
        self,
        roots: str,
        *,
        h3d_root_path: str,
        split: Optional[str] = "train",
        split_file: Optional[str] = None,
        edit_ids_file: Optional[str] = None,
        edit_types: Optional[Sequence[str]] = None,
        shards: Optional[Sequence[str]] = None,
        final_pass_only: bool = False,
        image_size: int = 518,
        random_cond_gt: bool = False,
        max_num_voxels: Optional[int] = None,
        max_samples: Optional[int] = None,
        filter_missing_members: bool = True,
        required_member_files: Optional[Sequence[str]] = None,
        filter_missing_workers: int = 64,
        dynamic_missing_retry: bool = True,
        missing_rescan_interval_steps: int = 200,
        missing_log_path: Optional[str] = None,
        missing_index_cache_path: Optional[str] = None,
        provide_case_dir: bool = False,
        case_dir_root: Optional[str] = None,
        mask_sidecar_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.roots = roots
        self.h3d_root_path = Path(h3d_root_path)
        # Optional on-disk per-case layout (must contain
        # <edit_type>/<shard>/<obj_id>/<edit_id>/{view.meta.json,after.png,...})
        # used by the train-time render loss. Tar-only training does NOT need this.
        self.provide_case_dir = bool(provide_case_dir)
        self.case_dir_root = Path(case_dir_root) if case_dir_root else None
        if self.provide_case_dir and self.case_dir_root is None:
            raise ValueError(
                "[H3D] provide_case_dir=True requires case_dir_root pointing at "
                "the expanded per-case directory layout (e.g. /mnt/.../data/H3D_v1)."
            )
        # Optional per-case mask sidecar directory (build_h3d_mask_full.py output);
        # contains <edit_type>/<shard>/<obj_id>/<edit_id>.npz with keys:
        #   mask_keep_ss            uint8 (16,16,16)
        #   mask_keep_slat          uint8 (N_after,)
        #   mask_keep_slat_before   uint8 (N_before,)
        # When None, mask_keep_* fields are not injected; trainer's mask loss
        # gates them out (mask_keep_slat==0 -> zero contribution).
        self.mask_sidecar_root = Path(mask_sidecar_root) if mask_sidecar_root else None
        self.split = split
        self.image_size = image_size
        self.random_cond_gt = random_cond_gt
        self.max_num_voxels = max_num_voxels
        self.filter_missing_members = filter_missing_members
        self.required_member_files = tuple(required_member_files or (
            "before.npz",
            "after.npz",
            "before.png",
            "after.png",
        ))
        self.filter_missing_workers = int(filter_missing_workers)
        self.dynamic_missing_retry = bool(dynamic_missing_retry)
        self.missing_rescan_interval_steps = int(missing_rescan_interval_steps)
        self.missing_log_path = Path(missing_log_path) if missing_log_path is not None else None
        self.missing_index_cache_path = Path(missing_index_cache_path) if missing_index_cache_path is not None else None
        self._active_records: List[dict] = []
        self._pending_records: Dict[str, dict] = {}
        self._pending_missing: Dict[str, List[str]] = {}
        self._skip_logged: set = set()
        self._last_missing_rescan_step = -1
        self._current_training_step = 0
        self.value_range = (0, 1)

        edit_type_filter = set(edit_types) if edit_types is not None else set(H3D_EDIT_TYPES)
        unknown_edit_types = edit_type_filter - set(H3D_EDIT_TYPES)
        if unknown_edit_types:
            raise ValueError(f"Unknown H3D edit types: {sorted(unknown_edit_types)}")
        shard_filter = set(shards) if shards is not None else None

        split_obj_ids = self._load_split_obj_ids(split, split_file)
        edit_ids = self._load_edit_ids(edit_ids_file)
        loaded_records = self._load_records(edit_type_filter, shard_filter, split_obj_ids, edit_ids, final_pass_only)
        n_before_filter = len(loaded_records)
        if self.filter_missing_members:
            self._active_records, pending = self._partition_records_by_existing_members_cached(loaded_records)
            self.records = list(self._active_records)
            self._pending_records = {self._record_key(record): record for record, _missing in pending}
            self._pending_missing = {self._record_key(record): list(_missing) for record, _missing in pending}
            self._write_missing_snapshot(reason="init")
        else:
            self.records = loaded_records
            self._active_records = list(self.records)
        if max_samples is not None:
            self.records = self.records[:max_samples]
            self._active_records = list(self.records)
        self.loads = [int(record.get("_load", 1)) for record in self.records]

        print("**********************")
        print(f"H3D editing data num: {len(self.records)}")
        if self.filter_missing_members:
            filtered_count = getattr(self, "missing_member_summary", {}).get("dropped", n_before_filter - len(self.records))
            print(f"  - filtered missing members: {filtered_count}")
        print(f"  - root: {self.h3d_root_path}")
        print(f"  - split: {split_file or split or 'all'}")
        print(f"  - final_pass_only: {final_pass_only}")
        if self.filter_missing_members and getattr(self, "missing_member_summary", None):
            summary = self.missing_member_summary
            if summary["dropped"]:
                print(f"  - missing patterns: {summary['by_pattern']}")
                print(f"  - missing examples: {summary['examples']}")
        print("**********************")

    def _resolve_h3d_path(self, path: str) -> Path:
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.h3d_root_path / path_obj

    def _load_split_obj_ids(self, split: Optional[str], split_file: Optional[str]) -> Optional[set]:
        if split_file is not None:
            path = self._resolve_h3d_path(split_file)
        elif split is not None:
            path = self.h3d_root_path / "data" / "splits" / f"{split}.obj_ids.txt"
        else:
            return None
        if not path.exists():
            raise FileNotFoundError(f"H3D split file not found: {path}")
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def _load_edit_ids(self, edit_ids_file: Optional[str]) -> Optional[set]:
        if edit_ids_file is None:
            return None
        path = self._resolve_h3d_path(edit_ids_file)
        if not path.exists():
            raise FileNotFoundError(f"H3D edit id file not found: {path}")
        if path.suffix == ".json":
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                ids = obj.get("edit_ids")
                if ids is None:
                    ids = obj.get("ids")
            else:
                ids = obj
            if ids is None:
                raise ValueError(f"Cannot find edit_ids in {path}")
            return {str(x).strip() for x in ids if str(x).strip()}
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def _manifest_paths(self) -> List[Path]:
        all_manifest = self.h3d_root_path / "data" / "manifests" / "all.jsonl"
        if all_manifest.exists():
            return [all_manifest]
        by_type = self.h3d_root_path / "data" / "manifests" / "by_type"
        if by_type.is_dir():
            paths = sorted(path for path in by_type.glob("*.jsonl") if path.name != ".gitkeep")
            if paths:
                return paths
        extracted_manifest_root = self.h3d_root_path / "manifests"
        extracted_manifest = extracted_manifest_root / "all.jsonl"
        if extracted_manifest.exists():
            return [extracted_manifest]
        if extracted_manifest_root.is_dir():
            return sorted(extracted_manifest_root.glob("*/*.jsonl"))
        raise FileNotFoundError(f"Cannot find H3D manifests under {self.h3d_root_path}")

    def _load_records(
        self,
        edit_type_filter: set,
        shard_filter: Optional[set],
        split_obj_ids: Optional[set],
        edit_ids: Optional[set],
        final_pass_only: bool,
    ) -> List[dict]:
        records: List[dict] = []
        for manifest_path in self._manifest_paths():
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("edit_type") not in edit_type_filter:
                        continue
                    if shard_filter is not None and record.get("shard") not in shard_filter:
                        continue
                    if split_obj_ids is not None and record.get("obj_id") not in split_obj_ids:
                        continue
                    if edit_ids is not None and record.get("edit_id") not in edit_ids:
                        continue
                    if final_pass_only and not (record.get("quality") or {}).get("final_pass", False):
                        continue
                    load = record.get("voxel_num")
                    if load is None:
                        load = record.get("views", {}).get("voxel_num")
                    record["_load"] = int(load) if load is not None else 1
                    records.append(record)
        return records

    def _member_name(self, record: dict, filename: str) -> str:
        return "/".join([record["edit_type"], record["shard"], record["obj_id"], record["edit_id"], filename])

    def _record_case_dir(self, record: dict) -> Optional[str]:
        if self.case_dir_root is None:
            return None
        return str(self.case_dir_root /
                   record["edit_type"] / str(record["shard"]) /
                   record["obj_id"] / record["edit_id"])

    def _mask_sidecar_path(self, record: dict) -> Optional[Path]:
        if self.mask_sidecar_root is None:
            return None
        return (self.mask_sidecar_root /
                record["edit_type"] / str(record["shard"]) /
                record["obj_id"] / f"{record['edit_id']}.npz")

    def _inject_mask_sidecar(self, record: dict,
                             before: Dict[str, np.ndarray],
                             after: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Inject mask_keep_ss / mask_keep_slat into the per-side npz dicts.

        Sidecar layout (build_h3d_mask_full.py):
            mask_keep_ss            uint8 (16,16,16)   # SS, lives on ori side
            mask_keep_slat          uint8 (N_after,)   # SLAT default x_0=after
            mask_keep_slat_before   uint8 (N_before,)  # SLAT swap   x_0=before
        Missing fields fall back to zero placeholders so collate_fn never sees a
        hole; mask=0 ⇒ this sample contributes 0 to the keep loss.
        """
        path = self._mask_sidecar_path(record)
        if path is not None and path.is_file():
            with np.load(path) as sc:
                files = set(sc.files)
                if "mask_keep_ss" in files:
                    m = np.asarray(sc["mask_keep_ss"])
                    before["mask_keep_ss"] = m
                    after["mask_keep_ss"] = m
                if "mask_keep_slat" in files:
                    after["mask_keep_slat"] = np.asarray(sc["mask_keep_slat"])
                if "mask_keep_slat_before" in files:
                    before["mask_keep_slat"] = np.asarray(sc["mask_keep_slat_before"])
        # placeholders
        if self.mask_sidecar_root is not None:
            if "mask_keep_ss" not in before:
                before["mask_keep_ss"] = np.zeros((16, 16, 16), dtype=np.uint8)
                after["mask_keep_ss"] = before["mask_keep_ss"]
            if "slat_coords" in after and "mask_keep_slat" not in after:
                after["mask_keep_slat"] = np.zeros((after["slat_coords"].shape[0],), dtype=np.uint8)
            if "slat_coords" in before and "mask_keep_slat" not in before:
                before["mask_keep_slat"] = np.zeros((before["slat_coords"].shape[0],), dtype=np.uint8)
        return before, after

    def _record_key(self, record: dict) -> str:
        return "/".join([record["edit_type"], str(record["shard"]), record["obj_id"], record["edit_id"]])

    def _member_path(self, record: dict, filename: str) -> Path:
        # Prefer case_dir_root (disk per-case layout) when set; otherwise the
        # legacy extracted-under-h3d_root_path convention.
        disk = self._disk_member_path(record, filename)
        if disk is not None:
            return disk
        return self.h3d_root_path / self._member_name(record, filename)

    def _missing_required_members(self, record: dict) -> List[str]:
        return [str(self._member_path(record, filename)) for filename in self.required_member_files if not self._member_path(record, filename).exists()]

    def _missing_cache_signature(self, records: List[dict]) -> dict:
        digest = hashlib.sha1()
        for record in records:
            digest.update(self._record_key(record).encode("utf-8"))
            digest.update(b"\n")
        return {
            "version": 1,
            "h3d_root_path": str(self.h3d_root_path),
            "required_member_files": list(self.required_member_files),
            "record_count": len(records),
            "record_keys_sha1": digest.hexdigest(),
        }

    def _cache_payload_matches(self, payload: dict, signature: dict) -> bool:
        return payload.get("signature") == signature

    def _load_missing_cache(self, records_by_key: Dict[str, dict], signature: dict) -> Optional[Tuple[List[dict], List[Tuple[dict, List[str]]]]]:
        if self.missing_index_cache_path is None or not self.missing_index_cache_path.exists():
            return None
        try:
            payload = json.loads(self.missing_index_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not self._cache_payload_matches(payload, signature):
            return None

        active_keys = payload.get("active_keys", [])
        pending_items = payload.get("pending", [])
        kept = [records_by_key[key] for key in active_keys if key in records_by_key]
        missing_items: List[Tuple[dict, List[str]]] = []
        for item in pending_items:
            key = item.get("key")
            if key in records_by_key:
                missing_items.append((records_by_key[key], list(item.get("missing_paths", []))))
        self.missing_member_summary = payload.get("missing_member_summary") or self._summarize_missing(missing_items)
        print(f"[H3D] loaded missing index cache: {self.missing_index_cache_path}", flush=True)
        return kept, missing_items

    def _write_missing_cache(self, kept: List[dict], missing_items: List[Tuple[dict, List[str]]], signature: dict) -> None:
        if self.missing_index_cache_path is None:
            return
        self.missing_index_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signature": signature,
            "active_keys": [self._record_key(record) for record in kept],
            "pending": [
                {
                    "key": self._record_key(record),
                    "edit_id": record.get("edit_id"),
                    "edit_type": record.get("edit_type"),
                    "shard": record.get("shard"),
                    "obj_id": record.get("obj_id"),
                    "missing_paths": list(missing),
                }
                for record, missing in missing_items
            ],
            "missing_member_summary": getattr(self, "missing_member_summary", self._summarize_missing(missing_items)),
        }
        tmp_path = self.missing_index_cache_path.with_suffix(self.missing_index_cache_path.suffix + f".tmp.{os.getpid()}")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.missing_index_cache_path)
        print(f"[H3D] wrote missing index cache: {self.missing_index_cache_path}", flush=True)

    def _partition_records_by_existing_members_cached(self, records: List[dict]) -> Tuple[List[dict], List[Tuple[dict, List[str]]]]:
        if self.missing_index_cache_path is None:
            return self._partition_records_by_existing_members(records)

        signature = self._missing_cache_signature(records)
        records_by_key = {self._record_key(record): record for record in records}
        cached = self._load_missing_cache(records_by_key, signature)
        if cached is not None:
            return cached

        lock_path = self.missing_index_cache_path.with_suffix(self.missing_index_cache_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"pid={os.getpid()} time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                break
            except FileExistsError:
                cached = self._load_missing_cache(records_by_key, signature)
                if cached is not None:
                    return cached
                try:
                    if time.time() - lock_path.stat().st_mtime > 3600:
                        lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(5)

        try:
            cached = self._load_missing_cache(records_by_key, signature)
            if cached is not None:
                return cached
            print(f"[H3D] building missing index cache: {self.missing_index_cache_path}", flush=True)
            kept, missing_items = self._partition_records_by_existing_members(records)
            self._write_missing_cache(kept, missing_items, signature)
            return kept, missing_items
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _summarize_missing(self, missing_items: List[Tuple[dict, List[str]]]) -> dict:
        missing_by_pattern: Counter = Counter()
        missing_by_type: Counter = Counter()
        examples: List[str] = []
        for record, missing in missing_items:
            names = [Path(path).name for path in missing]
            pattern = "+".join(names)
            missing_by_pattern[pattern] += 1
            missing_by_type[f"{record.get('edit_type')}:{pattern}"] += 1
            if len(examples) < 5:
                examples.append(f"{record.get('edit_id')} missing {pattern}")
        return {
            "dropped": len(missing_items),
            "by_pattern": dict(missing_by_pattern),
            "by_type": dict(missing_by_type),
            "examples": examples,
        }

    def _partition_records_by_existing_members(self, records: List[dict]) -> Tuple[List[dict], List[Tuple[dict, List[str]]]]:
        kept: List[dict] = []
        missing_items: List[Tuple[dict, List[str]]] = []

        def check(record: dict) -> Tuple[dict, List[str]]:
            return record, self._missing_required_members(record)

        max_workers = max(1, self.filter_missing_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record, missing in executor.map(check, records, chunksize=256):
                if not missing:
                    kept.append(record)
                else:
                    missing_items.append((record, missing))

        self.missing_member_summary = self._summarize_missing(missing_items)
        return kept, missing_items

    def _log_missing_event(self, event: str, record: dict, missing_paths: Sequence[str]) -> None:
        if self.missing_log_path is None:
            return
        self.missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "edit_id": record.get("edit_id"),
            "edit_type": record.get("edit_type"),
            "shard": record.get("shard"),
            "obj_id": record.get("obj_id"),
            "missing_paths": list(missing_paths),
            "step": int(getattr(self, "_current_training_step", 0)),
        }
        with self.missing_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_missing_snapshot(self, reason: str = "update") -> None:
        if self.missing_log_path is None:
            return
        snapshot_path = self.missing_log_path.with_name("missing_samples.json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "pending_count": len(self._pending_records),
            "pending": [
                {
                    "edit_id": record.get("edit_id"),
                    "edit_type": record.get("edit_type"),
                    "shard": record.get("shard"),
                    "obj_id": record.get("obj_id"),
                    "missing_paths": self._pending_missing.get(key, []),
                }
                for key, record in sorted(self._pending_records.items())
            ],
        }
        snapshot_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_training_step(self, step: int) -> None:
        self._current_training_step = int(step)

    def maybe_rescan_missing(self, step: int) -> int:
        if not self.dynamic_missing_retry or not self._pending_records:
            return 0
        interval = max(1, self.missing_rescan_interval_steps)
        if step - self._last_missing_rescan_step < interval:
            return 0
        self._last_missing_rescan_step = int(step)
        restored: List[str] = []
        for key, record in list(self._pending_records.items()):
            missing = self._missing_required_members(record)
            if missing:
                self._pending_missing[key] = missing
                continue
            self.records.append(record)
            self.loads.append(int(record.get("_load", 1)))
            restored.append(key)
            self._pending_records.pop(key, None)
            self._pending_missing.pop(key, None)
            self._log_missing_event("re-detected available sample; added back to training queue", record, [])
        if restored:
            self._write_missing_snapshot(reason="rescan")
            print(f"[H3D] re-detected available sample(s): {len(restored)} added back to training queue", flush=True)
        return len(restored)

    def _mark_runtime_missing(self, record: dict, missing_paths: Sequence[str], error: Optional[Exception] = None) -> None:
        key = self._record_key(record)
        if key not in self._pending_records:
            self._pending_records[key] = record
        self._pending_missing[key] = list(missing_paths)
        if key not in self._skip_logged:
            self._skip_logged.add(key)
            self._log_missing_event("skip missing sample", record, missing_paths)
            print(
                f"[H3D] skip missing sample edit_id={record.get('edit_id')} missing={list(missing_paths)}"
                + (f" error={error}" if error is not None else ""),
                flush=True,
            )
            self._write_missing_snapshot(reason="runtime_skip")

    def _filter_records_with_existing_members(self, records: List[dict]) -> List[dict]:
        kept, missing = self._partition_records_by_existing_members(records)
        return kept

    def _disk_member_path(self, record: dict, filename: str) -> Optional[Path]:
        """Path of <filename> under the per-case disk layout
        (<case_dir_root>/<edit_type>/<shard>/<obj_id>/<edit_id>/<filename>),
        or None if case_dir_root isn't configured."""
        if self.case_dir_root is None:
            return None
        return (
            self.case_dir_root /
            record["edit_type"] /
            str(record["shard"]) /
            record["obj_id"] /
            record["edit_id"] /
            filename
        )

    def _read_member(self, record: dict, filename: str) -> bytes:
        path = self._member_path(record, filename)
        if not path.exists():
            raise FileNotFoundError(f"H3D member not found: {path}")
        return path.read_bytes()

    def _load_npz(self, record: dict, filename: str) -> Dict[str, np.ndarray]:
        with np.load(io.BytesIO(self._read_member(record, filename))) as data:
            out = {key: np.asarray(data[key]) for key in ("ss", "slat_feats", "slat_coords")}
            # Optional mask keys: if pre-baked into before.npz/after.npz, pick them up here.
            for opt in ("mask_keep_ss", "mask_keep_slat"):
                if opt in data.files:
                    out[opt] = np.asarray(data[opt])
            return out

    def _load_image(self, record: dict, filename: str) -> Image.Image:
        image = Image.open(io.BytesIO(self._read_member(record, filename)))
        image.load()
        return image

    def __len__(self):
        return len(self.records)

    def __str__(self):
        return f"{self.__class__.__name__}\n  - Total instances: {len(self)}"

    def visualize_sample(self, sample):
        """Return a BCHW RGB tensor for `save_image` / `make_grid`.

        Handles (1) batched dataset dicts with `cond`, (2) dense SS latents
        ``[N, C, D, H, W]`` from flow snapshot (`sample_gt` / `sample`).

        Keeps the returned tensor on the same device as the input so DDP
        ``dist.gather`` (CUDA backend) still works.
        """
        import torch

        def _to_rgb_bchw(x: torch.Tensor) -> torch.Tensor:
            device = x.device
            x = x.detach().float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.ndim != 4:
                return x
            c = x.shape[1]
            if c >= 3:
                x = x[:, :3]
            elif c == 1:
                x = x.repeat(1, 3, 1, 1)
            else:
                pad_c = 3 - c
                pad = torch.zeros(x.shape[0], pad_c, x.shape[2], x.shape[3], dtype=x.dtype, device=device)
                x = torch.cat([x, pad], dim=1)
            return x.clamp(0.0, 1.0)

        if isinstance(sample, torch.Tensor):
            x = sample.detach().float()
            if x.ndim == 5:
                proj = x.mean(dim=(1, 2))
                proj = proj - proj.amin(dim=(1, 2), keepdim=True)
                denom = proj.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                proj = proj / denom
                return proj.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0)
            return _to_rgb_bchw(sample)

        if not isinstance(sample, dict) or "cond" not in sample:
            return sample

        cond = sample["cond"]
        if not isinstance(cond, torch.Tensor):
            return sample

        return _to_rgb_bchw(cond)


class H3DOriImageSparseStructureLatent(H3DEditingDatasetBase):
    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        record = self.records[index]
        missing = self._missing_required_members(record) if self.dynamic_missing_retry else []
        if missing:
            self._mark_runtime_missing(record, missing)
            return None
        try:
            before = self._load_npz(record, "before.npz")
            after = self._load_npz(record, "after.npz")
            before_img = _pil_to_tensor(self._load_image(record, "before.png"), self.image_size)
            after_img = _pil_to_tensor(self._load_image(record, "after.png"), self.image_size)
        except FileNotFoundError as exc:
            missing = self._missing_required_members(record) or [str(exc)]
            self._mark_runtime_missing(record, missing, exc)
            return None
        if self.random_cond_gt and np.random.rand() < 0.5:
            return {
                "x_0": torch.tensor(before["ss"]).float(),
                "ori_voxel": torch.tensor(after["ss"]).float(),
                "cond": before_img,
            }
        return {
            "x_0": torch.tensor(after["ss"]).float(),
            "ori_voxel": torch.tensor(before["ss"]).float(),
            "cond": after_img,
        }

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}


class H3DOriSLatImage(H3DEditingDatasetBase):
    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        record = self.records[index]
        missing = self._missing_required_members(record) if self.dynamic_missing_retry else []
        if missing:
            self._mark_runtime_missing(record, missing)
            return None
        try:
            before = self._load_npz(record, "before.npz")
            after = self._load_npz(record, "after.npz")
            before, after = self._inject_mask_sidecar(record, before, after)
            before_coords, before_feats = _slat_arrays(before)
            after_coords, after_feats = _slat_arrays(after)
            before_img = _pil_to_tensor(self._load_image(record, "before.png"), self.image_size)
            after_img = _pil_to_tensor(self._load_image(record, "after.png"), self.image_size)
        except FileNotFoundError as exc:
            missing = self._missing_required_members(record) or [str(exc)]
            self._mark_runtime_missing(record, missing, exc)
            return None
        case_dir = self._record_case_dir(record) if self.provide_case_dir else None
        edit_type = str(record.get("edit_type", "") or "")
        if self.random_cond_gt and np.random.rand() < 0.5:
            out: Dict[str, Any] = {
                "coords": before_coords,
                "feats": before_feats,
                "ori_coords": after_coords,
                "ori_feats": after_feats,
                "cond": before_img,
                "edit_type": edit_type,
            }
            if "mask_keep_slat" in before:
                out["mask_keep_slat"] = _slat_keep_mask_tensor(
                    before["mask_keep_slat"], before_coords.shape[0]
                )
        else:
            out = {
                "coords": after_coords,
                "feats": after_feats,
                "ori_coords": before_coords,
                "ori_feats": before_feats,
                "cond": after_img,
                "edit_type": edit_type,
            }
            if "mask_keep_slat" in after:
                out["mask_keep_slat"] = _slat_keep_mask_tensor(
                    after["mask_keep_slat"], after_coords.shape[0]
                )
        if case_dir is not None:
            out["case_dir"] = case_dir
        return out

    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list, feats_list, ori_coords_list, ori_feats_list, ori_occ_list = [], [], [], [], []
            layout, ori_layout = [], []
            start = ori_start = 0
            for i, b in enumerate(sub_batch):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                layout.append(slice(start, start + n))
                start += n
                aligned_ori_feats, ori_occ = _align_feats_to_coords(
                    b["ori_coords"], b["ori_feats"], b["coords"], return_occ=True
                )
                m = b["coords"].shape[0]
                ori_coords_list.append(torch.cat([torch.full((m, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                ori_feats_list.append(aligned_ori_feats)
                ori_occ_list.append(ori_occ)
                ori_layout.append(slice(ori_start, ori_start + m))
                ori_start += m

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub_batch[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)
            ori_slat = SparseTensor(coords=torch.cat(ori_coords_list), feats=torch.cat(ori_feats_list))
            ori_slat._shape = torch.Size([len(group), *sub_batch[0]["ori_feats"].shape[1:]])
            ori_slat.register_spatial_cache("layout", ori_layout)
            ori_occ = SparseTensor(coords=ori_slat.coords.clone(), feats=torch.cat(ori_occ_list))
            ori_occ._shape = torch.Size([len(group), 1])
            ori_occ.register_spatial_cache("layout", ori_layout)
            pack: Dict[str, Any] = {
                "x_0": x_0,
                "ori_slat": ori_slat,
                "ori_occ": ori_occ,
                "cond": torch.stack([b["cond"] for b in sub_batch]),
                "edit_type": [b.get("edit_type", "") for b in sub_batch],
            }
            if "case_dir" in sub_batch[0]:
                pack["case_dirs"] = [b["case_dir"] for b in sub_batch]
            if all("mask_keep_slat" in b for b in sub_batch):
                pack["mask_keep_slat"] = torch.cat(
                    [b["mask_keep_slat"] for b in sub_batch], dim=0
                )
            packs.append(pack)
        return packs[0] if split_size is None else packs


class H3DOriSLatImageTokenConcat(H3DOriSLatImage):
    """H3D SLat image dataset that keeps ori_slat on its original coords."""

    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            coords_list, feats_list = [], []
            ori_coords_list, ori_feats_list = [], []
            layout, ori_layout = [], []
            start = ori_start = 0
            for i, b in enumerate(sub_batch):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                layout.append(slice(start, start + n))
                start += n

                m = b["ori_coords"].shape[0]
                ori_coords_list.append(torch.cat([torch.full((m, 1), i, dtype=torch.int32), b["ori_coords"]], dim=-1))
                ori_feats_list.append(b["ori_feats"])
                ori_layout.append(slice(ori_start, ori_start + m))
                ori_start += m

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub_batch[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)

            ori_slat = SparseTensor(coords=torch.cat(ori_coords_list), feats=torch.cat(ori_feats_list))
            ori_slat._shape = torch.Size([len(group), *sub_batch[0]["ori_feats"].shape[1:]])
            ori_slat.register_spatial_cache("layout", ori_layout)

            pack_tc: Dict[str, Any] = {
                "x_0": x_0,
                "ori_slat": ori_slat,
                "cond": torch.stack([b["cond"] for b in sub_batch]),
                "edit_type": [b.get("edit_type", "") for b in sub_batch],
            }
            if "case_dir" in sub_batch[0]:
                pack_tc["case_dirs"] = [b["case_dir"] for b in sub_batch]
            if all("mask_keep_slat" in b for b in sub_batch):
                pack_tc["mask_keep_slat"] = torch.cat(
                    [b["mask_keep_slat"] for b in sub_batch], dim=0
                )
            packs.append(pack_tc)
        return packs[0] if split_size is None else packs


class _MixedDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        super().__init__()
        self.datasets = list(datasets)
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths).tolist()
        self.value_range = next(
            (dataset.value_range for dataset in self.datasets if hasattr(dataset, "value_range")),
            (0, 1),
        )
        self.loads = []
        for dataset in self.datasets:
            self.loads.extend(getattr(dataset, "loads", [1] * len(dataset)))
        print("**********************")
        print(f"Mixed data num: {len(self)}")
        for dataset, length in zip(self.datasets, self.lengths):
            print(f"  - {dataset.__class__.__name__}: {length}")
        print("**********************")

    def __len__(self):
        return sum(self.lengths)

    def _resolve_index(self, index):
        for dataset_idx, end in enumerate(self.cumulative_lengths):
            if index < end:
                start = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
                return self.datasets[dataset_idx], index - start
        raise IndexError(index)

    def __getitem__(self, index):
        dataset, local_index = self._resolve_index(index)
        return dataset[local_index]

    def visualize_sample(self, sample):
        """Return a torchvision-saveable CHW/BCHW tensor for dataset snapshots.

        `cond` should already be an RGB tensor in [0, 1], but keep this defensive.
        """
        import torch

        if not isinstance(sample, dict) or "cond" not in sample:
            return sample

        cond = sample["cond"]
        if not isinstance(cond, torch.Tensor):
            return sample

        x = cond.detach().float().cpu()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            return sample

        # x: [B, C, H, W]
        c = x.shape[1]
        if c >= 3:
            x = x[:, :3]
        elif c == 1:
            x = x.repeat(1, 3, 1, 1)
        else:
            pad_c = 3 - c
            pad = torch.zeros(x.shape[0], pad_c, x.shape[2], x.shape[3], dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = x.clamp(0.0, 1.0)
        return x


class MixedH3D3DEditVerseOriImageSparseStructureLatent(_MixedDataset):
    def __init__(self, roots: str, *, h3d_args: dict, editverse_args: dict, **kwargs):
        super().__init__(
            H3DOriImageSparseStructureLatent(roots, **h3d_args),
            Editing3DEditFormerImageSparseStructureLatent(roots, **editverse_args),
        )


class MixedH3D3DEditVerseOriSLatImage(_MixedDataset):
    collate_fn = staticmethod(H3DOriSLatImage.collate_fn)

    def __init__(self, roots: str, *, h3d_args: dict, editverse_args: dict, **kwargs):
        super().__init__(
            H3DOriSLatImage(roots, **h3d_args),
            Editing3DEditFormerSLatImage(roots, **editverse_args),
        )


class MixedH3D3DEditVerseOriSLatImageTokenConcat(_MixedDataset):
    collate_fn = staticmethod(H3DOriSLatImageTokenConcat.collate_fn)

    def __init__(self, roots: str, *, h3d_args: dict, editverse_args: dict, **kwargs):
        super().__init__(
            H3DOriSLatImageTokenConcat(roots, **h3d_args),
            Editing3DEditFormerSLatImage(roots, **editverse_args),
        )
