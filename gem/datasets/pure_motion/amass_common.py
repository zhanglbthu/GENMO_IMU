# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

from gem.utils.motion_utils import get_tgtcoord_rootparam
from gem.utils.pylogger import Log

from .utils import interpolate_smpl_params


AMASS_MOTION_FILENAME = "smplxpose_v2.pth"
DEFAULT_AMASS_ROOTS = (
    "/root/autodl-tmp/dataset/AMASS/hmr4d_support",
    "inputs/AMASS/hmr4d_support",
)


def resolve_amass_support_root(root=None):
    env_root = os.environ.get("GENMO_AMASS_ROOT")
    candidates = []
    if root is not None:
        candidates.append(Path(root))
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(p) for p in DEFAULT_AMASS_ROOTS)

    checked = []
    for candidate in candidates:
        resolved = _resolve_one_root(candidate)
        checked.append(str(candidate))
        if resolved is not None:
            return resolved

    checked_str = ", ".join(checked)
    raise FileNotFoundError(
        f"Unable to locate {AMASS_MOTION_FILENAME}. Checked: {checked_str}. "
        "Set GENMO_AMASS_ROOT or pass root=... to the dataset."
    )


def _resolve_one_root(candidate):
    candidate = Path(candidate).expanduser()
    if candidate.is_file():
        return _maybe_extract_and_find(candidate.parent, candidate)
    if candidate.is_dir():
        direct_file = candidate / AMASS_MOTION_FILENAME
        if direct_file.exists():
            return candidate
        nested = _find_motion_file(candidate)
        if nested is not None:
            return nested.parent
        tarball = candidate / "AMASS_hmr4d_support.tar.gz"
        if tarball.exists():
            return _maybe_extract_and_find(candidate, tarball)
    return None


def _maybe_extract_and_find(extract_dir, tarball):
    extract_dir = Path(extract_dir)
    tarball = Path(tarball)
    motion_file = extract_dir / AMASS_MOTION_FILENAME
    if motion_file.exists():
        return extract_dir

    Log.info(f"[AMASS] Extracting {tarball} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as handle:
        handle.extractall(path=extract_dir)

    if motion_file.exists():
        return extract_dir
    nested = _find_motion_file(extract_dir)
    if nested is not None:
        return nested.parent
    raise FileNotFoundError(f"Extracted {tarball}, but {AMASS_MOTION_FILENAME} was not found.")


def _find_motion_file(root):
    matches = sorted(Path(root).rglob(AMASS_MOTION_FILENAME))
    return matches[0] if matches else None


class AmassMotionMixin:
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,
        skip_moyo=True,
        random1024=False,
        root=None,
        split="train",
        val_ratio=0.1,
        split_seed=1234,
    ):
        self.root = resolve_amass_support_root(root)
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.random1024 = random1024
        self.skip_moyo = skip_moyo
        self.dataset_name = "AMASS"
        self.split = split
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

    def _load_dataset(self):
        filename = self.root / AMASS_MOTION_FILENAME
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplxpose_v2_random1024.pth")
            except Exception:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplxpose_v2_random1024.pth")
        else:
            self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(
            f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        seq_lengths = []
        self.idx2meta = []
        motion_start_id = {}
        all_vids = list(self.motion_files.keys())
        active_vids = self._select_split_vids(all_vids)
        for vid in active_vids:
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(
            f"[{self.dataset_name}][split={self.split}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _select_split_vids(self, vids):
        vids = sorted(vids)
        if self.split in (None, "all"):
            return vids
        if self.split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {self.split}")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio should be in (0,1), got {self.val_ratio}")

        rng = np.random.RandomState(self.split_seed)
        order = rng.permutation(len(vids))
        val_count = max(1, int(round(len(vids) * self.val_ratio)))
        val_indices = set(order[:val_count].tolist())
        if self.split == "val":
            return [vid for idx, vid in enumerate(vids) if idx in val_indices]
        return [vid for idx, vid in enumerate(vids) if idx not in val_indices]

    def load_amass_sequence(self, idx):
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        full_data, sample_info = self.sample_amass_window(raw_data, start_id)
        data = {
            k: v[sample_info["window_start"] : sample_info["window_end"]]
            for k, v in full_data.items()
        }
        tgt_len = sample_info["tgt_len"]
        data_interpolated = interpolate_smpl_params(data, tgt_len)
        data_interpolated["global_orient"], data_interpolated["transl"], _ = get_tgtcoord_rootparam(
            data_interpolated["global_orient"],
            data_interpolated["transl"],
            tsf="az->ay",
        )
        data_interpolated["data_name"] = "amass"
        data_interpolated["seq_name"] = mid
        return data_interpolated

    def sample_amass_window(self, raw_data, start_id=0):
        raw_len = raw_data["pose"].shape[0] - start_id
        full_data = {
            "body_pose": raw_data["pose"][start_id:, 3:],
            "betas": raw_data["beta"].repeat(raw_len, 1),
            "global_orient": raw_data["pose"][start_id:, :3],
            "transl": raw_data["trans"][start_id:],
        }

        tgt_len = self.motion_frames
        raw_subset_len = np.random.randint(
            int(tgt_len / self.l_factor), int(tgt_len * self.l_factor)
        )
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:
            start = 0
            end = raw_len
        sample_info = {
            "window_start": start,
            "window_end": end,
            "raw_len": raw_len,
            "tgt_len": tgt_len,
        }
        return full_data, sample_info
