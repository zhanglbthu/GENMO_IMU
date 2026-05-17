#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from gem.datasets.pure_motion.amass_common import AMASS_MOTION_FILENAME, resolve_amass_support_root
from gem.datasets.pure_motion.imu_utils import DEFAULT_CACHE_DIR, cache_file_for_sequence, compute_sequence_imu, load_smpl_models
from gem.utils.motion_utils import get_tgtcoord_rootparam
from gem.utils.pylogger import Log


def build_cache(args):
    root = resolve_amass_support_root(args.root)
    motion_path = root / AMASS_MOTION_FILENAME
    cache_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    motion_files = torch.load(motion_path)
    seq_names = sorted(motion_files.keys())
    if args.limit is not None:
        seq_names = seq_names[: args.limit]

    smplx, smplx2smpl = load_smpl_models()
    written = 0
    skipped = 0

    with torch.no_grad():
        for seq_name in tqdm(seq_names, desc="AMASS IMU cache"):
            cache_file = cache_file_for_sequence(cache_dir, seq_name)
            if cache_file.exists() and not args.overwrite:
                skipped += 1
                continue

            raw = motion_files[seq_name]
            seq_len = raw["pose"].shape[0]
            body_pose = raw["pose"][:, 3:].float()
            betas = raw["beta"].float().repeat(seq_len, 1)
            global_orient = raw["pose"][:, :3].float()
            transl = raw["trans"].float()
            global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="az->ay")

            imu_acc, imu_rot = compute_sequence_imu(
                body_pose,
                betas,
                global_orient,
                transl,
                smplx,
                smplx2smpl,
                smooth_n=args.smooth_n,
            )

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "seq_name": seq_name,
                    "imu_acc": imu_acc.cpu().half(),
                    "imu_rot": imu_rot.cpu().half(),
                },
                cache_file,
            )
            written += 1

    Log.info(
        f"[AMASS IMU cache] done. root={root} output_dir={cache_dir} written={written} skipped={skipped}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute per-sequence AMASS IMU cache for GENMO")
    parser.add_argument("--root", type=str, default="/root/autodl-tmp/dataset/AMASS/hmr4d_support")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--smooth_n", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
