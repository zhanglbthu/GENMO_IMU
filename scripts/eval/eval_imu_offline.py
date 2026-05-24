#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/home/project/GENMO")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.eval_imu_streaming import (
    body_params_to_full_pose,
    load_model,
    load_pose_evaluator,
    make_window_data,
    map_imuposer_to_f_imu,
    render_side_by_side,
    summarize_errors,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline whole-sequence IMU evaluation for GEM")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument("--exp", type=str, default="gem_imu")
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--postproc", action="store_true")
    parser.add_argument("--render", type=str2bool, default=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/eval_imuposer_offline_top3",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    seq_dir = out_dir / "sequences"
    vis_dir = out_dir / "videos"
    seq_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.exp)
    evaluator, body_model = load_pose_evaluator()
    test_data = torch.load(args.eval_pt, map_location="cpu")

    all_errors = []
    num_seqs = min(args.num_seqs, len(test_data["acc"]))
    for seq_idx in range(num_seqs):
        acc = test_data["acc"][seq_idx].float()
        ori = test_data["ori"][seq_idx].float()
        pose_t = test_data["pose"][seq_idx].float()
        tran_t = test_data["tran"][seq_idx].float()

        f_imu, sensor_mask = map_imuposer_to_f_imu(acc, ori, args.combo)
        data = make_window_data(f_imu, sensor_mask)
        pred = model.predict(data, static_cam=True, postproc=args.postproc)
        body = pred["body_params_global"]
        pose_p = body_params_to_full_pose(body["body_pose"], body["global_orient"]).cpu()
        tran_p = body["transl"].detach().cpu()

        err = evaluator.eval(pose_p.cuda(), pose_t.cuda(), tran_p=tran_p.cuda(), tran_t=tran_t.cuda()).cpu()
        all_errors.append(err)

        torch.save(
            {
                "pose_p": pose_p.cpu(),
                "pose_t": pose_t.cpu(),
                "tran_p": tran_p.cpu(),
                "tran_t": tran_t.cpu(),
                "error": err,
            },
            seq_dir / f"{seq_idx + 1}.pt",
        )
        if args.render:
            render_side_by_side(body_model, pose_t, tran_t, pose_p, tran_p, vis_dir / f"{seq_idx + 1}.mp4")
        print(f"seq{seq_idx + 1}:")
        print(err)

    summary = summarize_errors(all_errors)
    torch.save({"errors": torch.stack(all_errors), "summary": summary}, out_dir / "metrics.pt")
    names = [
        "SIP Error (deg)",
        "Angular Error (deg)",
        "Masked Angular Error (deg)",
        "Positional Error (cm)",
        "Masked Positional Error (cm)",
        "Mesh Error (cm)",
        "Jitter Error (100m/s^3)",
        "Distance Error (cm)",
    ]
    with open(out_dir / "metrics.txt", "w") as f:
        for idx, name in enumerate(names):
            line = f"{name}: {summary[idx,0].item():.4f} +/- {summary[idx,1].item():.4f}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
