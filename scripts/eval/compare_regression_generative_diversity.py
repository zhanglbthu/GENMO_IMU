#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/home/project/GENMO")
sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.rotation_conversions import matrix_to_axis_angle
from scripts.eval.aitviewer_render import render_meshes_side_by_side
from scripts.eval.eval_imu_offline import load_model
from scripts.eval.eval_imu_streaming import (
    body_params_to_full_pose,
    load_pose_evaluator,
    make_window_data,
    map_imuposer_to_f_imu,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare regression vs generative diversity on full IMUPoser sequences."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/checkpoints/last.ckpt",
    )
    parser.add_argument("--exp", type=str, default="gem_imu")
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument(
        "--regression-dir",
        type=str,
        default="/home/project/GENMO/outputs/directposer_transformer",
    )
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--diversity-metric",
        type=str,
        default="right_wrist_mean_deg",
        choices=["right_wrist_mean_deg", "all_mean_deg", "right_wrist_max_deg"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/compare_regression_generative_diversity",
    )
    return parser.parse_args()


def load_regression_pose(regression_dir: Path, seq_idx: int):
    payload = torch.load(regression_dir / f"{seq_idx + 1}.pt", map_location="cpu")
    pose_p = payload["pose_p"].float()
    pose_t = payload["pose_t"].float()
    if pose_t.ndim == 3 and pose_t.shape[0] == pose_p.shape[0] * 24:
        pose_t = pose_t.view(pose_p.shape[0], 24, 3, 3)
    return pose_p, pose_t


def sample_sequences(test_data, num_sequences, random_seed):
    del random_seed
    num_total = len(test_data["acc"])
    if num_total < num_sequences:
        raise ValueError(f"Need at least {num_sequences} sequences, got {num_total}")
    return list(range(num_sequences))


@torch.no_grad()
def run_seed(model, acc, ori, combo, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    f_imu, sensor_mask = map_imuposer_to_f_imu(acc, ori, combo)
    data = make_window_data(f_imu, sensor_mask)
    pred = model.predict(data, static_cam=True, postproc=True)
    body = pred["body_params_global"]
    return body_params_to_full_pose(body["body_pose"], body["global_orient"]).cpu()


def compute_diversity(reference_pose, pose):
    rel = torch.matmul(pose, reference_pose.transpose(-1, -2))
    ang = matrix_to_axis_angle(rel).norm(dim=-1) * 180.0 / torch.pi
    return {
        "all_mean_deg": float(ang.mean()),
        "right_wrist_mean_deg": float(ang[:, 19].mean()),
        "right_wrist_max_deg": float(ang[:, 19].max()),
    }


@torch.no_grad()
def render_pose_only_panel(body_model, poses, labels, output_path, fps=30, subject_spacing=1.05):
    zero_tran = torch.zeros(poses[0].shape[0], 3, dtype=poses[0].dtype)
    vertices = []
    for pose in poses:
        _, _, verts = body_model.forward_kinematics(pose=pose, tran=zero_tran, calc_mesh=True)
        vertices.append(verts.cpu())
    vertices = torch.stack(vertices, dim=0).numpy()
    faces = body_model.face.cpu().numpy() if torch.is_tensor(body_model.face) else body_model.face
    render_meshes_side_by_side(
        vertices=vertices,
        faces=faces,
        output_path=output_path,
        width=1920,
        height=1080,
        fps=fps,
        subject_spacing=subject_spacing,
    )


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = out_dir / "segments"
    videos_dir = out_dir / "videos"
    segments_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.exp)
    _, body_model = load_pose_evaluator()
    test_data = torch.load(args.eval_pt, map_location="cpu")
    regression_dir = Path(args.regression_dir)

    chosen_sequences = sample_sequences(
        test_data,
        num_sequences=args.num_sequences,
        random_seed=args.random_seed,
    )

    global_meta = {
        "eval_pt": args.eval_pt,
        "regression_dir": args.regression_dir,
        "num_sequences": args.num_sequences,
        "combo": args.combo,
        "diversity_metric": args.diversity_metric,
        "seeds": list(range(args.seed_start, args.seed_start + args.num_seeds)),
        "sequences": [],
    }

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    for item_id, seq_idx in enumerate(chosen_sequences, start=1):
        acc = test_data["acc"][seq_idx].float()
        ori = test_data["ori"][seq_idx].float()
        pose_t = test_data["pose"][seq_idx].float()
        reg_pose_p, _ = load_regression_pose(regression_dir, seq_idx)
        reg_pose_seq = reg_pose_p.float()

        pred_by_seed = {}
        for seed in seeds:
            pred_by_seed[seed] = run_seed(model, acc, ori, args.combo, seed)
            print(f"[sequence {item_id}/{args.num_sequences}] seq={seq_idx+1} seed={seed} done")

        ref_pose = pred_by_seed[seeds[0]]
        ranked = []
        for seed in seeds[1:]:
            stats = compute_diversity(ref_pose, pred_by_seed[seed])
            ranked.append({"seed": seed, **stats})
        ranked.sort(key=lambda x: x[args.diversity_metric], reverse=True)
        topk = ranked[: args.topk]

        poses = [pose_t, reg_pose_seq, pred_by_seed[seeds[0]]] + [pred_by_seed[x["seed"]] for x in topk]
        labels = ["GT", "Regression", f"seed{seeds[0]}"] + [f"seed{x['seed']}" for x in topk]

        video_path = videos_dir / f"sequence_{item_id:02d}_seq{seq_idx+1}.mp4"
        render_pose_only_panel(body_model, poses, labels, video_path)

        seq_meta = {
            "item_id": item_id,
            "sequence_index": seq_idx,
            "sequence_name": seq_idx + 1,
            "length": int(acc.shape[0]),
            "reference_seed": seeds[0],
            "topk": topk,
            "labels": labels,
            "video": str(video_path),
        }
        torch.save(
            {
                "pose_t": pose_t,
                "pose_reg": reg_pose_seq,
                "pose_seed0": pred_by_seed[seeds[0]],
                "topk": topk,
            },
            segments_dir / f"sequence_{item_id:02d}.pt",
        )
        with open(segments_dir / f"sequence_{item_id:02d}.json", "w") as f:
            json.dump(seq_meta, f, indent=2)
        global_meta["sequences"].append(seq_meta)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(global_meta, f, indent=2)
    print(json.dumps(global_meta, indent=2))


if __name__ == "__main__":
    main()
