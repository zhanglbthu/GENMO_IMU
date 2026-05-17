#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from gem.datasets.pure_motion.imu_utils import DEFAULT_SENSOR_COMBOS, build_f_imu
from gem.utils.rotation_conversions import axis_angle_to_matrix


WINDOW_FRAMES = 120
IMUPOSER_DEVICE_TO_SLOT = {
    0: 2,  # left phone -> left thigh / pocket
    1: 0,  # left watch -> left wrist
    2: 4,  # headphone -> head
    3: 3,  # right phone -> right thigh / pocket
    4: 1,  # right watch -> right wrist
}


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming IMU evaluation for GEM IMU model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/project/GENMO/outputs/gem_imu_amass/gem_imu_/version_10/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/AMASS/hmr4d_support/processed/eval/imuposer_test.pt",
    )
    parser.add_argument("--exp", type=str, default="gem_imu")
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES)
    parser.add_argument("--render-count", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to sibling directory next to checkpoint: eval_imuposer_streaming",
    )
    return parser.parse_args()


def load_pose_evaluator():
    repo_root = Path("/home/project/GENMO")
    mp_root = repo_root / "3rdparty" / "mobileposer"
    sys.path.insert(0, str(mp_root))
    from config import joint_set, paths, datasets  # type: ignore

    paths.smpl_file = mp_root / "smpl" / "basicmodel_m.pkl"
    import articulate as art  # type: ignore

    body_model = art.ParametricModel(paths.smpl_file)
    
    class PoseEvaluator:
        def __init__(self):
            self._eval_fn = art.FullMotionEvaluator(
                paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps
            )

        def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
            pose_p = pose_p.clone().view(-1, 24, 3, 3)
            pose_t = pose_t.clone().view(-1, 24, 3, 3)
            if tran_p is not None and tran_t is not None:
                tran_p = tran_p.clone().view(-1, 3)
                tran_t = tran_t.clone().view(-1, 3)
            else:
                tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device)
                tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device)

            ignored = torch.tensor(joint_set.ignored, device=pose_p.device, dtype=torch.long)
            identity_p = (
                torch.eye(3, device=pose_p.device)
                .view(1, 1, 3, 3)
                .expand(pose_p.shape[0], len(joint_set.ignored), 3, 3)
            )
            identity_t = (
                torch.eye(3, device=pose_t.device)
                .view(1, 1, 3, 3)
                .expand(pose_t.shape[0], len(joint_set.ignored), 3, 3)
            )
            pose_p = pose_p.index_copy(1, ignored, identity_p)
            pose_t = pose_t.index_copy(1, ignored.to(pose_t.device), identity_t)

            errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
            return torch.stack(
                [
                    errs[9],
                    errs[3],
                    errs[9],
                    errs[0] * 100,
                    errs[7] * 100,
                    errs[1] * 100,
                    errs[4] / 100,
                    errs[6],
                ]
            )

        @staticmethod
        def print(errors):
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
            for idx, name in enumerate(names):
                mean = errors[idx, 0].item()
                std = errors[idx, 1].item()
                print(f"{name}: {mean:.2f} (+/- {std:.2f})")

    evaluator = PoseEvaluator()
    return evaluator, body_model


def resolve_output_dir(args):
    if args.output_dir is not None:
        return Path(args.output_dir)
    ckpt_dir = Path(args.ckpt).resolve().parent.parent
    return ckpt_dir / "eval_imuposer_streaming"


def load_model(ckpt_path, exp_name):
    config_dir = Path("/home/project/GENMO/configs")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = hydra.compose(config_name="train", overrides=[f"exp={exp_name}", "use_wandb=false"])
    model = instantiate(cfg.model, _recursive_=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.pipeline.args.use_cfg_sampler_for_gen = False
    model.pipeline.denoiser3d.args.use_cfg_sampler_for_gen = False
    model = model.cuda().eval()
    return model


def body_params_to_full_pose(body_pose_aa, global_orient_aa):
    if body_pose_aa.ndim == 1:
        body_pose_aa = body_pose_aa[None]
    if global_orient_aa.ndim == 1:
        global_orient_aa = global_orient_aa[None]
    length = body_pose_aa.shape[0]
    full_pose = torch.eye(3, device=body_pose_aa.device, dtype=body_pose_aa.dtype).view(1, 1, 3, 3).repeat(length, 24, 1, 1)
    full_pose[:, 0] = axis_angle_to_matrix(global_orient_aa)
    full_pose[:, 1:22] = axis_angle_to_matrix(body_pose_aa.view(length, 21, 3))
    return full_pose


def map_imuposer_to_f_imu(acc5, ori5, combo_name):
    length = acc5.shape[0]
    acc7 = torch.zeros(length, 7, 3, dtype=torch.float32)
    ori7 = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(length, 7, 1, 1)
    available = torch.zeros(7, dtype=torch.float32)

    for src_idx, dst_idx in IMUPOSER_DEVICE_TO_SLOT.items():
        acc7[:, dst_idx] = acc5[:, src_idx]
        ori7[:, dst_idx] = ori5[:, src_idx]
        available[dst_idx] = 1.0

    sensor_mask = torch.zeros(7, dtype=torch.float32)
    sensor_mask[DEFAULT_SENSOR_COMBOS[combo_name]] = 1.0
    sensor_mask = sensor_mask * available

    for slot_idx in range(7):
        if sensor_mask[slot_idx] == 0:
            acc7[:, slot_idx] = 0
            ori7[:, slot_idx] = 0

    f_imu, _ = build_f_imu(acc7, ori7, sensor_mask, include_combo_mask=True)
    return f_imu, sensor_mask


def make_window_data(f_imu_window, sensor_mask):
    length = f_imu_window.shape[0]
    zeros_bool = torch.zeros(length, dtype=torch.bool)
    zeros_mask = {
        "valid": torch.ones(length, dtype=torch.bool),
        "humanoid": torch.zeros(length, dtype=torch.bool),
        "has_img_mask": zeros_bool.clone(),
        "has_2d_mask": zeros_bool.clone(),
        "has_cam_mask": zeros_bool.clone(),
        "has_audio_mask": zeros_bool.clone(),
        "has_music_mask": zeros_bool.clone(),
        "has_imu_mask": torch.ones(length, dtype=torch.bool),
        "2d_only": False,
        "vitpose": False,
        "bbx_xys": False,
        "f_imgseq": False,
        "spv_incam_only": False,
        "invalid_contact": False,
    }
    return {
        "meta": [{"mode": "default"}],
        "length": torch.tensor(length, dtype=torch.long),
        "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
        "bbx_xys": torch.zeros(length, 3, dtype=torch.float32),
        "K_fullimg": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1),
        "cam_angvel": torch.zeros(length, 6, dtype=torch.float32),
        "cam_tvel": torch.zeros(length, 3, dtype=torch.float32),
        "f_imgseq": torch.zeros(length, 1024, dtype=torch.float32),
        "f_imu": f_imu_window,
        "mask": zeros_mask,
        "caption": "",
        "has_text": torch.tensor(False),
        "imu_sensor_mask": sensor_mask,
    }


@torch.no_grad()
def predict_window_batch(model, windows, sensor_mask):
    from gem.utils.cam_utils import compute_bbox_info_bedlam
    from gem.utils.geo_transform import normalize_kp2d

    if model.endecoder.obs_indices_dict is None:
        model.endecoder.build_obs_indices_dict()

    B, L, _ = windows.shape
    device = next(model.parameters()).device
    kp2d = torch.zeros(B, L, 17, 3, device=device)
    bbx_xys = torch.zeros(B, L, 3, device=device)
    K_fullimg = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(B, L, 1, 1)
    cam_angvel = torch.zeros(B, L, 6, device=device)
    cam_tvel = torch.zeros(B, L, 3, device=device)
    f_imgseq = torch.zeros(B, L, 1024, device=device)
    has_text = torch.zeros(B, dtype=torch.bool, device=device)

    batch = {
        "length": torch.full((B,), L, dtype=torch.long, device=device),
        "obs": normalize_kp2d(kp2d, bbx_xys),
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_cam_angvel": cam_angvel.clone(),
        "cam_tvel": cam_tvel,
        "f_imgseq": f_imgseq,
        "has_text": has_text,
        "B": B,
        "L": L,
        "mode": "default",
        "target_x": torch.zeros(B, L, model.endecoder.get_motion_dim(), device=device),
        "sample_indices_dict": model.endecoder.obs_indices_dict,
        "f_imu": windows.to(device),
        "device": device,
        "meta": [{"mode": "default"} for _ in range(B)],
        "caption": [""] * B,
    }
    batch["encoded_text"] = model.encode_text(batch["caption"], batch["has_text"])
    batch["f_cliffcam"] = compute_bbox_info_bedlam(batch["bbx_xys"], batch["K_fullimg"]).to(device)

    false_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    condition_mask = {
        "has_img_mask": false_mask.clone(),
        "has_2d_mask": false_mask.clone(),
        "has_cam_mask": false_mask.clone(),
        "has_audio_mask": false_mask.clone(),
        "has_music_mask": false_mask.clone(),
        "has_imu_mask": torch.ones(B, L, dtype=torch.bool, device=device),
        "j2d_visible_mask": false_mask[:, :, None].repeat(1, 1, 17),
    }
    batch["condition_mask"] = condition_mask

    if model.model_cfg.normalize_cam_angvel:
        batch["f_cam_angvel"] = (batch["f_cam_angvel"] - model.cam_angvel_mean) / model.cam_angvel_std
    for k in model.normalizer_stats:
        if k in batch:
            batch[k] = model.normalize_attr(batch[k], k)

    batch = model.create_condition_mask(batch, cond_mask_cfg=None, mode=None, train=False)
    outputs = model.pipeline.forward(
        batch,
        train=False,
        postproc=False,
        static_cam=True,
        test_mode="default",
    )
    body = outputs["pred_body_params_global"]
    pose_p = body_params_to_full_pose(body["body_pose"][:, -1], body["global_orient"][:, -1]).cpu()
    tran_p = body.get("transl", torch.zeros(B, L, 3, device=device))[:, -1].detach().cpu()
    return pose_p, tran_p


@torch.no_grad()
def stream_predict_sequence(model, f_imu_seq, sensor_mask, window, batch_size=16):
    pred_pose = []
    pred_tran = []
    windows = []
    for frame_idx in range(f_imu_seq.shape[0]):
        left = max(0, frame_idx - window + 1)
        chunk = f_imu_seq[left : frame_idx + 1]
        if chunk.shape[0] < window:
            pad = chunk[:1].repeat(window - chunk.shape[0], 1)
            chunk = torch.cat([pad, chunk], dim=0)
        windows.append(chunk)

    for start in tqdm(range(0, len(windows), batch_size), leave=False):
        batch_windows = torch.stack(windows[start : start + batch_size], dim=0)
        with contextlib.redirect_stdout(io.StringIO()):
            pose_batch, tran_batch = predict_window_batch(model, batch_windows, sensor_mask)
        pred_pose.extend([x for x in pose_batch])
        pred_tran.extend([x for x in tran_batch])
    return torch.stack(pred_pose), torch.stack(pred_tran)


def skeleton_to_image(joints, parents, image_wh=(960, 540), color=(50, 90, 220)):
    width, height = image_wh
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    pts = joints[:, [0, 1]].copy()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6)
    pts = (pts - center) / scale
    pts[:, 1] *= -1
    pts[:, 0] = pts[:, 0] * (width * 0.65) + width / 2
    pts[:, 1] = pts[:, 1] * (height * 0.65) + height / 2
    pts = pts.astype(np.int32)
    for j, p in enumerate(parents):
        if p is None:
            continue
        cv2.line(canvas, tuple(pts[p]), tuple(pts[j]), color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, tuple(pt), 3, (20, 20, 20), -1, cv2.LINE_AA)
    return canvas


@torch.no_grad()
def render_side_by_side(body_model, pose_t, tran_t, pose_p, tran_p, output_path, fps=30):
    _, gt_joints = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=False)
    _, pred_joints = body_model.forward_kinematics(pose_p, tran=tran_p, calc_mesh=False)
    gt_joints = gt_joints.cpu().numpy()
    pred_joints = pred_joints.cpu().numpy()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1920, 540))
    parents = body_model.parent
    for frame_idx in range(min(len(gt_joints), len(pred_joints))):
        gt_img = skeleton_to_image(gt_joints[frame_idx], parents, color=(55, 125, 235))
        pred_img = skeleton_to_image(pred_joints[frame_idx], parents, color=(235, 110, 55))
        cv2.putText(gt_img, "GT", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(pred_img, "Pred", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        frame = np.concatenate([gt_img, pred_img], axis=1)
        writer.write(frame)
    writer.release()


def summarize_errors(errors):
    stacked = torch.stack(errors)
    return stacked.mean(dim=0)


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / "sequences"
    vis_dir = out_dir / "videos"
    seq_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.exp)
    evaluator, body_model = load_pose_evaluator()
    test_data = torch.load(args.eval_pt, map_location="cpu")

    all_errors = []
    for seq_idx in range(len(test_data["acc"])):
        acc = test_data["acc"][seq_idx].float()
        ori = test_data["ori"][seq_idx].float()
        pose_t = test_data["pose"][seq_idx].float()
        tran_t = test_data["tran"][seq_idx].float()

        f_imu, sensor_mask = map_imuposer_to_f_imu(acc, ori, args.combo)
        pose_p, tran_p = stream_predict_sequence(model, f_imu, sensor_mask, args.window)

        err = evaluator.eval(pose_p.cuda(), pose_t.cuda(), tran_p=tran_p.cuda(), tran_t=tran_t.cuda()).cpu()
        all_errors.append(err)

        torch.save(
            {
                "pose_p": pose_p.cpu(),
                "pose_t": pose_t.cpu(),
                "tran_p": tran_p.cpu(),
                "tran_t": tran_t.cpu(),
            },
            seq_dir / f"{seq_idx + 1}.pt",
        )

        if seq_idx < args.render_count:
            render_side_by_side(body_model, pose_t, tran_t, pose_p, tran_p, vis_dir / f"{seq_idx + 1}.mp4")

    summary = summarize_errors(all_errors)
    torch.save({"errors": torch.stack(all_errors), "summary": summary}, out_dir / "metrics.pt")
    evaluator.print(summary)
    with open(out_dir / "metrics.txt", "w") as handle:
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
        for idx, name in enumerate(names):
            mean = summary[idx, 0].item()
            std = summary[idx, 1].item()
            handle.write(f"{name}: {mean:.4f} +/- {std:.4f}\n")


if __name__ == "__main__":
    main()
