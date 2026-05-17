# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import torch

from gem.utils.body_model.smplx_lite import SmplxLite
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d

IMU_VERT_IDS = torch.tensor([1961, 5424, 876, 4362, 411, 3365, 6765], dtype=torch.long)
IMU_JOINT_IDS = torch.tensor([18, 19, 1, 2, 15, 7, 8], dtype=torch.long)
GRAVITY_VEC = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
TARGET_FPS = 30
DEFAULT_SENSOR_COMBOS = {
    "all": [0, 1, 2, 3, 4, 5, 6],
    "lw_rw_lp_rp_h_feet": [0, 1, 2, 3, 4, 5, 6],
    "lw_rw_lp_rp_h": [0, 1, 2, 3, 4],
    "lw_rw_lp_rp": [0, 1, 2, 3],
    "lw_rw_h_feet": [0, 1, 4, 5, 6],
    "lw_rw_feet": [0, 1, 5, 6],
    "lw_rp_h_feet": [0, 3, 4, 5, 6],
    "rw_rp_h_feet": [1, 3, 4, 5, 6],
    "lw_lp_h_feet": [0, 2, 4, 5, 6],
    "rw_lp_h_feet": [1, 2, 4, 5, 6],
    "lw_rp_h": [0, 3, 4],
    "rw_rp_h": [1, 3, 4],
    "lw_lp_h": [0, 2, 4],
    "rw_lp_h": [1, 2, 4],
    "lw_rp": [0, 3],
    "rw_rp": [1, 3],
    "lw_lp": [0, 2],
    "rw_lp": [1, 2],
}
DEFAULT_CACHE_DIR = Path("/root/autodl-tmp/dataset/processed/genmo/amass_imu_cache_v1")


def synthesize_acceleration(vertices, smooth_n=4):
    scale = TARGET_FPS**2
    acc = torch.stack(
        [(vertices[i] + vertices[i + 2] - 2 * vertices[i + 1]) * scale for i in range(vertices.shape[0] - 2)]
    )
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])), dim=0)

    if smooth_n > 0 and vertices.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (vertices[i] + vertices[i + smooth_n * 2] - 2 * vertices[i + smooth_n])
                * scale
                / smooth_n**2
                for i in range(vertices.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def compute_global_rotations(global_orient, body_pose, smplx_model):
    length = global_orient.shape[0]
    extra_pose = smplx_model.other_default_pose.view(33, 3).expand(length, -1, -1)
    local_pose = torch.cat(
        [global_orient[:, None], body_pose.view(length, 21, 3), extra_pose],
        dim=1,
    )
    local_rot = axis_angle_to_matrix(local_pose)

    global_rot = local_rot.clone()
    parents = smplx_model.parents.tolist()
    for joint_idx in range(1, len(parents)):
        parent_idx = parents[joint_idx]
        global_rot[:, joint_idx] = global_rot[:, parent_idx] @ local_rot[:, joint_idx]
    return global_rot


def load_smpl_models():
    smplx = SmplxLite()
    smplx2smpl = torch.load("gem/utils/body_model/smplx2smpl_sparse.pt")
    if smplx2smpl.is_sparse:
        smplx2smpl = smplx2smpl.to_dense()
    return smplx, smplx2smpl


def compute_sequence_imu(body_pose, betas, global_orient, transl, smplx_model, smplx2smpl, smooth_n=4):
    verts = smplx_model(
        body_pose.unsqueeze(0),
        betas.unsqueeze(0),
        global_orient.unsqueeze(0),
        transl.unsqueeze(0),
    )[0]
    smpl_verts = torch.einsum("sv,fvc->fsc", smplx2smpl, verts)
    imu_acc = synthesize_acceleration(smpl_verts[:, IMU_VERT_IDS], smooth_n=smooth_n)

    global_rot = compute_global_rotations(global_orient, body_pose, smplx_model)
    imu_rot = global_rot[:, IMU_JOINT_IDS]
    return imu_acc, imu_rot


def build_f_imu(imu_acc, imu_rot, sensor_mask, include_combo_mask=True):
    length = imu_acc.shape[0]
    imu_rot6d = matrix_to_rotation_6d(imu_rot).reshape(length, -1)
    parts = [imu_acc.reshape(length, -1), imu_rot6d]
    if include_combo_mask:
        parts.append(sensor_mask.float().unsqueeze(0).repeat(length, 1))
    return torch.cat(parts, dim=-1), imu_rot6d


def cache_file_for_sequence(cache_dir, seq_name):
    cache_dir = Path(cache_dir)
    digest = hashlib.sha1(seq_name.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.pt"
