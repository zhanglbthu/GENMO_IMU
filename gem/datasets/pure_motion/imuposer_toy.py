# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch
from torch.utils.data import Dataset

from gem.datasets.pure_motion.imu_utils import DEFAULT_SENSOR_COMBOS, build_f_imu
from gem.utils.motion_utils import get_tgtcoord_rootparam
from gem.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from gem.utils.pylogger import Log
from gem.utils.rotation_conversions import matrix_to_axis_angle


IMUPOSER_DEVICE_TO_SLOT = {
    0: 0,  # left wrist
    1: 1,  # right wrist
    2: 2,  # left phone / left thigh
    3: 3,  # right phone / right thigh
    4: 4,  # head
}


class IMUPoserToyDataset(Dataset):
    def __init__(
        self,
        data_path="/root/autodl-tmp/dataset/processed/eval/imuposer_toy_multimodal_train.pt",
        motion_frames=120,
        window_stride=120,
        combo_name="lw_rp_h",
        rotate_y_aug=True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.motion_frames = motion_frames
        self.window_stride = window_stride
        self.combo_name = combo_name
        self.rotate_y_aug = rotate_y_aug
        if combo_name not in DEFAULT_SENSOR_COMBOS:
            raise ValueError(f"Unknown combo_name: {combo_name}")
        self._load_dataset()
        self._build_index()

    def _load_dataset(self):
        if not self.data_path.exists():
            raise FileNotFoundError(self.data_path)
        Log.info(f"[IMUPoserToy] Loading {self.data_path} ...")
        self.data = torch.load(self.data_path, map_location="cpu")
        self.num_sequences = len(self.data["acc"])
        Log.info(f"[IMUPoserToy] {self.num_sequences} sequences loaded")

    def _build_index(self):
        self.idx2meta = []
        total_frames = 0
        for seq_idx in range(self.num_sequences):
            seq_len = self.data["acc"][seq_idx].shape[0]
            if seq_len < self.motion_frames:
                continue
            total_frames += seq_len
            for start in range(0, seq_len - self.motion_frames + 1, self.window_stride):
                self.idx2meta.append((seq_idx, start))
        Log.info(
            f"[IMUPoserToy] {total_frames / 30 / 3600:.2f} hours -> {len(self.idx2meta)} windows"
        )

    def __len__(self):
        return len(self.idx2meta)

    def _build_identity_camera(self, length):
        T_w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        K_fullimg = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        R_c2gv = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        return T_w2c, K_fullimg, R_c2gv

    def _sample_yaw_rot(self):
        angle = torch.rand(1).item() * 2 * torch.pi
        aa = torch.tensor([[0.0, angle, 0.0]], dtype=torch.float32)
        from gem.utils.rotation_conversions import axis_angle_to_matrix

        return axis_angle_to_matrix(aa).squeeze(0)

    def _apply_yaw_to_motion(self, global_orient, transl, yaw_rot):
        from gem.utils.rotation_conversions import axis_angle_to_matrix

        global_rot = yaw_rot.unsqueeze(0) @ axis_angle_to_matrix(global_orient)
        global_orient = matrix_to_axis_angle(global_rot)
        transl = (yaw_rot @ transl.T).T
        return global_orient, transl

    def _apply_yaw_to_imu(self, imu_acc, imu_rot, yaw_rot):
        imu_acc = torch.einsum("ab,lsb->lsa", yaw_rot, imu_acc)
        imu_rot = torch.einsum("ab,lsbc->lsac", yaw_rot, imu_rot)
        return imu_acc, imu_rot

    def _map_to_slots(self, acc5, ori5):
        length = acc5.shape[0]
        acc7 = torch.zeros(length, 7, 3, dtype=torch.float32)
        ori7 = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(length, 7, 1, 1)
        for src_idx, dst_idx in IMUPOSER_DEVICE_TO_SLOT.items():
            acc7[:, dst_idx] = acc5[:, src_idx]
            ori7[:, dst_idx] = ori5[:, src_idx]
        sensor_mask = torch.zeros(7, dtype=torch.float32)
        sensor_mask[DEFAULT_SENSOR_COMBOS[self.combo_name]] = 1.0
        for slot_idx in range(7):
            if sensor_mask[slot_idx] == 0:
                acc7[:, slot_idx] = 0
                ori7[:, slot_idx] = 0
        return acc7, ori7, sensor_mask

    def __getitem__(self, idx):
        seq_idx, start = self.idx2meta[idx]
        end = start + self.motion_frames
        pose_mats = self.data["pose"][seq_idx][start:end].float()
        global_orient = matrix_to_axis_angle(pose_mats[:, 0])
        body_pose = matrix_to_axis_angle(pose_mats[:, 1:22]).reshape(-1, 63)
        transl = self.data["tran"][seq_idx][start:end].float()
        global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="ay->ay")
        betas = torch.zeros(body_pose.shape[0], 10, dtype=torch.float32)

        imu_acc5 = self.data["acc"][seq_idx][start:end].float()
        imu_rot5 = self.data["ori"][seq_idx][start:end].float()
        imu_acc, imu_rot, sensor_mask = self._map_to_slots(imu_acc5, imu_rot5)

        if self.rotate_y_aug:
            yaw_rot = self._sample_yaw_rot()
            global_orient, transl = self._apply_yaw_to_motion(global_orient, transl, yaw_rot)
            imu_acc, imu_rot = self._apply_yaw_to_imu(imu_acc, imu_rot, yaw_rot)

        f_imu, imu_rot6d = build_f_imu(imu_acc, imu_rot, sensor_mask, include_combo_mask=True)
        length = body_pose.shape[0]
        T_w2c, K_fullimg, R_c2gv = self._build_identity_camera(length)
        smpl_params_w = {
            "body_pose": body_pose.clone(),
            "betas": betas.clone(),
            "global_orient": global_orient.clone(),
            "transl": transl.clone(),
        }
        smpl_params_c = {k: v.clone() for k, v in smpl_params_w.items()}

        return {
            "meta": {
                "data_name": "imuposer_toy",
                "seq_name": f"toy:{seq_idx}",
                "idx": idx,
                "imu_combo_name": self.combo_name,
                "imu_combo_sensor_ids": DEFAULT_SENSOR_COMBOS[self.combo_name],
            },
            "length": length,
            "smpl_params_c": repeat_to_max_len_dict(smpl_params_c, length),
            "smpl_params_w": repeat_to_max_len_dict(smpl_params_w, length),
            "R_c2gv": repeat_to_max_len(R_c2gv, length),
            "gravity_vec": torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32),
            "bbx_xys": torch.zeros((length, 3), dtype=torch.float32),
            "K_fullimg": repeat_to_max_len(K_fullimg, length),
            "f_imgseq": torch.zeros((length, 1024), dtype=torch.float32),
            "kp2d": torch.zeros((length, 17, 3), dtype=torch.float32),
            "cam_angvel": torch.zeros((length, 6), dtype=torch.float32),
            "cam_tvel": torch.zeros((length, 3), dtype=torch.float32),
            "noisy_cam_tvel": torch.zeros((length, 3), dtype=torch.float32),
            "T_w2c": repeat_to_max_len(T_w2c, length),
            "imu_acc": imu_acc,
            "imu_rot": imu_rot,
            "imu_rot6d": imu_rot6d,
            "imu_sensor_mask": sensor_mask,
            "imu_signal_mask": sensor_mask.unsqueeze(0).repeat(length, 1),
            "f_imu": f_imu,
            "mask": {
                "valid": get_valid_mask(length, length),
                "humanoid": get_valid_mask(length, 0),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(length, 0),
                "has_cam_mask": get_valid_mask(length, 0),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, 0),
                "has_imu_mask": get_valid_mask(length, length),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }
