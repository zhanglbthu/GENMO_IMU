# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from gem.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
from gem.utils.motion_utils import get_tgtcoord_rootparam
from gem.utils.pylogger import Log

from .amass_common import AmassMotionMixin
from .imu_utils import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SENSOR_COMBOS,
    GRAVITY_VEC,
    IMU_JOINT_IDS,
    IMU_VERT_IDS,
    build_f_imu,
    cache_file_for_sequence,
    compute_sequence_imu,
    load_smpl_models,
)
from .utils import augment_betas, interpolate_smpl_params


class IMUAmassDataset(AmassMotionMixin, Dataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,
        skip_moyo=True,
        random1024=False,
        limit_size=None,
        root=None,
        augment_betas_std=0.1,
        rotate_y_aug=True,
        smooth_n=4,
        sensor_combos=None,
        combo_names=None,
        sample_combo=True,
        include_combo_mask=True,
        window_stride=None,
        interpolate_to_motion_frames=False,
        split="train",
        val_ratio=0.1,
        split_seed=1234,
        use_processed=True,
        processed_root="/root/autodl-tmp/dataset/AMASS/hmr4d_support/processed",
        cache_dir=str(DEFAULT_CACHE_DIR),
        use_cache=True,
    ):
        Dataset.__init__(self)
        self.use_processed = use_processed
        self.processed_root = processed_root
        if self.use_processed:
            self.root = Path(processed_root).expanduser()
            self.motion_frames = motion_frames
            self.l_factor = l_factor
            self.random1024 = random1024
            self.skip_moyo = skip_moyo
            self.dataset_name = "AMASS"
            self.split = split
            self.val_ratio = float(val_ratio)
            self.split_seed = int(split_seed)
        else:
            AmassMotionMixin.__init__(
                self,
                motion_frames=motion_frames,
                l_factor=l_factor,
                skip_moyo=skip_moyo,
                random1024=random1024,
                root=root,
                split=split,
                val_ratio=val_ratio,
                split_seed=split_seed,
            )
        self.limit_size = limit_size
        self.augment_betas_std = augment_betas_std
        self.rotate_y_aug = rotate_y_aug
        self.smooth_n = smooth_n
        self.sample_combo = sample_combo
        self.include_combo_mask = include_combo_mask
        self.window_stride = motion_frames if window_stride is None else int(window_stride)
        self.interpolate_to_motion_frames = interpolate_to_motion_frames
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.sensor_combos = {k: list(v) for k, v in DEFAULT_SENSOR_COMBOS.items()}
        if sensor_combos is not None:
            for k, v in sensor_combos.items():
                self.sensor_combos[k] = list(v)
        self.combo_names = (
            list(combo_names) if combo_names is not None else list(self.sensor_combos.keys())
        )
        if len(self.combo_names) == 0:
            raise ValueError("combo_names must not be empty")

        self._load_dataset()
        self._get_idx2meta()
        if (not self.use_processed) and (not self.use_cache):
            self.smplx, self.smplx2smpl = load_smpl_models()

    def _load_dataset(self):
        if not self.use_processed:
            return AmassMotionMixin._load_dataset(self)

        processed_root = Path(self.processed_root).expanduser()
        if not processed_root.exists():
            raise FileNotFoundError(f"Processed AMASS root not found: {processed_root}")

        files = sorted(
            [p for p in processed_root.glob("*.pt") if p.is_file() and "eval" not in p.parts]
        )
        if not files:
            raise FileNotFoundError(f"No processed AMASS .pt files found under {processed_root}")

        self.processed_data = {}
        self.processed_seq_meta = []
        tic = Log.time()
        Log.info(f"[AMASS processed] Loading from {processed_root} ...")
        for file_path in tqdm(files):
            ds_name = os.path.splitext(file_path.name)[0]
            ds = torch.load(file_path, map_location="cpu")
            self.processed_data[ds_name] = ds
            n_seq = len(ds["pose"])
            for seq_idx in range(n_seq):
                self.processed_seq_meta.append((ds_name, seq_idx))
        self.seqs = [f"{ds_name}:{seq_idx}" for ds_name, seq_idx in self.processed_seq_meta]
        Log.info(
            f"[AMASS processed] {len(self.seqs)} sequences across {len(files)} datasets. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        if not self.use_processed:
            self.idx2meta = []
            seq_lengths = []
            active_vids = self._select_split_vids(list(self.motion_files.keys()))
            for vid in active_vids:
                if self.skip_moyo and "moyo_smplxn" in vid:
                    continue
                seq_len = self.motion_files[vid]["pose"].shape[0]
                starts = self._window_starts(seq_len)
                if not starts:
                    continue
                seq_lengths.append(seq_len)
                self.idx2meta.extend([(vid, start) for start in starts])
            hours = sum(seq_lengths) / 30 / 3600
            Log.info(
                f"[{self.dataset_name}][split={self.split}] has {hours:.1f} hours motion -> Windowed to {len(self.idx2meta)} samples."
            )
            return

        self.idx2meta = []
        active_seqs = self._select_split_vids(self.processed_seq_meta)
        seq_lengths = []
        for ds_name, seq_idx in active_seqs:
            seq_len = self.processed_data[ds_name]["pose"][seq_idx].shape[0]
            starts = self._window_starts(seq_len)
            if not starts:
                continue
            seq_lengths.append(seq_len)
            self.idx2meta.extend([((ds_name, seq_idx), start) for start in starts])
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(
            f"[{self.dataset_name}][processed][split={self.split}] has {hours:.1f} hours motion -> Windowed to {len(self.idx2meta)} samples."
        )

    def _window_starts(self, seq_len):
        if seq_len < self.motion_frames:
            return []
        return list(range(0, seq_len - self.motion_frames + 1, self.window_stride))

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _build_identity_camera(self, length):
        T_w2c = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        K_fullimg = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        R_c2gv = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
        return T_w2c, K_fullimg, R_c2gv

    def _choose_combo(self):
        if not self.sample_combo or len(self.combo_names) == 1:
            combo_name = self.combo_names[0]
        else:
            combo_name = np.random.choice(self.combo_names)
        sensor_ids = self.sensor_combos[combo_name]
        sensor_mask = torch.zeros(len(IMU_VERT_IDS), dtype=torch.bool)
        sensor_mask[sensor_ids] = True
        return combo_name, sensor_mask

    def _sample_yaw_rot(self):
        angle = torch.rand(1).item() * 2 * torch.pi
        aa = torch.tensor([[0.0, angle, 0.0]], dtype=torch.float32)
        return axis_angle_to_matrix(aa).squeeze(0)

    def _apply_yaw_to_motion(self, data, yaw_rot):
        global_rot = yaw_rot.unsqueeze(0) @ axis_angle_to_matrix(data["global_orient"])
        data["global_orient"] = matrix_to_axis_angle(global_rot)
        data["transl"] = (yaw_rot @ data["transl"].T).T
        return data

    def _apply_yaw_to_imu(self, imu_acc, imu_rot, yaw_rot):
        imu_acc = torch.einsum("ab,lsb->lsa", yaw_rot, imu_acc)
        imu_rot = torch.einsum("ab,lsbc->lsac", yaw_rot, imu_rot)
        return imu_acc, imu_rot

    def _apply_sensor_combo(self, imu_acc, imu_rot):
        combo_name, sensor_mask = self._choose_combo()
        imu_acc = imu_acc.clone()
        imu_rot = imu_rot.clone()
        imu_acc[:, ~sensor_mask] = 0
        imu_rot[:, ~sensor_mask] = 0
        return imu_acc, imu_rot, combo_name, sensor_mask

    def _interpolate_imu(self, imu_acc, imu_rot, tgt_len):
        imu_acc = rearrange(imu_acc, "l s c -> (s c) 1 l")
        imu_acc = F.interpolate(imu_acc, tgt_len, mode="linear", align_corners=True)
        imu_acc = rearrange(imu_acc, "(s c) 1 l -> l s c", s=len(IMU_VERT_IDS), c=3)

        imu_rot6d = matrix_to_rotation_6d(imu_rot).reshape(imu_rot.shape[0], len(IMU_JOINT_IDS), 6)
        imu_rot6d = rearrange(imu_rot6d, "l s c -> (s c) 1 l")
        imu_rot6d = F.interpolate(imu_rot6d, tgt_len, mode="linear", align_corners=True)
        imu_rot6d = rearrange(imu_rot6d, "(s c) 1 l -> l s c", s=len(IMU_JOINT_IDS), c=6)
        imu_rot = rotation_6d_to_matrix(imu_rot6d)
        return imu_acc, imu_rot

    def _load_processed_sequence(self, ds_name, seq_idx):
        seq = self.processed_data[ds_name]
        pose_mats = seq["pose"][seq_idx].float()
        global_orient = matrix_to_axis_angle(pose_mats[:, 0])
        body_pose = matrix_to_axis_angle(pose_mats[:, 1:22]).reshape(-1, 63)
        transl = seq["tran"][seq_idx].float()
        betas = seq["shape"][seq_idx].float().unsqueeze(0).repeat(body_pose.shape[0], 1)
        imu_acc = seq["acc"][seq_idx].float()
        imu_rot = seq["ori"][seq_idx].float()
        return {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient,
            "transl": transl,
            "imu_acc": imu_acc,
            "imu_rot": imu_rot,
            "data_name": "amass",
            "seq_name": f"{ds_name}:{seq_idx}",
        }

    def _get_cache_file(self, seq_name):
        if self.cache_dir is None:
            raise FileNotFoundError("cache_dir is not configured")
        return cache_file_for_sequence(self.cache_dir, seq_name)

    def _load_cached_sequence(self, seq_name):
        cache_file = self._get_cache_file(seq_name)
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Missing IMU cache for sequence {seq_name}: {cache_file}. "
                "Run scripts/preprocess_amass_imu_cache.py first, or set use_cache=false."
            )
        seq_cache = torch.load(cache_file, map_location="cpu")
        return seq_cache["imu_acc"].float(), seq_cache["imu_rot"].float()

    def _make_return_dict(self, idx, data, imu_acc, imu_rot, combo_name, sensor_mask):
        length = data["body_pose"].shape[0]
        T_w2c, K_fullimg, R_c2gv = self._build_identity_camera(length)
        smpl_params_w = {
            "body_pose": data["body_pose"].clone(),
            "betas": data["betas"].clone(),
            "global_orient": data["global_orient"].clone(),
            "transl": data["transl"].clone(),
        }
        smpl_params_c = {k: v.clone() for k, v in smpl_params_w.items()}
        imu_sensor_mask = sensor_mask.float().unsqueeze(0).repeat(length, 1)
        f_imu, imu_rot6d = build_f_imu(
            imu_acc, imu_rot, sensor_mask.float(), include_combo_mask=self.include_combo_mask
        )
        if self.include_combo_mask:
            pass
        has_imu_mask = get_valid_mask(length, length)

        return {
            "meta": {
                "data_name": data["data_name"],
                "seq_name": data["seq_name"],
                "idx": idx,
                "amass_root": str(self.root),
                "imu_combo_name": combo_name,
                "imu_combo_sensor_ids": sensor_mask.nonzero(as_tuple=False).flatten().tolist(),
            },
            "length": length,
            "smpl_params_c": repeat_to_max_len_dict(smpl_params_c, length),
            "smpl_params_w": repeat_to_max_len_dict(smpl_params_w, length),
            "R_c2gv": repeat_to_max_len(R_c2gv, length),
            "gravity_vec": GRAVITY_VEC.clone(),
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
            "imu_sensor_mask": sensor_mask.float(),
            "imu_signal_mask": imu_sensor_mask,
            "f_imu": f_imu,
            "imu_sensor_vertex_ids": IMU_VERT_IDS.clone(),
            "imu_sensor_joint_ids": IMU_JOINT_IDS.clone(),
            "mask": {
                "valid": get_valid_mask(length, length),
                "humanoid": get_valid_mask(length, 0),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(length, 0),
                "has_cam_mask": get_valid_mask(length, 0),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, 0),
                "has_imu_mask": has_imu_mask,
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }

    def __getitem__(self, idx):
        mid, start_id = self.idx2meta[idx]
        if self.use_processed:
            ds_name, seq_idx = mid
            seq = self._load_processed_sequence(ds_name, seq_idx)
            start = start_id
            end = start + self.motion_frames
            data = {
                "body_pose": seq["body_pose"][start:end],
                "betas": seq["betas"][start:end],
                "global_orient": seq["global_orient"][start:end],
                "transl": seq["transl"][start:end],
                "data_name": seq["data_name"],
                "seq_name": seq["seq_name"],
            }
            data_name = data["data_name"]
            seq_name = data["seq_name"]
            if self.interpolate_to_motion_frames:
                data = interpolate_smpl_params(data, self.motion_frames)
                data["data_name"] = data_name
                data["seq_name"] = seq_name
            data["betas"] = augment_betas(data["betas"], std=self.augment_betas_std)
            imu_acc = seq["imu_acc"][start:end]
            imu_rot = seq["imu_rot"][start:end]
            if self.interpolate_to_motion_frames:
                imu_acc, imu_rot = self._interpolate_imu(imu_acc, imu_rot, self.motion_frames)
        else:
            raw_data = self.motion_files[mid]
            raw_len = raw_data["pose"].shape[0] - start_id
            full_data = {
                "body_pose": raw_data["pose"][start_id:, 3:],
                "betas": raw_data["beta"].repeat(raw_len, 1),
                "global_orient": raw_data["pose"][start_id:, :3],
                "transl": raw_data["trans"][start_id:],
            }
            start = 0
            end = self.motion_frames
            data = {k: v[start:end] for k, v in full_data.items()}

            if self.interpolate_to_motion_frames:
                data = interpolate_smpl_params(data, self.motion_frames)
            data["global_orient"], data["transl"], _ = get_tgtcoord_rootparam(
                data["global_orient"], data["transl"], tsf="az->ay"
            )
            data["data_name"] = "amass"
            data["seq_name"] = mid
            data["betas"] = augment_betas(data["betas"], std=self.augment_betas_std)

            if self.use_cache:
                imu_acc_full, imu_rot_full = self._load_cached_sequence(mid)
                imu_acc = imu_acc_full[start_id : start_id + self.motion_frames]
                imu_rot = imu_rot_full[start_id : start_id + self.motion_frames]
                if self.interpolate_to_motion_frames:
                    imu_acc, imu_rot = self._interpolate_imu(imu_acc, imu_rot, self.motion_frames)
            else:
                imu_acc, imu_rot = compute_sequence_imu(
                    data["body_pose"],
                    data["betas"],
                    data["global_orient"],
                    data["transl"],
                    self.smplx,
                    self.smplx2smpl,
                    smooth_n=self.smooth_n,
                )

        if self.rotate_y_aug:
            yaw_rot = self._sample_yaw_rot()
            data = self._apply_yaw_to_motion(data, yaw_rot)
            imu_acc, imu_rot = self._apply_yaw_to_imu(imu_acc, imu_rot, yaw_rot)

        imu_acc, imu_rot, combo_name, sensor_mask = self._apply_sensor_combo(imu_acc, imu_rot)

        return self._make_return_dict(idx, data, imu_acc, imu_rot, combo_name, sensor_mask)
