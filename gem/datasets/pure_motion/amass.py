# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .amass_common import AmassMotionMixin
from .base_dataset import BaseDataset


class AmassDataset(AmassMotionMixin, BaseDataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
        root=None,
        split="train",
        val_ratio=0.1,
        split_seed=1234,
    ):
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
        BaseDataset.__init__(self, cam_augmentation, limit_size)

    def _load_data(self, idx):
        return self.load_amass_sequence(idx)
