#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from matplotlib import colormaps


TAB10_COLORS = tuple(tuple(float(c) for c in color) for color in colormaps["tab10"].colors)


@contextmanager
def managed_xvfb(display=":0", width=1920, height=1080):
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", f"{width}x{height}x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.0)
        old_display = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = display
        yield
    finally:
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _prepare_vertices(vertices, subject_spacing):
    vertices = vertices.copy()
    center = (vertices.reshape(-1, 3).min(axis=0) + vertices.reshape(-1, 3).max(axis=0)) / 2.0
    vertices -= center
    num_subjects = vertices.shape[0]
    offsets = (np.arange(num_subjects, dtype=np.float32) - (num_subjects - 1) / 2.0) * subject_spacing
    vertices[:, :, :, 0] += offsets[:, None, None]
    return vertices


def render_meshes_side_by_side(
    vertices,
    faces,
    output_path,
    width=1920,
    height=1080,
    fps=30,
    subject_spacing=1.4,
    xvfb_display=":0",
    xvfb_width=1920,
    xvfb_height=1080,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_prefix = output_path.with_suffix("")

    vertices = _prepare_vertices(vertices, subject_spacing)
    faces = np.asarray(faces, dtype=np.int32)

    with managed_xvfb(display=xvfb_display, width=xvfb_width, height=xvfb_height):
        viewer = HeadlessRenderer(size=(width, height))
        try:
            viewer.auto_set_camera_target = False
            for idx in range(vertices.shape[0]):
                mesh = Meshes(vertices[idx], faces, name=f"subject_{idx}", flat_shading=False, draw_edges=False)
                mesh.color = tuple(float(x) for x in TAB10_COLORS[idx % len(TAB10_COLORS)]) + (1.0,)
                viewer.scene.add(mesh)
            viewer._init_scene()
            viewer.scene.fps = fps
            viewer.playback_fps = fps
            viewer.export_fps = fps
            viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            camera_z = max(4.8, 3.4 + 0.85 * vertices.shape[0])
            viewer.scene.camera.position = np.array([0.0, 0.35, camera_z], dtype=np.float32)
            viewer.save_video(video_dir=str(tmp_prefix), output_fps=fps)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()

    produced = sorted(output_path.parent.glob(f"{tmp_prefix.name}_*.mp4"))
    if not produced:
        raise FileNotFoundError(f"AITViewer did not produce an output mp4 for prefix {tmp_prefix}.")
    shutil.move(str(produced[0]), str(output_path))
    for extra in produced[1:]:
        extra.unlink(missing_ok=True)


@torch.no_grad()
def render_smpl_side_by_side_aitviewer(
    body_model,
    pose_t,
    tran_t,
    pose_p,
    tran_p,
    output_path,
    fps=30,
    width=1920,
    height=1080,
    subject_spacing=1.4,
):
    _, _, gt_vertices = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=True)
    _, _, pred_vertices = body_model.forward_kinematics(pose_p, tran=tran_p, calc_mesh=True)
    vertices = torch.stack([gt_vertices, pred_vertices], dim=0).cpu().numpy()
    faces = body_model.face.cpu().numpy() if torch.is_tensor(body_model.face) else np.asarray(body_model.face)
    render_meshes_side_by_side(
        vertices=vertices,
        faces=faces,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        subject_spacing=subject_spacing,
        xvfb_width=width,
        xvfb_height=height,
    )
