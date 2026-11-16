"""读取 COLMAP 稀疏模型（text 格式 cameras.txt / images.txt / points3D.txt）。"""
from __future__ import annotations

import numpy as np


def read_cameras(path):
    """返回 {cam_id: (model, width, height, params_tuple)}"""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = tuple(float(x) for x in parts[4:])
            cameras[cam_id] = (model, w, h, params)
    return cameras


def read_images(path):
    """返回 {image_name: (cam_id, qvec(4), tvec(3), xys, point3D_ids)}"""
    images = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        cam_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        name = parts[9] if len(parts) > 9 else parts[-1]
        data = lines[i + 1].split()
        xys = np.array([float(x) for x in data[0::3]]).reshape(-1, 2) if data else np.zeros((0, 2))
        pids = np.array([int(x) for x in data[2::3]]) if data else np.zeros(0, dtype=int)
        images[name] = (cam_id, qvec, tvec, xys, pids)
    return images


def read_points3d(path):
    """返回 {pt_id: xyz(3)}"""
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            points[pid] = xyz
    return points


def load_sparse_model(sparse_dir):
    import os
    return (
        read_cameras(os.path.join(sparse_dir, "cameras.txt")),
        read_images(os.path.join(sparse_dir, "images.txt")),
        read_points3d(os.path.join(sparse_dir, "points3D.txt")),
    )
