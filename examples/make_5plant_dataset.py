# 从 5plant 真值点云生成基准数据集（渲染多视角 + 实例掩码 + COLMAP 格式）。

与 make_synthetic_dataset.py 的区别：几何与器官标签来自真实植物点云，
因此真值（器官身份、性状）是"真实"的，不是人工捏的。

用法：python examples/make_5plant_dataset.py <labeled_txt> <out_name> [n_views]
输出：examples/data/<out_name>/
  sparse/0/{cameras,images,points3D}.txt   COLMAP text
  masks/<view>.npz                          实例掩码
  dense.npy                                 稠密点云 + 重映射标签（茎=0，叶=1..K）
  ground_truth.json                         叶全局 id -> 可见视图列表

标签约定：5plant labeled 文件第 4 列 = 实例 id；茎 = z 跨度最大的实例。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT_BASE = os.path.join(HERE, "data")

W, H = 960, 720


def quat_from_rot(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i_m = int(np.argmax(np.diag(R)))
    j, k = (i_m + 1) % 3, (i_m + 2) % 3
    s = np.sqrt(max(1e-9, 1.0 + R[i_m, i_m] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i_m] = 0.25 * s
    q[1 + j] = (R[j, i_m] + R[i_m, j]) / s
    q[1 + k] = (R[k, i_m] + R[i_m, k]) / s
    return q


def make_cameras_ring(center, radius, n, z_c, f):
    """相机环绕环，两个俯仰角交替，看向植株中心。"""
    cameras, images = {}, {}
    for i in range(n):
        theta = 2 * np.pi * i / n
        el = np.radians(12.0 if i % 2 == 0 else -6.0)
        eye = np.array([center[0] + radius * np.cos(theta),
                        center[1] + radius * np.sin(theta),
                        z_c + radius * np.sin(el)])
        target = np.array([center[0], center[1], z_c])
        up = np.array([0, 0, 1.0])
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R = np.stack([right, true_up, fwd], axis=0)
        t = -R @ eye
        cameras[i + 1] = ("PINHOLE", W, H, (f, f, W / 2.0, H / 2.0))
        images[f"view_{i:02d}"] = (i + 1, quat_from_rot(R), t)
    return cameras, images


def project_pinhole(xyz, R, t, f):
    xc = (R @ xyz.T).T + t
    z = xc[:, 2]
    u = f * xc[:, 0] / np.maximum(z, 1e-9) + W / 2
    v = f * xc[:, 1] / np.maximum(z, 1e-9) + H / 2
    return u, v, z


def quat_to_rot(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    src = sys.argv[1]
    name = sys.argv[2]
    n_views = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    out = os.path.join(OUT_BASE, name)
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)

    data = np.loadtxt(src)
    xyz, inst = data[:, :3], data[:, 3].astype(int)
    n = len(xyz)

    # 茎 = z 跨度最大的实例；叶重映射为 1..K
    spans = {i: xyz[inst == i, 2].max() - xyz[inst == i, 2].min() for i in set(inst)}
    stem_inst = max(spans, key=lambda i: (spans[i], -i))
    leaf_insts = sorted(set(inst) - {stem_inst})
    remap = {stem_inst: 0}
    for k, i in enumerate(leaf_insts, start=1):
        remap[i] = k
    labels = np.array([remap[i] for i in inst])
    K = len(leaf_insts)

    center = xyz.mean(axis=0)
    r_plant = np.linalg.norm(xyz[:, :2] - center[:2], axis=1).max()
    f = float(W)
    radius = 3.0 * r_plant
    print(f"points={n} organs={K} leaves + stem(stem_inst={stem_inst}) "
          f"r_plant={r_plant:.1f} cam_radius={radius:.1f}")

    cameras, images_meta = make_cameras_ring(center, radius, n_views, center[2], f)

    # 稀疏支撑点 = 1/3 抽样；掩码由支撑点投影溅射
    sup_idx = np.arange(0, n, 3)
    sup_pids = sup_idx
    points3d = {int(k): xyz[k] for k in sup_pids}

    with open(os.path.join(out, "sparse", "0", "cameras.txt"), "w") as fp:
        fp.write("# id model width height params\n")
        for cid, (model, w, h, params) in cameras.items():
            fp.write(f"{cid} {model} {w} {h} {' '.join(f'{p:.6f}' for p in params)}\n")
    with open(os.path.join(out, "sparse", "0", "points3D.txt"), "w") as fp:
        fp.write("# id x y z\n")
        for p, pxyz in points3d.items():
            fp.write(f"{p} {pxyz[0]:.6f} {pxyz[1]:.6f} {pxyz[2]:.6f}\n")

    # 像素尺度：盘面半径对应多少像素，决定溅射圆盘半径
    px_per_unit = f * r_plant / radius          # 主视图里 r_plant 对应的像素数
    disk_r = max(3, int(0.06 * px_per_unit))    # 约为叶面尺度的 6%

    gt_groups = {k: [] for k in range(1, K + 1)}
    masks = {}
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as fp:
        fp.write("# id qvec tvec cam_id name\n")
        for i, (vname, (cid, q, t)) in enumerate(sorted(images_meta.items())):
            R = quat_to_rot(q)
            u_all, v_all, z_all = project_pinhole(xyz[sup_idx], R, t, f)
            fp.write(f"{i+1} {' '.join(f'{x:.8f}' for x in q)} "
                     f"{' '.join(f'{x:.8f}' for x in t)} {cid} {vname}\n")
            fp.write(" ".join(f"{u:.2f} {v:.2f} {p}" for u, v, p in
                              zip(u_all, v_all, sup_pids)) + "\n")
            img_masks = {}
            for k in range(1, K + 1):
                sel = labels[sup_idx] == k
                m = np.zeros((H, W), dtype=bool)
                for u, v, z in zip(u_all[sel], v_all[sel], z_all[sel]):
                    if z > 1e-6 and disk_r <= u < W - disk_r and disk_r <= v < H - disk_r:
                        yy, xx = np.ogrid[:H, :W]
                        m |= (xx - u) ** 2 + (yy - v) ** 2 <= disk_r ** 2
                if m.sum() >= 100:
                    img_masks[k] = m
                    gt_groups[k].append(vname)
            # 茎掩码（inst 0），不参与匹配但保留给后续分析
            sel = labels[sup_idx] == 0
            m = np.zeros((H, W), dtype=bool)
            for u, v, z in zip(u_all[sel], v_all[sel], z_all[sel]):
                if z > 1e-6 and disk_r <= u < W - disk_r and disk_r <= v < H - disk_r:
                    yy, xx = np.ogrid[:H, :W]
                    m |= (xx - u) ** 2 + (yy - v) ** 2 <= disk_r ** 2
            if m.sum() >= 100:
                img_masks[0] = m
            masks[vname] = img_masks

    for vname, im in masks.items():
        np.savez_compressed(os.path.join(out, "masks", f"{vname}.npz"),
                            **{str(k): v for k, v in im.items()})

    np.save(os.path.join(out, "dense.npy"), np.hstack([xyz, labels[:, None]]))
    with open(os.path.join(out, "ground_truth.json"), "w", encoding="utf-8") as fp:
        json.dump({"leaf_views": {str(k): v for k, v in gt_groups.items()},
                   "stem_label": 0, "src": os.path.basename(src)}, fp,
                  ensure_ascii=False, indent=2)
    print(f"dataset written to {out}  (disk_r={disk_r}px, "
          f"leaf visibility: {[len(gt_groups[k]) for k in range(1, K+1)]})")


if __name__ == "__main__":
    main()
