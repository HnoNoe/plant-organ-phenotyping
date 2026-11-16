"""生成合成演示数据集：3 个相机、6 片"叶子"簇、已知真值身份。

用法：python examples/make_synthetic_dataset.py
输出：examples/data/synthetic_demo/
  sparse/0/{cameras,images,points3D}.txt   COLMAP text 格式
  masks/<view>.npy                          每视图实例掩码（bool 数组字典存 npz）
  dense.npy                                 稠密点云（带真值标签，最后一列）
  ground_truth.json                         叶子分组的真值
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "synthetic_demo")

W, H = 640, 480
F = 500.0


def make_cameras_ring(n=3, radius=6.0, height=4.0):
    """相机环：看向原点，返回 (cameras, images) 的原始字段。"""
    cameras = {}
    images = {}
    for i in range(n):
        theta = 2 * np.pi * i / n
        eye = np.array([radius * np.cos(theta), radius * np.sin(theta), height])
        target = np.zeros(3)
        up = np.array([0, 0, 1.0])
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R = np.stack([right, true_up, fwd], axis=0)   # world->cam
        t = -R @ eye
        # COLMAP 四元数 q (w,x,y,z)，R 由 q 旋转得到
        tr = np.trace(R)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                          (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
        else:
            i_m = np.argmax(np.diag(R))
            j, k = (i_m + 1) % 3, (i_m + 2) % 3
            s = np.sqrt(max(1e-9, 1.0 + R[i_m, i_m] - R[j, j] - R[k, k])) * 2
            q = np.zeros(4)
            q[0] = (R[k, j] - R[j, k]) / s
            q[1 + i_m] = 0.25 * s
            q[1 + j] = (R[j, i_m] + R[i_m, j]) / s
            q[1 + k] = (R[k, i_m] + R[i_m, k]) / s
        cameras[i + 1] = ("PINHOLE", W, H, (F, F, W / 2, H / 2))
        images[f"view_{i}"] = (i + 1, q, t)
    return cameras, images


def project_pinhole(xyz, R, t):
    xc = (R @ xyz.T).T + t
    z = np.maximum(xc[:, 2], 1e-9)
    u = F * xc[:, 0] / z + W / 2
    v = F * xc[:, 1] / z + H / 2
    return u, v


def quat_to_rot(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    rng = np.random.default_rng(42)
    os.makedirs(os.path.join(OUT, "sparse", "0"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)

    # 6 片叶子，每片一个 3D 高斯簇；1 根茎
    leaves = [rng.normal(loc=[np.cos(a) * 1.2, np.sin(a) * 1.2, 1.5 + 0.3 * i],
                         scale=0.25, size=(120, 3))
              for i, a in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False))]
    stem = rng.normal(loc=[0, 0, 0.9], scale=[0.05, 0.05, 0.5], size=(80, 3))

    cameras, images_meta = make_cameras_ring(3)
    points3d = {}
    pid = 0
    for leaf in leaves:
        for p in leaf[::6]:  # 稀疏支撑点
            points3d[pid] = p
            pid += 1
    stem_pids = list(range(pid, pid + len(stem[::6])))
    for p in stem[::6]:
        points3d[pid] = p
        pid += 1

    # 写 COLMAP 风格文件
    with open(os.path.join(OUT, "sparse", "0", "cameras.txt"), "w") as f:
        f.write("# id model width height params\n")
        for cid, (model, w, h, params) in cameras.items():
            f.write(f"{cid} {model} {w} {h} {' '.join(f'{p:.6f}' for p in params)}\n")
    with open(os.path.join(OUT, "sparse", "0", "points3D.txt"), "w") as f:
        f.write("# id x y z\n")
        for p, xyz in points3d.items():
            f.write(f"{p} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")

    gt_groups = {i: [] for i in range(6)}
    masks = {}
    with open(os.path.join(OUT, "sparse", "0", "images.txt"), "w") as f:
        f.write("# id qvec tvec cam_id name\n")
        for i, (name, (cid, q, t)) in enumerate(sorted(images_meta.items())):
            R = quat_to_rot(q)
            u_all, v_all = project_pinhole(np.stack(list(points3d.values())), R, t)
            all_pids = np.array(sorted(points3d.keys()))
            img_masks = {}
            # 叶子实例：该叶子的支撑点投影邻域画圆盘
            for li, leaf in enumerate(leaves):
                sup_pids = all_pids[li * 20:(li + 1) * 20]
                m = np.zeros((H, W), dtype=bool)
                sel = np.isin(all_pids, sup_pids)
                for u, v in zip(u_all[sel], v_all[sel]):
                    if 8 <= u < W - 8 and 8 <= v < H - 8:
                        yy, xx = np.ogrid[:H, :W]
                        m |= (xx - u) ** 2 + (yy - v) ** 2 <= 12 ** 2
                if m.sum() > 0:
                    img_masks[li + 1] = m
                    gt_groups[li].append(name)
            masks[name] = img_masks
            f.write(f"{i+1} {' '.join(f'{x:.8f}' for x in q)} {' '.join(f'{x:.8f}' for x in t)} {cid} {name}\n")
            f.write(" ".join(f"{u:.2f} {v:.2f} {p}" for u, v, p in
                             zip(u_all, v_all, all_pids)) + "\n")

    for name, im in masks.items():
        np.savez_compressed(os.path.join(OUT, "masks", f"{name}.npz"),
                            **{str(k): v for k, v in im.items()})

    # 稠密点云 + 真值标签（叶 1..6，茎 0）
    dense, labels = [], []
    for li, leaf in enumerate(leaves):
        dense.append(leaf)
        labels += [li + 1] * len(leaf)
    dense.append(stem)
    labels += [0] * len(stem)
    dense = np.vstack(dense)
    labels = np.array(labels)
    np.save(os.path.join(OUT, "dense.npy"),
            np.hstack([dense, labels[:, None]]))

    with open(os.path.join(OUT, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump({"leaf_views": gt_groups, "stem_label": 0}, f, ensure_ascii=False, indent=2)
    print("synthetic dataset written to", OUT)


if __name__ == "__main__":
    main()
