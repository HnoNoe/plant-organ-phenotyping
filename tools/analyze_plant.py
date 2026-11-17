"""从场景点云中定位中央植株并做结构分析（密度峰定位 + 体素聚类）。

用法：python tools/analyze_plant.py <fused.ply 或 *_green.npy> <out_prefix> [--radius 1.2]

产物：<out_prefix>_plant.npy（裁剪后的单株 xyz）、<out_prefix>_plant.png
 stdout 输出株高/冠幅/聚类数等统计量。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud")
    ap.add_argument("out_prefix")
    ap.add_argument("--radius", type=float, default=1.2)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.cloud.endswith(".npy"):
        gp = np.load(args.cloud)
    else:
        from extract_green import green_mask, read_ply
        xyz, rgb = read_ply(args.cloud)
        gp = xyz[green_mask(rgb)]
    print(f"green points: {len(gp)}")

    # xy 密度峰定位
    H, xedges, yedges = np.histogram2d(gp[:, 0], gp[:, 1], bins=120)
    Hs = ndimage.gaussian_filter(H, sigma=2)
    pi, pj = np.unravel_index(np.argmax(Hs), Hs.shape)
    cx = (xedges[pi] + xedges[pi + 1]) / 2
    cy = (yedges[pj] + yedges[pj + 1]) / 2
    print(f"density peak at ({cx:.2f}, {cy:.2f})")

    d = np.linalg.norm(gp[:, :2] - [cx, cy], axis=1)
    plant = gp[d <= args.radius]
    if len(plant) < 2000:
        plant = gp[d <= args.radius * 2]
    print(f"cropped plant points: {len(plant)}")

    # 体素连通域聚类（叶片候选）
    voxel = 0.06
    cells = {}
    for k, p in enumerate(plant):
        cells.setdefault(tuple(np.floor(p / voxel).astype(int)), []).append(k)
    keys = list(cells)
    key2id = {c: i for i, c in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in keys:
        for dd in range(3):
            nb = list(c)
            nb[dd] += 1
            nb = tuple(nb)
            if nb in key2id:
                ra, rb = find(key2id[c]), find(key2id[nb])
                if ra != rb:
                    parent[rb] = ra
    clusters = {}
    for c in keys:
        clusters.setdefault(find(key2id[c]), []).append(c)
    sizes = sorted((sum(len(cells[c]) for c in comp)
                    for comp in clusters.values()), reverse=True)
    sig = [s for s in sizes if s >= 200]
    print(f"clusters (>=200 pts): {len(sig)}, top sizes: {sizes[:8]}")

    z0, z1 = plant[:, 2].min(), plant[:, 2].max()
    foot = np.linalg.norm(plant[:, :2] - plant[:, :2].mean(axis=0), axis=1)
    print(f"plant height span: {z1 - z0:.2f} (z {z0:.2f}..{z1:.2f}), "
          f"canopy diameter ~{2 * foot.mean():.2f}")

    s = plant if len(plant) <= 80000 else plant[np.random.default_rng(0).choice(
        len(plant), 80000, replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    step = max(1, len(gp) // 30000)
    axes[0].scatter(gp[::step, 0], gp[::step, 1], s=0.3, c="#3a5f3a", linewidths=0)
    axes[0].add_patch(plt.Circle((cx, cy), args.radius, color="red", fill=False))
    axes[0].set_title("green points + crop circle")
    for ax, (i, j), t in zip(axes[1:], [(0, 1), (0, 2)], ["plant top (xy)", "plant side (xz)"]):
        ax.scatter(s[:, i], s[:, j], s=1.0, c="#7ee081", linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_facecolor("#111111")
    axes[0].set_aspect("equal")
    axes[0].set_facecolor("#111111")
    fig.savefig(args.out_prefix + "_plant.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    np.save(args.out_prefix + "_plant.npy", plant)
    print("saved", args.out_prefix + "_plant.png / .npy")


if __name__ == "__main__":
    main()
