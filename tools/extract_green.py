"""从 PLY 点云按颜色筛出绿色植物点，另存 npy 并渲染核验图。

用法：python tools/extract_green.py fused.ply out_prefix
产物：<out_prefix>_green.npy（xyz 数组）、<out_prefix>_green.png
"""
from __future__ import annotations

import sys

import numpy as np


def read_ply(path):
    props = []
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, t))
            elif line == "end_header":
                break
        dtype = np.dtype([(nm, {"float": "<f4", "uchar": "u1", "int": "<i4"}[t])
                          for nm, t in props])
        data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(float)
    names = data.dtype.names
    if "red" in names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(float)
    else:
        rgb = np.ones((len(xyz), 3)) * 128
    return xyz, rgb


def green_mask(rgb):
    r_, g_, b_ = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return (g_ > r_ * 1.15) & (g_ > b_ * 1.15) & (g_ > 40)


def main():
    ply, out_prefix = sys.argv[1], sys.argv[2]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz, rgb = read_ply(ply)
    sel = xyz[green_mask(rgb)]
    print(f"green points: {len(sel)} / {len(xyz)}")
    np.save(out_prefix + "_green.npy", sel)

    s = sel if len(sel) <= 60000 else sel[np.random.default_rng(0).choice(
        len(sel), 60000, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (i, j), t in zip(axes, [(0, 1), (0, 2)], ["top (xy)", "side (xz)"]):
        ax.scatter(s[:, i], s[:, j], s=0.6, c="#7ee081", linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(t)
        ax.set_facecolor("#111111")
    fig.savefig(out_prefix + "_green.png", dpi=110, bbox_inches="tight",
                facecolor="white")
    print("saved", out_prefix + "_green.png")


if __name__ == "__main__":
    main()
