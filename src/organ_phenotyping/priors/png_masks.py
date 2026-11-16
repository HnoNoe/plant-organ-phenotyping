"""从 RGBA 透明 PNG 读取逐视图叶片实例掩码（frozen 2D priors 的一种载体）。

约定：alpha>0 为前景；每个实例用纯色区分（颜色即实例 id）。
输出 {img_name: {inst_id: bool array (H,W)}}，inst_id 从 1 开始（0 留给背景/茎，
与 5plant 语义约定一致：茎实例统一为 0）。
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image


def load_view_masks(png_dir, min_pixels=100):
    masks = {}
    for f in sorted(os.listdir(png_dir)):
        if not f.lower().endswith(".png"):
            continue
        img = np.array(Image.open(os.path.join(png_dir, f)).convert("RGBA"))
        alpha = img[..., 3] > 0
        rgb = img[..., :3]
        img_masks = {}
        colors = {}
        next_id = [1]
        ys, xs = np.where(alpha)
        for y, x in zip(ys, xs):
            color = tuple(rgb[y, x])
            if color not in colors:
                colors[color] = next_id[0]
                next_id[0] += 1
        for color, iid in colors.items():
            m = alpha & np.all(rgb == color, axis=-1)
            if m.sum() >= min_pixels:
                img_masks[iid] = m
        masks[os.path.splitext(f)[0]] = img_masks
    return masks


def load_instance_png(png_path, min_pixels=100):
    """单张'局部叶片实例'图：每种彩色 = 一个实例，白色/灰色 = 茎(0)。"""
    img = np.array(Image.open(png_path).convert("RGBA"))
    alpha = img[..., 3] > 0
    rgb = img[..., :3].astype(int)
    out = {}
    colors = {}
    ys, xs = np.where(alpha)
    for y, x in zip(ys, xs):
        colors.setdefault(tuple(rgb[y, x]), None)
    next_id = [1]
    for color in colors:
        m = alpha & np.all(rgb == color, axis=-1)
        if m.sum() < min_pixels:
            continue
        is_stem = max(color) - min(color) < 30  # 近白/近灰当茎
        iid = 0 if is_stem else next_id[0]
        if not is_stem:
            next_id[0] += 1
        out[iid] = m
    return out
