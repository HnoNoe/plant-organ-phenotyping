"""器官表型量测：叶面积/叶长/叶宽/叶倾角、株高、茎径。

定义（几何代理量，与人工协议的关系需在数据说明中交代——参见审稿意见 S3）：
- 叶面积：点云在 PCA 最优平面投影的 alpha 形状近似，这里用投影凸包面积；
- 叶长/叶宽：投影后沿第一/第二主方向的 95% 分位跨度（抗离群点）；
- 叶倾角：叶平面法向与竖直方向的夹角；
- 株高：茎基部参考面到全株最高点的 z 向跨度；
- 茎径：茎轴向 1/4 高度处截面径向距离的第 20 百分位 ×2。
坐标单位假定厘米（与 5plant 数据集一致）。
"""
from __future__ import annotations

import numpy as np


def pca_frame(xyz):
    c = xyz.mean(axis=0)
    X = xyz - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return c, vt  # vt 行向量为三个主方向


def measure_leaf(xyz):
    """返回 dict：area/length/width/inclination_deg。"""
    if len(xyz) < 10:
        return {"area": np.nan, "length": np.nan, "width": np.nan,
                "inclination_deg": np.nan, "n_points": len(xyz)}
    _c, vt = pca_frame(xyz)
    plane = vt[:2]                      # 叶片近似平面
    normal = vt[2]
    proj = (xyz - xyz.mean(axis=0)) @ plane.T
    # 凸包面积（单调链）
    pts = proj[np.lexsort((proj[:, 1], proj[:, 0]))]
    hull = _convex_hull(pts)
    area = _poly_area(hull) if len(hull) >= 3 else 0.0
    e1, e2 = proj[:, 0], proj[:, 1]
    length = np.percentile(e1, 97.5) - np.percentile(e1, 2.5)
    width = np.percentile(e2, 97.5) - np.percentile(e2, 2.5)
    cos = abs(normal @ np.array([0, 0, 1.0]))
    inclination = float(np.degrees(np.arccos(np.clip(cos, 0, 1))))
    return {"area": float(area), "length": float(length), "width": float(width),
            "inclination_deg": inclination, "n_points": int(len(xyz))}


def measure_plant_height(points_xyz, base_z=None):
    if base_z is None:
        base_z = np.percentile(points_xyz[:, 2], 1)
    return float(points_xyz[:, 2].max() - base_z)


def measure_stem_diameter(stem_xyz, percentile=20):
    """轴向取 z，1/4 高度处截面，径向距离的第 percentile 百分位 ×2。"""
    if len(stem_xyz) < 20:
        return np.nan
    z0, z1 = stem_xyz[:, 2].min(), stem_xyz[:, 2].max()
    band = stem_xyz[(stem_xyz[:, 2] >= z0 + 0.25 * (z1 - z0) - 0.1) &
                    (stem_xyz[:, 2] <= z0 + 0.25 * (z1 - z0) + 0.1)]
    if len(band) < 5:
        band = stem_xyz
    c = band.mean(axis=0)
    r = np.linalg.norm(band[:, :2] - c[:2], axis=1)
    return float(2 * np.percentile(r, percentile))


def measure_organs(points_xyz, labels, stem_label=-1):
    """全株量测入口。返回 (per_organ list, plant traits dict)。"""
    per_organ = []
    for gid in sorted(set(labels) - {-1}):
        xyz = points_xyz[labels == gid]
        if gid == stem_label:
            per_organ.append({"global_id": gid, "type": "stem",
                              "n_points": int(len(xyz))})
        else:
            rec = {"global_id": gid, "type": "leaf"}
            rec.update(measure_leaf(xyz))
            per_organ.append(rec)
    stem = points_xyz[labels == stem_label] if stem_label in set(labels) else points_xyz
    plant = {
        "height": measure_plant_height(points_xyz),
        "stem_diameter": measure_stem_diameter(stem),
        "n_organs": len(per_organ),
        "total_leaf_area": sum(o.get("area", 0) for o in per_organ if o["type"] == "leaf"),
    }
    return per_organ, plant


def _convex_hull(pts):
    """单调链凸包，返回顶点数组。"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _poly_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
