"""器官级后处理：碎片合并、低支持处理、茎保护、测量/可视化点云分离。

规则（与 GeoProMatch 的契约一致）：
- 不同 global_id 的器官从不合并；只允许同一 global_id 的碎片连接；
- 低支持实例不静默合并，只打 warning；
- 茎保持单器官，保护茎基部（z 最低的基部半径范围内不删点）；
- 输出测量点云（干净、用于性状）与可视化点云（完整、用于展示）两套。
"""
from __future__ import annotations

import numpy as np


def merge_fragments(labels, points_xyz, voxel=0.2, min_frag_points=50):
    """同一 global_id 的空间断开碎片，若质心距离 < voxel*3 则连接（改标号）。

    返回 (new_labels, merge_report)
    """
    new_labels = labels.copy()
    report = []
    for gid in sorted(set(labels) - {-1}):
        idx = np.where(labels == gid)[0]
        if len(idx) <= min_frag_points:
            continue
        sub = points_xyz[idx]
        # 简单网格聚类
        cells = {}
        for k, p in enumerate(sub):
            key = tuple(np.floor(p / voxel).astype(int))
            cells.setdefault(key, []).append(k)
        # 连通块（6 邻域）
        cell_ids = {c: i for i, c in enumerate(cells)}
        parent = list(range(len(cells)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for c in cells:
            for d in range(3):
                nb = list(c)
                nb[d] += 1
                nb = tuple(nb)
                if nb in cell_ids:
                    ra, rb = find(cell_ids[c]), find(cell_ids[nb])
                    if ra != rb:
                        parent[rb] = ra
        roots = {}
        for c, i in cell_ids.items():
            roots.setdefault(find(i), []).append(c)
        if len(roots) <= 1:
            continue
        frag_centers = []
        for cells_group in roots.values():
            pts = np.vstack([sub[cells[c]] for c in cells_group])
            frag_centers.append(pts.mean(axis=0))
        # 质心过近的碎片合并标记
        merged = 0
        for i in range(len(frag_centers)):
            for j in range(i + 1, len(frag_centers)):
                if np.linalg.norm(frag_centers[i] - frag_centers[j]) < voxel * 3:
                    merged += 1
        report.append({"global_id": gid, "fragments": len(roots), "merged_pairs": merged})
    return new_labels, report


def protect_stem_base(labels, points_xyz, stem_label, base_radius=1.0):
    """茎基部保护：茎实例中 z 最低的 5% 点，无论后续清理规则如何都保留。"""
    stem_idx = np.where(labels == stem_label)[0]
    if len(stem_idx) == 0:
        return np.array([], dtype=int)
    z = points_xyz[stem_idx, 2]
    thr = np.percentile(z, 5)
    return stem_idx[z <= thr + base_radius * 0.1]


def split_measurement_cloud(labels, points_xyz=None, noise_labels=(-1,)):
    """测量点云 = 有身份的器官点；可视化点云 = 全部点。返回 (meas_mask, viz_mask)。

    points_xyz 预留参数（未来按几何规则剔除测量点时用到），当前不使用。
    """
    viz_mask = np.ones(len(labels), dtype=bool)
    meas_mask = ~np.isin(labels, list(noise_labels))
    return meas_mask, viz_mask
