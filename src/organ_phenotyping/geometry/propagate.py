"""把全局叶片身份传播到稠密点云：逐点投影到各视图，按掩码命中投票。

对应 GeoProMatch 的机制：全局身份先建立，再传播到有尺度的稠密几何；
测量点云与可视化点云分离（artifact separation），本模块只负责"赋身份"这一步。
"""
from __future__ import annotations

import numpy as np

from ..matching.projection_hungarian import project_points


def propagate_labels(points_xyz, images, cameras, masks, global_instances,
                     top_k_views=4, min_votes=1):
    """每个稠密点投影到最多 top_k 个可见视图，收集命中的 (global_id) 投票。

    返回 (labels, votes)，labels[i] = global_id 或 -1（无身份）。
    """
    # 建立 (img, inst) -> global_id 反查表
    inst2global = {}
    for rec in global_instances:
        for img, inst in rec["members"].items():
            inst2global[(img, inst)] = rec["global_id"]

    n = len(points_xyz)
    labels = np.full(n, -1, dtype=int)
    votes = np.zeros(n, dtype=int)
    for img_name, (cam_id, qvec, tvec, _xys, _pids) in images.items():
        cam = cameras[cam_id]
        inst_masks = masks.get(img_name, {})
        if not inst_masks:
            continue
        u, v, z, valid = project_points(points_xyz, qvec, tvec, cam)
        uv = np.stack([u, v], axis=1)
        hit_gid = np.full(n, -1, dtype=int)
        for inst, m in inst_masks.items():
            gid = inst2global.get((img_name, inst), -1)
            if gid < 0:
                continue
            px = np.round(uv[valid]).astype(int)
            m_hits = np.zeros(int(valid.sum()), dtype=bool)
            inb = (px[:, 0] >= 0) & (px[:, 0] < m.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < m.shape[0])
            m_hits[inb] = m[px[inb, 1], px[inb, 0]]
            idx = np.where(valid)[0][m_hits]
            first = (hit_gid[idx] == -1)
            hit_gid[idx[first]] = gid
        gained = (hit_gid >= 0) & (labels == -1)
        labels[gained] = hit_gid[gained]
        votes[gained] += 1
        # 已有点若新视图给出一致身份，票数累加；不一致保留先到者（可在此处加冲突诊断）
        agree = (hit_gid >= 0) & (labels >= 0) & (hit_gid == labels)
        votes[agree] += 1

    labels[votes < min_votes] = -1
    return labels, votes
