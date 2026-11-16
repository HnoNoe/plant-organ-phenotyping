"""projection_hungarian: 跨视图叶片实例身份关联。

算法对应 GeoProMatch 论文 2.5 节的机制（本仓库为独立复现实现）：
1. 共享稀疏投影支撑：稀疏点若能投影进某视图的某实例掩码，则该点属于该实例的支撑集；
2. 逐视角对打分：支撑集 Jaccard 相似度；
3. 一对一指派：每对视角用匈牙利算法做一对一匹配（不会一个实例吞掉对面多个）；
4. 同视图冲突保护：一个组件（全局叶片）在同一个视角最多占一个实例，按分数从高到低
   建图，冲突的边直接拒绝；
5. 低支持实例只给 warning，不静默合并（审计要求）。

输入契约（与 COLMAP 稀疏模型对齐）：
  - cameras: {cam_id: (model, width, height, params)}
  - images:  {img_name: (cam_id, qvec(4), tvec(3), xys(N,2), point3D_ids(N))}
  - points3D: {pt_id: xyz(3)}
  - masks:   {img_name: {inst_id: mask_hxW bool}}
输出：
  - global_instances: [{global_id, members: {img_name: inst_id}, support_points, best_pair_score}]
  - diagnostics: 逐视角对匹配矩阵、接受/拒绝记录、低支持警告
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchConfig:
    score_threshold: float = 0.1      # 视角对匹配的最低 Jaccard
    min_support_points: int = 3       # 全局叶片至少需要的支撑点数，低于则 warning
    use_mask_iou: bool = False        # True 时用投影像素 IoU 代替支撑点 Jaccard


@dataclass
class Diagnostics:
    pair_reports: list = field(default_factory=list)
    rejected_edges: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "pair_reports": self.pair_reports,
            "rejected_edges": self.rejected_edges,
            "warnings": self.warnings,
        }


def _quat_to_rot(qvec):
    q = np.asarray(qvec, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    q = q / n
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_points(xyz, qvec, tvec, camera):
    """世界坐标点 -> 像素坐标，返回 (u, v, depth) 数组。相机外参 x_cam = R @ x + t。"""
    model, W, H, params = camera
    R = _quat_to_rot(qvec)
    xc = (R @ xyz.T).T + np.asarray(tvec)
    z = xc[:, 2]
    valid = z > 1e-6
    u = np.zeros(len(xyz))
    v = np.zeros(len(xyz))
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
        u = fx * xc[:, 0] / np.maximum(z, 1e-9) + cx
        v = fy * xc[:, 1] / np.maximum(z, 1e-9) + cy
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        u = f * xc[:, 0] / np.maximum(z, 1e-9) + cx
        v = f * xc[:, 1] / np.maximum(z, 1e-9) + cy
    else:
        raise ValueError(f"unsupported camera model: {model}")
    return u, v, z, valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def build_support(images, points3d, cameras, masks):
    """每个 (image, instance) 的支撑稀疏点集合。

    返回 support[img][inst] = set(point_id)
    """
    support = {}
    for img_name, (cam_id, qvec, tvec, _xys, _pids) in images.items():
        cam = cameras[cam_id]
        inst_masks = masks.get(img_name, {})
        if not inst_masks:
            continue
        pid_list = np.array(sorted(points3d.keys()))
        xyz = np.stack([points3d[p] for p in pid_list])
        u, v, _z, valid = project_points(xyz, qvec, tvec, cam)
        sup = {inst: set() for inst in inst_masks}
        uv = np.stack([u, v], axis=1)
        for idx, pid in enumerate(pid_list):
            if not valid[idx]:
                continue
            px, py = uv[idx]
            for inst, m in inst_masks.items():
                if m[int(round(py)), int(round(px))]:
                    sup[inst].add(int(pid))
                    break  # 一个点只记第一个命中的实例（掩码不应重叠）
        support[img_name] = sup
    return support


def _pair_score(sup_a, sup_b, cfg: MatchConfig):
    inter = len(sup_a & sup_b)
    union = len(sup_a | sup_b)
    if union == 0:
        return 0.0
    return inter / union


def match_all_pairs(images, support, cfg: MatchConfig, diag: Diagnostics):
    """逐视角对匈牙利指派，返回候选边列表 [(score, (img_a, inst_a), (img_b, inst_b))]。"""
    edges = []
    imgs = sorted(support.keys())
    for img_a, img_b in itertools.combinations(imgs, 2):
        sup_a = support[img_a]
        sup_b = support[img_b]
        if not sup_a or not sup_b:
            continue
        insts_a = sorted(sup_a.keys())
        insts_b = sorted(sup_b.keys())
        S = np.zeros((len(insts_a), len(insts_b)))
        for i, a in enumerate(insts_a):
            for j, b in enumerate(insts_b):
                S[i, j] = _pair_score(sup_a[a], sup_b[b], cfg)
        row, col = linear_sum_assignment(-S)
        pair_report = {"pair": [img_a, img_b], "accepted": [], "rejected": []}
        for i, j in zip(row, col):
            score = float(S[i, j])
            ea = (img_a, insts_a[i])
            eb = (img_b, insts_b[j])
            if score >= cfg.score_threshold:
                edges.append((score, ea, eb))
                pair_report["accepted"].append({"a": list(ea), "b": list(eb), "score": score})
            else:
                pair_report["rejected"].append({"a": list(ea), "b": list(eb), "score": score})
        diag.pair_reports.append(pair_report)
    edges.sort(key=lambda e: -e[0])
    return edges


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            return True
        return False


def build_global_instances(edges, support, cfg: MatchConfig, diag: Diagnostics):
    """按分数降序建图；同视图冲突（组件在某视角已占实例）的边拒绝。返回全局实例列表。"""
    nodes = [(img, inst) for img, sup in support.items() for inst in sup]
    uf = _UnionFind(nodes)
    comp_view_inst = {}  # root -> {img: inst}

    def root_of(x):
        return uf.find(x)

    for score, ea, eb in edges:
        ra, rb = root_of(ea), root_of(eb)
        merged_views_a = dict(comp_view_inst.get(ra, {ea[0]: ea[1]}))
        merged_views_b = comp_view_inst.get(rb, {eb[0]: eb[1]})
        conflict = False
        for img, inst in merged_views_b.items():
            if img in merged_views_a and merged_views_a[img] != inst:
                conflict = True
                break
        if conflict:
            diag.rejected_edges.append(
                {"edge": [list(ea), list(eb)], "score": score, "reason": "same_view_conflict"}
            )
            continue
        if ra != rb:
            uf.union(ea, eb)
            root = uf.find(ea)
            merged_views_a.update(merged_views_b)
            comp_view_inst.pop(rb, None)
            comp_view_inst[root] = merged_views_a

    groups = {}
    for node in nodes:
        groups.setdefault(uf.find(node), []).append(node)

    instances = []
    for gid, members in enumerate(sorted(groups.values(), key=lambda ms: -len(ms))):
        mem_map = {img: inst for img, inst in members}
        support_pts = set()
        for img, inst in members:
            support_pts |= support[img].get(inst, set())
        member_set = set(members)
        best = max(
            (s for s, ea, eb in edges if ea in member_set and eb in member_set),
            default=0.0,
        )
        rec = {
            "global_id": gid,
            "members": mem_map,
            "n_views": len(members),
            "support_points": len(support_pts),
            "best_pair_score": best,
        }
        if len(support_pts) < cfg.min_support_points:
            rec["warning"] = "low_support"
            diag.warnings.append(
                {"global_id": gid, "reason": "low_support",
                 "support_points": len(support_pts),
                 "threshold": cfg.min_support_points}
            )
        instances.append(rec)
    return instances


def run_matching(images, points3d, cameras, masks, cfg: MatchConfig | None = None):
    """端到端：支撑 -> 逐对匹配 -> 全局身份。返回 (instances, diagnostics)。"""
    cfg = cfg or MatchConfig()
    diag = Diagnostics()
    support = build_support(images, points3d, cameras, masks)
    edges = match_all_pairs(images, support, cfg, diag)
    instances = build_global_instances(edges, support, cfg, diag)
    return instances, diag, support
