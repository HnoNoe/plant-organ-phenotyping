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
    score_threshold: float = 0.1      # 视角对匹配的最低分数
    min_support_points: int = 3       # 全局叶片至少需要的支撑点数，低于则 warning
    use_mask_iou: bool = False        # True 时用投影像素 IoU 代替支撑点分数
    score_mode: str = "jaccard"       # jaccard | overlap（overlap=|∩|/min，抗碎片）
    multi_hit_support: bool = False   # True 时一个支撑点可属于多个重叠实例掩码
    rescue_threshold: float = 0.0     # >0 时启用"碎片抢救"：低于主阈值但高于此值的
                                      # 孤立实例二次并入最优组件（记录在 rescued_edges）


@dataclass
class Diagnostics:
    pair_reports: list = field(default_factory=list)
    rejected_edges: list = field(default_factory=list)
    rescued_edges: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "pair_reports": self.pair_reports,
            "rejected_edges": self.rejected_edges,
            "rescued_edges": self.rescued_edges,
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


def build_support(images, points3d, cameras, masks, cfg: MatchConfig | None = None):
    """每个 (image, instance) 的支撑稀疏点集合。

    返回 support[img][inst] = set(point_id)
    """
    cfg = cfg or MatchConfig()
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
            px, py = int(round(uv[idx][0])), int(round(uv[idx][1]))
            mh, mw = next(iter(inst_masks.values())).shape
            if not (0 <= px < mw and 0 <= py < mh):
                continue
            for inst, m in inst_masks.items():
                if m[py, px]:
                    sup[inst].add(int(pid))
                    if not cfg.multi_hit_support:
                        break  # 默认一个点只记第一个命中的实例；multi_hit 时计入所有命中
        support[img_name] = sup
    return support


def _pair_score(sup_a, sup_b, cfg: MatchConfig):
    inter = len(sup_a & sup_b)
    if cfg.score_mode == "overlap":
        # overlap 系数：|∩|/min(|A|,|B|)。对"小支撑集被掩码重叠切碎"鲁棒，
        # 代价是对方更大时会高估——靠 score_threshold 与冲突保护兜底。
        denom = min(len(sup_a), len(sup_b))
        return inter / denom if denom else 0.0
    union = len(sup_a | sup_b)
    if union == 0:
        return 0.0
    return inter / union


def match_all_pairs(images, support, cfg: MatchConfig, diag: Diagnostics):
    """逐视角对匈牙利指派，返回候选边列表 [(score, (img_a, inst_a), (img_b, inst_b))]。"""
    edges = []
    pair_scores = {}  # (ea, eb) -> score，含被拒边，供"碎片抢救"复用
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
            pair_scores[(ea, eb)] = score
            if score >= cfg.score_threshold:
                edges.append((score, ea, eb))
                pair_report["accepted"].append({"a": list(ea), "b": list(eb), "score": score})
            else:
                pair_report["rejected"].append({"a": list(ea), "b": list(eb), "score": score})
        diag.pair_reports.append(pair_report)
    edges.sort(key=lambda e: -e[0])
    return edges, pair_scores


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


def build_global_instances(edges, support, cfg: MatchConfig, diag: Diagnostics,
                           pair_scores=None):
    """按分数降序建图；同视图冲突（组件在某视角已占实例）的边拒绝。返回全局实例列表。"""
    pair_scores = pair_scores or {}
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

    # ---- 碎片抢救：孤立（单视图）组件二次并入最优匹配组件 ----
    if cfg.rescue_threshold > 0:
        comp_root = {node: uf.find(node) for node in nodes}
        comp_members = {}
        for node, r in comp_root.items():
            comp_members.setdefault(r, []).append(node)
        # 每个节点的最佳跨组件边分数
        best_cross = {}  # node -> (score, other_node)
        for (ea, eb), s in pair_scores.items():
            ra, rb = comp_root[ea], comp_root[eb]
            if ra == rb:
                continue
            for a, b in ((ea, eb), (eb, ea)):
                if s > best_cross.get(a, (0.0, None))[0]:
                    best_cross[a] = (s, b)
        for r, members in comp_members.items():
            if len(members) != 1:
                continue
            node = members[0]
            if node not in best_cross:
                continue
            s, other = best_cross[node]
            if s < cfg.rescue_threshold:
                continue
            ro = uf.find(other)
            if ro == r:
                continue
            views_mine = {node[0]: node[1]}
            views_other = comp_view_inst.get(ro, {})
            if any(img in views_mine and views_mine[img] != inst
                   for img, inst in views_other.items()):
                continue  # 同视图冲突，不抢救
            uf.union(node, other)
            root = uf.find(node)
            merged = dict(views_other)
            merged.update(views_mine)
            comp_view_inst.pop(ro, None)
            comp_view_inst[root] = merged
            diag.rescued_edges.append(
                {"edge": [list(node), list(other)], "score": s})

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
            "support_point_ids": sorted(support_pts),
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
    support = build_support(images, points3d, cameras, masks, cfg)
    edges, pair_scores = match_all_pairs(images, support, cfg, diag)
    instances = build_global_instances(edges, support, cfg, diag, pair_scores)
    return instances, diag, support
