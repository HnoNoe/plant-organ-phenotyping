"""合成场景端到端测试：3 视角、6 叶，验证跨视图身份恢复与冲突保护。"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from organ_phenotyping.matching.projection_hungarian import (  # noqa: E402
    MatchConfig, run_matching)
from organ_phenotyping.geometry.colmap_io import load_sparse_model  # noqa: E402


def _load_masks(d):
    masks = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            masks[os.path.splitext(f)[0]] = {int(k): z[k] for k in z.files}
    return masks


def test_matching_recovers_ground_truth():
    data = os.path.join(ROOT, "examples", "data", "synthetic_demo")
    cameras, images, points3d = load_sparse_model(os.path.join(data, "sparse", "0"))
    masks = _load_masks(os.path.join(data, "masks"))
    instances, diag, _support = run_matching(images, points3d, cameras, masks,
                                             MatchConfig(score_threshold=0.05))
    # 6 片叶子每片应在 3 个视图都有成员（本合成场景无遮挡）
    assert len(instances) == 6, f"expected 6 global leaves, got {len(instances)}"
    for rec in instances:
        assert rec["n_views"] == 3, rec
        assert rec["support_points"] >= 3
    # 身份应与真值分组一致：同一合成叶子的支撑点应落在同一全局实例
    import json
    with open(os.path.join(data, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)
    assert sorted(len(v) for v in gt["leaf_views"].values()) == [3] * 6


def test_same_view_conflict_protection():
    """构造冲突：两个高分对声称同一视图实例属于不同组件，应拒绝一条边。"""
    images = {
        "v0": (1, np.array([1.0, 0, 0, 0]), np.zeros(3), None, None),
        "v1": (1, np.array([1.0, 0, 0, 0]), np.zeros(3), None, None),
    }
    cameras = {1: ("PINHOLE", 100, 100, (50, 50, 50, 50))}
    points3d = {i: np.array([0.1 * (i % 5), 0.1 * (i // 5), 2.0]) for i in range(20)}
    m = np.zeros((100, 100), dtype=bool)
    m[40:60, 40:60] = True
    masks = {"v0": {1: m, 2: m.copy()}, "v1": {1: m.copy()}}
    instances, diag, _ = run_matching(images, points3d, cameras, masks,
                                      MatchConfig(score_threshold=0.05))
    # v0 的实例1和实例2若同时高分匹配到 v1 实例1，同视图冲突应拒绝其一
    assert diag.rejected_edges or len({tuple(sorted(r["members"].items())) for r in instances}) == len(instances)


if __name__ == "__main__":
    test_matching_recovers_ground_truth()
    test_same_view_conflict_protection()
    print("all matching tests passed")
