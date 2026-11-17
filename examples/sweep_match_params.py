"""基准扫描：在 5plant 数据集上扫匹配超参，观察实例数与准确率。

这是"算法革新"的迭代回路：改 MatchConfig -> 看 (global_instances, accuracy)。
支持指标：
  acc  = 稠密传播标签准确率（会受传播阶段投影污染影响，端到端视角）
  sup  = 支撑点级准确率（只看匹配算法产物，干净指标）
用法：python examples/sweep_match_params.py <dataset_name>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from organ_phenotyping.matching.projection_hungarian import MatchConfig, run_matching
from organ_phenotyping.geometry.colmap_io import load_sparse_model
from organ_phenotyping.geometry.propagate import propagate_labels


def load_masks(masks_dir):
    masks = {}
    for f in sorted(os.listdir(masks_dir)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(masks_dir, f))
            masks[os.path.splitext(f)[0]] = {int(k): z[k] for k in z.files}
    return masks


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def evaluate(dataset, **overrides):
    import tomllib
    with open(os.path.join(ROOT, "configs", "datasets", f"{dataset}.toml"), "rb") as f:
        dcfg = tomllib.load(f)["dataset"]
    masks = load_masks(os.path.join(ROOT, dcfg["masks_dir"]))
    cameras, images, points3d = load_sparse_model(os.path.join(ROOT, dcfg["sparse_dir"]))
    cfg = MatchConfig(**overrides)
    instances, diag, _ = run_matching(images, points3d, cameras, masks, cfg)
    dense = np.load(os.path.join(ROOT, dcfg["dense_path"]))
    pts, gt = dense[:, :3], dense[:, 3].astype(int)
    labels, _ = propagate_labels(pts, images, cameras, masks, instances)
    from scipy.optimize import linear_sum_assignment
    leaf = gt > 0
    pred_ids = sorted(set(labels[leaf]) - {-1})
    gt_ids = sorted(set(gt[leaf]))
    C = np.zeros((len(pred_ids), len(gt_ids)))
    for i, p in enumerate(pred_ids):
        for j, g in enumerate(gt_ids):
            C[i, j] = ((labels[leaf] == p) & (gt[leaf] == g)).sum()
    ri, cj = linear_sum_assignment(-C)
    acc = C[ri, cj].sum() / leaf.sum()
    # 支撑点级准确率：只看匹配算法自己的产物（支撑集），不受传播污染。
    # gid 与 GT 无对应，先匈牙利对齐再算。
    pred_rows, gt_rows = [], []
    for rec in instances:
        rows = [p for p in rec["support_point_ids"] if gt[p] > 0]
        pred_rows += [rec["global_id"]] * len(rows)
        gt_rows += gt[rows].tolist()
    if pred_rows:
        P = np.array(pred_rows)
        G = np.array(gt_rows)
        pids_p = sorted(set(P.tolist()))
        gids_g = sorted(set(G.tolist()))
        Cs = np.zeros((len(pids_p), len(gids_g)))
        for i, p in enumerate(pids_p):
            for j, g in enumerate(gids_g):
                Cs[i, j] = ((P == p) & (G == g)).sum()
        rs, cs = linear_sum_assignment(-Cs)
        sup_acc = float(Cs[rs, cs].sum() / len(P))
    else:
        sup_acc = 0.0
    return {
        "n_global": len(instances),
        "n_warnings": len(diag.warnings),
        "n_rejected": len(diag.rejected_edges),
        "n_rescued": len(diag.rescued_edges),
        "accuracy": float(acc),
        "support_acc": sup_acc,
    }


if __name__ == "__main__":
    dataset = sys.argv[1]
    print(f"# dataset={dataset}")
    hdr = f"{'mode':>8} {'multi':>5} {'rescue':>6} {'thr':>5} | {'global':>6} {'warn':>4} {'rej':>4} {'resc':>4} {'acc':>7} {'sup':>7}"
    print(hdr)
    variants = [
        # (score_mode, multi_hit_support, rescue_threshold, score_threshold)
        ("jaccard", False, 0.0, 0.1),    # 基线（论文机制）
        ("jaccard", False, 0.05, 0.1),   # +碎片抢救
        ("jaccard", True, 0.05, 0.1),    # +多点支撑
        ("overlap", False, 0.05, 0.1),   # overlap 打分
        ("overlap", True, 0.05, 0.1),    # 全开
        ("overlap", True, 0.10, 0.1),    # 抢救提到主阈值
    ]
    for mode, multi, rescue, thr in variants:
        r = evaluate(dataset, score_mode=mode, multi_hit_support=multi,
                     rescue_threshold=rescue, score_threshold=thr)
        print(f"{mode:>8} {str(multi):>5} {rescue:>6} {thr:>5} | "
              f"{r['n_global']:>6} {r['n_warnings']:>4} {r['n_rejected']:>4} "
              f"{r['n_rescued']:>4} {r['accuracy']:>7.4f} {r['support_acc']:>7.4f}")
