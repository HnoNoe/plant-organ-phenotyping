"""阶段式管线入口：从 configs 展开阶段命令并执行（argv 风格，不走 shell）。

用法：
  python scripts/pipeline.py --dataset synthetic_demo [--stages match,traits]

阶段：priors(载掩码) -> colmap(读稀疏模型) -> match(跨视图匹配) ->
      propagate(身份传播) -> postprocess(器官后处理) -> traits(表型) -> quality(报告)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from organ_phenotyping.matching.projection_hungarian import MatchConfig, run_matching
from organ_phenotyping.geometry.colmap_io import load_sparse_model
from organ_phenotyping.geometry.propagate import propagate_labels
from organ_phenotyping.organs.postprocess import (merge_fragments,
                                                  split_measurement_cloud)
from organ_phenotyping.traits.measure import measure_organs
from organ_phenotyping.quality.reports import (write_matching_diagnostics,
                                               write_postprocess_diagnostics,
                                               write_traits)


def load_masks(masks_dir):
    masks = {}
    for f in sorted(os.listdir(masks_dir)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(masks_dir, f))
            masks[os.path.splitext(f)[0]] = {int(k): z[k] for k in z.files}
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stages", default=None, help="逗号分隔，默认跑全部")
    ap.add_argument("--config", default="configs/pipeline.toml")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, args.config), "rb") as f:
        cfg = tomllib.load(f)
    with open(os.path.join(root, "configs", "datasets", f"{args.dataset}.toml"), "rb") as f:
        dcfg = tomllib.load(f)["dataset"]

    stages = args.stages.split(",") if args.stages else cfg["stages"]["order"]
    out_dir = os.path.join(root, "outputs", dcfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    ctx = {}
    mcfg = MatchConfig(**cfg.get("match", {}))

    if "priors" in stages:
        ctx["masks"] = load_masks(os.path.join(root, dcfg["masks_dir"]))
        print(f"[priors] loaded masks for {len(ctx['masks'])} views")
    if "colmap" in stages:
        ctx["sparse"] = load_sparse_model(os.path.join(root, dcfg["sparse_dir"]))
        print(f"[colmap] cameras={len(ctx['sparse'][0])} images={len(ctx['sparse'][1])} points={len(ctx['sparse'][2])}")
    if "match" in stages:
        cameras, images, points3d = ctx["sparse"]
        ctx["instances"], ctx["diag"], ctx["support"] = run_matching(
            images, points3d, cameras, ctx["masks"], mcfg)
        print(f"[match] global leaf instances: {len(ctx['instances'])}")
    if "propagate" in stages:
        cameras, images, _ = ctx["sparse"]
        dense = np.load(os.path.join(root, dcfg["dense_path"]))
        ctx["points"], gt = dense[:, :3], dense[:, 3].astype(int)
        ctx["labels"], ctx["votes"] = propagate_labels(
            ctx["points"], images, cameras, ctx["masks"], ctx["instances"],
            min_votes=cfg.get("propagate", {}).get("min_votes", 1))
        # 全局 id 与 GT id 无对应关系，用匈牙利找最优映射再算准确率
        from scipy.optimize import linear_sum_assignment
        leaf = gt > 0
        pred_leaf = ctx["labels"][leaf]
        gt_leaf = gt[leaf]
        pred_ids = sorted(set(pred_leaf) - {-1})
        gt_ids = sorted(set(gt_leaf))
        C = np.zeros((len(pred_ids), len(gt_ids)))
        for i, p in enumerate(pred_ids):
            for j, g in enumerate(gt_ids):
                C[i, j] = ((pred_leaf == p) & (gt_leaf == g)).sum()
        ri, cj = linear_sum_assignment(-C)
        matched = C[ri, cj].sum()
        acc = matched / leaf.sum()
        print(f"[propagate] dense points={len(ctx['points'])}, organ-label accuracy vs GT={acc:.4f}")
    if "postprocess" in stages:
        pcfg = cfg.get("postprocess", {})
        ctx["labels"], ctx["merge_report"] = merge_fragments(
            ctx["labels"], ctx["points"], voxel=pcfg.get("voxel", 0.2),
            min_frag_points=pcfg.get("min_frag_points", 50))
        ctx["meas_mask"], ctx["viz_mask"] = split_measurement_cloud(ctx["labels"])
        print(f"[postprocess] measurement points={int(ctx['meas_mask'].sum())}, "
              f"visualization points={int(ctx['viz_mask'].sum())}")
    if "traits" in stages:
        ctx["per_organ"], ctx["plant"] = measure_organs(
            ctx["points"], ctx["labels"], stem_label=int(dcfg.get("stem_label", 0)))
        print(f"[traits] organs={ctx['plant']['n_organs']} "
              f"height={ctx['plant']['height']:.2f} total_leaf_area={ctx['plant']['total_leaf_area']:.1f}")
    if "quality" in stages:
        write_matching_diagnostics(ctx["diag"], ctx["instances"], out_dir)
        write_postprocess_diagnostics(ctx["merge_report"],
                                      ctx["meas_mask"].sum(), ctx["viz_mask"].sum(), out_dir)
        write_traits(ctx["per_organ"], ctx["plant"], out_dir)
        print(f"[quality] reports written to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
