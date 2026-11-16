"""质量报告：matching_diagnostics.json / postprocess_diagnostics.json / traits.csv。

审计原则（对应论文"可审计"卖点与审稿 S1/S2 的要求）：
- 每个全局器官可追溯到它在哪些视图由哪些实例支持；
- 每个被拒绝的匹配边都有理由；
- 低支持实例是 warning 不是静默合并。
"""
from __future__ import annotations

import csv
import json
import os


def write_matching_diagnostics(diag, instances, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = diag.to_dict()
    payload["global_instances"] = instances
    with open(os.path.join(out_dir, "matching_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def write_postprocess_diagnostics(merge_report, meas_count, viz_count, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "fragment_merges": merge_report,
        "measurement_points": int(meas_count),
        "visualization_points": int(viz_count),
    }
    with open(os.path.join(out_dir, "postprocess_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=int)
    return payload


def write_traits(per_organ, plant, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "traits.csv"), "w", newline="encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["global_id", "type", "n_points", "area",
                                          "length", "width", "inclination_deg"])
        w.writeheader()
        for rec in per_organ:
            w.writerow({k: rec.get(k) for k in w.fieldnames})
    with open(os.path.join(out_dir, "plant_traits.json"), "w", encoding="utf-8") as f:
        json.dump(plant, f, ensure_ascii=False, indent=2)
