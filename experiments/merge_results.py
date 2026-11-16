"""Merge partial JSON outputs (ansatz_comparison_part*.json -> ansatz_comparison.json, etc.)."""
import glob
import json
import os
import sys

from common import MET

for base in ["ansatz_comparison", "physics_loss_ablation"]:
    parts = sorted(glob.glob(os.path.join(MET, f"{base}_part*.json")))
    if not parts:
        continue
    out = {}
    for pth in parts:
        d = json.load(open(pth))
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            elif isinstance(v, dict):
                out[k] = dict(v)
            elif k not in out:
                out[k] = v
            elif k == "total_time_s":
                out[k] = out[k] + v
    out["merged_from"] = [os.path.basename(p) for p in parts]
    json.dump(out, open(os.path.join(MET, f"{base}.json"), "w"), indent=2)
    print(base, {k: len(v) if isinstance(v, dict) else v for k, v in out.items()})
