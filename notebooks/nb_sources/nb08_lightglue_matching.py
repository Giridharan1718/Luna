# %% [markdown]
# ## 1. Objective
# Match query/reference feature sets with **LightGlue**, producing matched keypoint lists,
# match counts and line visualizations for every retrieval pair.

# %% [markdown]
# ## 2. Dataset Loading

# %%
import sys
from pathlib import Path

def _find_project_root() -> Path:
    # anchor at the directory that actually contains the project markers,
    # independent of the kernel's working directory
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "lunarai_lib").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import torch

from lunarai_lib.config import load_config
from lunarai_lib.matching import SuperPointLightGlue
from lunarai_lib.io_utils import save_json, save_rows_csv

cfg = load_config()
feat_files = sorted(cfg.MATCHING_DIR.glob("feat_*.npz"))
print(f"{len(feat_files)} feature pairs found")

# %% [markdown]
# ## 3. Configuration

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
matcher = SuperPointLightGlue(max_keypoints=int(cfg.MATCHING.get("superpoint_max_kp", 2048)),
                              match_threshold=float(cfg.MATCHING.get("lightglue_threshold", 0.2)),
                              device=device)
print("backend:", "superpoint+lightglue" if matcher.available else "sift fallback")

# %% [markdown]
# ## 4. Implementation

# %%
match_rows = []
for i, f in enumerate(feat_files):
    d = np.load(f, allow_pickle=True)
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    if q_img is None or r_img is None:
        continue
    m = matcher.match(q_img, r_img)
    np.savez_compressed(cfg.MATCHING_DIR / f"matches_{f.stem}.npz",
                        kp0=m.kp0, kp1=m.kp1, q_path=d["q_path"], r_path=d["r_path"],
                        q_id=d["q_id"], r_id=d["r_id"])
    match_rows.append({"pair": f.stem, "query_patch": str(d["q_id"]), "ref_patch": str(d["r_id"]),
                       "matches": m.n_matches, "matcher": m.matcher})
    print(f"  {f.stem}: {m.n_matches:4d} matches ({m.matcher})")

save_rows_csv(match_rows, cfg.MATCHING_DIR / "match_counts.csv")

# %% [markdown]
# ## 5. Visualization — match lines for the densest pair

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lunarai_lib.geometry import draw_matches

if match_rows:
    best = max(match_rows, key=lambda r: r["matches"])
    d = np.load(cfg.MATCHING_DIR / f"matches_{best['pair']}.npz", allow_pickle=True)
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    vis = draw_matches(q_img, r_img, d["kp0"], d["kp1"], max_draw=80)
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"LightGlue matches: {best['pair']} ({best['matches']} matches)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "lightglue_matches.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {
    "pairs_matched": len(match_rows),
    "total_matches": int(sum(r["matches"] for r in match_rows)),
    "mean_matches": round(float(np.mean([r["matches"] for r in match_rows])), 1) if match_rows else 0.0,
    "backend": match_rows[0]["matcher"] if match_rows else "none",
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/matching/matches_feat_XXX.npz`
# - `outputs/matching/match_counts.csv`
# - `outputs/visualizations/lightglue_matches.png`

# %%
save_json(metrics, cfg.METRICS_DIR / "lightglue_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Pairs with undecodable images are skipped.
# - Zero-match pairs are recorded (count=0) rather than dropped, so downstream statistics
#   reflect reality; MAGSAC in NB10 will treat them as failures.

# %% [markdown]
# ## 9. Explanation
# LightGlue attends jointly over both keypoint sets and assigns matches with confidence
# filtering (threshold 0.2), which tolerates the illumination/scale gaps between sensors far
# better than brute-force descriptor NNDR matching.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 09 applies ANMS to the matched keypoints for uniform spatial distribution.
