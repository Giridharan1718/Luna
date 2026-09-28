# %% [markdown]
# ## 1. Objective
# Apply **ANMS** (Adaptive Non-Maximal Suppression) to matched keypoints to obtain a uniform
# spatial distribution, and quantify Coverage Score + Uniformity Score with distribution
# visualizations.

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

from lunarai_lib.config import load_config
from lunarai_lib.matching import anms, coverage_score, uniformity_score
from lunarai_lib.geometry import coverage_heatmap
from lunarai_lib.io_utils import save_json, save_rows_csv

cfg = load_config()
match_files = sorted(cfg.MATCHING_DIR.glob("matches_feat_*.npz"))
print(f"{len(match_files)} match sets")

# %% [markdown]
# ## 3. Configuration

# %%
ANMS_TARGET = int(cfg.MATCHING.get("anms_target", 300))
GAMMA = float(cfg.MATCHING.get("anms_gamma", 1.6))
GRID = (8, 8)

# %% [markdown]
# ## 4. Implementation

# %%
anms_rows = []
for f in match_files:
    d = np.load(f, allow_pickle=True)
    kp0, kp1 = d["kp0"], d["kp1"]
    if len(kp0) == 0:
        continue
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    shape = q_img.shape if q_img is not None else (256, 256)
    scores = np.ones(len(kp0))
    sel_kp, sel_idx = anms(kp0, scores, target=ANMS_TARGET, gamma=GAMMA)
    sel_kp1 = kp1[sel_idx]
    pair_id = f.stem.replace("matches_", "")   # matches_feat_001 -> feat_001
    np.savez_compressed(cfg.MATCHING_DIR / f"anms_{pair_id}.npz",
                        kp0=sel_kp, kp1=sel_kp1, q_path=d["q_path"], r_path=d["r_path"],
                        q_id=d["q_id"], r_id=d["r_id"])
    cov = coverage_score(sel_kp, shape, GRID)
    uni = uniformity_score(sel_kp, shape, GRID)
    cov_raw = coverage_score(kp0, shape, GRID)
    anms_rows.append({"pair": pair_id, "matches_in": len(kp0), "matches_out": len(sel_kp),
                      "coverage_before": round(cov_raw, 3), "coverage_after": round(cov, 3),
                      "uniformity": round(uni, 3)})
    print(f"  {pair_id}: {len(kp0):4d} -> {len(sel_kp):3d} | coverage {cov_raw:.2f} -> {cov:.2f} "
          f"| uniformity {uni:.2f}")

save_rows_csv(anms_rows, cfg.MATCHING_DIR / "anms_report.csv")

# %% [markdown]
# ## 5. Visualization — before vs after ANMS + coverage heatmap

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if anms_rows:
    row = max(anms_rows, key=lambda r: r["matches_in"])
    d0 = np.load(cfg.MATCHING_DIR / f"matches_{row['pair']}.npz", allow_pickle=True)
    d1 = np.load(cfg.MATCHING_DIR / f"anms_{row['pair']}.npz", allow_pickle=True)
    q_img = cv2.imread(str(d0["q_path"]), cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].imshow(q_img, cmap="gray"); axes[0].scatter(d0["kp0"][:, 0], d0["kp0"][:, 1], s=4, c="#F59E0B")
    axes[0].set_title(f"Before ANMS ({len(d0['kp0'])} pts)")
    axes[1].imshow(q_img, cmap="gray"); axes[1].scatter(d1["kp0"][:, 0], d1["kp1"][:, 1], s=8, c="#22C55E")
    axes[1].set_title(f"After ANMS ({len(d1['kp0'])} pts, uniform)")
    hm = coverage_heatmap(d1["kp0"], q_img.shape, GRID)
    hm_n = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[2].imshow(cv2.applyColorMap(255 - hm_n, cv2.COLORMAP_JET))
    axes[2].set_title("Coverage heatmap")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "anms_distribution.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {
    "pairs_filtered": len(anms_rows),
    "mean_coverage_after": round(float(np.mean([r["coverage_after"] for r in anms_rows])), 3) if anms_rows else 0.0,
    "mean_uniformity": round(float(np.mean([r["uniformity"] for r in anms_rows])), 3) if anms_rows else 0.0,
    "anms_target": ANMS_TARGET,
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/matching/anms_feat_XXX.npz`
# - `outputs/matching/anms_report.csv`
# - `outputs/visualizations/anms_distribution.png`

# %%
save_json(metrics, cfg.METRICS_DIR / "anms_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Sets with fewer points than the target pass through unchanged (nothing to suppress).
# - Empty match sets are skipped and excluded from averages.

# %% [markdown]
# ## 9. Explanation
# ANMS grows a suppression radius around each keypoint in score order, keeping the strongest
# point in each neighborhood — the gamma exponent (>1) spreads selections further apart.
# Uniform correspondence coverage is a stated ISRO requirement: clustered matches bias the
# homography toward the cluster and leave the rest of the image unverified.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 10 runs MAGSAC++ outlier rejection on the ANMS-filtered correspondences.
