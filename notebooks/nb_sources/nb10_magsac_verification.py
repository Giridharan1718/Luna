# %% [markdown]
# ## 1. Objective
# Reject outliers with **MAGSAC++** (cv2.USAC_MAGSAC): compute inlier matches, inlier ratio
# and the initial geometric model for every ANMS-filtered pair.

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

import numpy as np
import pandas as pd

from lunarai_lib.config import load_config
from lunarai_lib.geometry import magsac_homography
from lunarai_lib.io_utils import save_json, save_rows_csv

cfg = load_config()
anms_files = sorted(cfg.MATCHING_DIR.glob("anms_feat_*.npz"))
print(f"{len(anms_files)} ANMS sets")

# %% [markdown]
# ## 3. Configuration

# %%
THRESH = float(cfg.MATCHING.get("magsac_threshold", 4.0))

# %% [markdown]
# ## 4. Implementation

# %%
rows = []
for f in anms_files:
    d = np.load(f, allow_pickle=True)
    kp0, kp1 = d["kp0"], d["kp1"]
    if len(kp0) < 4:
        print(f"  {f.stem}: skipped (<4 pts)")
        continue
    pair_id = f.stem.replace("anms_", "")   # anms_feat_001 -> feat_001
    H, mask, ratio, inliers = magsac_homography(kp0, kp1, threshold=THRESH)
    out = {"pair": pair_id, "matches": len(kp0), "inliers": inliers,
           "inlier_ratio": round(ratio, 3), "status": "ok" if H is not None else "failed"}
    if H is not None:
        np.savez_compressed(cfg.MATCHING_DIR / f"magsac_{pair_id}.npz",
                            kp0=kp0[mask > 0], kp1=kp1[mask > 0], H=H,
                            q_path=d["q_path"], r_path=d["r_path"],
                            q_id=d["q_id"], r_id=d["r_id"])
        out["H_det"] = round(float(np.linalg.det(H)), 3)
    rows.append(out)
    print(f"  {pair_id}: {inliers}/{len(kp0)} inliers ({ratio:.1%})")

save_rows_csv(rows, cfg.MATCHING_DIR / "magsac_report.csv")

# %% [markdown]
# ## 5. Visualization — inlier ratio distribution

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if rows:
    plt.figure(figsize=(7, 3.4))
    plt.hist([r["inlier_ratio"] for r in rows], bins=12, color="#38BDF8")
    plt.axvline(float(cfg.EVAL.get("inlier_ratio_target", 0.85)), color="#F59E0B",
                linestyle="--", label="target 0.85")
    plt.xlabel("inlier ratio"); plt.ylabel("pairs"); plt.legend()
    plt.title("MAGSAC++ inlier ratios")
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "magsac_inlier_ratios.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {
    "pairs_verified": len(rows),
    "mean_inlier_ratio": round(float(np.mean([r["inlier_ratio"] for r in rows])), 3) if rows else 0.0,
    "total_inliers": int(sum(r["inliers"] for r in rows)),
    "threshold_px": THRESH,
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/matching/magsac_feat_XXX.npz` (inliers + H)
# - `outputs/matching/magsac_report.csv`
# - `outputs/visualizations/magsac_inlier_ratios.png`

# %%
save_json(metrics, cfg.METRICS_DIR / "magsac_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Fewer than 4 points -> skipped (homography is underdetermined).
# - USAC failure degrades to classic RANSAC inside magsac_homography.
# - Failed pairs are listed with status="failed" for the dashboard to display.

# %% [markdown]
# ## 9. Explanation
# MAGSAC++ marginalizes over the noise scale instead of fixing one inlier threshold, which
# suits lunar imagery where localization error varies between 0.24 m/px OHRC and
# ~5-10 m/px TMC. The output inlier set seeds NB11's final homography.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 11 estimates the final homography from MAGSAC inliers and warps the query.
