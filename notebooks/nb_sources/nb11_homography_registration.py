# %% [markdown]
# ## 1. Objective
# Estimate the final **3x3 homography** from MAGSAC++ inliers, warp the query onto the
# reference frame and store the transformation matrix and registered image.

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

from lunarai_lib.config import load_config
from lunarai_lib.geometry import magsac_homography, warp_image, reprojection_rmse
from lunarai_lib.io_utils import save_json, save_rows_csv

cfg = load_config()
magsac_files = sorted(cfg.MATCHING_DIR.glob("magsac_feat_*.npz"))
print(f"{len(magsac_files)} verified pairs")

# %% [markdown]
# ## 3. Configuration

# %%
REG_DIR = cfg.REGISTRATION_DIR
THRESH = float(cfg.MATCHING.get("magsac_threshold", 4.0))

# %% [markdown]
# ## 4. Implementation

# %%
rows = []
for f in magsac_files:
    d = np.load(f, allow_pickle=True)
    kp0, kp1 = d["kp0"], d["kp1"]
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    if q_img is None or r_img is None or len(kp0) < 4:
        continue
    H, mask, ratio, inliers = magsac_homography(kp0, kp1, threshold=THRESH)
    if H is None:
        continue
    size = (r_img.shape[1], r_img.shape[0])
    warped = warp_image(q_img, H, size)
    rmse = reprojection_rmse(kp0, kp1, H)
    stem = f.stem.replace("magsac_", "")
    cv2.imwrite(str(REG_DIR / f"{stem}_registered.png"), warped)
    np.save(REG_DIR / f"{stem}_homography.npy", H)
    rows.append({"pair": stem, "inliers": inliers, "rmse_px": round(rmse, 3),
                 "H": np.round(H, 5).tolist(), "q_path": str(d["q_path"]),
                 "r_path": str(d["r_path"]), "registered_path": str(REG_DIR / f"{stem}_registered.png")})
    print(f"  {stem}: rmse={rmse:.3f}px  det(H)={np.linalg.det(H):.3e}")
    print("   H =", np.round(H, 5).tolist())

# %% [markdown]
# ## 5. Visualization — side-by-side reference vs registered

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if rows:
    row = rows[0]
    q_img = cv2.imread(row["q_path"], cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(row["r_path"], cv2.IMREAD_GRAYSCALE)
    warped = cv2.imread(row["registered_path"], cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].imshow(q_img, cmap="gray"); axes[0].set_title("Query (source)")
    axes[1].imshow(r_img, cmap="gray"); axes[1].set_title("Reference")
    axes[2].imshow(warped, cmap="gray"); axes[2].set_title("Registered (warped query)")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "homography_result.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {"pairs_registered": len(rows),
           "mean_rmse_px": round(float(np.mean([r["rmse_px"] for r in rows])), 3) if rows else None}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/registration/<pair>_registered.png`
# - `outputs/registration/<pair>_homography.npy`
# - `outputs/visualizations/homography_result.png`

# %%
save_rows_csv([{"pair": r["pair"], "rmse_px": r["rmse_px"], "inliers": r["inliers"],
                "H00": r["H"][0][0], "H01": r["H"][0][1], "H02": r["H"][0][2],
                "H10": r["H"][1][0], "H11": r["H"][1][1], "H12": r["H"][1][2],
                "H20": r["H"][2][0], "H21": r["H"][2][1], "H22": r["H"][2][2]}
               for r in rows], REG_DIR / "homography_report.csv")
save_json(metrics, cfg.METRICS_DIR / "homography_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Homography estimation can still fail after MAGSAC if inliers are degenerate
#   (collinear) - such pairs are skipped and absent from the report.
# - |det(H)| far from 1 signals a pathological warp; values are recorded for review.

# %% [markdown]
# ## 9. Explanation
# The homography maps query pixel coordinates into the reference frame via
# [x' y' w']^T = H [x y 1]^T. It models rotation/scale/translation plus mild perspective,
# which covers orbital viewpoint differences at these small off-nadir angles. Sub-pixel
# improvement is delegated to ECC in NB12.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 12 refines this warp with ECC to push alignment below the 1-pixel target.
