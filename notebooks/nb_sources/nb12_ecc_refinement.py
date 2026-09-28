# %% [markdown]
# ## 1. Objective
# Refine the homography registration with **ECC** (Enhanced Correlation Coefficient
# maximization) to reach **sub-pixel alignment**, saving refined warps and matrices.

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
from lunarai_lib.geometry import warp_image, reprojection_rmse, ecc_refine
from lunarai_lib.io_utils import save_json, save_rows_csv

cfg = load_config()
reg_files = sorted(cfg.REGISTRATION_DIR.glob("*_homography.npy"))
magsac_index = {f.stem.replace("magsac_", ""): f for f in cfg.MATCHING_DIR.glob("magsac_feat_*.npz")}
print(f"{len(reg_files)} homographies to refine")

# %% [markdown]
# ## 3. Configuration

# %%
ECC_ITERS = int(cfg.MATCHING.get("ecc_iterations", 100))
ECC_EPS = float(cfg.MATCHING.get("ecc_epsilon", 1e-4))

# %% [markdown]
# ## 4. Implementation

# %%
rows = []
for hf in reg_files:
    stem = hf.stem.replace("_homography", "")
    H = np.load(hf)
    pair = stem.replace("feat_", "feat_")
    mf = magsac_index.get(stem)
    if mf is None:
        hits = list(cfg.MATCHING_DIR.glob(f"magsac_{stem}.npz"))
        mf = hits[0] if hits else None
    if mf is None:
        continue
    d = np.load(mf, allow_pickle=True)
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    if q_img is None or r_img is None:
        continue
    size = (r_img.shape[1], r_img.shape[0])
    warped0 = warp_image(q_img, H, size)
    W, cc, ok = ecc_refine(r_img, warped0, np.eye(3, dtype=np.float32),
                           iterations=ECC_ITERS, eps=ECC_EPS)
    if ok:
        H_ref = (W @ H).astype(np.float32)
        rmse_ref = reprojection_rmse(d["kp0"], d["kp1"], H_ref)
    else:
        H_ref = H
        rmse_ref = reprojection_rmse(d["kp0"], d["kp1"], H)
    warped_ref = warp_image(q_img, H_ref, size)
    cv2.imwrite(str(cfg.REGISTRATION_DIR / f"{stem}_ecc_registered.png"), warped_ref)
    np.save(cfg.REGISTRATION_DIR / f"{stem}_ecc_warp.npy", H_ref)
    rows.append({"pair": stem, "ecc_converged": ok, "ecc_correlation": round(cc, 4),
                 "rmse_before_ecc": None, "rmse_after_ecc": round(rmse_ref, 3),
                 "delta_translation_px": round(float(np.hypot(H_ref[0, 2] - H[0, 2],
                                                              H_ref[1, 2] - H[1, 2])), 4)})
    print(f"  {stem}: ecc={ok} cc={cc:.4f} rmse={rmse_ref:.3f}px "
          f"delta_t={rows[-1]['delta_translation_px']}px")

# %% [markdown]
# ## 5. Visualization — difference maps before/after ECC

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if rows:
    stem = rows[0]["pair"]
    r_img = None
    for mf in magsac_index.values():
        d = np.load(mf, allow_pickle=True)
        if str(d["q_id"]).endswith(stem) or stem in str(d["q_id"]):
            r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
            break
    warped0 = cv2.imread(str(cfg.REGISTRATION_DIR / f"{stem}_registered.png"), cv2.IMREAD_GRAYSCALE)
    warped1 = cv2.imread(str(cfg.REGISTRATION_DIR / f"{stem}_ecc_registered.png"), cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].imshow(np.abs(warped0.astype(int) - (r_img or warped0).astype(int)) if r_img is not None else warped0,
                   cmap="inferno"); axes[0].set_title("|H warp - reference|")
    axes[1].imshow(np.abs(warped1.astype(int) - r_img.astype(int)) if r_img is not None else warped1,
                   cmap="inferno"); axes[1].set_title("|ECC warp - reference|")
    axes[2].imshow(warped1, cmap="gray"); axes[2].set_title("ECC-registered image")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "ecc_refinement.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {"pairs_refined": len(rows),
           "ecc_converged": sum(1 for r in rows if r["ecc_converged"]),
           "mean_ecc_correlation": round(float(np.mean([r["ecc_correlation"] for r in rows])), 4) if rows else None}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/registration/<pair>_ecc_registered.png`
# - `outputs/registration/<pair>_ecc_warp.npy`
# - `outputs/visualizations/ecc_refinement.png`

# %%
save_rows_csv(rows, cfg.REGISTRATION_DIR / "ecc_report.csv")
save_json(metrics, cfg.METRICS_DIR / "ecc_report.json")

# %% [markdown]
# ## 8. Error Handling
# - ECC non-convergence (cv2.error) keeps the homography result (plan 11 failure rule:
#   "ECC Failure -> Return Homography Result").
# - Correlation ~0 signals poor texture; flagged via ecc_correlation in the report.

# %% [markdown]
# ## 9. Explanation
# ECC maximizes correlation between reference and warped query over an 8-parameter warp.
# Operating on the homography-warped image, it only needs to recover the small residual
# (typically < 1 px), which is where its sub-pixel accuracy comes from. The refinement is
# composed as H_refined = W_ecc @ H so the full chain remains a single matrix.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 13 computes the complete metric suite (RMSE, SSIM, NCC, coverage, runtime) and
# builds the evaluation report + exports.
