# %% [markdown]
# ## 1. Objective
# Run **SuperPoint** feature extraction on query and retrieved patches, exporting keypoints
# and descriptors (with a SIFT fallback when offline) for the LightGlue stage.

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
from lunarai_lib.io_utils import save_json

cfg = load_config()
res_df = pd.read_csv(cfg.RETRIEVAL_DIR / "topk_results.csv")
df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
meta = df.set_index("patch_id")
print(f"{len(res_df)} retrieval rows | {res_df.query_patch.nunique()} queries")

# %% [markdown]
# ## 3. Configuration

# %%
MAX_KP = int(cfg.MATCHING.get("superpoint_max_kp", 2048))
TOP1_ONLY = True   # extract for best retrieval per query (pipeline demo scale)

device = "cuda" if torch.cuda.is_available() else "cpu"
matcher = SuperPointLightGlue(max_keypoints=MAX_KP,
                              match_threshold=float(cfg.MATCHING.get("lightglue_threshold", 0.2)),
                              device=device)
print("matcher backend:", "superpoint+lightglue" if matcher.available else "sift fallback",
      "| device:", matcher.device)

# %% [markdown]
# ## 4. Implementation

# %%
def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

pairs = []
if TOP1_ONLY:
    best = res_df[res_df["rank"] == 1]
    pairs = list(zip(best.query_patch, best.retrieved_patch))
else:
    pairs = list(zip(res_df.query_patch, res_df.retrieved_patch))

feature_rows = []
npz_dir = cfg.MATCHING_DIR
npz_dir.mkdir(parents=True, exist_ok=True)
for i, (qid, rid) in enumerate(pairs[:40]):
    qid_clean = str(qid).rsplit(".png", 1)[0]
    if qid_clean not in meta.index:
        continue
    qpath = meta.loc[qid_clean].patch_path
    rpath = cfg.PATCHES_DIR / rid if not Path(rid).is_absolute() else Path(rid)
    if not rpath.exists():
        # retrieved name may include sensor dir; search
        hits = list(cfg.PATCHES_DIR.rglob(Path(rid).name))
        if not hits:
            continue
        rpath = hits[0]
    q_img, r_img = load_gray(qpath), load_gray(rpath)
    fq = matcher.extract(q_img)
    fr = matcher.extract(r_img)
    tag = f"feat_{i:03d}"
    np.savez_compressed(npz_dir / f"{tag}.npz",
                        q_kps=fq["keypoints"], q_desc=fq["descriptors"],
                        r_kps=fr["keypoints"], r_desc=fr["descriptors"],
                        q_path=str(qpath), r_path=str(rpath),
                        q_id=qid_clean, r_id=rid)
    feature_rows.append({"feature_tag": tag, "query_patch": qid_clean, "ref_patch": rid,
                         "q_kps": len(fq["keypoints"]), "r_kps": len(fr["keypoints"]),
                         "desc_dim": int(fq["descriptors"].shape[1]) if fq["descriptors"] is not None else 0,
                         "backend": fq["backend"]})
    print(f"  {tag}: q_kps={feature_rows[-1]['q_kps']:4d} r_kps={feature_rows[-1]['r_kps']:4d} "
          f"({fq['backend']})")

# %% [markdown]
# ## 5. Visualization — keypoints on a sample pair

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if feature_rows:
    row = feature_rows[0]
    d = np.load(npz_dir / f"{row['feature_tag']}.npz", allow_pickle=True)
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, kps, name in [(axes[0], q_img, d["q_kps"], "query"),
                               (axes[1], r_img, d["r_kps"], "reference")]:
        ax.imshow(img, cmap="gray")
        ax.scatter(kps[:, 0], kps[:, 1], s=4, c="#38BDF8")
        ax.set_title(f"{name}: {len(kps)} keypoints")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "superpoint_keypoints.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {
    "pairs_extracted": len(feature_rows),
    "mean_kps_query": round(float(np.mean([r["q_kps"] for r in feature_rows])), 1) if feature_rows else 0.0,
    "mean_kps_ref": round(float(np.mean([r["r_kps"] for r in feature_rows])), 1) if feature_rows else 0.0,
    "backend": feature_rows[0]["backend"] if feature_rows else "none",
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/matching/feat_XXX.npz` (keypoints + descriptors per pair)
# - `outputs/visualizations/superpoint_keypoints.png`

# %%
save_json(metrics, cfg.METRICS_DIR / "superpoint_report.json")

# %% [markdown]
# ## 8. Error Handling
# - If SuperPoint/LightGlue weights cannot be downloaded, the module automatically falls
#   back to SIFT + FLANN so the pipeline completes offline (backend recorded in reports).
# - Missing retrieved patch paths are re-resolved by filename search; unresolved pairs skip.

# %% [markdown]
# ## 9. Explanation
# SuperPoint's self-supervised detector fires on crater rims and ridge lines; its 256-D
# descriptors are the input contract for LightGlue. Keypoint counts and per-pair NPZ files
# give the dashboard raw material to visualize detections independently of matching.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 08 feeds these NPZ keypoint/descriptor sets into LightGlue to produce matches.
