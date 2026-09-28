# %% [markdown]
# ## 1. Objective
# Apply the standardization pipeline (grayscale, CLAHE, histogram equalization, contrast
# normalization, noise reduction, resize) to every usable image, producing the enhanced
# imagery used for patch extraction in notebook 03.

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

from lunarai_lib.config import load_config
from lunarai_lib import metadata as md
from lunarai_lib import preprocessing as pp
from lunarai_lib.io_utils import load_image_any, load_browse_png, save_json, image_stats
import cv2
import numpy as np

cfg = load_config()
records_by_sensor = md.discover_all(cfg.DATA_ROOT)
all_records = [r for recs in records_by_sensor.values() for r in recs]
print(f"{len(all_records)} products loaded")

# %% [markdown]
# ## 3. Configuration

# %%
PROCESSED_DIR = cfg.PROCESSED_DIR
RESIZE_LONG = cfg.RESIZE_LONG_SIDE
PRE = cfg.PRE
print("CLAHE clip", PRE.get("clahe_clip_limit"), "| grid", PRE.get("clahe_grid"),
      "| denoise", PRE.get("denoise"))

# %% [markdown]
# ## 4. Implementation

# %%
def preprocess_record(record, processed_dir):
    """Returns dict with paths + before/after stats, or None if unusable."""
    img = None
    source = "binary"
    if record.binary_present:
        img = load_image_any(record, RESIZE_LONG)
    if img is None:
        img = load_browse_png(record, RESIZE_LONG)
        source = "browse_png_fallback"
    if img is None:
        return None

    enhanced = pp.full_preprocess(
        img,
        clahe_clip=float(PRE.get("clahe_clip_limit", 2.0)),
        clahe_grid=int(PRE.get("clahe_grid", 8)),
        denoise=str(PRE.get("denoise", "median")),
        denoise_ksize=int(PRE.get("denoise_ksize", 3)),
    )
    out_dir = processed_dir / record.sensor.replace(" ", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = record.image_id.rsplit(".", 1)[0][:60]
    out_path = out_dir / f"{stem}_proc.png"
    cv2.imwrite(str(out_path), enhanced)
    before, after = image_stats(img), image_stats(enhanced)
    if source == "browse_png_fallback":
        record.notes += "preprocessed from browse PNG; "
    return {"image_id": record.image_id, "sensor": record.sensor, "source": source,
            "processed_path": str(out_path),
            "contrast_before": round(before["std"], 2), "contrast_after": round(after["std"], 2),
            "mean_before": round(before["mean"], 2), "mean_after": round(after["mean"], 2)}

results = []
for r in all_records:
    res = preprocess_record(r, PROCESSED_DIR)
    if res:
        results.append(res)
        print(f"  [{res['sensor']:7s}] {res['image_id'][:44]:44s} "
              f"contrast {res['contrast_before']:6.1f} -> {res['contrast_after']:6.1f}")
    else:
        print(f"  [{r.sensor:7s}] {r.image_id[:44]:44s} SKIPPED (no decodable image)")

# %% [markdown]
# ## 5. Visualization — enhancement effect on a sample image

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sample = next((r for r in all_records if r.binary_present and r.sensor == "TMC-2"), all_records[0])
raw = load_image_any(sample, RESIZE_LONG)
enh = pp.full_preprocess(raw, clahe_clip=float(PRE.get("clahe_clip_limit", 2.0)),
                         clahe_grid=int(PRE.get("clahe_grid", 8)),
                         denoise=str(PRE.get("denoise", "median")),
                         denoise_ksize=int(PRE.get("denoise_ksize", 3)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(raw, cmap="gray"); axes[0].set_title("Raw")
axes[1].imshow(enh, cmap="gray"); axes[1].set_title("CLAHE + denoise + normalize")
axes[2].hist(raw.ravel(), bins=64, alpha=0.6, label="raw", color="#94A3B8")
axes[2].hist(enh.ravel(), bins=64, alpha=0.6, label="processed", color="#38BDF8")
axes[2].legend(); axes[2].set_title("Intensity histograms")
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(cfg.VISUALIZATIONS_DIR / "preprocessing_comparison.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
contrast_gain = [r["contrast_after"] / max(r["contrast_before"], 1e-6) for r in results]
metrics = {
    "images_processed": len(results),
    "images_skipped": len(all_records) - len(results),
    "mean_contrast_gain": round(float(np.mean(contrast_gain)), 3) if contrast_gain else 0.0,
    "fallback_to_browse_png": sum(1 for r in results if r["source"] == "browse_png_fallback"),
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/processed/{OHRC,TMC-2,IIRS,LRONAC}/*_proc.png`
# - `outputs/visualizations/preprocessing_comparison.png`

# %%
save_json({"metrics": metrics, "records": results},
          cfg.METRICS_DIR / "preprocessing_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Records without any decodable binary (IIRS without binaries) fall back to the browse PNG.
# - If both fail, the record is skipped and counted in `images_skipped`.
# - Extremely dark/flat images still pass through; contrast normalization clips to percentiles.

# %% [markdown]
# ## 9. Explanation
# The pipeline order is grayscale → CLAHE (adaptive local contrast for low-sun imagery) →
# median denoise (removes salt-and-pepper sensor noise) → percentile contrast normalization
# (harmonizes exposure across missions) → optional resize. CLAHE before denoise avoids
# amplifying noise; percentile normalization is robust to crater shadows saturating the range.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 03 extracts 256x256 patches from the *enhanced* images produced here, keeping the
# raw-patch store and the processed-patch store aligned by filename.
