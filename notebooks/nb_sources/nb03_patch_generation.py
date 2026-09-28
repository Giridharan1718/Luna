# %% [markdown]
# ## 1. Objective
# Generate lunar patches from all datasets at **256x256 with stride 128**, attach metadata
# (Patch_ID, Source_Image, Dataset_Name, Latitude, Longitude, Sun_Angle, Elevation,
# Patch_Path) and produce `patch_index.csv` plus the SQLite patch table.

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
from lunarai_lib import patching
from lunarai_lib.io_utils import save_json
from lunarai_lib.db import Database
import pandas as pd
import numpy as np

cfg = load_config()
records_by_sensor = md.discover_all(cfg.DATA_ROOT)
all_records = [r for recs in records_by_sensor.values() for r in recs]
elev_lookup = patching.build_elevation_lookup(cfg.DATA_ROOT / "ElevationProfile.csv")
print(f"{len(all_records)} products | elevation bins: {len(elev_lookup)}")

# %% [markdown]
# ## 3. Configuration

# %%
PATCH_SIZE = cfg.PATCH_SIZE          # 256
STRIDE = cfg.STRIDE                  # 128
MAX_PER_IMAGE = cfg.MAX_PATCHES_PER_IMAGE
MIN_STD = cfg.MIN_STD
print(f"patch {PATCH_SIZE}x{PATCH_SIZE} stride {STRIDE} cap {MAX_PER_IMAGE}/image")

# %% [markdown]
# ## 4. Implementation

# %%
all_metas = []
for r in all_records:
    metas = patching.generate_patches_for_record(
        r, cfg.PATCHES_DIR,
        patch_size=PATCH_SIZE, stride=STRIDE, max_patches=MAX_PER_IMAGE,
        min_std=MIN_STD, resize_long_side=cfg.RESIZE_LONG_SIDE,
        elev_lookup=elev_lookup,
        processed_dir=cfg.PROCESSED_DIR,
        preprocess_fn=None,   # patches inherit NB02-enhanced imagery; raw patches kept here
        sensor_tag=r.sensor.replace(" ", ""),
    )
    if r.notes:
        print(f"  [{r.sensor:7s}] {r.image_id[:40]:40s} -> {len(metas):3d} patches  ({r.notes.strip()})")
    else:
        print(f"  [{r.sensor:7s}] {r.image_id[:40]:40s} -> {len(metas):3d} patches")
    all_metas.extend(metas)

index_path = patching.write_patch_index(all_metas, cfg.PATCHES_DIR / "patch_index.csv")
df = pd.read_csv(index_path)
print(f"\nTotal patches: {len(df)}")
print(df.groupby("dataset_name").size())

# %% [markdown]
# ### SQLite persistence (Backend Schema: `patches`, `images` tables)

# %%
db = Database(cfg.DATABASE_DIR / "lunarai.db")
db.executemany(
    "INSERT OR REPLACE INTO images(image_id, filename, mission, sensor, latitude, longitude,"
    " resolution, sun_angle, acquisition_date) VALUES (?,?,?,?,?,?,?,?,?)",
    [(r.image_id, Path(r.path).name, r.mission, r.sensor,
      None if np.isnan(r.latitude) else float(r.latitude),
      None if np.isnan(r.longitude) else float(r.longitude),
      None if np.isnan(r.pixel_resolution) else float(r.pixel_resolution),
      None if np.isnan(r.sun_elevation) else float(r.sun_elevation),
      r.acquisition_date) for r in all_records])
def _n(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

db.executemany(
    "INSERT OR REPLACE INTO patches(patch_id, image_id, patch_path, latitude, longitude,"
    " patch_size, sensor) VALUES (?,?,?,?,?,?,?)",
    [(m.patch_id, m.source_image, m.patch_path, _n(m.latitude), _n(m.longitude),
      m.width, m.dataset_name) for m in all_metas])
print("DB rows:", db.query("SELECT sensor, COUNT(*) FROM patches GROUP BY sensor"))

# %% [markdown]
# ## 5. Visualization — patch mosaic per sensor

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, sensor in zip(axes, ["OHRC", "TMC-2", "IIRS", "LRO NAC"]):
    sub = df[df.dataset_name == sensor]
    if len(sub):
        img = cv2.imread(sub.patch_path.iloc[len(sub) // 2], cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{sensor}\n{len(sub)} patches")
    else:
        ax.set_title(f"{sensor}\n0 patches")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(cfg.VISUALIZATIONS_DIR / "patch_mosaic.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {
    "total_patches": int(len(df)),
    "per_sensor": df.groupby("dataset_name").size().to_dict(),
    "patch_size": PATCH_SIZE, "stride": STRIDE,
    "with_latitude": int(df.latitude.notna().sum()),
    "with_sun_angle": int(df.sun_angle.notna().sum()),
    "with_elevation": int(df.elevation.notna().sum()),
}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/patches/<sensor>/<patch_id>.png` (256x256)
# - `outputs/patches/patch_index.csv`
# - `database/lunarai.db` (patches + images tables populated)

# %%
save_json(metrics, cfg.METRICS_DIR / "patch_generation_report.json")

# %% [markdown]
# ## 8. Error Handling
# - Records with missing binaries fall back to browse PNGs (flagged in notes).
# - Patches with std < MIN_STD (blank/over-exposed) are rejected.
# - Oversized grids are subsampled evenly (linspace) to respect the per-image cap.

# %% [markdown]
# ## 9. Explanation
# Patches tile the image on a 128 grid; edges get a final aligned row/column so full
# coverage is kept. Geographic metadata is interpolated from PDS4 corner coordinates by
# patch-center position; sun angle is inherited from the product label; elevation comes
# from a dilated nearest-degree lookup into the LOLA DEM profile.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 04 consumes `patch_index.csv` to build cross-sensor geo-triplets and train LunaDNA.
