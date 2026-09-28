# %% [markdown]
# ## 1. Objective
# SIH Phase 1 — dataset integrity notebook: verify the 70/15/15 geographic
# train/val/test split (no location leakage), the ElevationProfile track-corridor
# integration, and the SunAngle.csv pass statistics + per-patch sun bins.
# Heavy lifting is done once by `scripts/run_sih_upgrade.py`; this notebook
# validates and visualizes its outputs.

# %% [markdown]
# ## 2. Prerequisites

# %%
import sys
from pathlib import Path

def _find_project_root() -> Path:
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "lunarai_lib").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np

from lunarai_lib.config import load_config

cfg = load_config()
SIH = cfg.OUTPUTS_ROOT / "sih_upgrade"
prereq = {
    "split_metadata.json": (cfg.OUTPUTS_ROOT / "splits" / "split_metadata.json").exists(),
    "patch_index_with_elevation.csv": (SIH / "patch_index_with_elevation.csv").exists(),
    "patch_index_with_sunbin.csv": (SIH / "patch_index_with_sunbin.csv").exists(),
}
print(json.dumps(prereq, indent=1))
missing = [k for k, v in prereq.items() if not v]
if missing:
    print(f"NOTE: {missing} not found -> run `python scripts/run_sih_upgrade.py` first; "
          "this notebook will recompute them below instead of failing.")

# %% [markdown]
# ## 3. Train / Validation / Test split — verification

# %%
import pandas as pd

from lunarai_lib.splits import assign_geo_splits, export_splits

df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
meta = export_splits(df, cfg.OUTPUTS_ROOT / "splits", seed=42)
split_df = pd.read_csv(cfg.OUTPUTS_ROOT / "splits" / "all_patches_with_splits.csv")

counts = split_df.split.value_counts()
per_sensor = split_df.groupby("dataset_name").split.value_counts().unstack(fill_value=0)
display(counts, per_sensor)

# LEAKAGE CHECK: a geo cell must never appear in two splits
per_cell = split_df.groupby("geo_cell").split.nunique()
assert (per_cell == 1).all(), "LOCATION LEAKAGE DETECTED"
print("leakage check PASSED: every geo cell lives in exactly one split")
print(json.dumps(meta, indent=1, default=str))

# %% [markdown]
# ## 4. Split visualization

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 5))
colors = {"train": "#38BDF8", "val": "#F59E0B", "test": "#22C55E"}
for name, g in split_df.groupby("split"):
    ax.scatter(g.longitude, g.latitude, s=12, alpha=0.7, c=colors[name], label=name)
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title("70/15/15 geographic split (patch footprints)")
ax.legend()
plt.tight_layout()
plt.savefig(SIH / "split_visualization.png", dpi=130)
plt.show()

# %% [markdown]
# ## 5. ElevationProfile integration — track corridor + coverage

# %%
from lunarai_lib.elevation import (elevation_band, load_elevation_track,
                                   map_elevations, track_stats)

pts = load_elevation_track(cfg.DATA_ROOT / "ElevationProfile.csv")
stats = track_stats(pts)
print("track stats:", json.dumps(stats, indent=1, default=str))

elevs, dists = map_elevations(df, pts, corridor_deg=1.0)
df["elevation_m"] = elevs
df["elevation_band"] = [elevation_band(e) for e in elevs]
covered = int(np.isfinite(elevs).sum())
print(f"patches within the 1-deg LOLA track corridor: {covered}/{len(df)} "
      f"({covered / max(len(df), 1):.1%}) — the rest stay NaN by design")

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(df.longitude, df.latitude, s=10, alpha=0.5, label="catalog patches", c="#94A3B8")
ok = df[np.isfinite(df.elevation_m)]
ax.scatter(ok.longitude, ok.latitude, s=22, c="#22C55E", label="inside track corridor")
ax.plot([p["lon"] for p in pts], [p["lat"] for p in pts], "r-", lw=1.5,
        label="LOLA ground track (Tycho)")
ax.set_xlabel("longitude (deg)"); ax.set_ylabel("latitude (deg)")
ax.set_title("ElevationProfile ground track vs patch catalog")
ax.legend()
plt.tight_layout()
plt.savefig(SIH / "elevation_track_coverage.png", dpi=130)
plt.show()

# %% [markdown]
# ## 6. SunAngle integration — pass statistics + per-patch bins

# %%
from lunarai_lib.sunangle import load_sunangle_stats, sun_bin_label

sa_stats = load_sunangle_stats(cfg.DATA_ROOT / "SunAngle.csv")
print("SunAngle pass stats:", json.dumps(sa_stats, indent=1, default=str))

df["sun_bin"] = [sun_bin_label(v) for v in df.sun_angle]
bin_counts = df.sun_bin.value_counts()
display(bin_counts)

fig, ax = plt.subplots(figsize=(6, 3.6))
bin_counts.plot.bar(ax=ax, color="#38BDF8")
ax.set_title("catalog patches per sun-elevation bin")
ax.set_ylabel("patches")
plt.tight_layout()
plt.savefig(SIH / "sunbin_counts.png", dpi=130)
plt.show()

# %% [markdown]
# ## 7. Summary
# - Split: 70/15/15 by geographic cell, leakage check enforced.
# - Elevation: LOLA Tycho-track corridor mapping with honest coverage reporting.
# - SunAngle: pass statistics + per-patch 5-bin labels feeding training
#   (sun-aware negatives) and retrieval (illumination-aware re-ranking).
print("nb17 dataset integrity: OK")
