# %% [markdown]
# ## 1. Objective
# Perform **full dataset inspection** of the Chandrayaan-2 (OHRC, TMC, IIRS) and LRO imagery
# and produce `dataset_report.json` with: total images, per-sensor counts, resolution and
# format distributions, missing/corrupted/duplicate files, metadata coverage, sun-angle and
# elevation statistics, and an overall dataset quality score.

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
from lunarai_lib.io_utils import save_json, save_rows_csv, image_stats, load_image_any
import cv2
import numpy as np

cfg = load_config()
cfg.ensure_dirs()
print("Data root:", cfg.DATA_ROOT)
print("Outputs:", cfg.OUTPUTS_ROOT)

# %% [markdown]
# ## 3. Configuration

# %%
ANALYSIS_DIR = cfg.DATASET_ANALYSIS_DIR
REPORT_PATH = ANALYSIS_DIR / "dataset_report.json"
HASH_FULL_IMAGES = False   # True = md5 of full binaries (slow for 1.2 GB images)

# %% [markdown]
# ## 4. Implementation

# %%
records_by_sensor = md.discover_all(cfg.DATA_ROOT)
all_records = [r for recs in records_by_sensor.values() for r in recs]
print(f"Discovered {len(all_records)} image products")
for sensor, recs in records_by_sensor.items():
    print(f"  {sensor:8s}: {len(recs)}")

# %%
inventory_rows = []
for r in all_records:
    row = r.to_row()
    row["dataset"] = r.sensor
    inventory_rows.append(row)
save_rows_csv(inventory_rows, ANALYSIS_DIR / "dataset_inventory.csv")
print("inventory saved:", ANALYSIS_DIR / "dataset_inventory.csv")

# %%
# Resolution + format distributions
res_dist: dict[str, int] = {}
fmt_dist: dict[str, int] = {}
for r in all_records:
    key = f"{r.width}x{r.height}" if r.width and r.height else "unknown"
    res_dist[key] = res_dist.get(key, 0) + 1
    fmt_dist[r.format] = fmt_dist.get(r.format, 0) + 1

# %% [markdown]
# **Corruption check** — read head bytes of every binary and attempt a full decode of
# PNG/browse images; verify .img byte size matches label expectation.

# %%
corrupted, missing, duplicates = [], [], []

for r in all_records:
    p = Path(r.path)
    if not p.exists():
        missing.append({"image_id": r.image_id, "reason": "file not on disk"})
        r.notes += "missing on disk; "
        continue
    if p.suffix.lower() == ".img":
        expected = r.width * r.height  # single-band UnsignedByte
        actual = p.stat().st_size
        if expected and actual < expected:
            corrupted.append({"image_id": r.image_id,
                              "reason": f"size {actual} < expected {expected}"})
            r.notes += "truncated binary; "
    elif p.suffix.lower() == ".png":
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            corrupted.append({"image_id": r.image_id, "reason": "undecodable PNG"})

# duplicates: md5 of small files; size+head-hash for the huge .img files
seen: dict[str, dict] = {}
for r in all_records:
    p = Path(r.path)
    if not p.exists():
        continue
    if p.stat().st_size < 64 * 1024 * 1024 or p.suffix.lower() == ".png":
        digest = md.md5_of_file(p)
    else:
        with open(p, "rb") as fh:
            digest = md.md5_of_file.__name__ + ":" + str(p.stat().st_size) + ":" + \
                     __import__("hashlib").md5(fh.read(4 * 1024 * 1024)).hexdigest()
    if digest in seen:
        duplicates.append({"image_id": r.image_id, "duplicate_of": seen[digest]["image_id"]})
        r.notes += "duplicate; "
    else:
        seen[digest] = {"image_id": r.image_id}
save_rows_csv(duplicates, ANALYSIS_DIR / "duplicate_report.csv")
save_rows_csv([{"image_id": c["image_id"], "reason": c["reason"]} for c in corrupted + missing],
              ANALYSIS_DIR / "missing_corrupt_report.csv")

# %% [markdown]
# **Metadata coverage** — fraction of products with lat/lon, sun angle, resolution filled.

# %%
def coverage(values):
    vals = [v for v in values if v is not None and isinstance(v, (int, float)) and np.isfinite(v)]
    return round(len(vals) / len(values), 3) if values else 0.0

meta_coverage = {
    "latitude": coverage([r.latitude for r in all_records]),
    "longitude": coverage([r.longitude for r in all_records]),
    "sun_elevation": coverage([r.sun_elevation for r in all_records]),
    "sun_azimuth": coverage([r.sun_azimuth for r in all_records]),
    "pixel_resolution": coverage([r.pixel_resolution for r in all_records]),
    "acquisition_date": coverage([r.acquisition_date for r in all_records]),
}

# %% [markdown]
# ## 5. Visualization — sun angle & elevation profiles

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sun = md.load_sunangle_csv(cfg.DATA_ROOT / "SunAngle.csv")
elev = md.load_elevation_csv(cfg.DATA_ROOT / "ElevationProfile.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
import csv as _csv
with open(cfg.DATA_ROOT / "SunAngle.csv") as fh:
    rows = list(_csv.DictReader(fh))
t = np.arange(len(rows))
e = [float(r["Elevation"]) for r in rows]
a = [float(r["Azimuth"]) for r in rows]
axes[0].plot(t, e, color="#38BDF8"); axes[0].set_title("Sun Elevation over pass"); axes[0].set_xlabel("sample")
axes[1].plot(t, a, color="#F59E0B"); axes[1].set_title("Sun Azimuth over pass"); axes[1].set_xlabel("sample")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "sunangle_profile.png", dpi=120)
plt.show()

with open(cfg.DATA_ROOT / "ElevationProfile.csv") as fh:
    erows = list(_csv.DictReader(fh))
ev = [float(r[list(erows[0].keys())[3]]) for r in erows if r[list(erows[0].keys())[3]].strip()]
edist = [float(r["Distance"]) for r in erows if r[list(erows[0].keys())[3]].strip()]
plt.figure(figsize=(10, 3.5))
plt.plot(np.array(edist) / 1000.0, ev, color="#22C55E")
plt.xlabel("distance (km)"); plt.ylabel("elevation (m)")
plt.title("LRO LOLA DEM elevation profile")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "elevation_profile.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics — dataset quality score

# %%
def quality_score(all_records, meta_coverage, corrupted, missing, duplicates) -> float:
    n = len(all_records) or 1
    s_binary = sum(1 for r in all_records if r.binary_present) / n            # 40%
    s_meta = float(np.mean(list(meta_coverage.values()))) if meta_coverage else 0.0  # 30%
    s_integrity = 1.0 - (len(corrupted) + len(missing)) / n                  # 20%
    s_uniq = 1.0 - len(duplicates) / n                                       # 10%
    return round(100 * (0.4 * s_binary + 0.3 * s_meta + 0.2 * s_integrity + 0.1 * s_uniq), 1)

report = {
    "problem_statement": "SIH 26166",
    "generated_by": "01_Dataset_Analysis",
    "total_images": len(all_records),
    "counts": {k: len(v) for k, v in records_by_sensor.items()},
    "ohrc_count": len(records_by_sensor["OHRC"]),
    "tmc_count": len(records_by_sensor["TMC-2"]),
    "iirs_count": len(records_by_sensor["IIRS"]),
    "lro_count": len(records_by_sensor["LRO NAC"]),
    "resolution_distribution": res_dist,
    "format_distribution": fmt_dist,
    "missing_files": missing,
    "corrupted_files": corrupted,
    "duplicate_files": duplicates,
    "metadata_coverage": meta_coverage,
    "sun_angle_statistics": sun,
    "elevation_statistics": elev,
    "dataset_quality_score": quality_score(all_records, meta_coverage, corrupted, missing, duplicates),
    "notes": {
        "iirs": "IIRS products ship ENVI .hdr labels; raw spectral binaries absent in this "
                "distribution - browse PNGs used downstream and flagged accordingly",
        "ohrc": "one OHRC product (20211228) contains labels only",
        "lro": "LRO contribution is a QuickMap LROC export (reference imagery)",
    },
}
save_json(report, REPORT_PATH)
print("dataset_report.json saved ->", REPORT_PATH)

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/dataset_analysis/dataset_report.json`
# - `outputs/dataset_analysis/dataset_inventory.csv`
# - `outputs/dataset_analysis/duplicate_report.csv`
# - `outputs/dataset_analysis/missing_corrupt_report.csv`
# - `outputs/dataset_analysis/sunangle_profile.png`, `elevation_profile.png`

# %% [markdown]
# ## 8. Error Handling
# - Files absent on disk are reported in `missing_files` rather than raising.
# - Truncated .img binaries (size < label expectation) flagged in `corrupted_files`.
# - Undecodable PNGs flagged; XML label parse failures degrade to header-only metadata.

# %% [markdown]
# ## 9. Explanation
# Discovery walks `data/` for PDS4 product folders, pairing each binary with its label.
# Integrity checks compare actual byte counts with PDS4 `Axis_Array` expectations, and
# duplicates are detected by content hash (full md5 for small files, size+head-md5 for
# multi-GB binaries). The quality score weights binary availability (40%), metadata
# completeness (30%), integrity (20%) and uniqueness (10%).

# %% [markdown]
# ## 10. Next-Step Integration
# `02_Preprocessing` consumes `dataset_inventory.csv` and the discovered records; the
# same `records_by_sensor` structure is re-derived there via `metadata.discover_all`.
