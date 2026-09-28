# %% [markdown]
# ## 1. Objective
# Run the **extended SIH validation program**: multi-modal CSV export, 5-bin sun-angle
# analysis, scale validation, sub-pixel accuracy, benchmark table, coverage analysis,
# confidence engine, lunar crater validation, geographic localization — producing
# `final_sih_results.md` plus every requested CSV/PNG in `outputs/reports/`.

# %% [markdown]
# ## 2. Dataset Loading — prerequisites

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

import json

from lunarai_lib.config import load_config

cfg = load_config()
prereq = {
    "patch_index.csv": (cfg.PATCHES_DIR / "patch_index.csv").exists(),
    "lunadna.pt": (cfg.MODELS_DIR / "lunadna.pt").exists(),
    "faiss_index.bin": (cfg.DATABASE_DIR / "faiss_index.bin").exists(),
    "SunAngle.csv": (cfg.DATA_ROOT / "SunAngle.csv").exists(),
    "ElevationProfile.csv": (cfg.DATA_ROOT / "ElevationProfile.csv").exists(),
}
print(json.dumps(prereq, indent=1))
assert all(prereq.values()), "Run notebooks 01-06 first (see README run order)"

# %% [markdown]
# ## 3. Configuration — outputs this notebook produces

# %%
REPORTS = cfg.OUTPUTS_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
EXPECTED = [
    "multi_modal_validation.csv", "multi_modal_validation.png",
    "sun_angle_analysis.csv", "sun_angle_performance.png",
    "scale_validation.csv", "scale_performance.png",
    "subpixel_accuracy_report.csv", "subpixel_accuracy_plot.png",
    "benchmark_results.csv", "benchmark_table.png",
    "coverage_analysis.csv", "coverage_visualization.png",
    "confidence_validation.csv",
    "crater_validation.csv", "crater_validation.png",
    "localization_report.csv",
    "final_sih_results.md", "extended_metrics.json",
]
print("outputs ->", REPORTS)
print(f"{len(EXPECTED)} extended deliverables expected")

# %% [markdown]
# ## 4. Implementation — run the extended validation driver
# The driver reuses the base ValidationSuite (model, FAISS index, matcher) so every
# measurement shares the controlled-GT protocol of Experiments 1-8. Set
# LUNARAI_FORCE_EXT=1 to force a re-run; otherwise an existing extended_metrics.json
# is treated as current.

# %%
import os
import subprocess

metrics_path = REPORTS / "extended_metrics.json"
if metrics_path.exists() and os.environ.get("LUNARAI_FORCE_EXT") != "1":
    print("extended_metrics.json already present -> extended validation already executed.")
    print("(set LUNARAI_FORCE_EXT=1 to force a re-run from this notebook)")
else:
    env = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONPATH": str(PROJECT_ROOT)}
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_extended_validation.py")],
                          capture_output=True, text=True, env={**os.environ, **env},
                          cwd=str(PROJECT_ROOT))
    print(proc.stdout[-6000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:])

# %% [markdown]
# ## 5. Visualization — deliverable inventory + extended figures

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

inventory = {name: (REPORTS / name).exists() for name in EXPECTED}
missing = [k for k, v in inventory.items() if not v]
print(f"{sum(inventory.values())}/{len(EXPECTED)} deliverables present")
if missing:
    print("missing:", missing)

figs = ["sun_angle_performance.png", "scale_performance.png", "benchmark_table.png",
        "coverage_visualization.png", "crater_validation.png",
        "multi_modal_validation.png", "subpixel_accuracy_plot.png"]
present = [REPORTS / f for f in figs if (REPORTS / f).exists()]
if present:
    cols = 2
    rows = (len(present) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.0 * rows))
    for ax, p in zip(fig.axes, present):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.name, fontsize=9)
        ax.axis("off")
    for ax in list(fig.axes)[len(present):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(REPORTS / "extended_validation_overview.png", dpi=110)
    plt.show()

# %% [markdown]
# ## 6. Metrics — headline numbers from extended_metrics.json

# %%
import pandas as pd

m = REPORTS / "extended_metrics.json"
if m.exists():
    xj = json.loads(m.read_text())
    print("pipeline runs:", xj.get("pipeline_runs"))
    print("sun-angle bins:")
    print(pd.DataFrame(xj.get("sun_angle_bins", [])).to_string(index=False))
    print("localization:", json.dumps(xj.get("localization_summary", {}), indent=1))
    print("confidence categories:", xj.get("confidence_category_counts"))
    print("craters:", xj.get("crater_matched_total"), "matched across",
          xj.get("crater_pairs"), "pairs")
else:
    print("extended_metrics.json missing")

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/reports/multi_modal_validation.{csv,png}`
# - `outputs/reports/sun_angle_analysis.csv`, `sun_angle_performance.png`
# - `outputs/reports/scale_validation.csv`, `scale_performance.png`
# - `outputs/reports/subpixel_accuracy_report.csv`, `subpixel_accuracy_plot.png`
# - `outputs/reports/benchmark_results.csv`, `benchmark_table.png`
# - `outputs/reports/coverage_analysis.csv`, `coverage_visualization.png`
# - `outputs/reports/confidence_validation.csv`
# - `outputs/reports/crater_validation.{csv,png}`
# - `outputs/reports/localization_report.csv`
# - `outputs/reports/final_sih_results.md`, `extended_metrics.json`

# %%
listing = sorted(p.name for p in REPORTS.glob("*") if p.is_file())
ext_files = [n for n in listing if n in EXPECTED]
print(f"{len(ext_files)}/{len(EXPECTED)} extended files in outputs/reports:")
for name in ext_files:
    print(" -", name)

# %% [markdown]
# ## 8. Error Handling
# - The driver wraps each section independently; a failing section logs a traceback
#   and the remaining sections still produce their deliverables.
# - Sun bins with no catalog patches are reported as empty rows, never interpolated.
# - Pairs rejected by the acceptance guard appear in the CSVs with their status so the
#   dashboard can show why, instead of silently vanishing.

# %% [markdown]
# ## 9. Explanation
# The extended modules add the mission-specific evidence the base suite does not
# measure: crater-consistency (Hough rims matched through the accepted homography),
# the SIH 5-bin sun-angle grouping with measured catalog shares, an a-priori-weighted
# confidence engine (25 sim + 30 inliers@3px + 30 RMSE + 15 coverage), and
# retrieval-as-localization with a label-shuffled chance baseline and similarity-band
# calibration. Every number is grounded in the same controlled-GT protocol as
# Experiments 1-8; nothing is tuned against its own criterion.

# %% [markdown]
# ## 10. Next-Step Integration
# These CSVs feed the **📊 Analytics** tabs and the **📍 Registration** sub-pixel tab of
# the seven-page app (generated by `app_build.py`), and `final_sih_results.md` is the SIH
# submission summary.
