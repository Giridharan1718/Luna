# %% [markdown]
# ## 1. Objective
# Run the complete SIH validation program (Experiments 1–8) against the built pipeline and
# produce every required report: multi-modal correspondence, sun-angle robustness, scale
# invariance, sub-pixel accuracy, benchmark comparison, retrieval validation, uniform
# distribution, ablation study, plus the final summary, metric tables and judges PDF.

# %% [markdown]
# ## 2. Dataset Loading — verify the pipeline artifacts the suite depends on

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
    "embeddings.npy": (cfg.DATABASE_DIR / "embeddings.npy").exists(),
    "dataset_report.json": (cfg.DATASET_ANALYSIS_DIR / "dataset_report.json").exists(),
}
print(json.dumps(prereq, indent=1))
assert all(prereq.values()), "Run notebooks 01-06 first (see README run order)"

# %% [markdown]
# ## 3. Configuration

# %%
REPORTS = cfg.OUTPUTS_ROOT / "reports"
FIGURES = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
print("reports ->", REPORTS)

# %% [markdown]
# ## 4. Implementation — run the validation suite
# The suite implements the *hybrid protocol*: the genuine TMC ncf↔ncn same-orbit pair is
# exercised for real; cross-sensor behaviour is measured on controlled pairs with a known
# ground-truth homography plus measured sensor-appearance/illumination transfer, because
# the distribution contains no spatially overlapping cross-sensor imagery.

# %%
import os
import subprocess

metrics_path = REPORTS / "final_metrics.json"
rerun = os.environ.get("LUNARAI_FORCE_VALIDATION") == "1"
if metrics_path.exists() and not rerun:
    print("final_metrics.json already present -> validation suite already executed.")
    print("(set LUNARAI_FORCE_VALIDATION=1 to re-run the full suite from this notebook)")
    proc = None
else:
    env = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONPATH": str(PROJECT_ROOT)}
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_validation.py")],
                          capture_output=True, text=True, env={**os.environ, **env},
                          cwd=str(PROJECT_ROOT))
    print(proc.stdout[-6000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:])

# %% [markdown]
# ## 5. Visualization — show the produced report figures

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

wanted = ["multimodal_summary.png", "sun_angle_validation.png", "scale_rmse.png",
          "subpixel_ecc.png", "benchmark_comparison.png", "retrieval_metrics.png",
          "uniformity_bars.png", "ablation.png"]
present = [FIGURES / w for w in wanted if (FIGURES / w).exists()]
if present:
    cols = 2
    rows = (len(present) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows))
    for ax, p in zip(fig.axes, present):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.name, fontsize=9)
        ax.axis("off")
    for ax in list(fig.axes)[len(present):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES / "validation_overview.png", dpi=110)
    plt.show()
else:
    print("no figures yet — rerun the implementation cell")

# %% [markdown]
# ## 6. Metrics — headline results from final_metrics.json

# %%
metrics_path = REPORTS / "final_metrics.json"
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    print("pipeline runs:", m.get("pipeline_runs"), "| matcher:", m.get("matcher_backend"))
    print("exp4 sub-pixel:", json.dumps(m.get("exp4_subpixel"), indent=1))
    print("exp6 retrieval:", json.dumps({k: v for k, v in (m.get("exp6_retrieval") or {}).items()
                                         if k != "per_query"}, indent=1))
else:
    print("final_metrics.json missing")

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/reports/{multimodal,sun_angle,scale,subpixel,benchmark,retrieval,uniform,ablation}_*.md`
# - `outputs/reports/final_results_summary.md` (with the 9 judge Q&A answers)
# - `outputs/reports/final_metrics.{json,csv}`, `final_presentation_tables.csv`
# - `outputs/reports/final_judges_report.pdf`
# - `outputs/reports/figures/*.png` (+ registration samples)

# %%
listing = sorted(p.name for p in REPORTS.glob("*"))
print(f"{len(listing)} files in outputs/reports:")
for name in listing:
    print(" -", name)

# %% [markdown]
# ## 8. Error Handling
# - The driver catches per-experiment exceptions and continues, recording the traceback
#   in the log so a single failing experiment cannot block the rest.
# - NaN metrics (failed homographies) are excluded from means and reported as `-`.
# - Missing prerequisites are asserted up front with an explicit run-order hint.

# %% [markdown]
# ## 9. Explanation
# Each experiment writes its own markdown report containing: the protocol note (real /
# retrieved / controlled), the metric table with all requested columns, figures and a
# per-run detail table. `final_results_summary.md` consolidates the headline numbers, the
# success-criteria table with measured evidence and honest limitations.

# %% [markdown]
# ## 10. Next-Step Integration
# These reports feed the SIH presentation and the dashboard's Metrics/Reports pages
# (notebook 14). Re-run anytime after new training or data: the suite memoizes pipeline
# runs so repeated experiments cost nothing.
