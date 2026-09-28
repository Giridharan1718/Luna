# %% [markdown]
# ## 1. Objective
# SIH Phases 3-5 — experiments and benchmarks: the new TMC-2 <-> IIRS pair,
# the full 5-bin sun-angle table (measured + synthetic-illumination protocol),
# scale 0.5x-8x with the multiscale retry ladder, the real viewpoint pair,
# AKAZE + SuperPoint+SuperGlue benchmarks, and the ISRO requirement matrix.
# All numbers are produced by `scripts/run_sih_upgrade.py`; this notebook
# loads, displays and plots them.

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

from lunarai_lib.config import load_config

cfg = load_config()
SIH = cfg.OUTPUTS_ROOT / "sih_upgrade"
metrics_p = SIH / "sih_upgrade_metrics.json"
if not metrics_p.exists():
    raise RuntimeError("run `python scripts/run_sih_upgrade.py` first")
R = json.loads(metrics_p.read_text(encoding="utf-8"))

# %% [markdown]
# ## 3. TMC-2 <-> IIRS — the previously missing pair

# %%
import pandas as pd

ti = pd.DataFrame(R.get("tmc2_iirs", {}).get("rows", []))
display(ti)

# %% [markdown]
# ## 4. Full sun-bin table (measured + synthetic-illumination protocol)

# %%
sb = pd.DataFrame(R.get("sun_bins", {}).get("rows", []))
display(sb)

import matplotlib.pyplot as plt

if len(sb):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ok = sb[sb.rmse_gt_px.notna()]
    colors = ["#38BDF8" if p == "measured" else "#A78BFA" for p in ok.protocol]
    ax1.bar(ok.sun_bin, ok.rmse_gt_px, color=colors, width=0.55)
    ax1.axhline(1.0, color="#EF4444", linestyle="--", label="1 px target")
    ax1.set_ylabel("GT RMSE (px)")
    ax1.set_xlabel("sun bin (blue = measured, purple = synthetic illumination)")
    ax2 = ax1.twinx()
    ax2.plot(ok.sun_bin, ok.inlier_ratio_3px, "o-", color="#22C55E")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("inlier ratio @3px", color="#22C55E")
    ax1.set_title("Sun-angle robustness across all 5 SIH bins")
    plt.tight_layout()
    plt.savefig(SIH / "sun_bins_full.png", dpi=130)
    plt.show()

# %% [markdown]
# ## 5. Scale 0.5x-8x with the multiscale retry ladder + retrieval@scale

# %%
sc = pd.DataFrame(R.get("scale", {}).get("rows", []))
display(sc)
ret = pd.DataFrame(R.get("scale", {}).get("retrieval", []))
display(ret)

if len(sc):
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ok = sc[sc.rmse_gt_px.notna()]
    ax1.bar(ok["scale"].astype(str), ok.rmse_gt_px, color="#38BDF8", width=0.5)
    ax1.set_xlabel("scale ratio (8x = guard-rejected or retry-solved)")
    ax1.set_ylabel("GT RMSE (px)")
    ax2 = ax1.twinx()
    if len(ret):
        ax2.plot(ret["scale"].astype(str), ret.retrieval_top1_correct,
                 "o-", color="#F59E0B")
    ax2.set_ylabel("retrieval top-1 (correct source)")
    ax1.set_title("Scale stress test: registration + retrieval")
    plt.tight_layout()
    plt.savefig(SIH / "scale_extended.png", dpi=130)
    plt.show()

# %% [markdown]
# ## 6. Viewpoint — the real TMC ncf<->ncn pair after the retry ladder

# %%
vp = pd.DataFrame(R.get("viewpoint", {}).get("rows", []))
display(vp)

# %% [markdown]
# ## 7. Benchmarks — AKAZE + SuperPoint+SuperGlue on the focused set

# %%
bm = pd.DataFrame(R.get("benchmarks", {}).get("rows", []))
display(bm)
if len(bm):
    piv = bm.pivot_table(index="method", values=["inlier_ratio_3px", "runtime_s"],
                         aggfunc="mean")
    display(piv)

# %% [markdown]
# ## 8. ISRO requirement matrix

# %%
matrix = pd.DataFrame(R.get("isro_matrix", {}).get("rows", []))
display(matrix)

# %% [markdown]
# ## 9. Summary
# - TMC-2 <-> IIRS closes the 5-pair sensor grid demanded by the PS.
# - Sun bins 10-20 / 60+ are now reported via the labelled synthetic-illumination
#   protocol instead of being empty.
# - 4x/8x scale rows show the retry ladder's effect (multiscale_retry column).
# - AKAZE and SuperGlue rows close the benchmark matrix.
print("nb19 experiments: OK")
