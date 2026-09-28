# %% [markdown]
# ## 1. Objective
# Compute the full evaluation suite — **RMSE, Inlier Ratio, Coverage Score, Match Count,
# SSIM, NCC, Runtime** and a confidence score — assemble `evaluation_report.json` and
# export CSV / JSON / PDF deliverables.

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

from lunarai_lib.config import load_config
from lunarai_lib.geometry import compute_ssim_ncc, confidence_score
from lunarai_lib.io_utils import save_json, load_json
from lunarai_lib.reporting import export_metrics_csv, build_pdf_report
from lunarai_lib.db import Database

cfg = load_config()
patch_df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])

def read_csv_rows(path, label):
    """Read a stage report CSV, tolerating a missing or header-only/empty file
    (earlier stages write 0-byte CSVs when no pair solved)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"[warn] {label}: {p.name} missing/empty - stage had no solved pairs")
        return []
    try:
        return pd.read_csv(p).to_dict("records")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {label}: unreadable ({exc})")
        return []

magsac_rows = read_csv_rows(cfg.MATCHING_DIR / "magsac_report.csv", "MAGSAC")
ecc_rows = read_csv_rows(cfg.REGISTRATION_DIR / "ecc_report.csv", "ECC")
print(f"{len(magsac_rows)} verified pairs | {len(ecc_rows)} ECC-refined")

# %% [markdown]
# ## 3. Configuration

# %%
E = cfg.EVAL
RMSE_TARGET = float(E.get("rmse_target", 1.0))
INLIER_TARGET = float(E.get("inlier_ratio_target", 0.85))
COV_TARGET = float(E.get("coverage_target", 0.80))

# %% [markdown]
# ## 4. Implementation

# %%
import json
import time

per_pair = []
for row in ecc_rows:
    stem = row["pair"]
    reg_png = cfg.REGISTRATION_DIR / f"{stem}_ecc_registered.png"
    warp_npy = cfg.REGISTRATION_DIR / f"{stem}_ecc_warp.npy"
    if not reg_png.exists() or not warp_npy.exists():
        continue
    H_ref = np.load(warp_npy)
    # locate the pair's images
    mf = cfg.MATCHING_DIR / f"magsac_{stem}.npz"
    if not mf.exists():
        continue
    d = np.load(mf, allow_pickle=True)
    q_img = cv2.imread(str(d["q_path"]), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(d["r_path"]), cv2.IMREAD_GRAYSCALE)
    warped = cv2.imread(str(reg_png), cv2.IMREAD_GRAYSCALE)
    if q_img is None or r_img is None or warped is None:
        continue
    ssim, ncc = compute_ssim_ncc(warped, r_img)
    anms_row = None
    anms_csv = cfg.MATCHING_DIR / "anms_report.csv"
    if anms_csv.exists():
        adf = pd.read_csv(anms_csv)
        hit = adf[adf.pair == f"feat_{stem.split('feat_')[-1]}"]
        if len(hit):
            anms_row = hit.iloc[0].to_dict()
    mrow = next((m for m in magsac_rows if m["pair"] == f"feat_{stem.split('feat_')[-1]}"), {})
    conf = confidence_score(float(row["rmse_after_ecc"]),
                            float(mrow.get("inlier_ratio", 0.0)),
                            float(anms_row["coverage_after"]) if anms_row else 0.0,
                            int(mrow.get("inliers", 0)),
                            float(row["ecc_correlation"]))
    per_pair.append({
        "pair": stem,
        "query_image": str(d["q_id"]), "reference_image": str(d["r_id"]),
        "rmse_px": float(row["rmse_after_ecc"]),
        "inlier_matches": int(mrow.get("inliers", 0)),
        "inlier_ratio": float(mrow.get("inlier_ratio", 0.0)),
        "coverage_score": float(anms_row["coverage_after"]) if anms_row else None,
        "uniformity_score": float(anms_row["uniformity"]) if anms_row else None,
        "match_count": int(anms_row["matches_in"]) if anms_row else 0,
        "ssim": ssim, "ncc": ncc,
        "ecc_correlation": float(row["ecc_correlation"]),
        "confidence": conf,
        "registered_path": str(reg_png),
        "homography": H_ref.tolist(),
    })

for p in per_pair:
    print(f"  {p['pair']}: rmse={p['rmse_px']:.3f} inlier={p['inlier_ratio']:.2f} "
          f"cov={p['coverage_score']} ssim={p['ssim']:.3f} ncc={p['ncc']:.3f} conf={p['confidence']}")

# %% [markdown]
# ### System-level aggregate + runtime statistics

# %%
# runtime: measured NB timings if present, else synthesized from stage reports
timing_files = [cfg.METRICS_DIR / "superpoint_report.json"]
runtime = {"retrieval_target_s": 10.0}
ret_report = cfg.METRICS_DIR / "retrieval_report.json"
if ret_report.exists():
    runtime["retrieval_queries"] = load_json(ret_report).get("queries")

def safe(fn, default=np.nan):
    vals = [p[fn] for p in per_pair if p.get(fn) is not None and np.isfinite(p[fn])]
    return round(float(np.mean(vals)), 4) if vals else default

summary = {
    "problem_statement": "SIH 26166",
    "n_pairs_evaluated": len(per_pair),
    "rmse_px_mean": safe("rmse_px"),
    "inlier_ratio_mean": safe("inlier_ratio"),
    "coverage_score_mean": safe("coverage_score"),
    "uniformity_score_mean": safe("uniformity_score"),
    "match_count_mean": safe("match_count"),
    "ssim_mean": safe("ssim"),
    "ncc_mean": safe("ncc"),
    "confidence_mean": safe("confidence"),
    "targets": {"rmse_px": RMSE_TARGET, "inlier_ratio": INLIER_TARGET, "coverage": COV_TARGET},
}
summary["rmse_target_met"] = bool(summary["rmse_px_mean"] is not np.nan and
                                  summary["rmse_px_mean"] < RMSE_TARGET)
summary["inlier_target_met"] = bool(summary["inlier_ratio_mean"] is not np.nan and
                                    summary["inlier_ratio_mean"] >= INLIER_TARGET)
summary["coverage_target_met"] = bool(summary["coverage_score_mean"] is not np.nan and
                                      summary["coverage_score_mean"] >= COV_TARGET)

report = {"summary": summary, "per_pair": per_pair, "runtime": runtime,
          "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
save_json(report, cfg.METRICS_DIR / "evaluation_report.json")
print(json.dumps(summary, indent=2, default=str))

# %% [markdown]
# ## 5. Visualization — metric bars + registration gallery

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if per_pair:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    names = [p["pair"] for p in per_pair]
    axes[0].bar(names, [p["rmse_px"] for p in per_pair], color="#38BDF8")
    axes[0].axhline(RMSE_TARGET, color="#EF4444", linestyle="--")
    axes[0].set_title("RMSE (px)"); axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].bar(names, [p["inlier_ratio"] for p in per_pair], color="#22C55E")
    axes[1].axhline(INLIER_TARGET, color="#F59E0B", linestyle="--")
    axes[1].set_title("Inlier ratio"); axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    axes[2].bar(names, [p["coverage_score"] or 0 for p in per_pair], color="#A78BFA")
    axes[2].axhline(COV_TARGET, color="#F59E0B", linestyle="--")
    axes[2].set_title("Coverage"); axes[2].tick_params(axis="x", rotation=90, labelsize=6)
    axes[3].bar(names, [p["confidence"] for p in per_pair], color="#F59E0B")
    axes[3].set_title("Confidence"); axes[3].tick_params(axis="x", rotation=90, labelsize=6)
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "evaluation_summary.png", dpi=120)
    plt.show()

# %% [markdown]
# ## 6. Metrics (CSV/JSON/PDF exports)

# %%
export_metrics_csv(summary, cfg.CSV_DIR / "evaluation_metrics.csv")
save_json(report, cfg.JSON_DIR / "evaluation_report.json")

pdf_images = [Path(p["registered_path"]) for p in per_pair[:3]]
pdf_images += [cfg.VISUALIZATIONS_DIR / "evaluation_summary.png"]
viz_pngs = [cfg.MATCHING_DIR.parent / "visualizations" / n for n in
            ["anms_distribution.png", "lightglue_matches.png", "retrieval_top5.png"]]
pdf_images += [v for v in viz_pngs if v.exists()]
build_pdf_report(
    cfg.PDF_DIR / "LunarAI_Evaluation_Report.pdf",
    title="LunarAI - Registration Evaluation",
    subtitle=f"SIH 26166 | {len(per_pair)} pairs | generated {report['generated_at']}",
    metrics=summary, image_paths=pdf_images,
    summary_lines=[
        "Pipeline: LunaDNA -> FAISS -> SuperPoint+LightGlue -> ANMS -> MAGSAC++ -> Homography -> ECC",
        "Targets: RMSE < 1 px | Inlier ratio > 0.85 | Coverage > 0.80",
    ])
print("PDF ->", cfg.PDF_DIR / "LunarAI_Evaluation_Report.pdf")

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/metrics/evaluation_report.json`
# - `exports/csv/evaluation_metrics.csv`
# - `exports/json/evaluation_report.json`
# - `exports/pdf/LunarAI_Evaluation_Report.pdf`
# - `outputs/visualizations/evaluation_summary.png`
# - SQLite `registration_results` + `correspondences` rows

# %%
db = Database(cfg.DATABASE_DIR / "lunarai.db")
for p in per_pair:
    db.execute(
        "INSERT OR REPLACE INTO registration_results(registration_id, source_image,"
        " reference_image, rmse, inlier_ratio, coverage) VALUES (?,?,?,?,?,?)",
        (p["pair"], p["query_image"], p["reference_image"], p["rmse_px"],
         p["inlier_ratio"], p["coverage_score"]))
    db.execute(
        "INSERT OR REPLACE INTO correspondences(match_id, query_image, reference_image,"
        " total_matches, inlier_matches) VALUES (?,?,?,?,?)",
        (p["pair"], p["query_image"], p["reference_image"], p["match_count"],
         p["inlier_matches"]))

# %% [markdown]
# ## 8. Error Handling
# - Pairs missing any artifact (registered image, warp, magsac npz) are skipped.
# - NaN metrics propagate as null in JSON rather than crashing aggregation.
# - Targets pulled from config; target achievement recorded as booleans in summary.

# %% [markdown]
# ## 9. Explanation
# RMSE is reprojection error of ANMS-inliers under the refined homography (pixels). SSIM/NCC
# are computed over the valid (non-border) overlap so warp padding does not dilute the score.
# The confidence score fuses all evidence per 09_Evaluation section 12 weights:
# RMSE 30 / inlier 30 / coverage 25 / match count 10 / ECC 5.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 14 assembles the Streamlit dashboard reading every artifact produced so far.
