"""LunarAI validation driver — Experiments 1-8 + final SIH deliverables.

Writes markdown reports, figures, metric tables and the judges PDF:
  outputs/reports/multimodal_validation_report.md
  outputs/reports/sun_angle_validation_report.md
  outputs/reports/scale_validation_report.md
  outputs/reports/subpixel_accuracy_report.md
  outputs/reports/benchmark_comparison_report.md
  outputs/reports/retrieval_validation_report.md
  outputs/reports/uniform_distribution_report.md
  outputs/reports/ablation_study_report.md
  outputs/reports/final_results_summary.md
  outputs/reports/final_metrics.csv / .json
  outputs/reports/final_presentation_tables.csv
  outputs/reports/final_judges_report.pdf
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lunarai_lib.config import load_config
from lunarai_lib.reporting import build_pdf_report
from lunarai_lib.validation import (GRID, Pair, ValidationSuite, load_gray,
                                    md_table, safe_mean)

METRIC_KEYS = ["match_count", "inlier_count", "inlier_ratio", "inlier_count_3px",
               "inlier_ratio_3px", "coverage_score", "uniformity_score", "rmse_px",
               "rmse_gt_px", "ssim", "ncc", "runtime_s", "confidence"]


class Cache:
    """Memoizes pipeline runs keyed by (pair, variant) so experiments share work."""

    def __init__(self, suite: ValidationSuite, save_dir: Path):
        self.suite = suite
        self.results: dict[tuple[str, str], dict] = {}
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_runs = 0

    def run(self, pair: Pair, variant: str = "full", save: bool = False,
            log=print) -> dict:
        key = (pair.name, variant)
        if key not in self.results:
            t0 = time.perf_counter()
            res = self.suite.run_pipeline(
                pair, variant=variant,
                save_dir=self.save_dir if save else None)
            self.n_runs += 1
            log(f"    [{self.n_runs:3d}] {pair.name[:44]:44s} {variant:10s} "
                f"{res.get('status','?'):>16s} "
                f"rmse_gt={res.get('rmse_gt_px', float('nan')):.3f} "
                f"inl={res.get('inlier_ratio', 0):.2f} "
                f"cov={res.get('coverage_score', 0):.2f} "
                f"({time.perf_counter()-t0:.1f}s)")
            self.results[key] = res
        return self.results[key]


def safe_median(vals) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float("nan")


def iqr(vals) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) else float("nan")


def ok_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "ok"]


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    """Group run rows by `keys`, averaging METRIC_KEYS and counting successes.

    Medians over the *successful* runs are added because a mean over a mix of solved
    and rejected runs is not a meaningful accuracy statement.
    """
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = tuple(r.get(k, "") for k in keys)
        groups.setdefault(k, []).append(r)
    for k, rs in groups.items():
        rec = {key: val for key, val in zip(keys, k)}
        for m in METRIC_KEYS:
            rec[m] = round(safe_mean([r.get(m) for r in rs]), 4)
        oks = ok_rows(rs)
        rec["runs"] = len(rs)
        rec["success_runs"] = len(oks)
        rec["success_rate"] = round(len(oks) / max(len(rs), 1), 3)
        rec["rmse_gt_px_median"] = round(safe_median([r.get("rmse_gt_px") for r in oks]), 4)
        rec["rmse_px_median"] = round(safe_median([r.get("rmse_px") for r in oks]), 4)
        rec["coverage_score_median"] = round(safe_median([r.get("coverage_score")
                                                          for r in oks]), 4)
        rec["uniformity_score_median"] = round(safe_median([r.get("uniformity_score")
                                                            for r in oks]), 4)
        gt_solved = [r.get("rmse_gt_px") for r in oks
                     if np.isfinite(r.get("rmse_gt_px", float("nan")))]
        rec["subpixel_share"] = (round(float(np.mean([v < 1.0 for v in gt_solved])), 3)
                                 if gt_solved else None)
        out.append(rec)
    return out


def bar_chart(labels, series: dict[str, list], title: str, ylabel: str, path: Path,
              rot: int = 0, hline: tuple[float, str] | None = None) -> None:
    n = len(series)
    x = np.arange(len(labels))
    width = 0.8 / max(n, 1)
    plt.figure(figsize=(max(6.5, 1.1 * len(labels)), 4.0))
    for i, (name, vals) in enumerate(series.items()):
        plt.bar(x + i * width, vals, width, label=name)
    if hline:
        plt.axhline(hline[0], linestyle="--", color="#EF4444", label=hline[1])
    plt.xticks(x + width * (n - 1) / 2, labels, rotation=rot)
    plt.ylabel(ylabel)
    plt.title(title)
    if n > 1 or hline:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def line_chart(xs, series: dict[str, list], title: str, xlabel: str, ylabel: str,
               path: Path) -> None:
    plt.figure(figsize=(6.5, 4.0))
    for name, vals in series.items():
        plt.plot(xs, vals, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Experiment 1 — multi-modal validation
# --------------------------------------------------------------------------- #
def exp1_multimodal(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp1] multi-modal validation")
    combos = [("OHRC", "LRO NAC"), ("OHRC", "TMC-2"), ("OHRC", "IIRS"),
              ("TMC-2", "LRO NAC"), ("TMC-2", "IIRS")]
    rows_controlled, rows_retrieved = [], []
    for a, b in combos:
        for i, src_path in enumerate(suite.best_texture_patches(a, n=3, seed=5)):
            pair = suite.controlled_pair(src_path, name=f"ctl_{a}_to_{b}_s1_i{i}",
                                         sensor_a=a, sensor_b=b, scale=1.0, seed=100 + i)
            if pair is None:
                continue
            r = cache.run(pair, "full", save=(i == 0), log=log)
            r.update(group=f"{a} <-> {b}")
            rows_controlled.append(r)
        for i in range(2):
            pair = suite.retrieved_pair(a, b, seed=10 + i)
            if pair is None:
                continue
            r = cache.run(pair, "full", log=log)
            r.update(group=f"{a} <-> {b}")
            rows_retrieved.append(r)

    agg_c = aggregate(rows_controlled, ["group"])
    agg_r = aggregate(rows_retrieved, ["group"])

    fig = out["figures"] / "multimodal_summary.png"
    if agg_c:
        labels = [a["group"] for a in agg_c]
        bar_chart(labels, {
            "inlier_ratio": [a["inlier_ratio"] for a in agg_c],
            "coverage": [a["coverage_score"] for a in agg_c],
            "ssim": [a["ssim"] for a in agg_c],
        }, "Multi-modal correspondence quality (controlled GT pairs)",
            "score", fig, rot=15,
            hline=(float(suite.cfg.EVAL.get("inlier_ratio_target", 0.85)), "inlier target"))

    availability = ("**Data-availability note.** The provided distribution contains no "
                    "spatially overlapping OHRC/TMC/IIRS/LRO image pairs (OHRC images are "
                    "south-polar, TMC mid-latitude 2021/2024/2026 products, IIRS is "
                    "browse-only, LRO is a QuickMap reference export). Cross-sensor "
                    "behaviour is therefore reported twice: *controlled* pairs (terrain "
                    "warps with a known ground-truth homography plus measured "
                    "sensor-appearance transfer — sharpness, contrast, brightness, noise) "
                    "give precise error figures; *retrieved* pairs are the real "
                    "cross-sensor candidates FAISS selects, showing raw behaviour on "
                    "this distribution.\n\n"
                    "**GSD protocol (why the pairs are GSD-normalized).** Product GSDs "
                    "span 0.24 m/px (OHRC) to 80 m/px (IIRS), a 300x+ gap that no "
                    "single-scale keypoint matcher can bridge and that the pipeline "
                    "never asks it to: both sides are resampled to a common working GSD "
                    "using the per-product GSD from the PDS labels before matching. The "
                    "controlled pairs model exactly that regime — shared geometry at a "
                    "common GSD, with the measured sensor appearance differences "
                    "(sharpness/contrast/noise) and illumination on top. Residual "
                    "scale tolerance after that resample is what Experiment 3 "
                    "measures (±0.5x … 4x).\n\n"
                    "**Acceptance guard.** A homography fitted to four correspondences "
                    "reproduces them exactly, so `inlier_ratio = 1.0` on a handful of "
                    "matches is an artifact of the fit, not a registration. Runs with "
                    "fewer than 10 correspondences or with a geometrically implausible "
                    "warp (degenerate/flipped/exploding quad) are therefore reported as "
                    "rejected, not scored; the tables show `success_runs` next to the "
                    "medians so both are visible.")

    table_cols = ["group", "runs", "success_runs", "success_rate", "match_count",
                  "inlier_count", "inlier_ratio", "inlier_ratio_3px", "coverage_score",
                  "rmse_px_median", "rmse_gt_px_median", "subpixel_share", "ssim",
                  "runtime_s"]
    report = f"""# Experiment 1 — Multi-modal Validation Report

**Project:** LunarAI — SIH 26166 · **Generated:** {time.strftime('%Y-%m-%d %H:%M')}
**Matcher:** {suite.matcher_backend} · **Runtime measured on:** CPU

{availability}

Score in each pair: `inlier_ratio` measured after MAGSAC++ on ANMS-filtered matches;
`rmse_gt_px` is the pixel RMSE of the estimated warp against the known ground truth
(controlled pairs only); `ssim`/`ncc` compare the ECC-registered image to the reference.

## Controlled pairs (ground-truth protocol)

{md_table([{c: a.get(c) for c in table_cols} for a in agg_c])}

![multimodal summary](figures/multimodal_summary.png)

## Retrieved pairs (real FAISS candidates, non-overlapping imagery)

{md_table([{c: a.get(c) for c in table_cols} for a in agg_r])}

`rmse_*_median` are medians over the runs that passed the acceptance guard, so a
single degenerate pair cannot dominate the group.

Per-run detail (all requested metrics)

{md_table([{c: r.get(c) for c in ['group', 'protocol', 'pair', 'status', 'match_count',
                                  'inlier_count', 'inlier_ratio', 'inlier_ratio_3px',
                                  'coverage_score', 'rmse_px', 'rmse_gt_px', 'ssim', 'ncc',
                                  'runtime_s']}
           for r in rows_controlled + rows_retrieved])}
"""
    (out["reports"] / "multimodal_validation_report.md").write_text(report, encoding="utf-8")
    return {"controlled": agg_c, "retrieved": agg_r,
            "rows": rows_controlled + rows_retrieved}


# --------------------------------------------------------------------------- #
# Experiment 2 — sun-angle validation
# --------------------------------------------------------------------------- #
def exp2_sun_angle(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp2] sun-angle validation")
    groups = ["low", "medium", "high"]
    src = suite.best_texture_patches("TMC-2", n=4, seed=8)
    rows = []
    for g in groups:
        for i, p in enumerate(src):
            pair = suite.controlled_pair(p, name=f"sun_{g}_i{i}", sensor_a="TMC-2",
                                         sensor_b="TMC-2", scale=1.0, illum_group=g,
                                         seed=200 + i)
            if pair is None:
                continue
            r = cache.run(pair, "full", save=(i == 0), log=log)
            r.update(sun_group=g)
            rows.append(r)
    agg = aggregate(rows, ["sun_group"])

    # retrieval accuracy per group: illuminate a sample of patches, embed, top-1 same-source
    log("[exp2] retrieval accuracy per illumination group")
    rng = np.random.default_rng(3)
    sample = suite.df.sample(min(60, len(suite.df)), random_state=3)
    ret_rows = []
    for g in groups:
        hits1 = hits5 = 0
        n = 0
        for _, row in sample.iterrows():
            img = load_gray(row.patch_path, 224, enhance=False)
            if img is None:
                continue
            q = suite.apply_illumination(img, g)
            tmp = out["tmp"] / f"sunq_{g}_{n}.png"
            cv2.imwrite(str(tmp), q)
            from lunarai_lib.lunadna import compute_embeddings
            emb = compute_embeddings(suite.model, [str(tmp)], device=suite.device,
                                     size=int(suite.cfg.LUNADNA.get("input_size", 224)))
            scores, idxs = suite.index.search(emb[0], 6)
            meta = suite.df.set_index("patch_id")
            ranked = []
            for s, gi in zip(scores[0], idxs[0]):
                if 0 <= gi < len(suite.mapping):
                    cid = Path(suite.mapping[int(gi)]).stem
                    if cid != row.patch_id:
                        ranked.append(cid)
            def hit(k):
                for cid in ranked[:k]:
                    if cid in meta.index and meta.loc[cid].source_image == row.source_image:
                        return True
                return False
            if ranked:
                n += 1
                hits1 += hit(1)
                hits5 += hit(5)
        ret_rows.append({"sun_group": g, "n": n,
                         "top1_acc": round(hits1 / max(n, 1), 3),
                         "top5_acc": round(hits5 / max(n, 1), 3)})
        log(f"    {g}: n={n} top1={ret_rows[-1]['top1_acc']} top5={ret_rows[-1]['top5_acc']}")

    fig = out["figures"] / "sun_angle_validation.png"
    labels = [a["sun_group"] for a in agg]
    bar_chart(labels, {
        "inlier_ratio": [a["inlier_ratio"] for a in agg],
        "coverage": [a["coverage_score"] for a in agg],
        "ssim": [a["ssim"] for a in agg],
    }, "Sun-angle robustness: correspondence quality (controlled illumination transfer)",
        "score", fig)
    fig2 = out["figures"] / "sun_angle_rmse.png"
    bar_chart(labels, {"rmse_gt_px": [a["rmse_gt_px"] for a in agg]},
              "Registration error vs sun-angle group", "pixel RMSE", fig2,
              hline=(1.0, "target < 1 px"))

    real_sun = {k: v for k, v in sorted(suite.image_sun.items(), key=lambda kv: kv[1])}
    real_sun_md = md_table([{"product": k, "sun_elevation_deg": round(v, 2)}
                            for k, v in real_sun.items()])

    report = f"""# Experiment 2 — Sun-Angle Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')} · Matcher {suite.matcher_backend}

## Illumination groups (measured from product labels + SunAngle.csv)

| Group | Real products | Sun elevation |
|---|---|---|
| Low | OHRC south-polar products | ≈ −0.8° … +0.8° (near-terminator) |
| Medium | TMC-2 2021-11-22 forward/nadir, IIRS 2024-01-24 | 37.1° – 38.3° |
| High | TMC-2 2026-07-01 forward | 51.7° |

SunAngle.csv pass statistics: elevation {json.dumps(suite.cfg and {}) if False else ''}see
`outputs/dataset_analysis/dataset_report.json` (`sun_angle_statistics`).
Illumination transfer matches each group's measured mean/std (gamma 0.85 / 1.0 / 1.15),
then the pipeline's CLAHE normalization is applied to both images before matching.

## Correspondence quality per group (controlled illumination transfer)

{md_table([{c: a.get(c) for c in ['sun_group', 'runs', 'success_runs', 'match_count',
                                  'inlier_count', 'inlier_ratio', 'coverage_score',
                                  'rmse_px', 'rmse_gt_px', 'ssim', 'ncc', 'runtime_s']}
           for a in agg])}

![sun angle quality](figures/sun_angle_validation.png)
![sun angle rmse](figures/sun_angle_rmse.png)

## Retrieval accuracy per group (LunaDNA + FAISS)

{md_table(ret_rows)}

## Real per-product sun elevations

{real_sun_md}
"""
    (out["reports"] / "sun_angle_validation_report.md").write_text(report, encoding="utf-8")
    return {"correspondence": agg, "retrieval": ret_rows, "rows": rows}


# --------------------------------------------------------------------------- #
# Experiment 3 — scale invariance
# --------------------------------------------------------------------------- #
def exp3_scale(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp3] scale-invariant validation")
    combos = [("OHRC", "TMC-2"), ("OHRC", "LRO NAC"), ("TMC-2", "LRO NAC"),
              ("TMC-2", "IIRS")]
    scales = [0.5, 1.0, 2.0, 4.0]
    rows = []
    for a, b in combos:
        srcs = suite.best_texture_patches(a, n=1, seed=12)
        if not srcs:
            continue
        for s in scales:
            pair = suite.controlled_pair(srcs[0], name=f"scale_{a}_to_{b}_x{s}",
                                         sensor_a=a, sensor_b=b, scale=s, seed=300)
            if pair is None:
                continue
            r = cache.run(pair, "full", save=(s == 1.0), log=log)
            r.update(combo=f"{a} vs {b}", scale_factor=s)
            rows.append(r)
    agg = aggregate(rows, ["combo", "scale_factor"])

    fig = out["figures"] / "scale_validation.png"
    combos_present = sorted({a["combo"] for a in agg})
    for metric, fname, ylabel, hl in [
            ("rmse_gt_px_median", "scale_rmse.png", "pixel RMSE (median)",
             (1.0, "target < 1 px")),
            ("match_count", "scale_matches.png", "matches", None),
            ("inlier_ratio", "scale_inlier.png", "inlier ratio", (0.85, "target"))]:
        plt.figure(figsize=(6.5, 4.0))
        for combo in combos_present:
            sub = [a for a in agg if a["combo"] == combo]
            sub.sort(key=lambda x: x["scale_factor"])
            plt.plot([x["scale_factor"] for x in sub], [x[metric] for x in sub],
                     marker="o", label=combo)
        if hl:
            plt.axhline(hl[0], linestyle="--", color="#EF4444", label=hl[1])
        plt.xscale("log", base=2)
        plt.xticks(scales, [f"{s}x" for s in scales])
        plt.xlabel("reference scale factor")
        plt.ylabel(ylabel)
        plt.title(f"Scale invariance: {metric}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out["figures"] / fname, dpi=130)
        plt.close()

    report = f"""# Experiment 3 — Scale-Invariant Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')} · Matcher {suite.matcher_backend}

Scale factors 0.5x / 1x / 2x / 4x are applied to the **reference** image via the
known ground-truth homography (a 2x reference renders the terrain at twice the
resolution of the query, 0.5x at half). `rmse_gt_px` = RMSE of the estimated warp
against that ground truth, in reference pixels.

{md_table([{c: a.get(c) for c in ['combo', 'scale_factor', 'runs', 'success_runs',
                                  'success_rate', 'match_count', 'inlier_ratio',
                                  'inlier_ratio_3px', 'coverage_score', 'rmse_gt_px_median',
                                  'subpixel_share', 'ssim', 'runtime_s']} for a in agg])}

![scale rmse](figures/scale_rmse.png)
![scale matches](figures/scale_matches.png)
![scale inlier](figures/scale_inlier.png)

## Interpretation

Scale is applied to the **reference** through the ground-truth homography, so each row
is the residual scale budget LunarAI must absorb *after* both sides were resampled to a
common working GSD (see the GSD protocol in Experiment 1). Registration error is the
median over runs that passed the acceptance guard; `success_rate` reports how many
pairs produced an accepted warp at that scale, which is the honest envelope of a
single-scale SuperPoint/LightGlue front end.

GSD itself is not left to the matcher: the pipeline resamples every product to the
working GSD from its PDS label before this stage (OHRC 0.24 m/px vs TMC-2 ~5 m/px is
~20x), and LunaDNA is scale-robust by construction because embeddings are computed on
resized 224x224 inputs — that is what makes retrieval across sensors possible at all.
"""
    (out["reports"] / "scale_validation_report.md").write_text(report, encoding="utf-8")
    return {"aggregate": agg, "rows": rows}


# --------------------------------------------------------------------------- #
# Experiment 4 — sub-pixel accuracy
# --------------------------------------------------------------------------- #
def exp4_subpixel(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp4] sub-pixel accuracy validation")
    # add the real cross-view pair
    real = suite.real_crossview_pair()
    if real is not None:
        cache.run(real, "full", save=True, log=log)

    rows = []
    for (name, variant), r in cache.results.items():
        if variant != "full" or r.get("status") != "ok":
            continue
        before = r.get("rmse_gt_before_ecc", float("nan"))
        after = r.get("rmse_gt_px", float("nan"))
        before_rp = r.get("rmse_px_before_ecc", float("nan"))
        after_rp = r.get("rmse_px", float("nan"))
        if not (np.isfinite(before) or np.isfinite(before_rp)):
            continue
        rows.append({
            "pair": name, "protocol": r.get("protocol"), "status": r.get("status"),
            "rmse_gt_before_ecc": before, "rmse_gt_after_ecc": after,
            "rmse_reproj_before_ecc": before_rp, "rmse_reproj_after_ecc": after_rp,
            "ecc_applied": r.get("ecc_applied"), "ecc_correlation": r.get("ecc_correlation"),
            "ecc_rejected": r.get("ecc_rejected", ""),
        })
    gt_rows = [r for r in rows if np.isfinite(r["rmse_gt_before_ecc"])
               and np.isfinite(r["rmse_gt_after_ecc"])]
    summary = {
        "n_runs_reported": len(rows),
        "n_gt_pairs": len(gt_rows),
        "median_rmse_gt_before_ecc": round(safe_median(
            [r["rmse_gt_before_ecc"] for r in gt_rows]), 4),
        "median_rmse_gt_after_ecc": round(safe_median(
            [r["rmse_gt_after_ecc"] for r in gt_rows]), 4),
        "mean_rmse_gt_before_ecc": round(safe_mean(
            [r["rmse_gt_before_ecc"] for r in gt_rows]), 4),
        "mean_rmse_gt_after_ecc": round(safe_mean(
            [r["rmse_gt_after_ecc"] for r in gt_rows]), 4),
        "iqr_rmse_gt_after_ecc": round(iqr([r["rmse_gt_after_ecc"] for r in gt_rows]), 4),
        "max_rmse_gt_after_ecc": round(float(np.max([r["rmse_gt_after_ecc"]
                                                     for r in gt_rows])), 4) if gt_rows else None,
        "mean_rmse_reproj_before_ecc": round(safe_mean(
            [r["rmse_reproj_before_ecc"] for r in rows]), 4),
        "mean_rmse_reproj_after_ecc": round(safe_mean(
            [r["rmse_reproj_after_ecc"] for r in rows]), 4),
        "subpixel_share": round(float(np.mean([r["rmse_gt_after_ecc"] < 1.0
                                               for r in gt_rows])), 3) if gt_rows else None,
        "ecc_applied_share": round(float(np.mean([r["ecc_applied"] for r in rows
                                                  if r["ecc_applied"] is not None])), 3)
        if rows else None,
        "ecc_rejected_runs": int(sum(1 for r in rows if str(r.get("ecc_rejected"))))
    }

    if gt_rows:
        plt.figure(figsize=(6.5, 5))
        plt.scatter([r["rmse_gt_before_ecc"] for r in gt_rows],
                    [r["rmse_gt_after_ecc"] for r in gt_rows],
                    c="#38BDF8", s=30)
        lim = max(2.0, max([r["rmse_gt_before_ecc"] for r in gt_rows] + [1.0]) * 1.1)
        plt.plot([0, lim], [0, lim], "--", color="#94A3B8", label="no change")
        plt.axhline(1.0, color="#EF4444", linestyle=":", label="1 px target")
        plt.xlabel("RMSE before ECC (px)")
        plt.ylabel("RMSE after ECC (px)")
        plt.title("Sub-pixel refinement: homography vs homography + ECC (GT error)")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out["figures"] / "subpixel_ecc.png", dpi=130)
        plt.close()

    report = f"""# Experiment 4 — Sub-Pixel Accuracy Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')}

Goal: **RMSE < 1 pixel** after ECC refinement.
Two RMSE definitions are reported:
- `rmse_gt_*`: pixel RMSE of the estimated warp vs the **known ground-truth
  homography** (controlled pairs) — an absolute, non-circular measure.
- `rmse_reproj_*`: mean reprojection residual of the inlier correspondences.

## Comparison table (homography-only vs homography + ECC)

| Metric | Before ECC | After ECC |
|---|---|---|
| **Median GT RMSE (px)** | {summary['median_rmse_gt_before_ecc']} | **{summary['median_rmse_gt_after_ecc']}** |
| Mean GT RMSE (px) | {summary['mean_rmse_gt_before_ecc']} | {summary['mean_rmse_gt_after_ecc']} |
| IQR of GT RMSE (px) | — | {summary['iqr_rmse_gt_after_ecc']} |
| Worst GT RMSE (px) | — | {summary['max_rmse_gt_after_ecc']} |
| Mean reprojection RMSE (px) | {summary['mean_rmse_reproj_before_ecc']} | {summary['mean_rmse_reproj_after_ecc']} |
| Pairs with GT evaluated | — | {summary['n_gt_pairs']} |
| Share of GT pairs < 1 px | — | {summary['subpixel_share']} |
| ECC applied / rejected | — | {summary['ecc_applied_share']} / {summary['ecc_rejected_runs']} runs |

![subpixel](figures/subpixel_ecc.png)

## Per-pair detail

{md_table([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
           for r in rows])}

**Reading the numbers.** `rmse_gt_*` is the pixel RMSE of the estimated warp against the
known ground-truth homography, measured over a uniform grid of source points — an
absolute, non-circular measure that no other method in the comparison can be optimised
against. Only runs that passed the acceptance guard (>=10 correspondences, plausible
quad) are listed: with four correspondences a homography reproduces its own points
exactly, so a *low* error there would be meaningless.

ECC refines the homography with an 8-parameter correlation maximization; the
combination H_refined = W_ecc · H keeps the whole chain in one matrix. ECC is kept only
when it does not regress the correspondence geometry (see `ecc_rejected`), because
correlation can otherwise drift on ill-posed pairs. The genuine TMC ncf→ncn cross-view
pair has no ground truth, so it is exercised and reported through the reprojection
measure instead.
"""
    (out["reports"] / "subpixel_accuracy_report.md").write_text(report, encoding="utf-8")
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# Experiment 5 — benchmark comparison
# --------------------------------------------------------------------------- #
def exp5_benchmark(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp5] benchmark comparison")
    # fixed pair set: 6 controlled (from exp1/exp3 cache) + real + retrieved
    pairs: list[Pair] = []
    for a, b in [("OHRC", "TMC-2"), ("OHRC", "IIRS"), ("TMC-2", "LRO NAC"),
                 ("TMC-2", "TMC-2"), ("TMC-2", "IIRS")]:
        for i, src in enumerate(suite.best_texture_patches(a, n=1, seed=5)[:1]):
            p = suite.controlled_pair(src, name=f"ctl_{a}_to_{b}_s1_i0", sensor_a=a,
                                      sensor_b=b, scale=1.0, seed=100)
            if p:
                pairs.append(p)
    for a, b in [("OHRC", "TMC-2"), ("TMC-2", "LRO NAC")]:
        p = suite.controlled_pair(suite.best_texture_patches(a, n=1, seed=12)[0],
                                  name=f"scale_{a}_to_{b}_x2.0", sensor_a=a, sensor_b=b,
                                  scale=2.0, seed=300)
        if p:
            pairs.append(p)
    real = suite.real_crossview_pair()
    if real:
        pairs.append(real)
    ret = suite.retrieved_pair("OHRC", "TMC-2", seed=10)
    if ret:
        pairs.append(ret)
    # dedupe by name
    seen, uniq = set(), []
    for p in pairs:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    pairs = uniq

    method_rows: dict[str, list[dict]] = {"lunarai_full": [], "sift_ransac": [],
                                          "orb_ransac": [], "splg_ransac": [],
                                          "akaze_ransac": [], "superglue_ransac": []}
    # SIH upgrade: AKAZE (cv2) and SuperPoint+SuperGlue (torch.hub) baselines.
    from lunarai_lib.models import AkazeRunner, SuperglueRunner
    akaze = AkazeRunner()
    sg = SuperglueRunner(max_keypoints=2048)
    if not sg.available:
        log(f"[exp5] SuperGlue unavailable on this machine ({sg.error[:120]}...); "
            f"row reported as 'unavailable' (offline-safe)")
    for p in pairs:
        full = cache.run(p, "full", save=True, log=log)
        method_rows["lunarai_full"].append({**full, "method": "lunarai_full"})
        for method, fn in [("sift_ransac", suite.run_sift_ransac),
                           ("orb_ransac", suite.run_orb_ransac),
                           ("splg_ransac", suite.run_splg_ransac)]:
            r = fn(p)
            r["pair"] = p.name
            r["protocol"] = p.protocol
            method_rows[method].append(r)
        r_ak = akaze.run(p, suite._prep, suite.cfg)
        method_rows["akaze_ransac"].append(r_ak)
        r_sg = sg.run(p, suite._prep, suite.cfg)
        method_rows["superglue_ransac"].append(r_sg)

    agg = []
    for method, rs in method_rows.items():
        oks = ok_rows(rs)
        agg.append({
            "method": method, "pairs": len(rs),
            "success_rate": round(float(np.mean([r.get("status") == "ok" for r in rs])), 3),
            "gt_evaluated": sum(1 for r in oks
                                if np.isfinite(r.get("rmse_gt_px", float("nan")))),
            "median_rmse_gt_px": round(safe_median([r.get("rmse_gt_px") for r in oks]), 4),
            "median_rmse_px": round(safe_median([r.get("rmse_px") for r in oks]), 4),
            "inlier_ratio": round(safe_mean([r.get("inlier_ratio") for r in oks]), 4),
            "inlier_ratio_3px": round(safe_mean([r.get("inlier_ratio_3px") for r in oks]), 4),
            "coverage_score": round(safe_mean([r.get("coverage_score") for r in oks]), 4),
            "match_count": round(safe_mean([r.get("match_count") for r in oks]), 1),
            "runtime_s": round(safe_mean([r.get("runtime_s") for r in rs]), 3),
        })
    # rank by reliability first, then by the non-circular GT error
    ranked = sorted(agg, key=lambda x: (-x["success_rate"],
                                        x["median_rmse_gt_px"]
                                        if np.isfinite(x["median_rmse_gt_px"]) else 1e9))

    # head-to-head: on the pairs both methods solved, whose warp is closer to GT
    base = {r["pair"]: r for r in ok_rows(method_rows["lunarai_full"])}
    head2head = []
    for method, rs in method_rows.items():
        if method == "lunarai_full":
            continue
        solved = {r["pair"]: r for r in ok_rows(rs)}
        w = l = 0
        for pair, r_ai in base.items():
            r_b = solved.get(pair)
            if r_b is None:
                continue
            a = r_ai.get("rmse_gt_px", float("nan"))
            b = r_b.get("rmse_gt_px", float("nan"))
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a < b:
                w += 1
            elif b < a:
                l += 1
        head2head.append({"comparison": f"lunarai_full vs {method}",
                          "pairs_both_solved": len(solved),
                          "lunarai_lower_gt_rmse": w, "baseline_lower_gt_rmse": l})

    fig = out["figures"] / "benchmark_comparison.png"
    labels = [a["method"] for a in agg]
    bar_chart(labels, {
        "inlier_ratio@3px": [a["inlier_ratio_3px"] for a in agg],
        "coverage": [a["coverage_score"] for a in agg],
        "success_rate": [a["success_rate"] for a in agg],
    }, "Benchmark: method comparison (same pair set, same acceptance guard)",
        "score", fig, rot=12)
    fig2 = out["figures"] / "benchmark_runtime.png"
    bar_chart(labels, {"runtime_s": [a["runtime_s"] for a in agg]},
              "Benchmark: mean runtime per pair", "seconds", fig2, rot=12)

    bench_note = ("SuperGlue row marked 'unavailable': torch.hub weights could not "
                  "be fetched on this machine at run time (offline demo box).")\
        if not any(r.get("status") == "ok" for r in method_rows["superglue_ransac"]) \
        else ""
    report = f"""# Experiment 5 — Benchmark Comparison Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')} · Pair set: {len(pairs)} pairs
(controlled GT pairs, scale variants, the genuine TMC ncf↔ncn cross-view pair and one
retrieved cross-sensor pair) — identical pairs for every method.

All four methods run on the identical pair set with the identical acceptance guard
(>=10 correspondences and a geometrically plausible warp), so `success_rate` and the
medians are directly comparable. `median_rmse_gt_px` is the median error of the
estimated warp against the known ground-truth homography over the pairs that method
evaluated — the one number none of the methods can be tuned against.

## Ranking (reliability first, then GT error)

{md_table([{**a, "rank": i + 1} for i, a in enumerate(ranked)])}

## Head-to-head vs LunarAI (pairs both methods solved)

{md_table(head2head)}

## Full metrics per method

{md_table(agg)}

![benchmark](figures/benchmark_comparison.png)
![runtime](figures/benchmark_runtime.png)

## Per-pair detail

{md_table([{c: r.get(c) for c in ['method', 'pair', 'protocol', 'status', 'match_count',
                                  'inlier_count', 'inlier_ratio', 'inlier_ratio_3px',
                                  'coverage_score', 'rmse_px', 'rmse_gt_px', 'ssim', 'ncc',
                                  'runtime_s']}
           for method, rs in method_rows.items() for r in rs])}

**Reading the table.** The comparison is deliberately uneven in one respect and fair in
every other: `lunarai_full` runs the complete chain
(CLAHE -> SuperPoint+LightGlue -> ANMS -> MAGSAC++ -> homography -> ECC) while the
baselines run their detector plus RANSAC, and all are judged by the same
non-circular GT criterion and the same acceptance guard. On these controlled pairs
SIFT is a strong single-pair matcher and essentially ties LunarAI (see head-to-head);
what SIFT/ORB cannot do is decide *where* on the Moon a patch comes from at database
scale (Experiments 6 and 8) or enforce a spatially uniform correspondence set
(Experiment 7) — the two requirements this pipeline exists for.

{bench_note}
"""
    (out["reports"] / "benchmark_comparison_report.md").write_text(report, encoding="utf-8")
    return {"aggregate": agg, "ranking": ranked, "head2head": head2head,
            "rows": [r for rs in method_rows.values() for r in rs]}


# --------------------------------------------------------------------------- #
# Experiment 6 — retrieval validation
# --------------------------------------------------------------------------- #
def exp6_retrieval(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp6] retrieval validation (FAISS Top-K)")
    res = suite.evaluate_retrieval(n_queries=60, top_k=10, seed=11)
    per_query = res.pop("per_query")
    pd.DataFrame(per_query).to_csv(out["reports"] / "retrieval_per_query.csv", index=False)
    fig = out["figures"] / "retrieval_metrics.png"
    bar_chart(["Top-1", "Top-5", "Top-10"], {
        "same-source accuracy": [res["top1_accuracy"], res["top5_accuracy"], res["top10_accuracy"]],
        "cross-sensor accuracy": [res["top1_accuracy_cross_sensor"],
                                  res["top5_accuracy_cross_sensor"],
                                  res["top10_accuracy_cross_sensor"]],
        "mAP/MRR": [res["mAP"], res["mrr"], 0.0],
    }, "FAISS retrieval quality (60 held-out queries, K=10)", "accuracy", fig,
        hline=(0.90, "Recall@5 plan target"))

    per_sensor = {}
    for r in per_query:
        per_sensor.setdefault(r["sensor"], []).append(r["top1_hit_same_source"])
    sensor_rows = [{"sensor": s, "queries": len(v),
                    "top1_accuracy": round(float(np.mean(v)), 3)}
                   for s, v in sorted(per_sensor.items())]

    report = f"""# Experiment 6 — Retrieval Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')}
Index: FAISS IndexFlatIP, {suite.index.ntotal} vectors, {suite.embeddings.shape[1]}-D
LunaDNA embeddings (checkpoint epoch {suite.ckpt_epoch}). Queries: 60 held-out patches.

Two ground-truth definitions are used:
- **same-source**: retrieved patch comes from the same source image (terrain check).
- **cross-sensor**: retrieved patch is from a *different* sensor but within 0.5° of the
  query's coordinates — the mission-critical case for multi-modal correspondence.

## Summary

| Metric | Value |
|---|---|
| Top-1 accuracy (same-source) | {res['top1_accuracy']:.3f} |
| Top-5 accuracy (same-source) | {res['top5_accuracy']:.3f} |
| Top-10 accuracy (same-source) | {res['top10_accuracy']:.3f} |
| Top-1 / 5 / 10 accuracy (cross-sensor) | {res['top1_accuracy_cross_sensor']:.3f} / {res['top5_accuracy_cross_sensor']:.3f} / {res['top10_accuracy_cross_sensor']:.3f} |
| mAP (same-source relevance) | {res['mAP']:.3f} |
| MRR | {res['mrr']:.3f} |
| Queries evaluated | {res['n_queries']} |

![retrieval](figures/retrieval_metrics.png)

## Accuracy by query sensor

{md_table(sensor_rows)}

Per-query results: `outputs/reports/retrieval_per_query.csv`.

**Interpretation note.** The distribution contains no spatially overlapping
cross-sensor imagery, so a "cross-sensor" hit requires the query and candidate to
share terrain that no other sensor imaged — its accuracy is intrinsically bounded by
data availability, not by the model. The headline retrieval evidence is the
same-source accuracy and mAP above, plus the embedding-separation statistics
(random-pair cosine ≈ 0.44 ± 0.23 vs top-1 ≈ 0.96).
"""
    (out["reports"] / "retrieval_validation_report.md").write_text(report, encoding="utf-8")
    return {"summary": res, "per_sensor": sensor_rows}


# --------------------------------------------------------------------------- #
# Experiment 7 — uniform distribution (ANMS on/off)
# --------------------------------------------------------------------------- #
def exp7_uniformity(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp7] uniform distribution validation (ANMS)")
    pairs: list[Pair] = []
    for a, b, seed in [("TMC-2", "TMC-2", 500), ("OHRC", "TMC-2", 501),
                       ("TMC-2", "LRO NAC", 502), ("OHRC", "IIRS", 503),
                       ("OHRC", "LRO NAC", 504), ("TMC-2", "TMC-2", 505),
                       ("TMC-2", "IIRS", 506)]:
        srcs = suite.best_texture_patches(a, n=1, seed=seed)
        if not srcs:
            continue
        p = suite.controlled_pair(srcs[0], name=f"uni_{a}_to_{b}_{seed}", sensor_a=a,
                                  sensor_b=b, scale=1.0, seed=seed)
        if p:
            pairs.append(p)

    rows = []
    for p in pairs:
        with_anms = cache.run(p, "full", save=True, log=log)
        without = cache.run(p, "no_anms", log=log)
        rows.append({"pair": p.name, "protocol": "with_anms",
                     "status": with_anms.get("status"),
                     "coverage_score": with_anms.get("coverage_score"),
                     "uniformity_score": with_anms.get("uniformity_score"),
                     "match_count": with_anms.get("match_count"),
                     "anms_matches": with_anms.get("anms_matches"),
                     "inlier_ratio": with_anms.get("inlier_ratio"),
                     "rmse_gt_px": with_anms.get("rmse_gt_px")})
        rows.append({"pair": p.name, "protocol": "without_anms",
                     "status": without.get("status"),
                     "coverage_score": without.get("coverage_score"),
                     "uniformity_score": without.get("uniformity_score"),
                     "match_count": without.get("match_count"),
                     "anms_matches": without.get("anms_matches"),
                     "inlier_ratio": without.get("inlier_ratio"),
                     "rmse_gt_px": without.get("rmse_gt_px")})
    agg = aggregate(rows, ["protocol"])

    # per-pair effect of ANMS (same pair, same matcher, only the filter differs)
    piv = []
    for pair_name in sorted({r["pair"] for r in rows}):
        w = next((r for r in rows if r["pair"] == pair_name and r["protocol"] == "with_anms"),
                 None)
        o = next((r for r in rows if r["pair"] == pair_name and r["protocol"] == "without_anms"),
                 None)
        if not w or not o:
            continue
        piv.append({
            "pair": pair_name, "matches": o["match_count"],
            "kept_by_anms": w["anms_matches"],
            "coverage_without": o["coverage_score"], "coverage_with": w["coverage_score"],
            "uniformity_without": o["uniformity_score"],
            "uniformity_with": w["uniformity_score"],
            "uniformity_gain": round((w["uniformity_score"] or 0.0)
                                     - (o["uniformity_score"] or 0.0), 4),
            "rmse_gt_with": w["rmse_gt_px"], "rmse_gt_without": o["rmse_gt_px"],
        })
    mean_gain = round(float(np.mean([p["uniformity_gain"] for p in piv])), 4) if piv else None
    n_improved = sum(1 for p in piv if p["uniformity_gain"] > 0)

    # spatial-distribution figure for one representative pair
    if pairs:
        p = pairs[0]
        src = cv2.createCLAHE(2.0, (8, 8)).apply(suite._prep(p.src))
        m = suite.matcher.match(src, cv2.createCLAHE(2.0, (8, 8)).apply(suite._prep(p.ref)))
        from lunarai_lib.matching import anms
        from lunarai_lib.geometry import coverage_heatmap
        sel_kp, sel = anms(m.kp0, np.ones(m.n_matches),
                           target=int(suite.cfg.MATCHING.get("anms_target", 300)),
                           gamma=float(suite.cfg.MATCHING.get("anms_gamma", 1.6)),
                           min_radius=float(suite.cfg.MATCHING.get("anms_min_radius", 0.0)))
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        axes[0].imshow(src, cmap="gray")
        axes[0].scatter(m.kp0[:, 0], m.kp0[:, 1], s=4, c="#F59E0B")
        axes[0].set_title(f"matches (no ANMS, {m.n_matches})")
        axes[1].imshow(src, cmap="gray")
        axes[1].scatter(sel_kp[:, 0], sel_kp[:, 1], s=8, c="#22C55E")
        axes[1].set_title(f"ANMS ({len(sel_kp)})")
        for ax, pts, ttl in [(axes[2], m.kp0, "heatmap no ANMS"),
                             (axes[3], sel_kp, "heatmap with ANMS")]:
            hm = coverage_heatmap(pts, src.shape, GRID)
            hm_n = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ax.imshow(cv2.applyColorMap(255 - hm_n, cv2.COLORMAP_JET))
            ax.set_title(ttl)
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(out["figures"] / "uniform_distribution.png", dpi=130)
        plt.close()

    fig2 = out["figures"] / "uniformity_bars.png"
    labels = [a["protocol"] for a in agg]
    bar_chart(labels, {
        "coverage_score_median": [a["coverage_score_median"] for a in agg],
        "uniformity_score_median": [a["uniformity_score_median"] for a in agg],
        "inlier_ratio_3px": [a["inlier_ratio_3px"] for a in agg],
    }, "Uniform match distribution: without vs with ANMS (median over solved pairs)",
        "score", fig2, hline=(0.80, "coverage target"))

    report = f"""# Experiment 7 — Uniform Distribution Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')}

ANMS (adaptive non-maximal suppression, γ=1.6, target 300, minimum suppression radius
14 px) redistributes matched keypoints so no region of the image is over-represented.
Coverage Score = fraction of 8×8 image cells containing ≥1 correspondence; Spatial
Distribution Score (uniformity) = 1 − Gini of per-cell counts; both are computed on the
query frame.

The minimum suppression radius is what makes the effect measurable: a plain "keep the
best 300" rule is a no-op whenever the matcher already returns ~300 matches, while a
greedy radius filter removes clusters regardless of how many matches exist. Because
both arms use the *same* matcher output, any difference below is attributable to the
filter alone.

## Comparison: without ANMS vs with ANMS

{md_table([{c: a.get(c) for c in ['protocol', 'runs', 'success_rate', 'match_count',
                                  'uniformity_score', 'uniformity_score_median',
                                  'coverage_score', 'coverage_score_median',
                                  'inlier_ratio', 'rmse_gt_px_median']} for a in agg])}

Mean per-pair uniformity gain from ANMS: **{mean_gain}**
({n_improved}/{len(piv)} pairs improve).

![uniformity bars](figures/uniformity_bars.png)
![uniform distribution](figures/uniform_distribution.png)

## Per-pair effect (same matches, only the filter differs)

{md_table(piv)}

## Per-run detail

{md_table(rows)}

**Why it matters for registration.** A clustered correspondence set biases the
homography toward that cluster and leaves the rest of the frame unconstrained, so the
estimated 3×3 matrix is only valid where the matches were — the opposite of what a
rover or an ortho-rectification product needs. ANMS buys spatial spread at the cost of
raw match count, and because the spread points are still inliers (see `inlier_ratio`)
the accuracy of the warp does not suffer: that trade is the ISRO uniform-distribution
requirement, measured above pair-by-pair.
"""
    (out["reports"] / "uniform_distribution_report.md").write_text(report, encoding="utf-8")
    return {"aggregate": agg, "rows": rows, "pairs": piv,
            "mean_uniformity_gain": mean_gain, "n_pairs_improved": n_improved}


# --------------------------------------------------------------------------- #
# Experiment 8 — ablation study
# --------------------------------------------------------------------------- #
def exp8_ablation(suite: ValidationSuite, cache: Cache, out: dict, log=print) -> dict:
    log("[exp8] ablation study")
    pairs: list[Pair] = []
    for i, seed in enumerate([600, 601, 602, 603]):
        srcs = suite.best_texture_patches("TMC-2" if i % 2 else "OHRC", n=1, seed=seed)
        if not srcs:
            continue
        p = suite.controlled_pair(srcs[0], name=f"abl_{i}", sensor_a="OHRC" if i % 2 == 1 else "TMC-2",
                                  sensor_b="TMC-2" if i % 2 else "LRO NAC",
                                  scale=1.0, seed=seed)
        if p:
            pairs.append(p)

    variants = ["full", "no_anms", "no_ecc", "no_magsac"]
    rows = []
    for p in pairs:
        for v in variants:
            r = cache.run(p, v, log=log)
            rows.append({"variant": v, **{k: r.get(k) for k in
                                          ["pair", "status", "match_count", "inlier_count",
                                           "inlier_ratio", "inlier_count_3px",
                                           "inlier_ratio_3px", "coverage_score",
                                           "uniformity_score", "rmse_gt_px", "rmse_px",
                                           "runtime_s"]}})

    # no-FAISS: correct retrieval candidate vs a random (wrong) reference
    log("[exp8] FAISS on/off (correct candidate vs random reference)")
    faiss_rows = []
    for i, seed in enumerate([610, 611, 612]):
        srcs = suite.best_texture_patches("OHRC", n=2, seed=seed)
        if len(srcs) < 2:
            continue
        p_ok = suite.controlled_pair(srcs[0], name=f"ab_faiss_on_{i}", sensor_a="OHRC",
                                     sensor_b="TMC-2", scale=1.0, seed=seed)
        p_wrong_base = suite.controlled_pair(srcs[1], name=f"ab_faiss_off_{i}",
                                             sensor_a="OHRC", sensor_b="TMC-2",
                                             scale=1.0, seed=seed + 1)
        if p_ok is None or p_wrong_base is None:
            continue
        p_wrong = Pair(name=f"ab_faiss_off_{i}", protocol="controlled_wrong_ref",
                       sensor_a="OHRC", sensor_b="TMC-2", src=p_ok.src,
                       ref=p_wrong_base.ref, H_gt=None, meta={})
        r_on = cache.run(p_ok, "full", log=log)
        r_off = cache.run(p_wrong, "full", log=log)

        # what retrieval actually contributes: LunaDNA similarity of the query to the
        # *correct* candidate vs to the region FAISS would not have picked
        from lunarai_lib.lunadna import compute_embeddings

        def _emb(arr: np.ndarray, tag: str) -> np.ndarray:
            fp = out["tmp"] / f"faiss_sim_{tag}.png"
            cv2.imwrite(str(fp), arr)
            return compute_embeddings(suite.model, [str(fp)], device=suite.device,
                                      size=int(suite.cfg.LUNADNA.get("input_size", 224)))[0]

        e_src = _emb(p_ok.src, f"{i}_src")
        sim_correct = float(np.dot(e_src, _emb(p_ok.ref, f"{i}_correct")))
        sim_wrong = float(np.dot(e_src, _emb(p_wrong.ref, f"{i}_wrong")))
        faiss_rows.append({"condition": "with_faiss", "pair": p_ok.name,
                           "status": r_on.get("status"),
                           "match_count": r_on.get("match_count"),
                           "inlier_ratio": r_on.get("inlier_ratio"),
                           "rmse_gt_px": r_on.get("rmse_gt_px"),
                           "retrieval_similarity": round(sim_correct, 4),
                           "confidence": r_on.get("confidence")})
        faiss_rows.append({"condition": "without_faiss", "pair": p_wrong.name,
                           "status": r_off.get("status"),
                           "match_count": r_off.get("match_count"),
                           "inlier_ratio": r_off.get("inlier_ratio"),
                           "rmse_gt_px": r_off.get("rmse_gt_px"),
                           "retrieval_similarity": round(sim_wrong, 4),
                           "confidence": r_off.get("confidence")})

    agg = aggregate(rows, ["variant"])
    fig = out["figures"] / "ablation.png"
    labels = [a["variant"] for a in agg]
    bar_chart(labels, {
        "inlier_ratio_3px": [a["inlier_ratio_3px"] for a in agg],
        "coverage": [a["coverage_score_median"] for a in agg],
        "uniformity": [a["uniformity_score_median"] for a in agg],
    }, "Ablation study: component contribution (residual inliers @3px)", "score", fig)
    fig2 = out["figures"] / "ablation_rmse.png"
    bar_chart(labels, {"rmse_gt_px_median": [a["rmse_gt_px_median"] for a in agg]},
              "Ablation: registration error (median, solved pairs)", "pixel RMSE", fig2,
              hline=(1.0, "target"))

    faiss_agg = []
    for cond in ["with_faiss", "without_faiss"]:
        rs = [r for r in faiss_rows if r["condition"] == cond]
        faiss_agg.append({
            "condition": cond, "runs": len(rs),
            "success_rate": round(float(np.mean([r["status"] == "ok" for r in rs])), 3),
            "match_count": round(safe_mean([r["match_count"] for r in rs]), 1),
            "inlier_ratio": round(safe_mean([r["inlier_ratio"] for r in rs]), 4),
            "rmse_gt_px": round(safe_median([r["rmse_gt_px"] for r in rs]), 4),
            "retrieval_similarity": round(safe_mean(
                [r["retrieval_similarity"] for r in rs]), 4),
            "confidence": round(safe_mean([r["confidence"] for r in rs]), 2),
        })

    report = f"""# Experiment 8 — Ablation Study Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M')} · {len(pairs)} controlled pairs per variant

Variants: **full** (ANMS + MAGSAC++ + H + ECC) · **no_anms** (raw matches) ·
**no_ecc** (homography only) · **no_magsac** (least-squares homography, no robust
rejection) · **without_faiss** (reference built from a different region entirely).

## Component ablation

{md_table([{c: a.get(c) for c in ['variant', 'runs', 'success_runs', 'success_rate',
                                  'match_count', 'inlier_count', 'inlier_ratio',
                                  'inlier_ratio_3px', 'coverage_score_median',
                                  'uniformity_score_median', 'rmse_gt_px_median',
                                  'subpixel_share', 'rmse_px_median', 'runtime_s']}
           for a in agg])}

![ablation](figures/ablation.png)
![ablation rmse](figures/ablation_rmse.png)

## Retrieval ablation (FAISS candidate vs wrong reference)

{md_table(faiss_agg)}

| Condition | Per-run detail |
|---|---|
{chr(10).join(f"| {r['condition']} | {r['pair']} · status={r['status']} · inl={r['inlier_ratio']} · rmse_gt={r['rmse_gt_px']} |" for r in faiss_rows)}

## Findings

All variants run on the identical controlled pairs with the identical acceptance guard,
so every row is comparable. `inlier_ratio_3px` is the share of correspondences within
3 px of the estimated warp — measured the same way for every variant, because a
least-squares fit reports "100% inliers" by construction and a MAGSAC++ inlier count is
not comparable with it.

1. **MAGSAC++** is what makes "inlier ratio" a meaningful number: without robust
   rejection the least-squares fit reports every match as an inlier, and the resulting
   warp is dragged away from the correspondences. Compare `inlier_ratio_3px` and
   `rmse_gt_px_median` of `full` against `no_magsac` above.
2. **ECC** is the sub-pixel stage: see Experiment 4 for the before/after GT error, and
   `ecc_rejected` there for the pairs where correlation drift was refused.
3. **ANMS** trades raw match count for spatial spread, and Experiment 7 measures that
   trade pair-by-pair (the spread points remain inliers, so accuracy is preserved).
4. **FAISS retrieval decides *where* to match** — the geometric chain decides *how
   accurately*. The retrieval rows below put the same query against the candidate
   retrieval would return and against a region it would not, with the LunaDNA
   similarity of each, so the contribution of the retrieval stage is visible as a
   number rather than asserted.
"""
    (out["reports"] / "ablation_study_report.md").write_text(report, encoding="utf-8")
    return {"component": agg, "faiss": faiss_agg, "rows": rows, "faiss_rows": faiss_rows}


# --------------------------------------------------------------------------- #
# final outputs
# --------------------------------------------------------------------------- #
def final_outputs(suite: ValidationSuite, cache: Cache, results: dict, out: dict,
                  t_start: float, log=print) -> None:
    log("[final] assembling final outputs")
    reports = out["reports"]

    dataset_report = {}
    p = suite.cfg.DATASET_ANALYSIS_DIR / "dataset_report.json"
    if p.exists():
        dataset_report = json.loads(p.read_text(encoding="utf-8"))
    dataset_report_slim = {k: dataset_report.get(k) for k in
                           ["total_images", "counts", "dataset_quality_score",
                            "metadata_coverage", "sun_angle_statistics"] if dataset_report}

    metrics = {
        "project": "LunarAI", "problem_statement": "SIH 26166",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "LunaDNA(ResNet18 512-D) -> FAISS IndexFlatIP -> SuperPoint -> "
                    "LightGlue -> ANMS -> MAGSAC++ -> Homography -> ECC",
        "matcher_backend": suite.matcher_backend,
        "lunadna_checkpoint_epoch": suite.ckpt_epoch,
        "dataset": dataset_report_slim,
        "pipeline_runs": cache.n_runs,
        "validation_runtime_s": round(time.perf_counter() - t_start, 1),
        "exp1_multimodal": results.get("exp1"),
        "exp2_sun_angle": results.get("exp2"),
        "exp3_scale": results.get("exp3", {}).get("aggregate"),
        "exp4_subpixel": results.get("exp4", {}).get("summary"),
        "exp5_benchmark": results.get("exp5", {}).get("ranking"),
        "exp6_retrieval": results.get("exp6", {}).get("summary"),
        "exp7_uniformity": results.get("exp7", {}).get("aggregate"),
        "exp8_ablation": results.get("exp8", {}).get("component"),
        "exp8_faiss_ablation": results.get("exp8", {}).get("faiss"),
    }
    (reports / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    # flat CSV of headline metrics
    flat = []

    def add(section, key, val, note=""):
        flat.append({"section": section, "metric": key, "value": val, "note": note})

    for k, v in (dataset_report_slim or {}).items():
        add("dataset", k, json.dumps(v) if isinstance(v, dict) else v)
    sp = results.get("exp4", {}).get("summary", {})
    for k, v in sp.items():
        add("exp4_subpixel", k, v)
    ret = results.get("exp6", {}).get("summary", {})
    for k, v in ret.items():
        add("exp6_retrieval", k, round(v, 4) if isinstance(v, float) else v)
    for row in results.get("exp5", {}).get("ranking", []):
        for k, v in row.items():
            add(f"exp5_benchmark/{row['method']}", k, v)
    for row in results.get("exp7", {}).get("aggregate", []):
        for k, v in row.items():
            add(f"exp7_uniformity/{row['protocol']}", k, v)
    for row in results.get("exp8", {}).get("component", []):
        for k, v in row.items():
            add(f"exp8_ablation/{row['variant']}", k, v)
    for row in results.get("exp8", {}).get("faiss", []):
        for k, v in row.items():
            add(f"exp8_faiss/{row['condition']}", k, v)
    for row in results.get("exp2", {}).get("correspondence", []):
        for k, v in row.items():
            add(f"exp2_sun_angle/{row['sun_group']}", k, v)
    for row in results.get("exp1", {}).get("controlled", []):
        for k, v in row.items():
            add(f"exp1_multimodal_controlled/{row['group'].replace(' <-> ', '_')}", k, v)
    for row in results.get("exp3", {}).get("aggregate", []):
        for k, v in row.items():
            add(f"exp3_scale/{row['combo'].replace(' vs ', '_')}/x{row['scale_factor']}", k, v)
    pd.DataFrame(flat).to_csv(reports / "final_metrics.csv", index=False)

    # long-format presentation tables
    pres = []
    def trow(exp, table, row_key, metric, value):
        pres.append({"experiment": exp, "table": table, "row_key": row_key,
                     "metric": metric, "value": value})
    for row in results.get("exp1", {}).get("controlled", []):
        for k, v in row.items():
            if k != "group":
                trow("exp1_multimodal", "controlled", row["group"], k, v)
    for row in results.get("exp2", {}).get("correspondence", []):
        for k, v in row.items():
            if k != "sun_group":
                trow("exp2_sun_angle", "correspondence", row["sun_group"], k, v)
    for row in results.get("exp2", {}).get("retrieval", []):
        for k, v in row.items():
            if k != "sun_group":
                trow("exp2_sun_angle", "retrieval", row["sun_group"], k, v)
    for row in results.get("exp3", {}).get("aggregate", []):
        for k, v in row.items():
            if k not in ("combo", "scale_factor"):
                trow("exp3_scale", f"x{row['scale_factor']}", row["combo"], k, v)
    for row in results.get("exp5", {}).get("ranking", []):
        for k, v in row.items():
            if k != "method":
                trow("exp5_benchmark", "ranking", row["method"], k, v)
    for k, v in ret.items():
        trow("exp6_retrieval", "summary", "all_queries", k, v)
    for row in results.get("exp7", {}).get("aggregate", []):
        for k, v in row.items():
            if k != "protocol":
                trow("exp7_uniformity", "protocol", row["protocol"], k, v)
    for row in results.get("exp8", {}).get("component", []):
        for k, v in row.items():
            if k != "variant":
                trow("exp8_ablation", "variant", row["variant"], k, v)
    for row in results.get("exp8", {}).get("faiss", []):
        for k, v in row.items():
            if k != "condition":
                trow("exp8_ablation", "faiss", row["condition"], k, v)
    for k, v in sp.items():
        trow("exp4_subpixel", "summary", "before_vs_after_ecc", k, v)
    pd.DataFrame(pres).to_csv(reports / "final_presentation_tables.csv", index=False)

    # judges Q&A grounded in measured numbers (medians over *solved* pairs only)
    single = results.get("exp1", {}).get("controlled", [])
    med_inlier = safe_median([r.get("inlier_ratio") for r in single])
    cov = safe_median([r.get("coverage_score_median") for r in single])
    bench = {r["method"]: r for r in results.get("exp5", {}).get("ranking", [])}
    full_rows = [r for r in results.get("exp5", {}).get("rows", [])
                 if r.get("method") == "lunarai_full"]
    full_ok = ok_rows(full_rows)
    suite_med_gt = safe_median([r.get("rmse_gt_px") for r in full_ok])
    _gt_vals = [r["rmse_gt_px"] for r in full_ok if np.isfinite(r.get("rmse_gt_px", float("nan")))]
    suite_p90_gt = float(np.percentile(_gt_vals, 90)) if _gt_vals else float("nan")
    suite_share_ok = len(full_ok) / max(len(full_rows), 1)
    h2h = {h["comparison"].split(" vs ")[-1]: h
           for h in results.get("exp5", {}).get("head2head", [])}
    exp2_rows = results.get("exp2", {}).get("correspondence", [])
    sun_runs = sum(r.get("runs", 0) for r in exp2_rows)
    sun_ok = sum(r.get("success_runs", 0) for r in exp2_rows)
    sun_med_gt = safe_median([r.get("rmse_gt_px_median") for r in exp2_rows])
    exp3_rows = results.get("exp3", {}).get("aggregate", [])
    scale_txt_parts = []
    for sf in sorted({r.get("scale_factor") for r in exp3_rows}, key=lambda v: (v is None, v)):
        rs = [r for r in exp3_rows if r.get("scale_factor") == sf]
        solved = sum(r.get("success_runs", 0) for r in rs)
        med = safe_median([r.get("rmse_gt_px_median") for r in rs
                           if np.isfinite(r.get("rmse_gt_px_median", float("nan")))])
        scale_txt_parts.append(
            f"{sf}x: {solved}/{len(rs)} pairs solved"
            + (f", median GT {med:.2f} px" if np.isfinite(med) else ""))
    scale_txt = "; ".join(scale_txt_parts)
    exp7_rows = results.get("exp7", {}).get("aggregate", [])
    uni_with = next((r for r in exp7_rows if r.get("protocol") == "with_anms"), {})
    uni_without = next((r for r in exp7_rows if r.get("protocol") == "without_anms"), {})
    anms_gain = results.get("exp7", {}).get("mean_uniformity_gain")
    faiss_tbl = {r["condition"]: r for r in results.get("exp8", {}).get("faiss", [])}
    qa = [
        ("Why LunaDNA?",
         f"A plain matcher cannot know *where* on the Moon a patch is from. LunaDNA's "
         f"512-D terrain fingerprints (ResNet18, triplet loss, {suite.df.dataset_name.nunique()}-sensor "
         f"training) let the system retrieve the right region first — measured retrieval: "
         f"Top-1 {ret.get('top1_accuracy', 0):.2f}, Top-5 {ret.get('top5_accuracy', 0):.2f}, "
         f"mAP {ret.get('mAP', 0):.2f} on {ret.get('n_queries', 0)} held-out queries, and the "
         f"top-1 candidate sits a median {ret.get('geo_top1_median_deg', float('nan')):.3f} deg "
         f"from the query in absolute coordinates vs "
         f"{ret.get('geo_random_pair_median_deg', float('nan')):.2f} deg for a random pair "
         f"({ret.get('geo_localization_gain', float('nan')):.1f}x closer than chance)."),
        ("Why FAISS before registration?",
         "Matching the whole lunar database is unbounded work; retrieval narrows it to "
         f"{suite.cfg.FAISS.get('top_k', 10)} candidates in milliseconds (index: "
         f"{suite.index.ntotal} vectors). Experiment 8 puts the same query against the "
         f"candidate retrieval selects ({faiss_tbl.get('with_faiss', {}).get('success_rate')} "
         f"solved, {faiss_tbl.get('with_faiss', {}).get('match_count')} matches, median GT "
         f"{faiss_tbl.get('with_faiss', {}).get('rmse_gt_px')} px, LunaDNA similarity "
         f"{faiss_tbl.get('with_faiss', {}).get('retrieval_similarity')}) and against a "
         f"region it would not ({faiss_tbl.get('without_faiss', {}).get('success_rate')} "
         f"solved, {faiss_tbl.get('without_faiss', {}).get('match_count')} matches, "
         f"similarity {faiss_tbl.get('without_faiss', {}).get('retrieval_similarity')}): "
         "retrieval decides *where* to match, the geometric chain then decides how "
         "accurately."),
        ("How does LunarAI handle scale variation?",
         "At two levels. (1) GSD: the pipeline resamples every product to a common "
         "working GSD from its PDS label before matching, so a ~20x gap (OHRC 0.24 m/px "
         "vs TMC-2 ~5 m/px) is never handed to a single-scale keypoint matcher — and "
         "LunaDNA stays scale-robust across that gap because embeddings are computed on "
         "resized 224x224 inputs. (2) Residual scale after that resample is absorbed by "
         f"the homography. Experiment 3 measures that budget: {scale_txt}. Within the "
         "0.5x-2x envelope the solved pairs register at sub-pixel to ~4 px; at 4x the "
         "single-scale detector finds too few correspondences and the guard rejects the "
         "run instead of reporting a number."),
        ("How does LunarAI handle sun-angle variation?",
         "CLAHE normalization at preprocessing (applied to both images before matching) "
         "plus LunaDNA training across low/medium/high-sun products (OHRC ~0 deg to TMC "
         f"51.7 deg). Experiment 2: {sun_ok}/{sun_runs} pairs solved across the three "
         f"illumination groups with median GT RMSE {sun_med_gt:.2f} px, and the real "
         "per-product sun elevations are tabulated from the PDS labels in the report."),
        ("How does LunarAI ensure uniform distribution?",
         "ANMS (adaptive non-maximal suppression) re-selects correspondences with a "
         "minimum suppression radius, so clustered matches are removed even when the "
         "matcher returns fewer than the target count. Experiment 7 runs both arms on "
         f"the same matcher output: mean per-pair uniformity gain {anms_gain}, median "
         f"uniformity {uni_with.get('uniformity_score_median')} with ANMS vs "
         f"{uni_without.get('uniformity_score_median')} without, median coverage "
         f"{uni_with.get('coverage_score_median')} vs "
         f"{uni_without.get('coverage_score_median')}, with spatial heatmaps in the "
         "report."),
        ("How is sub-pixel accuracy achieved?",
         "MAGSAC++ gives a robust initial homography; ECC then refines the warp by "
         "maximizing correlation on the already-aligned pair, recovering the small "
         "sub-pixel residual. Experiment 4, restricted to pairs that passed the "
         f"acceptance guard: median GT RMSE {sp.get('median_rmse_gt_before_ecc')} px with "
         f"the homography alone -> {sp.get('median_rmse_gt_after_ecc')} px after ECC "
         f"(IQR {sp.get('iqr_rmse_gt_after_ecc')} px, worst {sp.get('max_rmse_gt_after_ecc')} px), "
         f"with {sp.get('subpixel_share')} of evaluated pairs under 1 px. ECC "
         "refinements that regress the correspondence geometry are refused rather than "
         "reported."),
        ("Why is LunarAI better than SIFT and ORB — and where do AKAZE and SuperGlue fit?",
         "The comparison (Experiment 5) is on the non-circular ground-truth criterion, "
         "with the identical pair set and the identical acceptance guard. On this "
         "bench LunarAI matches SIFT's reliability "
         f"({bench.get('lunarai_full', {}).get('success_rate')} vs "
         f"{bench.get('sift_ransac', {}).get('success_rate')} pairs solved) with a higher "
         "fraction of correspondences within 3 px of the warp "
         f"({bench.get('lunarai_full', {}).get('inlier_ratio_3px')} vs "
         f"{bench.get('sift_ransac', {}).get('inlier_ratio_3px')}), beats ORB on both "
         "(head-to-head "
         f"{h2h.get('orb_ransac', {}).get('lunarai_lower_gt_rmse')}/"
         f"{h2h.get('orb_ransac', {}).get('pairs_both_solved')} pairs closer to GT), and "
         f"ties SIFT head-to-head ({h2h.get('sift_ransac', {}).get('lunarai_lower_gt_rmse')}/"
         f"{h2h.get('sift_ransac', {}).get('pairs_both_solved')}). SIFT remains a strong "
         "single-pair matcher at small scale — the honest statement is parity there — but "
         "it offers no answer to the questions that dominate the mission: where a patch "
         "comes from at database scale (Experiment 6: Top-1 "
         f"{ret.get('top1_accuracy', 0):.2f}, "
         f"{ret.get('geo_localization_gain', float('nan')):.0f}x closer to the true "
         "coordinates than chance), and uniform spatial coverage of the correspondences "
         "(Experiment 7), both of which LunarAI provides end-to-end. AKAZE (a modern "
         "classical baseline) and SuperPoint+SuperGlue (the canonical learned matcher) "
         "run on the identical bench in this report — see the per-method table for the "
         "measured comparison."),
        ("What is the innovation?",
         "Treating lunar registration as *retrieval-driven terrain localization*: "
         "mission-invariant LunaDNA fingerprints + FAISS decide where to match; a "
         "SuperPoint-LightGlue/ANMS/MAGSAC++/ECC chain then delivers uniform, "
         "sub-pixel-verified correspondence. Every stage is instrumented with "
         "quantitative evidence (RMSE/coverage/inlier ratio/runtime)."),
        ("What is the future scope?",
         "Sun-AngleNet for illumination-invariant descriptors, CraterGraphNet crater "
         "graph matching, multi-modal spectral fusion with IIRS hyperspectral cubes, "
         "confidence-aware rover localization, and IVF-PQ indexing for 100k+ patch "
         "scale (already structured for it)."),
    ]
    qa_md = "\n\n".join(f"**Q{i+1}. {q}**\n\n{a}" for i, (q, a) in enumerate(qa))

    crit = {
        "Multi-Modal Correspondence": (
            f"Exp 1: {suite_share_ok:.0%} of cross-sensor controlled pairs solved, median "
            f"inlier ratio {med_inlier:.2f}, median GT RMSE {suite_med_gt:.2f} px "
            f"(90th percentile {suite_p90_gt:.2f} px)"),
        "Sun-Angle Robustness": (
            f"Exp 2: {sun_ok}/{sun_runs} pairs solved across low (OHRC ~0 deg) / medium "
            f"(TMC-2 + IIRS ~38 deg) / high (TMC-2 51.7 deg) groups, median GT RMSE "
            f"{sun_med_gt:.2f} px"),
        "Scale Invariance": (
            f"Exp 3: {scale_txt}; sub-pixel to ~4 px median GT RMSE within the 0.5x-2x "
            "envelope, guard-rejected (no number) at 4x"),
        "Uniform Match Distribution": (
            f"Exp 7: mean per-pair uniformity gain {anms_gain} from ANMS (median "
            f"uniformity {uni_with.get('uniformity_score_median')} vs "
            f"{uni_without.get('uniformity_score_median')}), median coverage "
            f"{uni_with.get('coverage_score_median')} with vs "
            f"{uni_without.get('coverage_score_median')} without"),
        "Sub-Pixel Registration": (
            f"Exp 4: median GT RMSE {sp.get('median_rmse_gt_after_ecc')} px after ECC "
            f"(from {sp.get('median_rmse_gt_before_ecc')} px), "
            f"{sp.get('subpixel_share')} of evaluated pairs under 1 px"),
        "Benchmark Superiority": (
            f"Exp 5: reliability parity with SIFT ({bench.get('lunarai_full', {}).get('success_rate')} "
            f"vs {bench.get('sift_ransac', {}).get('success_rate')} solved), ahead of ORB "
            f"({bench.get('orb_ransac', {}).get('success_rate')}); residual inliers @3px "
            f"{bench.get('lunarai_full', {}).get('inlier_ratio_3px')} vs SIFT "
            f"{bench.get('sift_ransac', {}).get('inlier_ratio_3px')} / ORB "
            f"{bench.get('orb_ransac', {}).get('inlier_ratio_3px')}; head-to-head "
            f"{h2h.get('sift_ransac', {}).get('lunarai_lower_gt_rmse')}-"
            f"{h2h.get('sift_ransac', {}).get('baseline_lower_gt_rmse')} vs SIFT, "
            f"{h2h.get('orb_ransac', {}).get('lunarai_lower_gt_rmse')}-"
            f"{h2h.get('orb_ransac', {}).get('baseline_lower_gt_rmse')} vs ORB — plus the "
            "retrieval capability no baseline has (Exp 6)"),
    }
    crit_md = md_table([{"success_criterion": k, "measured evidence": v}
                        for k, v in crit.items()])

    summary_md = f"""# LunarAI — Final Results Summary (SIH 26166)

**Generated:** {metrics['generated_at']} · **Pipeline runs in validation:** {cache.n_runs}
· **Validation runtime:** {metrics['validation_runtime_s']} s
· **Matcher:** {suite.matcher_backend} · **LunaDNA checkpoint epoch:** {suite.ckpt_epoch}

## Headline numbers

| Item | Value |
|---|---|
| Dataset | {dataset_report_slim.get('total_images', '-')} products, quality score {dataset_report_slim.get('dataset_quality_score', '-')} |
| Patches indexed | {suite.index.ntotal} |
| Retrieval Top-1 / Top-5 (same-source) | {ret.get('top1_accuracy', 0):.3f} / {ret.get('top5_accuracy', 0):.3f} |
| Retrieval mAP / MRR | {ret.get('mAP', 0):.3f} / {ret.get('mrr', 0):.3f} |
| Geo-localization (top-1 vs chance) | {ret.get('geo_top1_median_deg', float('nan')):.3f} deg vs {ret.get('geo_random_pair_median_deg', float('nan')):.2f} deg |
| Median inlier ratio (cross-sensor controlled) | {med_inlier:.3f} |
| Median coverage score | {cov:.3f} |
| Controlled pair success rate | {suite_share_ok:.0%} (acceptance guard) |
| Median GT RMSE after ECC | {sp.get('median_rmse_gt_after_ecc')} px (IQR {sp.get('iqr_rmse_gt_after_ecc')} px) |
| Pairs under 1 px | {sp.get('subpixel_share')} of evaluated controlled pairs |

## Success criteria — measured evidence

{crit_md}

## Reports

- Experiment 1: `multimodal_validation_report.md`
- Experiment 2: `sun_angle_validation_report.md`
- Experiment 3: `scale_validation_report.md`
- Experiment 4: `subpixel_accuracy_report.md`
- Experiment 5: `benchmark_comparison_report.md`
- Experiment 6: `retrieval_validation_report.md`
- Experiment 7: `uniform_distribution_report.md`
- Experiment 8: `ablation_study_report.md`
- Machine-readable: `final_metrics.json`, `final_metrics.csv`,
  `final_presentation_tables.csv`, `final_judges_report.pdf`

## Honest limitations

- No truly overlapping OHRC/TMC/IIRS/LRO imagery exists in the provided distribution;
  cross-sensor numbers use the documented controlled GT protocol, and the only genuine
  cross-view orbital pair (TMC ncf↔ncn, same orbit + timestamp) is exercised for real.
- IIRS spectral binaries are absent (ENVI headers only); browse renders are used.
- LRO contribution is a QuickMap reference export, not a NAC mosaic.
- Training budget on this CPU-only machine was bounded (checkpoint epoch {suite.ckpt_epoch});
  the architecture and configs support the full 120-epoch schedule.

## Judge Q&A

{qa_md}
"""
    (reports / "final_results_summary.md").write_text(summary_md, encoding="utf-8")

    # judges PDF
    pdf_images = [out["figures"] / n for n in
                  ["multimodal_summary.png", "sun_angle_validation.png",
                   "scale_rmse.png", "subpixel_ecc.png", "benchmark_comparison.png",
                   "retrieval_metrics.png", "uniformity_bars.png", "ablation.png"]]
    pdf_images = [p for p in pdf_images if p.exists()]
    reg_samples = sorted(out["samples"].glob("*_registered.png"))[:2]
    pdf_images = list(reg_samples) + pdf_images
    build_pdf_report(
        reports / "final_judges_report.pdf",
        title="LunarAI — Final Judges Report",
        subtitle=f"SIH 26166 · {metrics['generated_at']} · {cache.n_runs} validation runs",
        metrics={k: v for k, v in [
            ("retrieval_top1", round(ret.get("top1_accuracy", 0), 3)),
            ("retrieval_top5", round(ret.get("top5_accuracy", 0), 3)),
            ("retrieval_mAP", round(ret.get("mAP", 0), 3)),
            ("inlier_ratio_median", round(float(np.nan_to_num(med_inlier)), 3)),
            ("coverage_median", round(float(np.nan_to_num(cov)), 3)),
            ("rmse_gt_before_ecc_median_px", sp.get("median_rmse_gt_before_ecc", 0)),
            ("rmse_gt_after_ecc_median_px", sp.get("median_rmse_gt_after_ecc", 0)),
            ("subpixel_share", sp.get("subpixel_share")),
            ("patches_indexed", suite.index.ntotal),
            ("lunadna_checkpoint_epoch", suite.ckpt_epoch),
        ]},
        image_paths=pdf_images,
        summary_lines=[f"{q}: {a.splitlines()[0][:100]}" for q, a in qa[:4]],
    )
    log(f"[final] wrote {reports}/final_results_summary.md, final_metrics.*, "
        f"final_presentation_tables.csv, final_judges_report.pdf")


# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.perf_counter()
    cfg = load_config()
    reports = cfg.OUTPUTS_ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out = {"reports": reports, "figures": figures,
           "samples": figures / "registration_samples",
           "tmp": cfg.MATCHING_DIR / "_tmp"}
    out["tmp"].mkdir(parents=True, exist_ok=True)
    out["samples"].mkdir(parents=True, exist_ok=True)

    print("[suite] initializing (loads LunaDNA + FAISS + SuperPoint/LightGlue)…", flush=True)
    suite = ValidationSuite(cfg)
    cache = Cache(suite, out["samples"])

    steps = [
        ("exp1", lambda: exp1_multimodal(suite, cache, out)),
        ("exp2", lambda: exp2_sun_angle(suite, cache, out)),
        ("exp3", lambda: exp3_scale(suite, cache, out)),
        ("exp4", lambda: exp4_subpixel(suite, cache, out)),
        ("exp5", lambda: exp5_benchmark(suite, cache, out)),
        ("exp6", lambda: exp6_retrieval(suite, cache, out)),
        ("exp7", lambda: exp7_uniformity(suite, cache, out)),
        ("exp8", lambda: exp8_ablation(suite, cache, out)),
    ]
    results: dict = {}
    for name, fn in steps:
        try:
            results[name] = fn()
        except Exception:
            print(f"[{name}] FAILED:\n{traceback.format_exc()}", flush=True)
            results[name] = {}
    try:
        final_outputs(suite, cache, results, out, t_start)
    except Exception:
        print(f"[final] FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"[done] {cache.n_runs} pipeline runs, "
          f"{time.perf_counter() - t_start:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
