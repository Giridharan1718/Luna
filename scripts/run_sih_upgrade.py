"""SIH Grand Finale upgrade driver — runs every new Phase 1-5 experiment and
writes outputs/sih_upgrade/ + sih_upgrade_report.md.

Sections (each independently try/except-guarded; the driver never dies midway):
  1. splits        70/15/15 geo split export + leakage check
  2. elevation     ElevationProfile track-corridor mapping onto patches
  3. sunangle      SunAngle.csv pass stats + per-patch sun-bin enrichment
  4. pca           no-PCA vs 256 vs 128 (fit on train split, eval on test)
  5. tmc2_iirs     TMC-2 <-> IIRS controlled + retrieval experiments
  6. sun_bins      full 5-bin table: measured bins + SYNTHETIC-ILLUMINATION
                   protocol for bins with no catalog imagery (labelled)
  7. scale         0.5x-8x with the multiscale-retry ladder + retrieval@scale
  8. viewpoint     real TMC ncf<->ncn pair with the retry ladder (bounded H)
  9. benchmarks    AKAZE + SuperPoint+SuperGlue on the focused pair set
 10. isro_matrix   requirement-by-requirement status with evidence pointers

Run:  python scripts/run_sih_upgrade.py
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
from lunarai_lib.elevation import (elevation_band, load_elevation_track, map_elevations,
                                   track_stats)
from lunarai_lib.models import AkazeRunner, SuperglueRunner
from lunarai_lib.pca import pca_ablation
from lunarai_lib.splits import export_splits
from lunarai_lib.sunangle import SUN_BINS_5, load_sunangle_stats, sun_bin_label
from lunarai_lib.validation import GRID, Pair, ValidationSuite, load_gray

OUT = ROOT / "outputs" / "sih_upgrade"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[sih] {msg}", flush=True)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_none_"
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(
            ("" if r.get(c) is None else
             (f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r.get(c))))
            for c in cols) + " |")
    return "\n".join([head, sep] + body)


# --------------------------------------------------------------------------- #
def sec_splits(cfg) -> dict:
    df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
    meta = export_splits(df, cfg.OUTPUTS_ROOT / "splits", seed=42)
    return {"meta": meta,
            "table": [{"split": k, "patches": v}
                      for k, v in meta["patch_counts"].items()]}


def sec_elevation(cfg) -> dict:
    pts = load_elevation_track(cfg.DATA_ROOT / "ElevationProfile.csv")
    stats = track_stats(pts)
    df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv")
    elevs, dists = map_elevations(df, pts, corridor_deg=1.0)
    df["elevation_m"] = np.round(elevs, 1)
    df["track_distance_deg"] = np.round(dists, 4)
    df["elevation_band"] = [elevation_band(e) for e in elevs]
    covered = int(np.isfinite(elevs).sum())
    df.to_csv(OUT / "patch_index_with_elevation.csv", index=False)
    return {"stats": stats, "covered": covered, "total": len(df),
            "coverage": round(covered / max(len(df), 1), 4)}


def sec_sunangle(cfg) -> dict:
    stats = load_sunangle_stats(cfg.DATA_ROOT / "SunAngle.csv")
    df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv")
    df["sun_bin"] = [sun_bin_label(v) for v in df.sun_angle]
    counts = df.sun_bin.value_counts().to_dict()
    df.to_csv(OUT / "patch_index_with_sunbin.csv", index=False)
    return {"stats": stats, "bins": counts}


def _load_model_index(suite: ValidationSuite):
    return suite.model, suite.embeddings, suite.mapping, suite.df


def sec_pca(suite: ValidationSuite, cfg) -> dict:
    from lunarai_lib.lunadna import compute_embeddings
    from lunarai_lib.splits import assign_geo_splits
    df = assign_geo_splits(suite.df.copy(), seed=42)
    train_df = df[df.split == "train"]
    test_df = df[df.split == "test"]
    model = suite.model
    train_embs = compute_embeddings(model, train_df.patch_path.tolist(),
                                    device=suite.device, size=224)
    test_embs = compute_embeddings(model, test_df.patch_path.tolist(),
                                   device=suite.device, size=224)
    sensors = test_df.dataset_name.to_numpy()
    coords = test_df[["latitude", "longitude"]].to_numpy(dtype=float)
    rows, _fitted = pca_ablation(train_embs, test_embs, sensors, coords,
                                 dims=(512, 256, 128),
                                 out_dir=cfg.DATABASE_DIR)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "pca_ablation.csv", index=False)
    return {"rows": rows}


def sec_tmc2_iirs(suite: ValidationSuite, cache) -> dict:
    rows = []
    srcs = suite.best_texture_patches("TMC-2", n=1, seed=17)
    if srcs:
        p = suite.controlled_pair(srcs[0], name="sih_TMC2_to_IIRS", sensor_a="TMC-2",
                                  sensor_b="IIRS", scale=1.0, seed=900)
        if p is not None:
            r = cache.run(p, "full")
            rows.append({"pair": p.name, "protocol": p.protocol, "status": r.get("status"),
                         "match_count": r.get("match_count"),
                         "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                         "coverage_score": r.get("coverage_score"),
                         "rmse_gt_px": r.get("rmse_gt_px"),
                         "confidence": r.get("confidence"),
                         "runtime_s": r.get("runtime_s")})
        r2 = cache.run(p, "no_anms") if p is not None else {}
    ret = suite.retrieved_pair("TMC-2", "IIRS", seed=33)
    if ret is not None:
        r = suite.run_pipeline(ret, variant="full")
        rows.append({"pair": ret.name, "protocol": ret.protocol, "status": r.get("status"),
                     "match_count": r.get("match_count"),
                     "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                     "coverage_score": r.get("coverage_score"),
                     "rmse_gt_px": float("nan"),
                     "confidence": r.get("confidence"),
                     "runtime_s": r.get("runtime_s")})
    return {"rows": rows}


def sec_sun_bins(suite: ValidationSuite, cache, ext) -> dict:
    """Measured bins from real catalog patches; empty bins via the documented
    SYNTHETIC-ILLUMINATION protocol (labelled 'synthetic' in the table)."""
    from lunarai_lib.validation_ext import sun_angle_bin_catalog
    cat = sun_angle_bin_catalog(ext.base.df)
    rows = []
    for name, _lo, _hi in SUN_BINS_5:
        sub = cat[cat.sun_bin == name] if "sun_bin" in cat.columns else cat.iloc[:0]
        if len(sub) >= 6:
            # measured protocol: real patches from this bin
            src_path = sub.iloc[int(np.argmax(
                [load_gray(p, 256).std() if load_gray(p, 256) is not None else 0
                 for p in sub.patch_path]))].patch_path
            p = suite.controlled_pair(src_path, name=f"sih_sun_{name}", sensor_a="TMC-2",
                                      sensor_b="TMC-2", scale=1.0, seed=41)
            if p is not None:
                r = cache.run(p, "full")
                rows.append({"sun_bin": name, "protocol": "measured",
                             "catalog_patches": int(len(sub)),
                             "status": r.get("status"),
                             "rmse_gt_px": r.get("rmse_gt_px"),
                             "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                             "coverage_score": r.get("coverage_score")})
                continue
        # synthetic-illumination protocol (empty bins): TMC-2 mid-latitude terrain
        # resampled to the bin's radiometric regime (mean/std + gamma), via the
        # SAME illumination model the suite uses for low/medium/high groups.
        srcs = suite.best_texture_patches("TMC-2", n=1, seed=55)
        if not srcs:
            continue
        p = suite.controlled_pair(srcs[0], name=f"sih_sun_syn_{name}", sensor_a="TMC-2",
                                  sensor_b="TMC-2", scale=1.0, seed=42)
        if p is None:
            continue
        gamma = {"0-10": 0.85, "10-20": 0.92, "20-40": 1.0, "40-60": 1.08, "60+": 1.15}.get(name, 1.0)
        target_mean = {"0-10": 70.0, "10-20": 95.0, "20-40": 120.0,
                       "40-60": 150.0, "60+": 165.0}.get(name, 120.0)
        target_std = {"0-10": 28.0, "10-20": 38.0, "20-40": 48.0,
                      "40-60": 52.0, "60+": 55.0}.get(name, 48.0)
        ref = p.ref.astype(np.float32) / 255.0
        ref = np.clip((ref ** gamma) * 255.0, 0, 255)
        ref = np.clip((ref - ref.mean()) * (target_std / max(ref.std(), 1e-6)) + target_mean,
                      0, 255).astype(np.uint8)
        p2 = Pair(name=f"sih_sun_syn_{name}", protocol="controlled_synthetic_illum",
                  sensor_a="TMC-2", sensor_b="TMC-2", src=p.src, ref=ref,
                  H_gt=p.H_gt, meta={**p.meta, "sun_bin": name, "synthetic": True})
        r = cache.run(p2, "full")
        rows.append({"sun_bin": name, "protocol": "synthetic_illumination",
                     "catalog_patches": int(len(sub)),
                     "status": r.get("status"),
                     "rmse_gt_px": r.get("rmse_gt_px"),
                     "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                     "coverage_score": r.get("coverage_score")})
    return {"rows": rows}


def retrieval_at_scale(suite: ValidationSuite) -> list[dict]:
    """LunaDNA retrieval Top-1 under reference scale change (0.5x-8x)."""
    srcs = suite.best_texture_patches("TMC-2", n=2, seed=9)
    rows = []
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        for src_path in srcs[:1]:
            img = load_gray(src_path, enhance=False)
            if img is None:
                continue
            h, w = img.shape
            ref = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            top1 = suite.retrieve_top1_correct(ref, src_path)
            rows.append({"scale": s, "retrieval_top1_correct": int(top1)})
    return rows


def sec_scale(suite: ValidationSuite, cache) -> dict:
    rows = []
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        srcs = suite.best_texture_patches("OHRC", n=1, seed=19)
        if not srcs:
            continue
        p = suite.controlled_pair(srcs[0], name=f"sih_scale_OHRC_to_TMC2_x{s}",
                                  sensor_a="OHRC", sensor_b="TMC-2", scale=s, seed=77)
        if p is None:
            continue
        r = cache.run(p, "full")
        rows.append({"scale": s, "status": r.get("status"),
                     "multiscale_retry": (r.get("scale_pyramid_retry", "")
                                          or r.get("multiscale_retry", "")),
                     "match_count": r.get("match_count"),
                     "inlier_count": r.get("inlier_count"),
                     "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                     "rmse_gt_px": r.get("rmse_gt_px"),
                     "coverage_score": r.get("coverage_score")})
    ret_rows = retrieval_at_scale(suite)
    ret_agg = [{"scale": s,
                "retrieval_top1_correct": round(float(np.mean([
                    x["retrieval_top1_correct"] for x in ret_rows
                    if x["scale"] == s]) or np.nan), 3)}
               for s in (0.5, 1.0, 2.0, 4.0, 8.0)]
    return {"rows": rows, "retrieval": ret_agg}


def sec_viewpoint(suite: ValidationSuite, cache) -> dict:
    real = suite.real_crossview_pair()
    if real is None:
        return {"rows": [{"pair": "real_ncf_ncn", "status": "pair_unavailable"}]}
    r = cache.run(real, "full")
    return {"rows": [{"pair": real.name, "protocol": real.protocol,
                      "status": r.get("status"),
                      "multiscale_retry": r.get("multiscale_retry", ""),
                      "match_count": r.get("match_count"),
                      "inlier_count": r.get("inlier_count"),
                      "rmse_px": r.get("rmse_px"),
                      "coverage_score": r.get("coverage_score"),
                      "runtime_s": r.get("runtime_s")}]}


def sec_benchmarks(suite: ValidationSuite, cache) -> dict:
    rows = []
    srcs_t = suite.best_texture_patches("TMC-2", n=1, seed=5)
    srcs_o = suite.best_texture_patches("OHRC", n=1, seed=5)
    pairs = []
    if srcs_t:
        p = suite.controlled_pair(srcs_t[0], name="sih_bench_TMC2_to_LRO",
                                  sensor_a="TMC-2", sensor_b="LRO NAC", scale=1.0, seed=11)
        if p:
            pairs.append(p)
    if srcs_o:
        p = suite.controlled_pair(srcs_o[0], name="sih_bench_OHRC_to_TMC2",
                                  sensor_a="OHRC", sensor_b="TMC-2", scale=1.0, seed=12)
        if p:
            pairs.append(p)
    akaze = AkazeRunner()
    sg = SuperglueRunner(max_keypoints=2048)
    log(f"superglue available={sg.available}")
    for p in pairs:
        r_full = cache.run(p, "full")
        rows.append({"pair": p.name, "method": "lunarai_full", "status": r_full.get("status"),
                     "inlier_ratio_3px": r_full.get("inlier_ratio_3px"),
                     "rmse_gt_px": r_full.get("rmse_gt_px"),
                     "coverage_score": r_full.get("coverage_score"),
                     "runtime_s": r_full.get("runtime_s")})
        for runner in (akaze, sg):
            r = runner.run(p, suite._prep, suite.cfg)
            rows.append({"pair": p.name, "method": runner.method, "status": r.get("status"),
                         "inlier_ratio_3px": r.get("inlier_ratio_3px"),
                         "rmse_gt_px": r.get("rmse_gt_px"),
                         "coverage_score": r.get("coverage_score"),
                         "runtime_s": r.get("runtime_s")})
    return {"rows": rows, "superglue_available": sg.available}


ISRO_MATRIX = [
    {"requirement": "Illumination Variation", "status": "implemented (measured bins) + synthetic protocol for empty bins",
     "evidence": "sun_angle bins table; CLAHE; sun-aware triplets; rerank_by_sun_distance"},
    {"requirement": "Viewpoint Variation", "status": "improved — bounded-homography multiscale retry on the real ncf<->ncn pair",
     "evidence": "viewpoint section rows (status + multiscale_retry)"},
    {"requirement": "Scale Variation", "status": "implemented 0.5x-8x with retry ladder; retrieval scale-invariant by 224x224 resize",
     "evidence": "scale section rows + retrieval@scale"},
    {"requirement": "Multi-Modal Correspondence", "status": "implemented incl. TMC-2<->IIRS (grayscale browse caveat remains)",
     "evidence": "tmc2_iirs section + multi_modal_validation.csv"},
    {"requirement": "Uniform Match Distribution", "status": "implemented (ANMS min-radius greedy)",
     "evidence": "uniform_distribution_report.md; coverage_analysis.csv"},
    {"requirement": "Sub-Pixel Registration", "status": "implemented (median GT RMSE 0.2338 px pre-upgrade; ECC + regression guard)",
     "evidence": "subpixel_accuracy_report.csv"},
]


def main() -> int:
    t0 = time.perf_counter()
    cfg = load_config()
    suite = ValidationSuite(cfg)

    class _Cache:
        def __init__(self):
            self.results = {}

        def run(self, pair: Pair, variant: str = "full", save: bool = False, log=log):
            key = (pair.name, variant)
            if key not in self.results:
                self.results[key] = suite.run_pipeline(
                    pair, variant=variant, save_dir=OUT if save else None)
                log(f"run {pair.name[:44]:44s} {variant:9s} -> {self.results[key].get('status')}")
            return self.results[key]

    cache = _Cache()

    # retrieval Top-1 correctness helper (used by retrieval@scale)
    from lunarai_lib.lunadna import compute_embeddings as _cemb
    import tempfile

    def retrieve_top1_correct(self_suite, ref_img, true_path):
        tmp = Path(tempfile.gettempdir()) / "sih_query.png"
        cv2.imwrite(str(tmp), ref_img)
        emb = _cemb(self_suite.model, [str(tmp)], device=self_suite.device, size=224)
        scores, idx = self_suite.index.search(emb[0], 1)
        if not len(idx[0]) or idx[0][0] < 0 or idx[0][0] >= len(self_suite.mapping):
            return False
        hit = Path(self_suite.mapping[int(idx[0][0])]).name == Path(true_path).name
        return bool(hit)

    # patch the helper onto the suite (kept local to this driver)
    suite.retrieve_top1_correct = retrieve_top1_correct.__get__(suite)  # type: ignore

    from lunarai_lib.validation_ext import ExtSuite
    ext = ExtSuite(suite)

    results: dict[str, dict] = {}
    sections = [
        ("splits", lambda: sec_splits(cfg)),
        ("elevation", lambda: sec_elevation(cfg)),
        ("sunangle", lambda: sec_sunangle(cfg)),
        ("pca", lambda: sec_pca(suite, cfg)),
        ("tmc2_iirs", lambda: sec_tmc2_iirs(suite, cache)),
        ("sun_bins", lambda: sec_sun_bins(suite, cache, ext)),
        ("scale", lambda: sec_scale(suite, cache)),
        ("viewpoint", lambda: sec_viewpoint(suite, cache)),
        ("benchmarks", lambda: sec_benchmarks(suite, cache)),
    ]
    for name, fn in sections:
        log(f"section {name}")
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": f"{exc}", "trace": traceback.format_exc()[-1500:]}
            log(f"ERROR in {name}: {exc}")

    results["isro_matrix"] = {"rows": ISRO_MATRIX}

    # machine-readable dump
    (OUT / "sih_upgrade_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    # CSVs for the dashboard
    if "pca" in results and "rows" in results["pca"]:
        pd.DataFrame(results["pca"]["rows"]).to_csv(OUT / "pca_ablation.csv", index=False)
    for key, fname in [("tmc2_iirs", "tmc2_iirs_validation.csv"),
                       ("sun_bins", "sun_bins_full.csv"),
                       ("scale", "scale_extended.csv"),
                       ("viewpoint", "viewpoint_real_pair.csv"),
                       ("benchmarks", "benchmarks_extended.csv")]:
        sec = results.get(key, {})
        rows = sec.get("rows") if isinstance(sec, dict) else None
        if rows:
            pd.DataFrame(rows).to_csv(OUT / fname, index=False)

    # ---------------- markdown report ---------------- #
    md = [f"# LunarAI SIH Grand Finale Upgrade Report",
          f"",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} · runtime "
          f"{time.perf_counter() - t0:.0f}s · matcher {suite.matcher_backend} · "
          f"checkpoint epoch {suite.ckpt_epoch}",
          ""]
    md += ["## 1. Train/Val/Test split (70/15/15, geographic, no leakage)", ""]
    sp = results.get("splits", {})
    if "meta" in sp:
        md += [f"- cells: {sp['meta']['total_geo_cells']} · patches: {sp['meta']['total_patches']}",
               "", md_table(sp["table"]), ""]
    elif "error" in sp:
        md += [f"- ERROR: {sp['error']}", ""]
    md += ["## 2. ElevationProfile integration (LOLA Tycho track corridor)", ""]
    el = results.get("elevation", {})
    if "stats" in el:
        md += [f"- track points: {el['stats'].get('points')} · covered patches: "
               f"{el['covered']}/{el['total']} ({el['coverage']:.1%})",
               f"- track stats: {json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in el['stats'].items()})}",
               "- remaining patches stay NaN by design: they lie outside the single LOLA ground track.", ""]
    md += ["## 3. SunAngle integration", ""]
    sa = results.get("sunangle", {})
    if "stats" in sa:
        md += [f"- pass stats: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in sa['stats'].items()})}",
               f"- patch sun bins: {json.dumps(sa['bins'])}", ""]
    md += ["## 4. PCA compression ablation (fit on train, eval on test)", ""]
    pc = results.get("pca", {})
    md += [md_table(pc.get("rows", [])) if "rows" in pc else f"- ERROR: {pc.get('error')}", ""]
    md += ["## 5. TMC-2 <-> IIRS (new pair)", ""]
    ti = results.get("tmc2_iirs", {})
    md += [md_table(ti.get("rows", [])) if "rows" in ti else f"- ERROR: {ti.get('error')}", ""]
    md += ["## 6. Full sun-bin table (measured + synthetic-illumination protocol)", ""]
    sb = results.get("sun_bins", {})
    md += [md_table(sb.get("rows", [])) if "rows" in sb else f"- ERROR: {sb.get('error')}",
           "", "_synthetic_illumination rows: no catalog imagery exists in that sun bin; "
           "TMC-2 terrain was resampled to the bin's radiometric regime (gamma + mean/std) "
           "with the known GT transform preserved. They measure matcher robustness to the "
           "illumination regime, NOT end-to-end catalog coverage._", ""]
    md += ["## 7. Scale 0.5x-8x (multiscale retry ladder) + retrieval@scale", ""]
    sc = results.get("scale", {})
    if "rows" in sc:
        md += [md_table(sc["rows"]), "", "Retrieval Top-1 under reference scale change:", "",
               md_table(sc.get("retrieval", [])), ""]
    md += ["## 8. Viewpoint: real TMC ncf<->ncn pair with retry ladder", ""]
    vp = results.get("viewpoint", {})
    md += [md_table(vp.get("rows", [])) if "rows" in vp else f"- ERROR: {vp.get('error')}", ""]
    md += ["## 9. Benchmarks: AKAZE + SuperPoint+SuperGlue on the focused set", ""]
    bm = results.get("benchmarks", {})
    md += [md_table(bm.get("rows", [])) if "rows" in bm else f"- ERROR: {bm.get('error')}", ""]
    md += ["## 10. ISRO requirement matrix", "", md_table(ISRO_MATRIX), ""]

    (OUT / "sih_upgrade_report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"done in {time.perf_counter() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
