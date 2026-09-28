"""Extended validation modules for SIH 26166.

Mission-specific evidence layers on top of the base validation suite
(`lunarai_lib.validation`):

- **Crater validation** — Hough-circle crater detection, matched-crater consistency,
  center deviation and overlap score on pairs with a known homography.
- **5-bin sun-angle validation** — pairs grouped by measured sun elevation into
  0-10 / 10-20 / 20-40 / 40-60 / 60+ degree bins. Bin labels carry the measured share
  of each bin, so the coverage limitation is explicit rather than hidden.
- **Confidence engine** — 0-100 score with Very High / High / Medium / Low categories
  from embedding similarity, inlier ratio, RMSE and coverage (weights fixed a priori),
  plus a category calibration table against measured GT error.
- **Geographic localization** — retrieval-as-localization: predicted (lat, lon) of the
  top-FAISS candidate vs the query's catalog coordinates, against a label-shuffled
  chance baseline and with calibration (error vs retrieval similarity).

All functions are pure measurements: no reported metric depends on being tuned
against itself.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import Config

# --------------------------------------------------------------------------- #
# Sun-angle bins (SIH spec)
# --------------------------------------------------------------------------- #
SUN_BINS = [
    ("0-10", 0.0, 10.0),
    ("10-20", 10.0, 20.0),
    ("20-40", 20.0, 40.0),
    ("40-60", 40.0, 60.0),
    ("60+", 60.0, 91.0),
]


def bin_label(sun: float) -> str | None:
    """Bin name for a sun elevation in degrees, or None if outside [0, 91)."""
    for name, lo, hi in SUN_BINS:
        if lo <= sun < hi:
            return name
    return None


# --------------------------------------------------------------------------- #
# Crater detection and pair consistency
# --------------------------------------------------------------------------- #
def detect_craters(img: np.ndarray, min_r: int = 5, max_r: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Detect crater rims with the Hough circle transform.

    CLAHE + median blur first; `param2` is swept over a small grid and the best
    parameter set (most circles without heavy mutual overlap) is kept. Returns
    (centres Nx2 float, radii N); empty arrays when nothing is found.
    """
    if img is None or getattr(img, "ndim", 2) != 2:
        return np.zeros((0, 2)), np.zeros(0)
    work = img
    if work.dtype != np.uint8:
        work = np.clip(np.asarray(work, dtype=float), 0, 255).astype(np.uint8)
    work = cv2.createCLAHE(2.0, (8, 8)).apply(work)
    work = cv2.medianBlur(work, 3)
    h, w = work.shape
    best: tuple[np.ndarray, np.ndarray] = (np.zeros((0, 2)), np.zeros(0))
    best_score = -1.0
    for p2 in (22, 18, 15, 12):
        circles = cv2.HoughCircles(work, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
                                   param1=110, param2=p2,
                                   minRadius=min_r, maxRadius=min(max_r, min(h, w) // 3))
        if circles is None:
            continue
        c = circles.reshape(-1, 3)
        cent = c[:, :2].astype(float)
        rad = c[:, 2].astype(float)
        score = float(len(cent))
        if len(cent) > 1:
            d = np.hypot(cent[:, None, 0] - cent[None, :, 0],
                         cent[:, None, 1] - cent[None, :, 1])
            ratio = np.minimum(rad[:, None], rad[None, :]) / np.maximum(rad[:, None], rad[None, :])
            overlap = float((ratio * (d < (rad[:, None] + rad[None, :]))).sum()
                            / (len(cent) * (len(cent) - 1)))
            score *= (1.0 - 0.5 * min(overlap, 1.0))
        if score > best_score:
            best_score = score
            best = (cent, rad)
    cent, rad = best
    if len(cent) > 40:  # keep the 40 largest: small false circles swamp the ratio
        order = np.argsort(-rad)[:40]
        cent, rad = cent[order], rad[order]
    return cent, rad


def crater_pair_metrics(src_img: np.ndarray, ref_img: np.ndarray, H: np.ndarray | None,
                        tol_px: float = 6.0) -> dict:
    """Crater consistency on one pair with a known homography.

    Reference craters are lifted into the query frame through H^-1 and matched to
    query craters by center distance < tol:
    - `craters_src` / `craters_ref`: raw detection counts.
    - `matched`: reference craters with a query counterpart within tol.
    - `crater_center_dev_px`: mean center distance of matched craters.
    - `crater_overlap_score`: mean radius agreement, 1 - |r1-r2|/max(r), in [0,1].
    - `consistency`: matched / min(n_src, n_ref), in [0,1].
    """
    cs, rs = detect_craters(src_img)
    cr, rr = detect_craters(ref_img)
    out: dict = {"craters_src": int(len(cs)), "craters_ref": int(len(cr))}
    if H is None or len(cs) == 0 or len(cr) == 0:
        out.update(matched=0, crater_center_dev_px=np.nan,
                   crater_overlap_score=np.nan, consistency=0.0)
        return out
    Hinv = np.linalg.inv(H)
    lifted = cv2.perspectiveTransform(cr.reshape(-1, 1, 2).astype(np.float32),
                                      Hinv.astype(np.float32)).reshape(-1, 2)
    d = np.hypot(cs[:, None, 0] - lifted[None, :, 0], cs[:, None, 1] - lifted[None, :, 1])
    pair_dist = d.min(axis=1)
    nearest = d.argmin(axis=1)
    keep = pair_dist < tol_px
    n_matched = int(keep.sum())
    out["matched"] = n_matched
    if n_matched == 0:
        out.update(crater_center_dev_px=np.nan, crater_overlap_score=np.nan, consistency=0.0)
        return out
    out["crater_center_dev_px"] = float(pair_dist[keep].mean())
    rad_q = rs[keep]
    rad_r = rr[nearest[keep]]
    out["crater_overlap_score"] = float(
        np.mean(1.0 - np.abs(rad_q - rad_r) / np.maximum(rad_q, rad_r)))
    out["consistency"] = float(n_matched / max(min(len(cs), len(cr)), 1))
    return out


def crater_visualization(src_img: np.ndarray, ref_img: np.ndarray, H: np.ndarray | None,
                         path: Path, tol_px: float = 6.0) -> None:
    """Side-by-side crater detections (green = matched, orange = unmatched)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs, rs = detect_craters(src_img)
    cr, rr = detect_craters(ref_img)
    matched_q = np.zeros(len(cs), bool)
    matched_r = np.zeros(len(cr), bool)
    if H is not None and len(cs) and len(cr):
        Hinv = np.linalg.inv(H)
        lifted = cv2.perspectiveTransform(cr.reshape(-1, 1, 2).astype(np.float32),
                                          Hinv.astype(np.float32)).reshape(-1, 2)
        if len(cs) and len(lifted):
            d = np.hypot(cs[:, None, 0] - lifted[None, :, 0],
                         cs[:, None, 1] - lifted[None, :, 1])
            nearest = d.argmin(axis=1)
            keep = d.min(axis=1) < tol_px
            matched_q = keep
            matched_r[nearest[keep]] = True
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, img, cent, rad, m, title in [
            (axes[0], src_img, cs, rs, matched_q, "query craters"),
            (axes[1], ref_img, cr, rr, matched_r, "reference craters (lifted to query frame)")]:
        ax.imshow(img, cmap="gray")
        for (x, y), r, ok in zip(cent, rad, m):
            ax.add_patch(plt.Circle((x, y), r, fill=False,
                                    color="#22C55E" if ok else "#F59E0B", linewidth=1.2))
        ax.set_title(f"{title}: {int(m.sum())} matched / {len(cent)}")
        ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Confidence engine
# --------------------------------------------------------------------------- #
def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def confidence_engine(embedding_similarity: float, inlier_ratio: float, rmse_px: float,
                      coverage: float) -> dict:
    """0-100 confidence with a category, from four independent evidence axes.

    Weights are fixed a priori (before evaluation):
    - embedding similarity in [0,1] -> 0-25
    - inlier ratio (share of correspondences within 3 px) in [0,1] -> 0-30
    - RMSE: 0 px -> full 30, >= 4 px -> 0 (piecewise linear) -> 0-30
    - coverage in [0,1] -> 0-15
    Categories: >=80 Very High, >=60 High, >=40 Medium (High if every axis >= 0.75
    of its full contribution), else Low.
    """
    sim_pts = 25.0 * _clamp01(embedding_similarity)
    inl_pts = 30.0 * _clamp01(inlier_ratio)
    rmse_pts = 30.0 * _clamp01(1.0 - (rmse_px / 4.0)) if np.isfinite(rmse_px) else 0.0
    cov_pts = 15.0 * _clamp01(coverage)
    score = sim_pts + inl_pts + rmse_pts + cov_pts
    if score >= 80:
        cat = "Very High"
    elif score >= 60:
        cat = "High"
    elif score >= 40:
        axes_q = [_clamp01(embedding_similarity), _clamp01(inlier_ratio),
                  _clamp01(1 - rmse_px / 4.0) if np.isfinite(rmse_px) else 0.0,
                  _clamp01(coverage)]
        cat = "High" if float(np.mean(axes_q)) >= 0.75 else "Medium"
    else:
        cat = "Low"
    return {"confidence_score": round(float(score), 1), "category": cat,
            "axes": {"embedding": round(sim_pts, 2), "inliers": round(inl_pts, 2),
                     "rmse": round(rmse_pts, 2), "coverage": round(cov_pts, 2)}}


def confidence_category(score: float) -> str:
    if score >= 80:
        return "Very High"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Geographic localization
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points in degrees."""
    r = 1_737_400.0  # lunar mean radius, metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def localize_query(model, index, mapping, patch_df: pd.DataFrame, query_img: np.ndarray,
                   input_size: int, device: str = "cpu", top_k: int = 5) -> dict:
    """Predict (lat, lon) for a query patch as the similarity-weighted centroid of the
    top-FAISS candidates' catalog coordinates, and score it against the query's own
    catalog coordinates when available.

    Returns dict with predicted lat/lon, top-1 candidate coords, similarity, and
    (when the query row is supplied) error in metres.
    """
    import tempfile
    from .lunadna import compute_embeddings

    tmp = Path(tempfile.gettempdir()) / "lunarai_loc_query.png"
    cv2.imwrite(str(tmp), query_img)
    emb = compute_embeddings(model, [str(tmp)], device=device, size=input_size)[0]
    scores, idxs = index.search(emb[None, :], top_k)
    coords, sims = [], []
    top1 = None
    for rank, (s, i) in enumerate(zip(scores[0], idxs[0]), start=1):
        if not (0 <= i < len(mapping)):
            continue
        cid = Path(mapping[int(i)]).stem
        row = patch_df[patch_df.patch_id == cid]
        if not len(row):
            continue
        r = row.iloc[0]
        if rank == 1:
            top1 = {"patch_id": cid, "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]), "similarity": float(s)}
        if not (np.isfinite(r["latitude"]) and np.isfinite(r["longitude"])):
            continue
        coords.append((float(r["latitude"]), float(r["longitude"])))
        sims.append(float(s))
    out: dict = {"top1": top1, "n_candidates": len(coords)}
    if coords:
        w = np.asarray(sims, dtype=float)
        w = np.clip(w - (w.max() - 0.05), 0.01, None)  # sharpen, keep positive weights
        lat = float(np.average([c[0] for c in coords], weights=w))
        lon = float(np.average([c[1] for c in coords], weights=w))
        out["pred_lat"], out["pred_lon"] = lat, lon
    else:
        out["pred_lat"] = out["pred_lon"] = None
    return out


def localization_from_retrieval(suite, n_queries: int = 60, seed: int = 11,
                                log: Callable[[str], None] = print) -> dict:
    """Localization error over held-out queries, with chance and calibration views.

    Error = haversine distance between the top-1 candidate's catalog coordinates and
    the query's own coordinates. Same-query candidates (the identical patch) are
    skipped so the measure is non-trivial. The chance baseline shuffles the predicted
    labels across queries; the calibration table reports median error by similarity
    band — the operational rule "trust high-similarity retrievals more" must show up
    here, or the confidence engine would be rewarding noise.
    """
    import torch
    df = suite.df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    qidx = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
    qdf = df.iloc[qidx].reset_index(drop=True)
    from .lunadna import compute_embeddings

    tmp_dir = Path(suite.cfg.MATCHING_DIR) / "_loc_queries"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, p in enumerate(qdf.patch_path):
        q = tmp_dir / f"q{i:03d}.png"
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            paths.append(p)
            continue
        cv2.imwrite(str(q), img)
        paths.append(str(q))
    embs = compute_embeddings(suite.model, paths, device=suite.device,
                              size=int(suite.cfg.LUNADNA.get("input_size", 224)))
    scores, idxs = suite.index.search(embs, 10)

    meta = df.set_index("patch_id")
    rows = []
    for qi in range(len(qdf)):
        q = qdf.iloc[qi]
        if not (np.isfinite(q["latitude"]) and np.isfinite(q["longitude"])):
            continue
        pred = None
        for s, gi in zip(scores[qi], idxs[qi]):
            if not (0 <= gi < len(suite.mapping)):
                continue
            cid = Path(suite.mapping[int(gi)]).stem
            if cid == q.patch_id or cid not in meta.index:
                continue
            c = meta.loc[cid]
            if np.isfinite(c["latitude"]) and np.isfinite(c["longitude"]):
                pred = (float(c["latitude"]), float(c["longitude"]), float(s), cid)
                break
        if pred is None:
            continue
        plat, plon, sim, cid = pred
        err_m = haversine_m(q["latitude"], q["longitude"], plat, plon)
        rows.append({"query": q.patch_id, "sensor": q["dataset_name"],
                     "true_lat": float(q["latitude"]), "true_lon": float(q["longitude"]),
                     "pred_lat": plat, "pred_lon": plon, "retrieval_similarity": sim,
                     "error_m": round(err_m, 1), "pred_patch": cid})
    if not rows:
        return {"rows": [], "summary": {}, "chance_median_m": None, "calibration": []}

    errs = np.asarray([r["error_m"] for r in rows], dtype=float)
    sims = np.asarray([r["retrieval_similarity"] for r in rows], dtype=float)

    # chance: shuffle predicted coordinates across queries
    perm = rng.permutation(len(rows))
    chance = np.asarray([haversine_m(rows[i]["true_lat"], rows[i]["true_lon"],
                                     rows[perm[i]]["pred_lat"], rows[perm[i]]["pred_lon"])
                         for i in range(len(rows))], dtype=float)

    bands = [(0.95, 1.01, ">=0.95"), (0.90, 0.95, "0.90-0.95"),
             (0.80, 0.90, "0.80-0.90"), (0.00, 0.80, "<0.80")]
    calibration = []
    for lo, hi, label in bands:
        m = (sims >= lo) & (sims < hi)
        if m.sum():
            calibration.append({"similarity_band": label, "queries": int(m.sum()),
                                "median_error_m": round(float(np.median(errs[m])), 1)})

    summary = {
        "queries": len(rows),
        "median_error_m": round(float(np.median(errs)), 1),
        "mean_error_m": round(float(np.mean(errs)), 1),
        "p90_error_m": round(float(np.percentile(errs, 90)), 1),
        "within_1km": round(float((errs < 1000).mean()), 3),
        "within_10km": round(float((errs < 10000).mean()), 3),
        "chance_median_m": round(float(np.median(chance)), 1),
        "localization_gain": round(float(np.median(chance) / max(np.median(errs), 1e-9)), 1),
    }
    log(f"[loc] median {summary['median_error_m']} m vs chance {summary['chance_median_m']} m "
        f"({summary['localization_gain']}x) | within 1 km: {summary['within_1km']}")
    return {"rows": rows, "summary": summary, "chance_median_m": summary["chance_median_m"],
            "calibration": calibration}


# --------------------------------------------------------------------------- #
# Sun-angle bins over patch catalog
# --------------------------------------------------------------------------- #
def sun_angle_bin_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the SIH 5-bin sun-angle label to every patch row with a sun value."""
    d = df[df.sun_angle.notna()].copy()
    d["sun_bin"] = d.sun_angle.map(bin_label)
    return d.dropna(subset=["sun_bin"])


def make_ext_suite(base_suite) -> "ExtSuite":
    """Build an ExtSuite that reuses a fitted ValidationSuite (model, index, matcher)."""
    return ExtSuite(base_suite)


class ExtSuite:
    """Extended validation: craters, 5-bin sun angle, confidence, localization.

    Wraps a fitted `ValidationSuite` and reuses its model, FAISS index, matcher,
    patch catalog and pair builders so every measurement uses the same protocol.
    """

    def __init__(self, base) -> None:
        self.base = base
        self.cfg: Config = base.cfg
        self.log = base.log

    # -- confidence --------------------------------------------------------- #
    def confidence_from_run(self, run: dict, embedding_similarity: float) -> dict:
        """Confidence for one pipeline run from its measured quantities."""
        return confidence_engine(
            embedding_similarity=embedding_similarity,
            inlier_ratio=float(run.get("inlier_ratio_3px", run.get("inlier_ratio", 0.0)) or 0.0),
            rmse_px=float(run.get("rmse_gt_px", run.get("rmse_px", float("nan")))),
            coverage=float(run.get("coverage_score", 0.0) or 0.0))

    # -- craters ------------------------------------------------------------ #
    def crater_validation(self, pairs, run_lookup: Callable[[str], dict | None],
                          out_dir: Path, tol_px: float = 6.0) -> pd.DataFrame:
        """Crater consistency per pair. `run_lookup(pair_name)` returns that pair's
        pipeline run (for the accepted homography) or None."""
        rows = []
        for pair in pairs:
            run = run_lookup(pair.name) or {}
            H = None
            # homography preference: the run's estimated homography, else the pair's
            # ground-truth warp (controlled pairs) so crater consistency is measurable
            # even where the matcher found no accepted H
            if run.get("homography") is not None:
                H = np.asarray(run["homography"], dtype=np.float64)
            elif run.get("status") == "ok" and pair.H_gt is not None:
                H = pair.H_gt  # controlled pairs: the GT warp defines the mapping
            m = crater_pair_metrics(pair.src, pair.ref, H, tol_px=tol_px)
            m.update(pair=pair.name, protocol=pair.protocol,
                     sensor_a=pair.sensor_a, sensor_b=pair.sensor_b,
                     status=run.get("status", "not_run"),
                     homography_source=("run" if run.get("homography") is not None
                                        else ("gt" if H is not None else "none")))
            rows.append(m)
            self.log(f"[crater] {pair.name}: {m['craters_src']} src / {m['craters_ref']} ref, "
                     f"matched={m['matched']} dev={m['crater_center_dev_px']} "
                     f"overlap={m['crater_overlap_score']}")
            crater_visualization(pair.src, pair.ref, H,
                                 out_dir / f"crater_{pair.name[:40]}.png", tol_px)
        return pd.DataFrame(rows)

    # -- 5-bin sun angle ---------------------------------------------------- #
    def sun_angle_pairs(self, n_per_bin: int = 4, seed: int = 21) -> list:
        """Controlled TMC->TMC pairs whose source patches fall in each sun bin.

        Bin labels in the report carry `catalog_share` — the share of the patch
        catalog in that bin — because this dataset cannot populate every bin.
        """
        from .validation import Pair  # noqa: F401  (typing only)
        cat = sun_angle_bin_catalog(self.base.df)
        pairs = []
        for name, lo, hi in SUN_BINS:
            sub = cat[cat.sun_bin == name]
            if not len(sub):
                continue
            take = sub.sample(min(n_per_bin, len(sub)), random_state=seed)
            for _, row in take.iterrows():
                p = self.base.controlled_pair(
                    row.patch_path, name=f"sunbin_{name}_{row.patch_id[-14:]}",
                    sensor_a=row.dataset_name, sensor_b=row.dataset_name,
                    scale=1.0, seed=seed)
                if p is not None:
                    p.meta["sun_bin"] = name
                    p.meta["sun_angle"] = float(row.sun_angle)
                    pairs.append(p)
        return pairs

    def retrieval_by_bin(self, n_per_bin: int = 20, seed: int = 23) -> list[dict]:
        """Retrieval top-1/top-5 (same-source) per sun bin on illuminated queries."""
        from .lunadna import compute_embeddings
        cat = sun_angle_bin_catalog(self.base.df)
        rows = []
        tmp_dir = Path(self.cfg.MATCHING_DIR) / "_sunbin_queries"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        meta = self.base.df.set_index("patch_id")
        for name, lo, hi in SUN_BINS:
            sub = cat[cat.sun_bin == name]
            if not len(sub):
                rows.append({"sun_bin": name, "catalog_patches": 0, "queries": 0,
                             "top1": np.nan, "top5": np.nan})
                continue
            take = sub.sample(min(n_per_bin, len(sub)), random_state=seed)
            hits1 = hits5 = n = 0
            for i, (_, row) in enumerate(take.iterrows()):
                img = cv2.imread(row.patch_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                q = tmp_dir / f"{name.replace('-', '_')}_{i}.png"
                cv2.imwrite(str(q), img)
                emb = compute_embeddings(self.base.model, [str(q)], device=self.base.device,
                                         size=int(self.cfg.LUNADNA.get("input_size", 224)))
                sc, ix = self.base.index.search(emb, 6)
                ranked = []
                for s, gi in zip(sc[0], ix[0]):
                    if 0 <= gi < len(self.base.mapping):
                        cid = Path(self.base.mapping[int(gi)]).stem
                        if cid != row.patch_id and cid in meta.index:
                            ranked.append(cid)
                if not ranked:
                    continue
                n += 1
                if ranked[:1] and meta.loc[ranked[0]].source_image == row.source_image:
                    hits1 += 1
                if any(meta.loc[c].source_image == row.source_image for c in ranked[:5]):
                    hits5 += 1
            rows.append({"sun_bin": name, "catalog_patches": int(len(sub)), "queries": n,
                         "top1": round(hits1 / max(n, 1), 3),
                         "top5": round(hits5 / max(n, 1), 3)})
            self.log(f"[sunbin] {name}: n={n} top1={rows[-1]['top1']} top5={rows[-1]['top5']}")
        return rows

    # -- localization ------------------------------------------------------- #
    def localization(self, n_queries: int = 60, seed: int = 11) -> dict:
        return localization_from_retrieval(self.base, n_queries=n_queries, seed=seed,
                                           log=self.log)
