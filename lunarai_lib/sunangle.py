"""SunAngle integration — SIH Phase 1.3.

data/SunAngle.csv: 5000 rows (Time, Elevation, Azimuth) sampled over one lunar
day at the survey site (elev 26.8-34.2 deg, azimuth 298-309 deg).

Three concrete uses, all wired end to end:
1. TRAINING: sun-aware triplet sampling — negatives are preferentially chosen
   with a sun-elevation difference > sun_neg_delta_deg (see
   make_sun_aware_triplets), so illumination invariance is *trained*, not
   assumed. Patches keep their per-product sun_elevation from the PDS labels.
2. FEATURE ENGINEERING: every patch keeps sun_elevation/sun_azimuth metadata
   plus a derived sun_bin; SunAngle.csv supplies the pass statistics
   (min/mean/max/std of elevation+azimuth) exported alongside.
3. RETRIEVAL: illumination-aware re-ranking — candidates are re-ordered by
   similarity minus a small penalty proportional to the sun-elevation
   difference between query and candidate (rerank_by_sun_distance).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd

SUN_BINS_5 = [("0-10", 0.0, 10.0), ("10-20", 10.0, 20.0), ("20-40", 20.0, 40.0),
              ("40-60", 40.0, 60.0), ("60+", 60.0, 90.0)]


def load_sunangle_stats(csv_path: Path) -> dict:
    """(Time, Elevation, Azimuth) -> pass statistics used in reports/metadata."""
    if not csv_path.exists():
        return {"rows": 0}
    elevs: list[float] = []
    azims: list[float] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                elevs.append(float(row[1]))
                azims.append(float(row[2]))
            except (ValueError, IndexError):
                continue
    if not elevs:
        return {"rows": 0}
    e = np.array(elevs, dtype=float)
    z = np.array(azims, dtype=float)
    return {
        "rows": len(elevs),
        "elev_min": float(e.min()), "elev_mean": float(e.mean()),
        "elev_max": float(e.max()), "elev_std": float(e.std()),
        "azim_min": float(z.min()), "azim_mean": float(z.mean()),
        "azim_max": float(z.max()), "azim_std": float(z.std()),
    }


def sun_distance(sun_a: float, sun_b: float) -> float:
    """Absolute sun-elevation difference in degrees; NaN-safe."""
    try:
        a = float(sun_a)
        b = float(sun_b)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(a - b)


def sun_bin_label(elev: float) -> str:
    """5-bin SIH label for a sun elevation in degrees ('unknown' when NaN)."""
    try:
        e = float(elev)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(e):
        return "unknown"
    for name, lo, hi in SUN_BINS_5:
        if lo <= e < hi:
            return name
    return "unknown"


def _to_float_array(series) -> np.ndarray:
    return pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)


def make_sun_aware_triplets(df, n_negatives: int = 1, seed: int = 42,
                            sun_neg_delta_deg: float = 10.0
                            ) -> list[tuple[str, str, str]]:
    """Triplets whose negatives come preferentially from a *different sun regime*.

    Positives: adjacent patches of the SAME source image (50% overlap = same
    terrain) — identical to make_adjacent_triplets. Using cross-sensor
    geo-proximity positives instead would teach the trivial collapse noted in
    lunadna.make_geo_triplets (no truly overlapping cross-sensor imagery exists).

    Negatives: patches from a DIFFERENT source image, preferring sun-elevation
    delta > sun_neg_delta_deg -> the embedding must separate terrain while
    staying invariant to illumination.

    df needs: patch_path, source_image, row, col, sun_angle.
    """
    rng = random.Random(seed)
    rows = df.dropna(subset=["patch_path", "row", "col"]).reset_index(drop=True)
    if not len(rows):
        return []
    suns = _to_float_array(rows["sun_angle"])
    by_image = {name: grp.reset_index(drop=True)
                for name, grp in rows.groupby("source_image")}
    image_names = [k for k, v in by_image.items() if len(v) >= 2]
    if len(image_names) < 2:
        return []
    # per-image median sun elevation drives the sun-hard negative choice
    img_sun = {k: float(np.nanmedian(_to_float_array(v["sun_angle"])))
               for k, v in by_image.items()}

    triplets: list[tuple[str, str, str]] = []
    for img in image_names:
        g = by_image[img]
        rows_arr = g["row"].to_numpy(dtype=float)
        cols_arr = g["col"].to_numpy(dtype=float)
        others = [k for k in image_names if k != img]
        sun_here = img_sun.get(img, float("nan"))
        # negatives from images with a different sun regime, fallback: any other
        sun_far_imgs = [k for k in others
                        if np.isfinite(sun_here) and np.isfinite(img_sun[k])
                        and abs(sun_here - img_sun[k]) > sun_neg_delta_deg]
        neg_imgs = sun_far_imgs or others
        for i in range(len(g)):
            d = np.abs(rows_arr - rows_arr[i]) + np.abs(cols_arr - cols_arr[i])
            d[i] = np.inf
            j = int(np.argmin(d))
            if not np.isfinite(d[j]):
                continue
            neg_img = rng.choice(neg_imgs)
            ng = by_image[neg_img]
            k = rng.randrange(len(ng))
            for _ in range(n_negatives):
                triplets.append((g.patch_path.iloc[i], g.patch_path.iloc[j],
                                 ng.patch_path.iloc[k]))
    return triplets


def rerank_by_sun_distance(scores, cand_suns, query_sun: float,
                           lambda_sun: float = 0.002) -> np.ndarray:
    """Illumination-aware re-ranking: similarity - lambda * |Δ sun elevation|.

    scores: (N,) similarities; cand_suns: (N,) candidate sun elevations
    (NaN suns get zero penalty). Returns the adjusted scores.
    """
    scores = np.asarray(scores, dtype=float)
    cand = np.asarray(cand_suns, dtype=float)
    with np.errstate(invalid="ignore"):
        d = np.abs(cand - float(query_sun))
    d = np.nan_to_num(d, nan=0.0)
    return scores - lambda_sun * d
