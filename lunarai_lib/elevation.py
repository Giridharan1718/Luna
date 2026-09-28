"""ElevationProfile integration — SIH Phase 1.2.

Reality of data/ElevationProfile.csv: it is an LRO LOLA DEM ground track over
Tycho crater (col 1 = Longitude, col 2 = Latitude, col 4 = Elevation in meters,
quoted headers). 100 rows along a single ground track. It is NOT a global DEM,
so a global lon/lat grid lookup can never resolve (the pre-upgrade integration
filled 0/940 patches).

What we implement instead (honest + useful):
- Track loading with tolerance to quoted headers / malformed rows.
- Track-corridor mapping: a patch maps to an elevation sample iff it lies within
  `corridor_deg` of some track point (haversine on the lunar sphere).
- Track statistics (min/mean/max/std elevation) exported with every summary.
- Coverage is reported truthfully; patches far from the track stay NaN. For the
  SIH story the feature is also used as a *retrieval metadata filter* feature
  (elevation_band) — see metadata.py helpers here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

MOON_RADIUS_M = 1_737_400.0
DEG_TO_M = MOON_RADIUS_M * np.pi / 180.0  # ~30334 m per degree (surface)


def load_elevation_track(csv_path: Path) -> list[dict]:
    """Parse ElevationProfile.csv -> [{lat, lon, elev_m}] (skips bad rows)."""
    pts: list[dict] = []
    if not csv_path.exists():
        return pts
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header (quoted, sometimes malformed)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                lon = float(row[0])
                lat = float(row[1])
                elev = float(row[3])
            except ValueError:
                continue
            if not all(np.isfinite(v) for v in (lat, lon, elev)):
                continue
            pts.append({"lat": lat, "lon": lon, "elev_m": elev})
    return pts


def track_stats(pts: list[dict]) -> dict:
    if not pts:
        return {"points": 0}
    vals = np.array([p["elev_m"] for p in pts], dtype=float)
    return {
        "points": len(pts),
        "elev_min_m": float(vals.min()),
        "elev_mean_m": float(vals.mean()),
        "elev_max_m": float(vals.max()),
        "elev_std_m": float(vals.std()),
        "lat_min": float(min(p["lat"] for p in pts)),
        "lat_max": float(max(p["lat"] for p in pts)),
        "lon_min": float(min(p["lon"] for p in pts)),
        "lon_max": float(max(p["lon"] for p in pts)),
    }


def nearest_track_distance_deg(lat: float, lon: float, pts: list[dict]) -> float:
    """Min great-circle distance (deg on lunar sphere) from (lat, lon) to track."""
    la = np.radians(lat)
    lo = np.radians(lon)
    best = float("inf")
    for p in pts:
        plat, plon = np.radians(p["lat"]), np.radians(p["lon"])
        # central angle via haversine (robust for small separations)
        dlat = plat - la
        dlon = plon - lo
        h = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(plat) * np.sin(dlon / 2) ** 2
        ang = 2 * np.arcsin(min(1.0, np.sqrt(h)))
        if ang < best:
            best = ang
    return float(np.degrees(best))


def map_elevations(df, pts: list[dict], corridor_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """For a dataframe with latitude/longitude columns return (elev_m, dist_deg).

    elev_m is set only when the patch lies within corridor_deg of the track;
    dist_deg is always the distance to the nearest track point.
    """
    elevs = np.full(len(df), np.nan)
    dists = np.full(len(df), np.nan)
    for i, (la, lo) in enumerate(zip(df.latitude, df.longitude)):
        try:
            la_f, lo_f = float(la), float(lo)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(la_f) and np.isfinite(lo_f)):
            continue
        d = nearest_track_distance_deg(la_f, lo_f, pts)
        dists[i] = d
        if d <= corridor_deg:
            # nearest sample's elevation
            best_j, best_d = 0, float("inf")
            for j, p in enumerate(pts):
                plat, plon = np.radians(p["lat"]), np.radians(p["lon"])
                dlat = plat - np.radians(la_f)
                dlon = plon - np.radians(lo_f)
                h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(la_f)) * np.cos(plat) * np.sin(dlon / 2) ** 2
                ang = 2 * np.arcsin(min(1.0, np.sqrt(h)))
                if ang < best_d:
                    best_d, best_j = ang, j
            elevs[i] = pts[best_j]["elev_m"]
    return elevs, dists


def elevation_band(elev_m: float) -> str:
    """Coarse terrain band for retrieval metadata filtering."""
    if not np.isfinite(elev_m):
        return "unknown"
    if elev_m < -3000:
        return "deep_terrain"
    if elev_m < 0:
        return "low_terrain"
    if elev_m < 3000:
        return "mid_terrain"
    return "high_terrain"
