"""Patch generation: sliding-window extraction with metadata.

256x256 patches, stride 128 (configurable). Uses numpy reads for
large PDS4 binaries, honors a per-image cap for laptop feasibility, and writes
patch_index.csv with Patch_ID / Source_Image / Dataset_Name / Latitude /
Longitude / Sun_Angle / Elevation / Patch_Path.

Layout note: patches are stored FLAT under outputs/patches/<patch_id>.png and
the sensor tag is encoded in the patch id (e.g. OHRC_xxx). This keeps the
dashboard and FAISS mapping simple.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import load_browse_png, load_image_any, save_rows_csv
from .metadata import ImageRecord

PATCH_FIELDS = ["patch_id", "source_image", "dataset_name", "latitude", "longitude",
                "sun_angle", "elevation", "patch_path", "row", "col", "width", "height"]


@dataclass
class PatchMeta:
    patch_id: str
    source_image: str
    dataset_name: str
    latitude: float
    longitude: float
    sun_angle: float
    elevation: float
    patch_path: str
    row: int
    col: int
    width: int
    height: int


def _elevation_lookup(elev_lookup: dict[tuple[int, int], float] | None,
                      lat: float, lon: float) -> float:
    """Nearest-bin lookup from ElevationProfile (keys are 1-degree lat/lon bins)."""
    if not elev_lookup or not np.isfinite(lat) or not np.isfinite(lon):
        return np.nan
    return elev_lookup.get((round(lat), round(lon)), np.nan)


def generate_patches_for_record(
    record: ImageRecord,
    out_dir: Path,
    *,
    patch_size: int = 256,
    stride: int = 128,
    max_patches: int = 150,
    min_std: float = 4.0,
    resize_long_side: int = 2048,
    elev_lookup: dict[tuple[int, int], float] | None = None,
    processed_dir: Path | None = None,
    preprocess_fn=None,
    sensor_tag: str = "",
    flat_layout: bool = True,
) -> list[PatchMeta]:
    """Extract patches from one image record; returns metadata list.

    If the primary binary is missing (IIRS), falls back to the browse PNG.
    With flat_layout=True all PNGs go directly into out_dir and the sensor is
    encoded in the patch id.
    """
    img = None
    used_fallback = False
    if record.binary_present:
        img = load_image_any(record, resize_long_side)
    if img is None:
        img = load_browse_png(record, resize_long_side)
        used_fallback = img is not None
    if img is None:
        return []

    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        # upscale thin browse strips so they still contribute patches
        scale = max(patch_size / h, patch_size / w)
        img = cv2.resize(img, (max(patch_size, int(w * scale)),
                               max(patch_size, int(h * scale))),
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    sensor_tag = sensor_tag or record.sensor.replace(" ", "")
    stem = record.image_id.rsplit(".", 1)[0][:44]
    patch_out = out_dir if flat_layout else out_dir / sensor_tag
    patch_out.mkdir(parents=True, exist_ok=True)

    rows = list(range(0, max(h - patch_size, 0) + 1, stride))
    cols = list(range(0, max(w - patch_size, 0) + 1, stride))
    if rows and rows[-1] != h - patch_size and h >= patch_size:
        rows.append(h - patch_size)
    if cols and cols[-1] != w - patch_size and w >= patch_size:
        cols.append(w - patch_size)

    grid = [(r, c) for r in rows for c in cols]
    if len(grid) > max_patches:
        idx = np.linspace(0, len(grid) - 1, max_patches).astype(int)
        grid = [grid[i] for i in idx]

    metas: list[PatchMeta] = []
    for r, c in grid:
        patch = img[r:r + patch_size, c:c + patch_size]
        if patch.std() < min_std:
            continue
        # fractional geo position: linear interp of patch center across image extent
        frac_y = (r + patch_size / 2) / h
        frac_x = (c + patch_size / 2) / w
        lat = record.latitude
        lon = record.longitude
        if np.isfinite(lat) and np.isfinite(lon):
            # spread patches +-0.25 deg around the image center (coarse but consistent)
            lat = lat + (0.5 - frac_y) * 0.5
            lon = (lon + (frac_x - 0.5) * 0.5) % 360.0
        patch_id = f"{sensor_tag}_{stem}_{r:05d}_{c:05d}"
        fname = f"{patch_id}.png"
        fpath = patch_out / fname
        if preprocess_fn is not None:
            patch = preprocess_fn(patch)
            if processed_dir is not None:
                pdir = processed_dir / sensor_tag
                pdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(pdir / fname), patch)
        cv2.imwrite(str(fpath), patch)
        metas.append(PatchMeta(
            patch_id=patch_id, source_image=record.image_id,
            dataset_name=record.sensor, latitude=round(float(lat), 6),
            longitude=round(float(lon), 6),
            sun_angle=(record.sun_elevation if np.isfinite(record.sun_elevation) else np.nan),
            elevation=_elevation_lookup(elev_lookup, lat, lon),
            patch_path=str(fpath), row=r, col=c, width=patch_size, height=patch_size,
        ))
    if used_fallback:
        record.notes += "patches from browse PNG fallback; "
    return metas


def build_elevation_lookup(elevation_csv: Path, radius_bins: int = 2) -> dict[tuple[int, int], float]:
    """Map 1-degree lat/lon bins -> elevation, dilated by radius_bins to cover gaps."""
    import csv as _csv
    lookup: dict[tuple[int, int], float] = {}
    if not elevation_csv.exists():
        return lookup
    with open(elevation_csv, newline="", encoding="utf-8") as fh:
        reader = _csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                lon, lat = float(row[0]), float(row[1])
            except ValueError:
                continue
            v = row[3].strip()
            if not v:
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            lookup[(round(lat), round(lon))] = val
    if radius_bins > 0:
        dilated: dict[tuple[int, int], float] = {}
        for (la, lo), v in lookup.items():
            for dla in range(-radius_bins, radius_bins + 1):
                for dlo in range(-radius_bins, radius_bins + 1):
                    key = (la + dla, lo + dlo)
                    dilated.setdefault(key, v)
        return dilated
    return lookup


def write_patch_index(metas: list[PatchMeta], path: Path) -> Path:
    rows = []
    for m in metas:
        d = asdict(m)
        rows.append({k: d.get(k) for k in PATCH_FIELDS})
    return save_rows_csv(rows, path)
