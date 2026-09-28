"""IO helpers: PDS4 binary readers, image loading, report saving."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .metadata import ImageRecord

ENVI_DTYPE = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
              5: np.float64, 12: np.uint16, 13: np.uint32}





def load_image_any(record: ImageRecord, resize_long_side: int = 2048) -> np.ndarray | None:
    """Load any discovered image to grayscale uint8, downsampled if huge.

    IIRS records have no binary: returns None (caller uses browse PNG fallback).
    """
    path = Path(record.path)
    try:
        if path.suffix.lower() == ".img":
            h, w = record.height, record.width
            if not h or not w:
                return None
            expected = h * w
            raw = np.fromfile(str(path), dtype=np.uint8, count=expected)
            if raw.size < expected:
                return None
            img = raw.reshape(h, w)
            return _downsample(img, resize_long_side)
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return _downsample(img, resize_long_side)
    except Exception:
        return None
    return None


def load_browse_png(record: ImageRecord, resize_long_side: int = 2048) -> np.ndarray | None:
    """Search for the product's browse PNG, walking up from the data file to the
    product folder root (IIRS keeps browse under <product>/browse/raw/<date>/)."""
    start = Path(record.path).parent
    for folder in [start, *list(start.parents)[:4]]:
        for png in sorted(folder.glob("*_brw_*.png")):
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return _downsample(img, resize_long_side)
        # one recursive scan at the product root, then stop climbing
        if record.image_id[:20] in folder.name:
            for png in sorted(folder.rglob("*_brw_*.png")):
                img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return _downsample(img, resize_long_side)
            break
    return None


def _downsample(img: np.ndarray, long_side: int) -> np.ndarray:
    if long_side <= 0:
        return img
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def image_stats(img: np.ndarray) -> dict[str, float]:
    f = img.astype(np.float32)
    return {
        "mean": float(f.mean()), "std": float(f.std()),
        "min": float(f.min()), "max": float(f.max()),
        "p1": float(np.percentile(f, 1)), "p99": float(np.percentile(f, 99)),
    }


def save_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_rows_csv(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    """Write list-of-dicts rows to CSV, unioning keys across rows.

    Rows in pipeline reports are heterogeneous (e.g. a row gains `H_det` only when a
    homography was found), so the field set is the union over all rows and missing
    keys are written as empty cells.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path
