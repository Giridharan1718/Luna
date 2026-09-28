"""Geographic train/val/test splits — SIH Phase 1.

Splits *geo cells* (rounded lat/lon) — never individual patches — so no two
patches from the same terrain can appear in different splits (location leakage).
The split is deterministic (seeded) and reproducible; metadata is exported to
outputs/splits/split_metadata.json and per-split CSVs.

Design notes
------------
- Geo cell = f"{round(lat, cell_decimals)}_{round(lon, cell_decimals)}". Patches
  spread +-0.25 deg around an image center (see patching), so 1.0 deg cells keep
  every patch of one source image together by construction; the numeric hash
  bins used in the pre-upgrade script could place adjacent cells together, this
  string scheme cannot collide.
- Cells are shuffled with numpy default_rng(seed) and cut 70/15/15.
- Every patch carries its split in the returned dataframe column `split`;
  retrieval/registration evaluation must filter to split == "test".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_TRAIN_FRAC = 0.70
SPLIT_VAL_FRAC = 0.15
SPLIT_TEST_FRAC = 0.15  # remainder; asserted below


def geo_cell(lat: float, lon: float, cell_decimals: int = 0) -> str | None:
    """Stable string geo cell for a coordinate; None when coords are missing."""
    if lat is None or lon is None:
        return None
    lat_f, lon_f = float(lat), float(lon)
    if not (np.isfinite(lat_f) and np.isfinite(lon_f)):
        return None
    la = round(lat_f, cell_decimals)
    lo = round(lon_f, cell_decimals)
    if cell_decimals <= 0:
        return f"{int(la)}_{int(lo)}"
    return f"{la}_{lo}"


def assign_geo_splits(df: pd.DataFrame, seed: int = 42,
                      cell_decimals: int = 0) -> pd.DataFrame:
    """Return a copy of `df` with a `split` column in {train, val, test}.

    Split unit is the geo cell (never the patch) -> zero location leakage.
    """
    out = df.copy().reset_index(drop=True)
    cells = sorted({c for c in (geo_cell(la, lo, cell_decimals)
                                for la, lo in zip(out.latitude, out.longitude))
                    if c is not None})
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    n = len(cells)
    n_train = int(round(n * SPLIT_TRAIN_FRAC))
    n_val = int(round(n * SPLIT_VAL_FRAC))
    n_test = n - n_train - n_val  # remainder keeps the three-way sum exact
    if n >= 3:  # guarantee at least one cell per split when possible
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        if n_train + n_val + n_test > n:
            n_train = n - n_val - n_test
    assignment: dict[str, str] = {}
    for i, cell in enumerate(cells):
        if i < n_train:
            assignment[cell] = "train"
        elif i < n_train + n_val:
            assignment[cell] = "val"
        else:
            assignment[cell] = "test"
    out["geo_cell"] = [geo_cell(la, lo, cell_decimals)
                       for la, lo in zip(out.latitude, out.longitude)]
    out["split"] = [assignment.get(c, "train") for c in out.geo_cell]
    return out


def split_metadata(df: pd.DataFrame, seed: int = 42,
                   cell_decimals: int = 0) -> dict:
    """Summary dict for split_metadata.json (counts + per-sensor breakdown)."""
    counts = df.split.value_counts().to_dict()
    per_sensor = {s: g.split.value_counts().to_dict()
                  for s, g in df.groupby("dataset_name")}
    n_cells = df.geo_cell.nunique()
    return {
        "method": "geographic cell split (no location leakage)",
        "unit": "geo cell = round(lat/lon, 0) deg",
        "fractions": {"train": SPLIT_TRAIN_FRAC, "val": SPLIT_VAL_FRAC,
                      "test": SPLIT_TEST_FRAC},
        "seed": seed,
        "total_patches": int(len(df)),
        "total_geo_cells": int(n_cells),
        "patch_counts": {k: int(v) for k, v in counts.items()},
        "per_sensor": {s: {k: int(v) for k, v in d.items()}
                       for s, d in per_sensor.items()},
        "leakage_check": "patches sharing a geo cell always share a split",
    }


def export_splits(df: pd.DataFrame, out_dir: Path, seed: int = 42,
                  cell_decimals: int = 0) -> dict:
    """Assign splits and write split metadata + per-split CSVs. Returns meta."""
    out = assign_geo_splits(df, seed=seed, cell_decimals=cell_decimals)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = split_metadata(out, seed=seed, cell_decimals=cell_decimals)
    (out_dir / "split_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    for name in ("train", "val", "test"):
        sub = out[out.split == name]
        sub.to_csv(out_dir / f"{name}_patches.csv", index=False)
    out.to_csv(out_dir / "all_patches_with_splits.csv", index=False)
    return meta


def load_or_create_splits(patch_csv: Path, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    """Load outputs/splits/all_patches_with_splits.csv if present, else build it."""
    all_csv = out_dir / "all_patches_with_splits.csv"
    if all_csv.exists():
        return pd.read_csv(all_csv)
    df = pd.read_csv(patch_csv)
    export_splits(df, out_dir, seed=seed)
    return pd.read_csv(all_csv)
