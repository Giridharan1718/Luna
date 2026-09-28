"""Read-only, fault-tolerant access to LunarAI's generated artifacts.

The Streamlit dashboard is a *view* over artifacts produced by ``scripts/`` and
``notebooks/``.  Every getter in this module returns ``None`` / an empty
container when an artifact is missing or unreadable, so an absent file degrades
into a caption in the UI instead of an exception.

Nothing here writes to disk and nothing here loads a model: the heavy live
inference path lives in :mod:`lunarai_lib.live`.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, load_config

# --------------------------------------------------------------------------- #
# artifact roots
# --------------------------------------------------------------------------- #
ROOTS: dict[str, str] = {
    "reports": "reports",
    "splits": "splits",
    "upgrade": "sih_upgrade",
    "complete": "sih_complete",
    "metrics": "metrics",
    "patches": "patches",
    "retrieval": "retrieval",
    "matching": "matching",
    "registration": "registration",
    "dataset": "dataset_analysis",
    "visualizations": "visualizations",
    "exports": "exports",
    "processed": "processed",
    "embeddings": "embeddings",
    # Additional mappings for files in different locations
    "subpixel": "reports",  # subpixel_accuracy_report.csv is in reports
    "validation": "reports",  # multi_modal_validation.csv is in reports
}

# deliverable files surfaced on the Reports page: (display group, root key, name)
REPORT_GROUPS: list[tuple[str, str, str]] = [
    ("Summary", "reports", "final_sih_results.md"),
    ("Summary", "reports", "final_results_summary.md"),
    ("Summary", "reports", "FINAL_SIH_AUDIT.md"),
    ("Summary", "reports", "final_judges_report.pdf"),
    ("Validation report", "reports", "multimodal_validation_report.md"),
    ("Validation report", "reports", "sun_angle_validation_report.md"),
    ("Validation report", "reports", "scale_validation_report.md"),
    ("Validation report", "reports", "subpixel_accuracy_report.md"),
    ("Validation report", "reports", "retrieval_validation_report.md"),
    ("Validation report", "reports", "uniform_distribution_report.md"),
    ("Validation report", "reports", "benchmark_comparison_report.md"),
    ("Validation report", "reports", "ablation_study_report.md"),
    ("Metrics", "reports", "final_metrics.json"),
    ("Metrics", "reports", "extended_metrics.json"),
    ("Metrics", "reports", "final_metrics.csv"),
    ("Metrics", "reports", "final_presentation_tables.csv"),
    ("Upgrade report", "upgrade", "sih_upgrade_report.md"),
    ("Upgrade report", "upgrade", "sih_upgrade_metrics.json"),
    ("Exports", "exports", "pdf/LunarAI_Evaluation_Report.pdf"),
    ("Exports", "exports", "csv/evaluation_metrics.csv"),
    ("Exports", "exports", "json/evaluation_report.json"),
]


def root(key: str = "reports", cfg: Config | None = None) -> Path:
    """Resolve a short root key (see :data:`ROOTS`) to an absolute directory."""
    cfg = cfg or load_config()
    sub = ROOTS.get(key)
    if sub is None:
        path = Path(key)
        return path if path.is_absolute() else cfg.OUTPUTS_ROOT / path
    return cfg.OUTPUTS_ROOT / sub


def resolve(name: str, key: str = "reports", cfg: Config | None = None) -> Path:
    """Resolve ``name`` under a root key.  ``name`` may itself be nested."""
    p = Path(name)
    return p if p.is_absolute() else root(key, cfg) / p


def exists(name: str, key: str = "reports", cfg: Config | None = None) -> bool:
    try:
        p = resolve(name, key, cfg)
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def size_mb(path: str | Path) -> float:
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def text(name: str, key: str = "reports", cfg: Config | None = None,
         max_chars: int | None = None) -> str | None:
    p = resolve(name, key, cfg)
    try:
        body = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return body[:max_chars] if max_chars else body


def js(name: str, key: str = "reports", cfg: Config | None = None) -> Any | None:
    """Load a JSON artifact, or ``None`` when missing/corrupt."""
    p = resolve(name, key, cfg)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def table(name: str, key: str = "reports", cfg: Config | None = None) -> pd.DataFrame | None:
    """Load a CSV artifact, tolerating ``#`` comment rows and empty files.
    
    Tries multiple locations if the file is not found in the default location.
    """
    # Try default location first
    p = resolve(name, key, cfg)
    if p.is_file() and p.stat().st_size > 0:
        try:
            df = pd.read_csv(p, comment="#")
            return df if len(df.columns) else None
        except Exception:  # noqa: BLE001 - pandas raises many types on odd CSVs
            try:
                df = pd.read_csv(p, comment="#", on_bad_lines="skip")
                return df if len(df.columns) else None
            except Exception:  # noqa: BLE001
                return None
    
    # If not found, try other common locations
    cfg = cfg or load_config()
    alternative_roots = ["reports", "sih_complete", "sih_upgrade", "matching", "registration", "retrieval"]
    for alt_key in alternative_roots:
        if alt_key == key:
            continue
        p = resolve(name, alt_key, cfg)
        if p.is_file() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p, comment="#")
                return df if len(df.columns) else None
            except Exception:  # noqa: BLE001
                try:
                    df = pd.read_csv(p, comment="#", on_bad_lines="skip")
                    return df if len(df.columns) else None
                except Exception:  # noqa: BLE001
                    return None
    
    return None


def pngs(key: str = "visualizations", pattern: str = "*.png",
         cfg: Config | None = None) -> list[Path]:
    d = root(key, cfg)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(pattern) if p.is_file() and p.stat().st_size > 0)


def report_inventory(cfg: Config | None = None) -> list[dict[str, Any]]:
    """Every listed deliverable that actually exists on disk, with its size."""
    cfg = cfg or load_config()
    rows: list[dict[str, Any]] = []
    for group, key, name in REPORT_GROUPS:
        p = resolve(name, key, cfg)
        if not p.is_file() or p.stat().st_size == 0:
            continue
        rows.append({"group": group, "file": name, "path": p,
                     "size_mb": size_mb(p), "suffix": p.suffix.lower()})
    return rows


# --------------------------------------------------------------------------- #
# catalogue helpers
# --------------------------------------------------------------------------- #
def patch_index(cfg: Config | None = None) -> pd.DataFrame | None:
    return table("patch_index.csv", "patches", cfg)


# the checkpoint is ~45 MB, so read its metadata once per process
_CKPT_EPOCH: dict[str, int | None] = {}


def checkpoint_epoch(cfg: Config | None = None) -> int | None:
    """Epoch actually stored inside ``models/lunadna.pt`` (the deployed weights).

    Reports are snapshots and go stale: ``final_metrics.json`` still says epoch 1
    while the checkpoint on disk is epoch 13.  The weights win.
    """
    cfg = cfg or load_config()
    key = str(cfg.MODELS_DIR / "lunadna.pt")
    if key in _CKPT_EPOCH:
        return _CKPT_EPOCH[key]
    value: int | None = None
    try:
        import torch  # local import: keep module import light

        ckpt = torch.load(key, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            value = ckpt.get("epoch", ckpt.get("best_epoch"))
    except Exception:  # noqa: BLE001
        value = None
    _CKPT_EPOCH[key] = value
    return value


def sensor_patch_counts(cfg: Config | None = None) -> dict[str, int]:
    df = patch_index(cfg)
    if df is None or "dataset_name" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df.dataset_name.value_counts().items()}


def _median(df: pd.DataFrame | None, col: str) -> float | None:
    if df is None or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.median()) if len(vals) else None


def _share_below(df: pd.DataFrame | None, col: str, thr: float = 1.0) -> float | None:
    """Fraction of numeric ``col`` values strictly below ``thr``."""
    if df is None or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float((vals < thr).mean()) if len(vals) else None


def _affirmative_share(df: pd.DataFrame | None, col: str) -> float | None:
    """Fraction of *text* verdict cells that are affirmative (``MET``/``yes``…).

    ``subpixel_accuracy_report.csv`` stores its per-pair target verdict as the
    string ``MET``, so a numeric mean would either fail or silently mislead.
    """
    if df is None or col not in df.columns:
        return None
    vals = df[col].dropna().astype(str).str.strip().str.lower()
    yes = {"met", "true", "yes", "y", "1", "pass", "passed"}
    no = {"not met", "false", "no", "n", "0", "fail", "failed"}
    known = vals[vals.isin(yes | no)]
    return float(known.isin(yes).mean()) if len(known) else None


def headline(cfg: Config | None = None) -> dict[str, Any]:
    """Collect the real, evidence-backed headline numbers for the Dashboard.

    Anything unavailable stays ``None`` and the UI prints ``-``.
    """
    cfg = cfg or load_config()
    out: dict[str, Any] = {
        "images": None, "counts": {}, "quality": None, "patches": None,
        "runs": None, "matcher": None, "epoch": None, "index_ntotal": None,
        "embed_dim": None, "index_mb": None, "rmse_after_ecc_px": None,
        "share_err_lt_1px": None, "target_met": None, "subpixel_pairs": None,
        "localization_m": None, "chance_m": None,
        "localization_gain": None, "within_1km": None, "sensor_patches": {},
        "split_counts": {}, "elevation_coverage": None,
        "pca_best": None, "robust_cases": None, "robust_ok": None,
        "runtime_s": None, "sun_bins_reported": None, "cross_mission_ok": None,
    }

    ds = js("dataset_report.json", "dataset", cfg) or {}
    out["images"] = ds.get("total_images")
    out["counts"] = ds.get("counts") or {}
    out["quality"] = ds.get("dataset_quality_score")

    patches = js("patch_generation_report.json", "metrics", cfg) or {}
    out["patches"] = patches.get("total_patches")
    out["sensor_patches"] = sensor_patch_counts(cfg)
    if out["patches"] is None and out["sensor_patches"]:
        out["patches"] = sum(out["sensor_patches"].values())

    fm = js("final_metrics.json", "reports", cfg) or {}
    out["runs"] = fm.get("pipeline_runs")
    out["matcher"] = fm.get("matcher_backend")
    out["epoch"] = checkpoint_epoch(cfg)
    if out["epoch"] is None:
        out["epoch"] = fm.get("lunadna_checkpoint_epoch") or fm.get("checkpoint_epoch")
    out["runtime_s"] = fm.get("validation_runtime_s")

    try:
        embs = np.load(cfg.DATABASE_DIR / "embeddings.npy", mmap_mode="r")
        out["index_ntotal"], out["embed_dim"] = int(embs.shape[0]), int(embs.shape[1])
    except Exception:  # noqa: BLE001
        pass
    out["index_mb"] = round(size_mb(cfg.DATABASE_DIR / "faiss_index.bin"), 2) or None

    sub = table("subpixel_accuracy_report.csv", "reports", cfg)
    out["rmse_after_ecc_px"] = _median(sub, "rmse_after_ecc_px")
    out["share_err_lt_1px"] = _share_below(sub, "rmse_after_ecc_px", 1.0)
    out["target_met"] = _affirmative_share(sub, "target_lt_1px")
    out["subpixel_pairs"] = None if sub is None else int(len(sub))

    ext = js("extended_metrics.json", "reports", cfg) or {}
    loc = ext.get("localization_summary") or {}
    out["localization_m"] = loc.get("median_error_m")
    out["chance_m"] = loc.get("chance_median_m")
    out["localization_gain"] = loc.get("localization_gain")
    out["within_1km"] = loc.get("within_1km")
    out["cross_mission_ok"] = loc.get("n_queries") or loc.get("queries")

    up = js("sih_upgrade_metrics.json", "upgrade", cfg) or {}
    meta = (up.get("splits") or {}).get("meta") or {}
    out["split_counts"] = meta.get("patch_counts") or {}
    out["elevation_coverage"] = (up.get("elevation") or {}).get("coverage")
    rows = ((up.get("pca") or {}).get("rows") or [])
    if rows:
        best = min(rows, key=lambda r: r.get("recall@1") or 0) if len(rows) > 1 else rows[0]
        out["pca_best"] = best
    sun_bins = up.get("sun_bins")
    if isinstance(sun_bins, list):
        out["sun_bins_reported"] = sum(
            1 for r in sun_bins if str(r.get("status", "")).lower() in {"ok", "solved"})
    elif isinstance(sun_bins, dict):
        rows_sb = sun_bins.get("rows") or []
        out["sun_bins_reported"] = sum(
            1 for r in rows_sb if str(r.get("status", "")).lower() in {"ok", "solved"})

    rob = table("robustness_validation.csv", "complete", cfg)
    if rob is not None:
        out["robust_cases"] = int(len(rob))
        if "success" in rob.columns:
            out["robust_ok"] = int(pd.to_numeric(rob["success"], errors="coerce")
                                   .fillna(0).sum())

    return out


def db_rows(query: str, cfg: Config | None = None) -> list[dict[str, Any]]:
    """Run a read-only query against ``database/lunarai.db`` (empty on failure)."""
    cfg = cfg or load_config()
    db = cfg.DATABASE_DIR / "lunarai.db"
    if not db.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in con.execute(query).fetchall()]
        finally:
            con.close()
        return rows
    except Exception:  # noqa: BLE001
        return []
