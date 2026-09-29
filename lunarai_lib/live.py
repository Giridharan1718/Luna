"""Live inference for the dashboard: uploaded image -> retrieval -> registration.

This is the only place in the UI layer that touches the model, the FAISS index
or the matcher.  Everything is lazy: importing this module loads no weights, and
:func:`session` builds the (single, cached) :class:`ValidationSuite` on first
use, which is what the Streamlit pages wrap in ``st.cache_resource``.

Preprocessing is deliberately *not* re-implemented here: embeddings go through
``lunadna.compute_embeddings`` / ``PatchDataset`` so an uploaded image sees
exactly the training-time transform, and registration reuses
``ValidationSuite.run_pipeline`` unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from .config import Config, load_config
from .lunadna import compute_embeddings
from .validation import Pair, ValidationSuite, load_gray

SUPPORTED_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".img"}

# one suite per process; the Streamlit layer also caches on top of this
_SUITE: ValidationSuite | None = None


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def uploads_dir(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    d = cfg.OUTPUTS_ROOT / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir(cfg: Config | None = None) -> Path:
    d = uploads_dir(cfg) / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(data: bytes, filename: str, cfg: Config | None = None) -> Path:
    """Persist an uploaded file (basename only, sanitised) and return its path."""
    name = Path(filename or "upload.png").name
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    dest = uploads_dir(cfg) / safe
    dest.write_bytes(data)
    return dest


# --------------------------------------------------------------------------- #
# image handling
# --------------------------------------------------------------------------- #
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch any bit depth to 8-bit for display/matching."""
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(f.min()), float(f.max()) or 1.0
    return np.clip((f - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def read_upload(path: str | Path) -> tuple[np.ndarray | None, str]:
    """Load an uploaded image as an 8-bit array.

    Returns ``(image, note)``.  ``image`` is ``None`` with an explanation when
    the file cannot be decoded — notably a raw PDS3 ``.img`` whose label was not
    supplied, which is the honest outcome rather than a garbage array.
    """
    p = Path(path)
    if not p.is_file():
        return None, f"file not found: {p.name}"
    if p.suffix.lower() == ".img":
        return None, (
            f"`{p.name}` is a raw PDS3 image: the pixel data cannot be interpreted "
            "without its PDS label. Upload the matching `.lbl`/`.xml` (or a "
            "PNG/TIFF export), or run the pipeline against a catalog patch instead."
        )
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, f"could not decode `{p.name}` as an image."
    img = _to_uint8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, "ok"


def describe_image(img: np.ndarray, raw_dtype: str | None = None) -> dict[str, Any]:
    f = img.astype(np.float32)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "bit_depth": raw_dtype or f"{img.dtype}",
        "mean": round(float(f.mean()), 2),
        "std": round(float(f.std()), 2),
        "laplacian_var": round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 2),
        "p1": round(float(np.percentile(f, 1)), 1),
        "p99": round(float(np.percentile(f, 99)), 1),
    }


# --------------------------------------------------------------------------- #
# model / index / matcher session
# --------------------------------------------------------------------------- #
def session(cfg: Config | None = None, log: Callable[[str], None] = print,
            refresh: bool = False) -> ValidationSuite:
    """Build (once) and return the shared :class:`ValidationSuite`."""
    global _SUITE
    if _SUITE is None or refresh:
        cfg = cfg or load_config()
        _SUITE = ValidationSuite(cfg, log=log)
    return _SUITE


def session_info(cfg: Config | None = None) -> dict[str, Any]:
    """Describe the live session without building it."""
    cfg = cfg or load_config()
    info: dict[str, Any] = {
        "loaded": _SUITE is not None,
        "device": "cpu",
        "matcher": None,
        "checkpoint_epoch": None,
        "ntotal": None,
    }
    if _SUITE is not None:
        info.update(
            loaded=True,
            device=getattr(_SUITE, "device", "cpu"),
            matcher=getattr(_SUITE, "matcher_backend", None),
            checkpoint_epoch=getattr(_SUITE, "ckpt_epoch", None),
            ntotal=int(getattr(_SUITE.index, "ntotal", 0) or 0),
        )
    if info["ntotal"] is None:
        try:
            embs = np.load(cfg.DATABASE_DIR / "embeddings.npy", mmap_mode="r")
            info["ntotal"] = int(embs.shape[0])
        except Exception:  # noqa: BLE001
            pass
    return info


# --------------------------------------------------------------------------- #
# embedding + retrieval
# --------------------------------------------------------------------------- #
def embed(paths: list[str | Path], cfg: Config | None = None,
          suite: ValidationSuite | None = None) -> np.ndarray:
    """Embed images with the deployed LunaDNA checkpoint (training transform)."""
    suite = suite or session(cfg)
    return compute_embeddings(suite.model, [str(p) for p in paths],
                             device=getattr(suite, "device", "cpu"))


def stored_embedding(patch_id: str, cfg: Config | None = None,
                     suite: ValidationSuite | None = None) -> np.ndarray | None:
    """Return the already-indexed embedding of a catalog patch, if present."""
    try:
        suite = suite or session(cfg)
        idx = list(suite.mapping).index(patch_id)
        return np.asarray(suite.embeddings[idx], dtype=np.float32)[None, :]
    except Exception:  # noqa: BLE001
        return None


def catalogue_row(patch_id: str, cfg: Config | None = None) -> dict[str, Any]:
    from . import appdata

    df = appdata.patch_index(cfg)
    if df is None or "patch_id" not in df.columns:
        return {}
    hit = df[df.patch_id == patch_id]
    return {} if not len(hit) else {k: v for k, v in hit.iloc[0].to_dict().items()}


def retrieve(query: str | Path | np.ndarray, k: int = 10, cfg: Config | None = None,
             suite: ValidationSuite | None = None,
             drop_self: bool = True) -> pd.DataFrame:
    """Top-K catalog patches for a query image path or an embedding vector."""
    from . import appdata

    suite = suite or session(cfg)
    if isinstance(query, np.ndarray):
        emb = np.atleast_2d(query.astype(np.float32))
        self_path = None
    else:
        emb = embed([query], cfg=cfg, suite=suite)
        # suite.mapping stores full patch paths, so the query must be dropped
        # by exact path match (a filename-stem comparison never matches and the
        # query used to come back as a ghost rank-1 hit with empty metadata).
        self_path = str(Path(query).resolve()) if drop_self else None
    scores, idx = suite.index.search(emb, max(k + 1, k))
    idx_df = appdata.patch_index(cfg)
    meta = {}
    if idx_df is not None and "patch_id" in idx_df.columns:
        meta = idx_df.set_index("patch_id").to_dict("index")

    rows: list[dict[str, Any]] = []
    for score, i in zip(np.ravel(scores), np.ravel(idx)):
        if i < 0 or i >= len(suite.mapping):
            continue
        mapped = str(suite.mapping[i])
        # drop the query itself (works whether mapping stores ids or full paths)
        if self_path is not None and (
                Path(mapped).resolve() == self_path or Path(mapped).stem == Path(str(query)).stem):
            continue
        # mapping may hold patch ids OR full paths; resolve metadata via stem
        m = meta.get(mapped) or meta.get(Path(mapped).stem, {})
        rows.append({
            "rank": len(rows) + 1,
            "patch_id": Path(mapped).stem,
            "sensor": m.get("dataset_name", "-"),
            "similarity": round(float(score), 4),
            "latitude": m.get("latitude"),
            "longitude": m.get("longitude"),
            "sun_angle": m.get("sun_angle"),
            "patch_path": m.get("patch_path", mapped),
        })
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def register(src: str | Path | np.ndarray, ref: str | Path | np.ndarray, *,
             name: str = "live_pair", sensor_a: str = "uploaded",
             sensor_b: str = "LRO NAC", cfg: Config | None = None,
             suite: ValidationSuite | None = None, save: bool = True,
             variant: str = "full", H_gt: np.ndarray | None = None,
             extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full correspondence + registration chain on one pair.

    Returns a dict with ``result`` (the raw :meth:`run_pipeline` dict), ``H``
    (3x3, recovered from the saved ``*_H.npy`` so no change to
    ``validation.py`` is needed) and the artifact paths for the UI.
    """
    suite = suite or session(cfg)
    cfg = cfg or load_config()

    src_img = load_gray(src) if isinstance(src, (str, Path)) else src
    ref_img = load_gray(ref) if isinstance(ref, (str, Path)) else ref
    if src_img is None or ref_img is None:
        return {"result": {"status": "error: unreadable input image",
                           "pair": name, "confidence": 0.0},
                "H": None, "images": {}, "run_dir": None, "is_gt": False}

    pair = Pair(name=name, protocol="uploaded", sensor_a=sensor_a, sensor_b=sensor_b,
                src=src_img, ref=ref_img, H_gt=H_gt, meta=dict(extra_meta or {}))
    run_dir = runs_dir(cfg) if save else None
    out = suite.run_pipeline(pair, variant=variant, save_dir=run_dir, tag=name)

    H = None
    images: dict[str, Path] = {}
    if run_dir is not None:
        safe = f"{name}".replace("/", "_")[:60]
        h_file = run_dir / f"{safe}_H.npy"
        if h_file.is_file():
            try:
                H = np.load(h_file)
            except Exception:  # noqa: BLE001
                H = None
        for suffix in ("src", "ref", "registered", "gt_warp"):
            f = run_dir / f"{safe}_{suffix}.png"
            if f.is_file():
                images[suffix] = f
    return {"result": out, "H": H, "images": images, "run_dir": run_dir,
            "is_gt": H_gt is not None}
