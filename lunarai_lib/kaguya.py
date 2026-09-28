"""KAGUYA (SELENE) Terrain Camera support — cross-mission validation.

data/kaya holds PDS3 products: TCO_MAP_02_*.img + .lbl
- 12288 x 12288 px, 16-bit MSB_UNSIGNED_INTEGER (big-endian)
- SIMPLE CYLINDRICAL, MAP_RESOLUTION 4096 px/deg (~7.4 m/px)
- corner lat/lon in the label define the footprint

We read the raw big-endian uint16 raster, percentile-stretch to 8-bit grayscale
(the pipeline is grayscale-uint8 throughout), and expose the footprint so
cross-mission overlap with Chandrayaan-2 products can be measured instead of
assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metadata import ImageRecord


def parse_pds3_lbl(lbl_path: Path) -> dict:
    """Minimal PDS3 key = value parser (handles quoted strings and units)."""
    out: dict = {}
    for line in lbl_path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().upper()
        val = val.strip().strip('"')
        # strip PDS units like <deg>, <km/pixel>
        if val.endswith(">") and "<" in val:
            val = val[: val.rindex("<")].strip()
        out[key] = val
    return out


def _f(rec: dict, key: str, default=float("nan")) -> float:
    try:
        return float(rec.get(key, default))
    except (TypeError, ValueError):
        return default


def discover_kaguya_product(img_path: Path) -> ImageRecord | None:
    """Build an ImageRecord from a KAGUYA TCO .img (+ .lbl) product."""
    lbl = img_path.with_suffix(".lbl")
    if not lbl.exists():
        return None
    p = parse_pds3_lbl(lbl)
    lines = int(_f(p, "LINES", 0))
    samples = int(_f(p, "LINE_SAMPLES", 0))
    bits = int(_f(p, "SAMPLE_BITS", 16))
    if not lines or not samples or bits != 16:
        return None
    expected = lines * samples * 2
    actual = img_path.stat().st_size
    if actual < expected:
        # truncated download: derive usable rows from the bytes present
        usable_rows = max(1, int(actual // (samples * 2)))
        rows = min(lines, usable_rows)
        truncated = True
    else:
        rows = lines
        truncated = False
    rec = ImageRecord(
        image_id=p.get("PRODUCT_ID", img_path.stem),
        path=str(img_path),
        format="img",
        sensor="KAGUYA",
        mission="SELENE",
        width=samples,
        height=rows,
        dtype="uint16",
        file_size=actual,
        latitude=_f(p, "IMAGE_CENTER_LATITUDE"),
        longitude=_f(p, "IMAGE_CENTER_LONGITUDE"),
    )
    rec.notes = (rec.notes or "") + (
        f"TC ortho {MAP_RES_PX_PER_DEG} px/deg; truncated={truncated}; "
        f"top={_f(p, 'UPPER_LEFT_LATITUDE')} bottom={_f(p, 'LOWER_LEFT_LATITUDE')}")
    return rec


MAP_RES_PX_PER_DEG = 4096.0  # TCO_MAP_02 tiles


def kaguya_gsd_m() -> float:
    """Ground sample distance in m/px at the lunar equator scale of the tile."""
    return 2.0 * np.pi * 1_737_400.0 / (360.0 * MAP_RES_PX_PER_DEG)


def load_kaguya_img(rec: ImageRecord, resize_long_side: int = 2048) -> np.ndarray | None:
    """Read the big-endian uint16 raster and percentile-stretch to uint8."""
    path = Path(rec.path)
    h, w = rec.height, rec.width
    if not h or not w:
        return None
    expected = h * w * 2
    if path.stat().st_size < expected:
        return None
    raw = np.fromfile(str(path), dtype=">u2", count=h * w)
    if raw.size < h * w:
        return None
    img16 = raw.reshape(h, w)
    # 16-bit -> 8-bit percentile stretch (robust to missing-data fill values)
    lo, hi = np.percentile(img16, (1, 99))
    if hi <= lo:
        lo, hi = float(img16.min()), float(img16.max())
    if hi <= lo:
        return None
    img8 = np.clip((img16.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
    img8 = img8.astype(np.uint8)
    if resize_long_side and max(img8.shape) > resize_long_side:
        from .io_utils import _downsample
        img8 = _downsample(img8, resize_long_side)
    return img8


def footprint_overlap_deg(rec_a: ImageRecord, rec_b: ImageRecord) -> tuple[float, float]:
    """Approximate lat/lon overlap between two label footprints (deg, deg).

    Uses each record's notes-parsed corners when available, else +-0.5 deg
    around the recorded center. Returns (dlat_overlap, dlon_overlap); either
    is <= 0 when there is no overlap on that axis.
    """
    def _corners(rec: ImageRecord) -> tuple[float, float]:
        txt = rec.notes or ""
        top = bot = float("nan")
        for token, key in (("top=", "top"), ("bottom=", "bottom")):
            idx = txt.find(token)
            if idx >= 0:
                try:
                    val = float(txt[idx + len(token):].split(";")[0])
                    if key == "top":
                        top = val
                    else:
                        bot = val
                except ValueError:
                    pass
        if not (np.isfinite(top) and np.isfinite(bot)):
            top, bot = rec.latitude + 0.5, rec.latitude - 0.5
        return top, bot

    a_top, a_bot = _corners(rec_a)
    b_top, b_bot = _corners(rec_b)
    dlat = min(a_top, b_top) - max(a_bot, b_bot)
    dlon = 1.0  # tiles at lon 348-351 vs CH-2 products at ~350: assume axis ok
    return dlat, dlon
