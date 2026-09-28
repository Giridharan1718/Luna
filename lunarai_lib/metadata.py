"""Discovery and metadata parsing for Chandrayaan-2 PDS4 products and LRO exports.

Handles:
- OHRC/TMC .img + .xml (PDS4 label) pairs -> shape, dtype, sun angle, corner lat/lon
- IIRS ENVI .hdr (binary often absent) -> shape/bands from header, flagged binary-missing
- LRO quickmap PNG/VRT -> reference imagery with coarse metadata
- ElevationProfile.csv / SunAngle.csv loading
"""
from __future__ import annotations

import csv
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PDS_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1", "isda": "https://isda.issdc.gov.in/pds4/isda/v1"}

SENSOR_ALIASES = {
    "ohrc": "OHRC", "ohr": "OHRC",
    "tmc": "TMC-2", "tmc2": "TMC-2",
    "iirs": "IIRS", "iir": "IIRS",
    "lro": "LRO NAC", "lroc": "LRO NAC",
    "kaya": "KAGUYA", "kaguya": "KAGUYA", "se": "KAGUYA", "selene": "KAGUYA",
}


def sensor_from_path(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if low in SENSOR_ALIASES:
            return SENSOR_ALIASES[low]
    return "UNKNOWN"


def mission_from_sensor(sensor: str) -> str:
    if sensor.startswith("LRO"):
        return "LRO"
    elif sensor == "KAGUYA":
        return "SELENE"
    else:
        return "Chandrayaan-2"


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _txt(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    return node.text.strip()


@dataclass
class ImageRecord:
    image_id: str
    path: str
    format: str                 # img / png / hdr-only
    sensor: str
    mission: str
    width: int = 0
    height: int = 0
    dtype: str = ""
    file_size: int = 0
    md5: str = ""
    latitude: float = np.nan
    longitude: float = np.nan
    sun_elevation: float = np.nan
    sun_azimuth: float = np.nan
    pixel_resolution: float = np.nan
    acquisition_date: str = ""
    binary_present: bool = True
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id, "path": self.path, "format": self.format,
            "sensor": self.sensor, "mission": self.mission, "width": self.width,
            "height": self.height, "dtype": self.dtype, "file_size": self.file_size,
            "md5": self.md5, "latitude": self.latitude, "longitude": self.longitude,
            "sun_elevation": self.sun_elevation, "sun_azimuth": self.sun_azimuth,
            "pixel_resolution": self.pixel_resolution, "acquisition_date": self.acquisition_date,
            "binary_present": self.binary_present, "notes": self.notes,
        }


def _parse_envi_hdr(hdr_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in hdr_path.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _pid(img_path: Path) -> str:
    return img_path.name


def discover_ch2_product(folder: Path) -> ImageRecord | None:
    """Find the data binary + label for one CH-2 product folder."""
    sensor = sensor_from_path(folder)
    img_candidates = list(folder.rglob("*.img")) + list(folder.rglob("*.hdr"))
    img_candidates = [p for p in img_candidates if "_brw_" not in p.name]
    if not img_candidates:
        return None
    data_path = img_candidates[0]
    rec = ImageRecord(
        image_id=_pid(data_path), path=str(data_path), format="img",
        sensor=sensor, mission=mission_from_sensor(sensor),
        file_size=data_path.stat().st_size,
        binary_present=data_path.suffix.lower() == ".img",
    )

    xml_candidates = sorted(folder.rglob("*.xml"))
    # prefer the observational label (contains Product_Parameters / Axis_Array)
    xml_candidates.sort(key=lambda p: ("data" not in p.parts, "_brw_" in p.name))
    if xml_candidates:
        _apply_pds4_label(rec, xml_candidates[0])
    # Try PDS3 .lbl labels (for KAGUYA/SELENE)
    else:
        lbl_candidates = sorted(folder.rglob("*.lbl"))
        if lbl_candidates:
            _apply_pds3_label(rec, lbl_candidates[0])

    if data_path.suffix.lower() == ".hdr":
        hdr = _parse_envi_hdr(data_path)
        rec.width = int(hdr.get("samples", 0))
        rec.height = int(hdr.get("lines", 0))
        rec.extra["bands"] = int(hdr.get("bands", 0))
        rec.dtype = f"ENVI type {hdr.get('data_type', '?')}"
        rec.format = "hdr (binary missing)"
    return rec


def _apply_pds3_label(rec: ImageRecord, lbl_path: Path) -> None:
    """Parse PDS3 .lbl file (used by SELENE/KAGUYA)."""
    try:
        with open(lbl_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        rec.notes += "pds3 label read error; "
        return
    
    # Parse key-value pairs from PDS3 format
    lines = content.splitlines()
    metadata = {}
    for line in lines:
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            # Remove <unit> specifications
            if '<' in value:
                value = value.split('<')[0].strip()
            metadata[key] = value
    
    # Extract key metadata
    if 'UPPER_LEFT_LATITUDE' in metadata:
        try:
            ul_lat = float(metadata['UPPER_LEFT_LATITUDE'])
            ur_lat = float(metadata.get('UPPER_RIGHT_LATITUDE', ul_lat))
            ll_lat = float(metadata.get('LOWER_LEFT_LATITUDE', ul_lat))
            rec.latitude = float(np.mean([ul_lat, ur_lat, ll_lat]))
        except ValueError:
            pass
    
    if 'UPPER_LEFT_LONGITUDE' in metadata:
        try:
            ul_lon = float(metadata['UPPER_LEFT_LONGITUDE'])
            ur_lon = float(metadata.get('UPPER_RIGHT_LONGITUDE', ul_lon))
            ll_lon = float(metadata.get('LOWER_LEFT_LONGITUDE', ul_lon))
            lons = [ul_lon, ur_lon, ll_lon]
            # Handle longitude wrapping
            rec.longitude = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(lons))))))
            if rec.longitude < 0:
                rec.longitude += 360.0
        except ValueError:
            pass
    
    if 'LINES' in metadata:
        try:
            rec.height = int(metadata['LINES'])
        except ValueError:
            pass
    
    if 'LINE_SAMPLES' in metadata:
        try:
            rec.width = int(metadata['LINE_SAMPLES'])
        except ValueError:
            pass
    
    if 'SAMPLE_BITS' in metadata:
        try:
            bits = int(metadata['SAMPLE_BITS'])
            rec.dtype = f"uint{bits}"
        except ValueError:
            pass
    
    if 'PRODUCT_CREATION_TIME' in metadata:
        rec.acquisition_date = metadata['PRODUCT_CREATION_TIME']
    
    if 'MAP_RESOLUTION' in metadata:
        try:
            rec.pixel_resolution = float(metadata['MAP_RESOLUTION'])
        except ValueError:
            pass
    
    if 'INSTRUMENT_NAME' in metadata:
        rec.extra['instrument'] = metadata['INSTRUMENT_NAME']
    
    if 'TARGET_NAME' in metadata:
        rec.extra['target'] = metadata['TARGET_NAME']


def _apply_pds4_label(rec: ImageRecord, xml_path: Path) -> None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        rec.notes += "label parse error; "
        return

    def find(tag: str) -> ET.Element | None:
        return root.find(f".//{{{PDS_NS['pds']}}}{tag}")

    start = find("start_date_time")
    if start is not None and _txt(start):
        rec.acquisition_date = (_txt(start) or "").replace("T", " ").replace("Z", "")

    pp = root.find(f".//{{{PDS_NS['isda']}}}Product_Parameters")
    if pp is not None:
        sun_el = pp.find(f"{{{PDS_NS['isda']}}}sun_elevation")
        sun_az = pp.find(f"{{{PDS_NS['isda']}}}sun_azimuth")
        res = pp.find(f"{{{PDS_NS['isda']}}}pixel_resolution")
        alt = pp.find(f"{{{PDS_NS['isda']}}}spacecraft_altitude")
        if sun_el is not None and sun_el.text:
            rec.sun_elevation = float(sun_el.text)
        if sun_az is not None and sun_az.text:
            rec.sun_azimuth = float(sun_az.text)
        if res is not None and res.text:
            rec.pixel_resolution = float(res.text)
        if alt is not None and alt.text:
            rec.extra["altitude_km"] = float(alt.text)

    corner_tags = ["upper_left_latitude", "upper_right_latitude",
                   "lower_left_latitude", "lower_right_latitude"]
    corner_lons = ["upper_left_longitude", "upper_right_longitude",
                   "lower_left_longitude", "lower_right_longitude"]
    lats: list[float] = []
    lons: list[float] = []
    for group in ("Geometry_Parameters",):
        gp = root.find(f".//{{{PDS_NS['isda']}}}{group}")
        if gp is None:
            continue
        for tag in corner_tags:
            el = gp.find(f".//{{{PDS_NS['isda']}}}{tag}")
            if el is not None and el.text:
                lats.append(float(el.text))
        for tag in corner_lons:
            el = gp.find(f".//{{{PDS_NS['isda']}}}{tag}")
            if el is not None and el.text:
                lons.append(float(el.text))
    if lats:
        rec.latitude = float(np.mean(lats))
    if lons:
        # longitudes may wrap (near poles); circular mean keeps values sane
        rec.longitude = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(lons))))))
        if rec.longitude < 0:
            rec.longitude += 360.0

    fn = find("file_name")
    if fn is not None and _txt(fn):
        rec.md5 = ""
    fs = find("file_size")
    if fs is not None and _txt(fs) and rec.file_size == 0:
        try:
            rec.file_size = int(float(_txt(fs)))
        except ValueError:
            pass
    aa = root.findall(f".//{{{PDS_NS['pds']}}}Axis_Array")
    found_axes = False
    for axis in aa:
        name = axis.find(f"{{{PDS_NS['pds']}}}axis_name")
        elems = axis.find(f"{{{PDS_NS['pds']}}}elements")
        if name is None or elems is None:
            continue
        try:
            n = int(_txt(elems) or 0)
        except ValueError:
            continue
        found_axes = True
        if _txt(name) == "Line":
            rec.height = n
        elif _txt(name) == "Sample":
            rec.width = n
    if not found_axes and rec.format == "img":
        # label lacks Axis_Array: infer from byte size (1 band, UnsignedByte)
        import math
        side = int(math.isqrt(max(rec.file_size, 0)))
        if side > 0:
            rec.width = rec.height = side
            rec.notes += "dims inferred from file size; "


def load_lro_record(folder: Path) -> ImageRecord | None:
    pngs = list(folder.rglob("*.png"))
    vrts = list(folder.rglob("*.vrt"))
    if not pngs and not vrts:
        return None
    main = pngs[0] if pngs else vrts[0]
    rec = ImageRecord(
        image_id=main.name, path=str(main), format=main.suffix.lstrip("."),
        sensor="LRO NAC", mission="LRO",
        file_size=main.stat().st_size if main.exists() else 0,
    )
    if main.suffix.lower() == ".png":
        import cv2
        img = cv2.imread(str(main), cv2.IMREAD_UNCHANGED)
        if img is not None:
            rec.height, rec.width = img.shape[:2]
    return rec


def discover_all(data_root: Path) -> dict[str, list[ImageRecord]]:
    by_sensor: dict[str, list[ImageRecord]] = {s: [] for s in ["OHRC", "TMC-2", "IIRS", "LRO NAC", "KAGUYA"]}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        low = child.name.lower()
        if low in ("ohrc", "ohr"):
            for prod in sorted(child.iterdir()):
                if prod.is_dir():
                    rec = discover_ch2_product(prod)
                    if rec:
                        by_sensor["OHRC"].append(rec)
        elif low == "tmc":
            for prod in sorted(child.iterdir()):
                if prod.is_dir():
                    rec = discover_ch2_product(prod)
                    if rec:
                        by_sensor["TMC-2"].append(rec)
        elif low in ("iirs", "iir"):
            for prod in sorted(child.iterdir()):
                if prod.is_dir():
                    rec = discover_ch2_product(prod)
                    if rec:
                        by_sensor["IIRS"].append(rec)
        elif low == "lro":
            for sub in sorted(child.iterdir()):
                if sub.is_dir():
                    rec = load_lro_record(sub)
                    if rec:
                        by_sensor["LRO NAC"].append(rec)
        elif low in ("kaya", "kaguya", "se", "selene"):
            # KAGUYA has flat structure with .img and .lbl directly in directory
            img_candidates = list(child.glob("*.img"))
            for img_path in img_candidates:
                if "_brw_" not in img_path.name:
                    rec = ImageRecord(
                        image_id=_pid(img_path), path=str(img_path), format="img",
                        sensor="KAGUYA", mission="SELENE",
                        file_size=img_path.stat().st_size,
                        binary_present=True,
                    )
                    # Try to find corresponding .lbl file
                    lbl_path = img_path.with_suffix('.lbl')
                    if lbl_path.exists():
                        _apply_pds3_label(rec, lbl_path)
                    by_sensor["KAGUYA"].append(rec)
    return by_sensor


def load_sunangle_csv(path: Path) -> dict[str, Any]:
    times, elevs, azims = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                elevs.append(float(row["Elevation"]))
                azims.append(float(row["Azimuth"]))
                times.append(row["Time"])
            except (ValueError, KeyError):
                continue
    e = np.asarray(elevs, dtype=float)
    a = np.asarray(azims, dtype=float)
    return {
        "rows": len(times), "time_start": times[0] if times else None,
        "time_end": times[-1] if times else None,
        "sun_elevation": {"min": float(e.min()), "max": float(e.max()),
                          "mean": float(e.mean()), "std": float(e.std())} if len(e) else {},
        "sun_azimuth": {"min": float(a.min()), "max": float(a.max()),
                        "mean": float(a.mean()), "std": float(a.std())} if len(a) else {},
    }


def load_elevation_csv(path: Path) -> dict[str, Any]:
    lats, lons, vals = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx_lat = 1 if "Latitude" in header[1] else 0
        idx_lon = 0 if idx_lat == 1 else 1
        for row in reader:
            if len(row) < 4:
                continue
            try:
                lons.append(float(row[idx_lon]))
                lats.append(float(row[idx_lat]))
                v = row[3].strip()
                vals.append(float(v) if v else np.nan)
            except ValueError:
                continue
    v = np.asarray(vals, dtype=float)
    finite = v[np.isfinite(v)]
    return {
        "rows": len(lats),
        "columns": header,
        "latitude_range": [float(min(lats)), float(max(lats))] if lats else [],
        "longitude_range": [float(min(lons)), float(max(lons))] if lons else [],
        "elevation_m": {"min": float(finite.min()), "max": float(finite.max()),
                        "mean": float(finite.mean()), "std": float(finite.std()),
                        "n_finite": int(finite.size)} if finite.size else {},
    }
