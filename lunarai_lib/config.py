"""LunarAI configuration loader.

Loads configs/config.yaml, resolves absolute paths, and exposes convenient
directory attributes that are created on demand.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class Config:
    """Attribute-style access to the YAML config with resolved paths."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self._raw: dict[str, Any] = yaml.safe_load(fh) or {}

        p = self._raw.get("paths", {})
        self.PROJECT_ROOT = PROJECT_ROOT
        self.DATA_ROOT = _resolve(p.get("data_root", PROJECT_ROOT / "data"))
        self.OUTPUTS_ROOT = _resolve(p.get("outputs_root", PROJECT_ROOT / "outputs"))
        self.MODELS_DIR = _resolve(p.get("models_dir", PROJECT_ROOT / "models"))
        self.DATABASE_DIR = _resolve(p.get("database_dir", PROJECT_ROOT / "database"))
        self.EXPERIMENTS_DIR = _resolve(p.get("experiments_dir", PROJECT_ROOT / "experiments"))
        self.LOGS_DIR = _resolve(p.get("logs_dir", PROJECT_ROOT / "logs"))

        # Frequently used output subdirectories
        self.DATASET_ANALYSIS_DIR = self.OUTPUTS_ROOT / "dataset_analysis"
        self.PATCHES_DIR = self.OUTPUTS_ROOT / "patches"
        self.PROCESSED_DIR = self.OUTPUTS_ROOT / "processed"
        self.EMBEDDINGS_DIR = self.OUTPUTS_ROOT / "embeddings"
        self.RETRIEVAL_DIR = self.OUTPUTS_ROOT / "retrieval"
        self.MATCHING_DIR = self.OUTPUTS_ROOT / "matching"
        self.REGISTRATION_DIR = self.OUTPUTS_ROOT / "registration"
        self.METRICS_DIR = self.OUTPUTS_ROOT / "metrics"
        self.VISUALIZATIONS_DIR = self.OUTPUTS_ROOT / "visualizations"
        self.EXPORTS_DIR = self.OUTPUTS_ROOT / "exports"
        self.PDF_DIR = self.EXPORTS_DIR / "pdf"
        self.CSV_DIR = self.EXPORTS_DIR / "csv"
        self.JSON_DIR = self.EXPORTS_DIR / "json"
        self.IMAGES_EXPORT_DIR = self.EXPORTS_DIR / "images"

        ds = self._raw.get("dataset", {})
        self.SENSORS: list[str] = ds.get("sensors", ["ohrc", "tmc", "iirs", "lro"])
        self.PATCH_SIZE: int = int(ds.get("patch_size", 256))
        self.STRIDE: int = int(ds.get("stride", 128))
        self.MAX_PATCHES_PER_IMAGE: int = int(ds.get("max_patches_per_image", 150))
        self.MIN_STD: float = float(ds.get("min_std", 4.0))
        self.RESIZE_LONG_SIDE: int = int(ds.get("resize_long_side", 2048))
        self.PROCESSED_FORMAT: str = str(ds.get("processed_format", "png"))

        self.PRE = self._raw.get("preprocessing", {})
        self.LUNADNA = self._raw.get("lunadna", {})
        self.FAISS = self._raw.get("faiss", {})
        self.MATCHING = self._raw.get("matching", {})
        self.EVAL = self._raw.get("evaluation", {})

    def ensure_dirs(self) -> None:
        for d in [
            self.OUTPUTS_ROOT, self.MODELS_DIR, self.DATABASE_DIR,
            self.EXPERIMENTS_DIR, self.LOGS_DIR, self.DATASET_ANALYSIS_DIR,
            self.PATCHES_DIR, self.PROCESSED_DIR, self.EMBEDDINGS_DIR,
            self.RETRIEVAL_DIR, self.MATCHING_DIR, self.REGISTRATION_DIR,
            self.METRICS_DIR, self.VISUALIZATIONS_DIR, self.EXPORTS_DIR,
            self.PDF_DIR, self.CSV_DIR, self.JSON_DIR, self.IMAGES_EXPORT_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

    # dict-like passthrough for nested sections
    def section(self, key: str) -> dict[str, Any]:
        return self._raw.get(key, {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(data={self.DATA_ROOT}, outputs={self.OUTPUTS_ROOT})"


def load_config(config_path: str | Path | None = None) -> Config:
    cfg = Config(config_path)
    cfg.ensure_dirs()
    return cfg
