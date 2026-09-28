"""Preprocessing: grayscale, CLAHE, histogram equalization, contrast
normalization, noise reduction, resize — per 02_TRD Module 1."""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return clahe.apply(img)


def apply_hist_eq(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


def contrast_normalize(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    out = (f - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def reduce_noise(img: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(img, ksize if ksize % 2 == 1 else ksize + 1)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def full_preprocess(img: np.ndarray, *, clahe_clip: float = 2.0, clahe_grid: int = 8,
                    denoise: str = "median", denoise_ksize: int = 3,
                    size: tuple[int, int] | None = None) -> np.ndarray:
    """Ordered pipeline: grayscale -> CLAHE -> denoise -> contrast norm -> resize."""
    g = to_grayscale(img)
    g = apply_clahe(g, clahe_clip, clahe_grid)
    g = reduce_noise(g, denoise, denoise_ksize)
    g = contrast_normalize(g)
    if size is not None and (g.shape[1], g.shape[0]) != tuple(size):
        g = resize(g, size)
    return g
