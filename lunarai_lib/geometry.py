"""Geometric verification and registration: MAGSAC++, homography, ECC,
RMSE, SSIM/NCC similarity, and match visualizations."""
from __future__ import annotations

import time

import cv2
import numpy as np


def magsac_homography(kp0: np.ndarray, kp1: np.ndarray, threshold: float = 4.0):
    """MAGSAC++ (cv2.USAC_MAGSAC) homography estimation.

    Returns (H, mask, inlier_ratio, inlier_count) or (None, ..., 0.0, 0) on failure.
    """
    if len(kp0) < 4:
        return None, None, 0.0, 0
    src = np.ascontiguousarray(kp0.reshape(-1, 1, 2).astype(np.float32))
    dst = np.ascontiguousarray(kp1.reshape(-1, 1, 2).astype(np.float32))
    try:
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold,
                                     maxIters=20000, confidence=0.9995)
    except Exception:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
    if H is None or mask is None:
        return None, None, 0.0, 0
    inliers = int(mask.sum())
    return H, mask.reshape(-1), inliers / len(kp0), inliers


def ecc_refine(template: np.ndarray, moving: np.ndarray, warp: np.ndarray,
               iterations: int = 100, eps: float = 1e-4):
    """ECC sub-pixel refinement. `moving` is warped toward `template`.

    Returns (warp, ecc_corr, success).
    """
    templ = template if template.ndim == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    mov = moving if moving.ndim == 2 else cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    templ_f = templ.astype(np.float32) / 255.0
    mov_f = mov.astype(np.float32) / 255.0
    w = np.eye(3, dtype=np.float32) if warp is None else warp.astype(np.float32).copy()
    try:
        cc, w = cv2.findTransformECC(mov_f, templ_f, w, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                      iterations, eps), None, 5)
        return w, float(cc), True
    except cv2.error as exc:
        return w, 0.0, False


def warp_image(img: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(img, H, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def reprojection_rmse(kp0: np.ndarray, kp1: np.ndarray, H: np.ndarray) -> float:
    """RMSE (px) between kp1 and kp0 projected by H — registration accuracy."""
    if H is None or len(kp0) == 0:
        return float("nan")
    pts = cv2.perspectiveTransform(kp0.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float32))
    err = pts.reshape(-1, 2) - kp1.reshape(-1, 2).astype(np.float32)
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def compute_ssim_ncc(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """SSIM and NCC between two equal-size grayscale images (valid-pixel masked)."""
    from skimage.metrics import structural_similarity
    a = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    valid = (a > 0) & (b > 0)
    if valid.sum() < 100:
        return float("nan"), float("nan")
    av, bv = a[valid], b[valid]
    ssim = structural_similarity(a, b)
    avf, bvf = av.astype(np.float64), bv.astype(np.float64)
    denom = avf.std() * bvf.std()
    ncc = float(((avf - avf.mean()) * (bvf - bvf.mean())).mean() / denom) if denom > 1e-9 else 0.0
    return float(ssim), float(max(-1.0, min(1.0, ncc)))


def confidence_score(rmse: float, inlier_ratio: float, coverage: float,
                     match_count: int, ecc_score: float | None = None) -> float:
    """0-100 confidence per 09_Evaluation section 12."""
    score = 0.0
    score += 30.0 * max(0.0, 1.0 - min(rmse, 3.0) / 3.0) if np.isfinite(rmse) else 0.0
    score += 30.0 * max(0.0, min(inlier_ratio, 1.0))
    score += 25.0 * max(0.0, min(coverage, 1.0))
    score += 10.0 * min(match_count / 100.0, 1.0)
    if ecc_score is not None and np.isfinite(ecc_score):
        score += 5.0 * max(0.0, min(ecc_score, 1.0))
    else:
        score += 2.5  # ECC unavailable: neutral half-credit
    return round(score, 1)


def draw_matches(img0: np.ndarray, img1: np.ndarray, kp0: np.ndarray, kp1: np.ndarray,
                 max_draw: int = 150) -> np.ndarray:
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0
    canvas[:h1, w0:] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(kp0))[:max_draw] if len(kp0) > max_draw else np.arange(len(kp0))
    for i in idx:
        p0 = (int(kp0[i][0]), int(kp0[i][1]))
        p1 = (int(kp1[i][0]) + w0, int(kp1[i][1]))
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    return canvas


def coverage_heatmap(points: np.ndarray, shape: tuple[int, int],
                     grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Integer count heatmap of point density per grid cell."""
    gh, gw = grid
    h, w = shape
    hm = np.zeros((gh, gw), dtype=np.float32)
    for x, y in points:
        cx = min(int(x / w * gw), gw - 1)
        cy = min(int(y / h * gh), gh - 1)
        hm[cy, cx] += 1
    return cv2.resize(hm, (w, h), interpolation=cv2.INTER_NEAREST)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False
