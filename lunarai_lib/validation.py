"""LunarAI validation suite — experiments 1-8 for SIH 26166.

Protocols
---------
real      : genuine imagery pairs from the provided distribution
            (anchor: TMC ncf/ncn same orbit+timestamp cross-view pair)
retrieved : real cross-sensor candidates selected by LunaDNA + FAISS
controlled: a source patch transformed under a KNOWN ground-truth homography
            with sensor-simulated appearance (resolution / contrast / noise) and
            illumination transfer matched to a target sun-angle group.
            Gives precise RMSE-vs-GT numbers where no real overlap exists.

All experiment functions return list-of-dict rows; markdown rendering is handled
by scripts/run_validation.py.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch

from .config import Config
from .geometry import (compute_ssim_ncc, confidence_score, ecc_refine,
                       magsac_homography, warp_image)
from .lunadna import LunaDNA, compute_embeddings
from .matching import SuperPointLightGlue, anms, coverage_score, uniformity_score
from .faiss_db import VectorIndex

WORK_SIZE = 384            # matching working resolution (patches upscaled)
GRID = (8, 8)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def load_gray(path: str | Path, size: int = WORK_SIZE, enhance: bool = True) -> np.ndarray | None:
    """Load a patch as grayscale; CLAHE-enhance so faint lunar features are matchable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if size and (img.shape[0] != size or img.shape[1] != size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC if size > img.shape[0] else cv2.INTER_AREA)
    if enhance:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    return img


def lap_var(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def md_table(rows: list[dict], floatfmt: str = "%.3f", missing: str = "-") -> str:
    if not rows:
        return "_(no data)_\n"
    cols = list(rows[0].keys())
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, missing)
            if isinstance(v, float):
                cells.append(missing if not np.isfinite(v) else floatfmt % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def safe_mean(vals: list[float]) -> float:
    arr = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else float("nan")


@dataclass
class Pair:
    name: str
    protocol: str           # real | retrieved | controlled
    sensor_a: str
    sensor_b: str
    src: np.ndarray
    ref: np.ndarray
    H_gt: np.ndarray | None = None
    meta: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


# --------------------------------------------------------------------------- #
# suite
# --------------------------------------------------------------------------- #
class ValidationSuite:
    def __init__(self, cfg: Config, device: str | None = None,
                 max_keypoints: int = 2048, log: Callable[[str], None] = print) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.log = log
        self.df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
        self.runs: list[dict] = []          # every pipeline run (tagged) for reuse

        # LunaDNA
        self.model = LunaDNA(pretrained=False, grayscale=True).to(self.device)
        ckpt = torch.load(cfg.MODELS_DIR / "lunadna.pt", map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ckpt_epoch = ckpt.get("epoch")

        # retrievable database
        self.index, self.embeddings, self.mapping = VectorIndex.load(cfg.DATABASE_DIR)

        # matcher
        self.matcher = SuperPointLightGlue(
            max_keypoints=max_keypoints,
            match_threshold=float(cfg.MATCHING.get("lightglue_threshold", 0.2)),
            device=self.device)
        self.matcher_backend = "superpoint+lightglue" if self.matcher.available else "sift+flann"

        # sensor appearance profiles (measured from actual patches)
        self.profiles: dict[str, dict] = {}
        self.image_sun: dict[str, float] = {}
        self._build_profiles()
        log(f"[suite] device={self.device} matcher={self.matcher_backend} "
            f"checkpoint_epoch={self.ckpt_epoch} patches={len(self.df)}")

    # ----------------------------- profiles -------------------------------- #
    def _build_profiles(self) -> None:
        for sensor in ["OHRC", "TMC-2", "IIRS", "LRO NAC"]:
            sub = self.df[self.df.dataset_name == sensor]
            if not len(sub):
                continue
            sample = sub.sample(min(30, len(sub)), random_state=0)
            means, stds, laps, px = [], [], [], []
            for p in sample.patch_path:
                img = load_gray(p)
                if img is None:
                    continue
                means.append(img.mean()); stds.append(img.std()); laps.append(lap_var(img))
            # pixel resolution (m/px) from dataset report if available
            pxr = {"OHRC": 0.24, "TMC-2": 5.0, "IIRS": 80.0, "LRO NAC": 2.0}.get(sensor, 5.0)
            self.profiles[sensor] = {
                "mean": float(np.mean(means)) if means else 128.0,
                "std": float(np.mean(stds)) if stds else 40.0,
                "lap_var": float(np.median(laps)) if laps else 100.0,
                "px_m": pxr,
                "n_patches": int(len(sub)),
            }
        # real sun elevations from product labels (measured via metadata module)
        try:
            from . import metadata as md
            recs = md.discover_all(self.cfg.DATA_ROOT)
            for sensor, rs in recs.items():
                for r in rs:
                    if r.sun_elevation == r.sun_elevation:
                        self.image_sun[f"{sensor}:{r.image_id}"] = float(r.sun_elevation)
        except Exception as exc:
            self.log(f"[suite] sun metadata unavailable: {exc}")

    # --------------------------- source patches ---------------------------- #
    def source_patches(self, sensor: str, n: int = 3, seed: int = 7) -> list[str]:
        sub = self.df[self.df.dataset_name == sensor]
        if not len(sub):
            return []
        return sub.sample(min(n, len(sub)), random_state=seed).patch_path.tolist()

    def best_texture_patches(self, sensor: str, n: int = 3, pool: int = 40,
                             seed: int = 7) -> list[str]:
        """Prefer high-contrast patches so matching has features to lock onto."""
        sub = self.df[self.df.dataset_name == sensor]
        if not len(sub):
            return []
        sample = sub.sample(min(pool, len(sub)), random_state=seed)
        scored = []
        for p in sample.patch_path:
            img = load_gray(p, 256)
            if img is None:
                continue
            scored.append((img.std(), p))
        scored.sort(reverse=True)
        return [p for _, p in scored[:n]]

    # ------------------------- controlled pairs ---------------------------- #
    def controlled_pair(self, src_path: str, *, name: str, sensor_a: str,
                        sensor_b: str, scale: float = 1.0, illum_group: str | None = None,
                        seed: int = 0) -> Pair | None:
        src = load_gray(src_path, enhance=False)
        if src is None:
            return None
        rng = np.random.default_rng(seed)
        # known ground-truth transform: scale + small rotation + translation
        sx = scale
        sy = scale * float(rng.uniform(0.97, 1.03))
        rot = float(rng.uniform(-3, 3)) * np.pi / 180.0
        cos, sin = np.cos(rot), np.sin(rot)
        A = np.array([[sx * cos, -sx * sin, 0.0],
                      [sy * sin, sy * cos, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        h, w = src.shape
        # center the transformed content in the canvas
        cx, cy = w / 2.0, h / 2.0
        tc = A @ np.array([cx, cy, 1.0])
        A[0, 2] = cx - tc[0] + float(rng.uniform(-12, 12))
        A[1, 2] = cy - tc[1] + float(rng.uniform(-12, 12))
        ref = warp_image(src, A, (w, h))

        # appearance: sensor look + illumination transfer
        ref = self.apply_sensor_look(ref, sensor_a, sensor_b, seed=seed)
        if illum_group:
            ref = self.apply_illumination(ref, illum_group)
        sa_px = self.profiles.get(sensor_a, {}).get("px_m")
        sb_px = self.profiles.get(sensor_b, {}).get("px_m")
        return Pair(name=name, protocol="controlled", sensor_a=sensor_a, sensor_b=sensor_b,
                    src=src, ref=ref, H_gt=A,
                    meta={"scale": scale, "illum_group": illum_group, "src_path": src_path,
                          "gsd_ratio": (sb_px / sa_px) if (sa_px and sb_px) else None,
                          "gsd_normalized": True})

    def apply_sensor_look(self, img: np.ndarray, src_sensor: str, dst_sensor: str,
                          seed: int = 0) -> np.ndarray:
        """Simulate how dst_sensor would render the same terrain: point-spread
        (sharpness), contrast, brightness and noise, guided by measured profiles.

        **Why the resolution transfer is bounded.** Raw GSD ratios between these
        sensors reach 330x (OHRC 0.24 m/px vs IIRS 80 m/px). Resampling by that factor
        inside a 384-px canvas destroys the terrain signal rather than modelling it, and
        it is also not what the pipeline does: both sides are resampled to a common
        working GSD *before* matching (product GSD comes from the PDS labels, `px_m`
        below). The controlled pairs are therefore **GSD-normalized**: geometry is
        shared, and what varies between the two sides is the sensor's appearance. The
        residual MTF difference is modelled as a bounded (<= 2x) low-pass plus the
        measured `lap_var` sharpness target -- the part that CLAHE + LightGlue
        actually have to overcome.
        """
        sp = self.profiles.get(src_sensor)
        dp = self.profiles.get(dst_sensor)
        if sp is None or dp is None:
            return img
        out = img.copy()
        h, w = out.shape
        ratio = dp["px_m"] / sp["px_m"]        # >1: target sensor is coarser
        if ratio > 1.2:
            f = min(ratio, 2.0)               # bounded MTF loss, never signal collapse
            small = cv2.resize(out, (max(8, int(round(w / f))), max(8, int(round(h / f)))),
                               interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        # sharpen/blur to match target sharpness
        target_lap = dp["lap_var"]
        for sigma in (0.0, 0.6, 1.2, 2.0, 3.0):
            if sigma > 0:
                blurred = cv2.GaussianBlur(out, (0, 0), sigma)
            else:
                blurred = out
            if lap_var(blurred) <= target_lap or sigma == 3.0:
                out = blurred
                break
        # contrast / brightness match (affine map on percentiles)
        p1, p99 = np.percentile(out, (1, 99))
        if p99 - p1 > 1e-6:
            out = np.clip((out.astype(np.float32) - p1) / (p99 - p1) * 255.0, 0, 255)
            out = (out - out.mean()) * (dp["std"] / max(out.std(), 1e-6)) + dp["mean"]
            out = np.clip(out, 0, 255).astype(np.uint8)
        # sensor noise
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1.5, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    def illum_stats(self, group: str) -> tuple[float, float]:
        """Mean/std of real patches belonging to a sun-angle group."""
        product_filters = {
            "low": ["ch2_ohr_ncp"],                                     # ~0 deg sun elevation
            "medium": ["ch2_tmc_ncf_20211122", "ch2_tmc_ncn_20211122",
                       "ch2_iir_nri_20240124"],                         # 37-38 deg
            "high": ["ch2_tmc_ncf_20260701"],                           # ~52 deg
        }
        prefixes = product_filters.get(group, [])
        if prefixes:
            sub = self.df[self.df.source_image.str.startswith(tuple(prefixes))]
            if len(sub):
                sample = sub.sample(min(25, len(sub)), random_state=1)
                means, stds = [], []
                for p in sample.patch_path:
                    im = load_gray(p, enhance=False)   # raw stats, not CLAHE-equalized
                    if im is not None:
                        means.append(im.mean()); stds.append(im.std())
                if means:
                    return float(np.mean(means)), float(np.mean(stds))
        fallback = {"low": (70.0, 28.0), "medium": (120.0, 48.0), "high": (150.0, 52.0)}
        return fallback.get(group, (128.0, 40.0))

    def apply_illumination(self, img: np.ndarray, group: str) -> np.ndarray:
        target_mean, target_std = self.illum_stats(group)
        # gamma models the sun-angle contrast change; gain matches the mean
        gamma = {"low": 0.85, "medium": 1.0, "high": 1.15}.get(group, 1.0)
        out = (img.astype(np.float32) / 255.0) ** gamma * 255.0
        out = (out - out.mean()) * (target_std / max(out.std(), 1e-6)) + target_mean
        return np.clip(out, 0, 255).astype(np.uint8)

    # --------------------------- real pairs -------------------------------- #
    def real_crossview_pair(self) -> Pair | None:
        """TMC forward (ncf) vs nadir (ncn) same orbit+timestamp — real overlap."""
        a_img = "ch2_tmc_ncf_20211122T1925079181_d_img_d18"
        b_img = "ch2_tmc_ncn_20211122T1925079214_d_img_d18"
        a = self.df[self.df.source_image.str.startswith(a_img[:38])]
        b = self.df[self.df.source_image.str.startswith(b_img[:38])]
        if not len(a) or not len(b):
            return None
        best, best_d = None, 1e9
        for _, ra in a.iterrows():
            for _, rb in b.iterrows():
                d = np.hypot(ra.latitude - rb.latitude, ra.longitude - rb.longitude)
                if d < best_d:
                    best_d, best = d, (ra, rb)
        ra, rb = best
        sa, sb = ra.dataset_name, rb.dataset_name
        src = load_gray(ra.patch_path, enhance=False)
        ref = load_gray(rb.patch_path, enhance=False)
        if src is None or ref is None:
            return None
        return Pair(name=f"real_ncf{ncf_tag(ra.patch_id)}_vs_ncn{ncf_tag(rb.patch_id)}",
                    protocol="real", sensor_a=sa, sensor_b=sb, src=src, ref=ref, H_gt=None,
                    meta={"geo_dist": float(best_d),
                          "src_image": ra.source_image, "ref_image": rb.source_image})

    def retrieved_pair(self, sensor_a: str, sensor_b: str, n_candidates: int = 3,
                       seed: int = 3) -> Pair | None:
        """Real cross-sensor candidate: query patch from sensor_a, reference from
        sensor_b chosen by LunaDNA + FAISS (best of n_candidates by score)."""
        pool = self.source_patches(sensor_a, n=1, seed=seed)
        if not pool:
            return None
        src_path = pool[0]
        src = load_gray(src_path, enhance=False)
        if src is None:
            return None
        tmp = Path(self.cfg.MATCHING_DIR) / "_query_tmp.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp), src)
        emb = compute_embeddings(self.model, [str(tmp)], device=self.device,
                                 size=int(self.cfg.LUNADNA.get("input_size", 224)))
        scores, idx = self.index.search(emb[0], 40)
        picked = None
        for s, i in zip(scores[0], idx[0]):
            if not (0 <= i < len(self.mapping)):
                continue
            cand_path = self.mapping[int(i)]
            cid = Path(cand_path).stem
            row = self.df[self.df.patch_id == cid]
            if not len(row) or row.iloc[0].dataset_name != sensor_b:
                continue
            picked = (cand_path, float(s))
            break
        if picked is None:
            return None
        cand_path, score = picked
        ref = load_gray(cand_path, enhance=False)
        if ref is None:
            return None
        return Pair(name=f"retrieved_{sensor_a}_to_{sensor_b}", protocol="retrieved",
                    sensor_a=sensor_a, sensor_b=sensor_b, src=src, ref=ref, H_gt=None,
                    meta={"similarity": score, "src_path": src_path, "ref_path": cand_path})

    # ---------------------------- runners ---------------------------------- #
    def _prep(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("image missing")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def run_pipeline(self, pair: Pair, variant: str = "full", save_dir: Path | None = None,
                     tag: str = "") -> dict:
        """Run the correspondence/registration chain with stage toggles."""
        src, ref = self._prep(pair.src), self._prep(pair.ref)
        # preprocessing stage: CLAHE normalization on both images before matching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src, ref = clahe.apply(src), clahe.apply(ref)
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        out: dict[str, Any] = {
            "pair": pair.name, "protocol": pair.protocol, "sensor_a": pair.sensor_a,
            "sensor_b": pair.sensor_b, "variant": variant, "scale": pair.meta.get("scale"),
            "illum_group": pair.meta.get("illum_group"),
        }
        min_matches = int(self.cfg.MATCHING.get("min_matches", 10))
        try:
            t = time.perf_counter()
            m = self.matcher.match(src, ref)
            timings["matching"] = time.perf_counter() - t
            out["matcher"] = m.matcher
            out["match_count"] = m.n_matches
            kp0, kp1 = m.kp0, m.kp1

            if out["match_count"] < min_matches:
                # --- SIH upgrade: equalized-pyramid retry BEFORE giving up ---
                # At 4x/8x reference scale (or 0.5x) the descriptor neighborhoods
                # no longer correspond, so no matches are found at all. Equalize
                # by matching FULL-src against ref/_rs (which also normalizes a
                # 0.5x reference). The rescued correspondences are lifted into
                # the full frames (kp0 is already full-src; ref points scale by
                # _rs) and then flow through the STANDARD chain: ANMS -> MAGSAC
                # -> guard -> (affine rescue) -> ECC, so no special-casing downstream.
                rescued = False
                if variant != "no_magsac":
                    # Two equalization arms per rung, whichever restores matches:
                    #   arm A (ref too coarse): match full-src vs ref/_rs,
                    #        lift ref points by _rs.
                    #   arm B (ref too fine, e.g. 4x/8x upscale): match src/_rs vs
                    #        full-ref, lift src points by _rs.
                    for _rs in (2.0, 4.0, 8.0):
                        for arm in ("A", "B"):
                            if arm == "A":
                                img_q = src
                                img_r = cv2.resize(ref, (max(1, int(ref.shape[1] / _rs)),
                                                         max(1, int(ref.shape[0] / _rs))),
                                                   interpolation=cv2.INTER_AREA)
                            else:
                                img_q = cv2.resize(src, (max(1, int(src.shape[1] / _rs)),
                                                         max(1, int(src.shape[0] / _rs))),
                                                   interpolation=cv2.INTER_AREA)
                                img_r = ref
                            m_s = self.matcher.match(img_q, img_r)
                            if m_s.n_matches < min_matches:
                                continue
                            k0_s, k1_s = m_s.kp0, m_s.kp1
                            if arm == "A":
                                kp0 = k0_s.astype(np.float32)
                                kp1 = (k1_s * _rs).astype(np.float32)
                            else:
                                kp0 = (k0_s * _rs).astype(np.float32)
                                kp1 = k1_s.astype(np.float32)
                            out["match_count"] = m_s.n_matches
                            out["scale_pyramid_retry"] = f"{_rs:g}{arm}"
                            timings["matching"] = time.perf_counter() - t
                            rescued = True
                            break
                        if rescued:
                            break
                if not rescued:
                    out.update(status="insufficient_matches", anms_matches=0,
                               inlier_count=0, inlier_ratio=0.0, inlier_count_3px=0,
                               inlier_ratio_3px=0.0, coverage_score=0.0,
                               uniformity_score=0.0,
                               rmse_px=float("nan"), rmse_gt_px=float("nan"),
                               rmse_px_before_ecc=float("nan"),
                               rmse_gt_before_ecc=float("nan"),
                               ssim=float("nan"), ncc=float("nan"), confidence=0.0,
                               runtime_s=time.perf_counter() - t0, timings=timings)
                    self.runs.append(out)
                    return out

            if variant != "no_anms" and len(kp0):
                t = time.perf_counter()
                sel_kp, sel_idx = anms(kp0, np.ones(len(kp0)),
                                       target=int(self.cfg.MATCHING.get("anms_target", 300)),
                                       gamma=float(self.cfg.MATCHING.get("anms_gamma", 1.6)),
                                       min_radius=float(self.cfg.MATCHING.get("anms_min_radius", 0.0)))
                kp0, kp1 = sel_kp, kp1[sel_idx]
                timings["anms"] = time.perf_counter() - t
            out["anms_matches"] = len(kp0)
            out["coverage_score"] = coverage_score(kp0, src.shape, GRID)
            out["uniformity_score"] = uniformity_score(kp0, src.shape, GRID)

            t = time.perf_counter()
            if variant == "no_magsac":
                H, mask, ratio, inliers = self._plain_least_squares_h(kp0, kp1)
            else:
                H, mask, ratio, inliers = magsac_homography(
                    kp0, kp1, threshold=float(self.cfg.MATCHING.get("magsac_threshold", 4.0)))
            timings["magsac" if variant != "no_magsac" else "least_squares"] = time.perf_counter() - t
            # --- multiscale retry ladder (SIH upgrade) ---
            # On extreme scale ratios (4x/8x) or oblique reference geometry the
            # first homography can blow up (quad guard) or find too few inliers.
            # Retry matching against a 2x/4x INTER_AREA-downsampled reference
            # before giving up; the retried warp must pass the SAME guard.
            min_inliers = int(self.cfg.MATCHING.get("min_inliers", 8))
            _ok_h, _why_h = (False, "none") if H is None else self._h_sanity(H, src.shape)
            _h_bad = (H is None) or (not _ok_h) or (int(inliers) < min_inliers)
            if variant != "no_magsac" and _h_bad:
                for _rs in (2.0, 4.0):
                    ref_s = cv2.resize(ref, (max(1, int(ref.shape[1] / _rs)),
                                             max(1, int(ref.shape[0] / _rs))),
                                       interpolation=cv2.INTER_AREA)
                    m_s = self.matcher.match(src, ref_s)
                    if m_s.n_matches < min_matches:
                        continue
                    k0_s, k1_s = m_s.kp0, m_s.kp1
                    if variant != "no_anms" and len(k0_s):
                        sel_s, idx_s = anms(k0_s, np.ones(len(k0_s)),
                                            target=int(self.cfg.MATCHING.get("anms_target", 300)),
                                            gamma=float(self.cfg.MATCHING.get("anms_gamma", 1.6)),
                                            min_radius=float(self.cfg.MATCHING.get("anms_min_radius", 0.0)))
                        k0_s, k1_s = sel_s, k1_s[idx_s]
                    H_s, mask_s, ratio_s, inl_s = magsac_homography(
                        k0_s, k1_s,
                        threshold=float(self.cfg.MATCHING.get("magsac_threshold", 4.0)))
                    ok_s, _why_s = self._h_sanity(H_s, src.shape)
                    if H_s is not None and ok_s and int(inl_s) >= min_inliers:
                        # lift retry results back to the FULL-ref frame: H maps
                        # src -> ref, so compose with the _rs scale; kp1 scale too.
                        S = np.diag([_rs, _rs, 1.0]).astype(np.float32)
                        kp0, kp1 = k0_s, (k1_s * _rs).astype(np.float32)
                        H = (S @ H_s).astype(np.float32)
                        mask, ratio, inliers = mask_s, ratio_s, inl_s
                        out["multiscale_retry"] = _rs
                        out["match_count"] = m_s.n_matches
                        out["anms_matches"] = len(k0_s)
                        out["coverage_score"] = coverage_score(kp0, src.shape, GRID)
                        out["uniformity_score"] = uniformity_score(kp0, src.shape, GRID)
                        timings["multiscale_retry"] = time.perf_counter() - t
                        break
            out["inlier_count"] = int(inliers)
            out["inlier_ratio"] = float(ratio)
            out["inlier_ratio_3px"], out["inlier_count_3px"] = self._residual_ratio(kp0, kp1, H)
            # pre-ECC reference points for the sub-pixel comparison (experiment 4)
            out["rmse_px_before_ecc"] = (self._rmse_reproj(kp0, kp1, H)
                                         if H is not None else float("nan"))
            out["rmse_gt_before_ecc"] = (self._rmse_vs_gt(H, pair.H_gt, src.shape)
                                         if (H is not None and pair.H_gt is not None)
                                         else float("nan"))

            if H is None:
                out.update(status="failed_homography", rmse_px=float("nan"),
                           rmse_gt_px=float("nan"), ssim=float("nan"), ncc=float("nan"),
                           confidence=0.0, runtime_s=time.perf_counter() - t0,
                           timings=timings)
                self.runs.append(out)
                return out

            # geometric plausibility: a 4-point fit reproduces its own points exactly,
            # so "inlier_ratio = 1.0" on a handful of matches is an artifact, not a
            # registration; such runs are reported as rejected instead of scored.
            ok_geom, why = self._h_sanity(H, src.shape)
            min_inliers = int(self.cfg.MATCHING.get("min_inliers", 8))
            if not ok_geom or int(inliers) < min_inliers:
                # --- SIH upgrade: bounded-affine rescue (P5 viewpoint fix) ---
                # A full 8-DoF homography is ill-posed on oblique real pairs with
                # few matches (the TMC ncf<->ncn quad blow-up). A 6-DoF affine is
                # the classic next step (same family as the classic constrained-
                # homography ladder) and cannot diverge the same way. The result
                # is embedded into a homography and must STILL clear the guard.
                A_pair = cv2.estimateAffine2D(kp0, kp1,
                                              method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                              maxIters=5000)
                A_res = None
                aff_inliers = 0
                if isinstance(A_pair, tuple) and len(A_pair) == 2:
                    A_res, aff_mask = A_pair[0], A_pair[1]
                    aff_inliers = int(np.asarray(aff_mask).sum()) if aff_mask is not None else 0
                out["affine_rescue_inliers"] = aff_inliers
                if A_res is None or aff_inliers < min_inliers:
                    # second escalation: re-match with a looser LightGlue filter
                    # (0.08 vs 0.2). Forward-looking vs nadir real views produce
                    # few high-confidence matches; this is the last cheap arm
                    # before declaring the pair unsolvable at this guard.
                    try:
                        from .matching import SuperPointLightGlue
                        loose = SuperPointLightGlue(max_keypoints=2048,
                                                    match_threshold=0.08)
                        m_loose = loose.match(src, ref)
                        if m_loose.n_matches >= min_matches:
                            A2_pair = cv2.estimateAffine2D(
                                m_loose.kp0, m_loose.kp1,
                                method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                maxIters=5000)
                            A2, m2_mask = (A2_pair if isinstance(A2_pair, tuple)
                                           else (None, None))
                            inl2 = int(np.asarray(m2_mask).sum()) \
                                if m2_mask is not None else 0
                            if A2 is not None:
                                H2 = np.vstack([A2, [0.0, 0.0, 1.0]]).astype(np.float64)
                                ok2, _w2 = self._h_sanity(H2, src.shape)
                                if ok2 and inl2 >= min_inliers:
                                    kp0, kp1 = m_loose.kp0, m_loose.kp1
                                    H = H2
                                    mask = None
                                    ratio = inl2 / max(len(kp0), 1)
                                    inliers = inl2
                                    ok_geom, why = True, "ok"
                                    out["affine_rescue"] = True
                                    out["loose_matcher"] = True
                                    out["affine_rescue_inliers"] = inl2
                                    A_res, aff_inliers = A2, inl2
                    except Exception:  # noqa: BLE001 - rescue is best-effort
                        pass
                if A_res is not None:
                    H_aff = np.vstack([A_res, [0.0, 0.0, 1.0]]).astype(np.float64)
                    ok_aff, why_aff = self._h_sanity(H_aff, src.shape)
                    if ok_aff and aff_inliers >= min_inliers:
                        H = H_aff
                        mask = None
                        ratio = aff_inliers / max(len(kp0), 1)
                        inliers = aff_inliers
                        ok_geom, why = True, "ok"
                        out["affine_rescue"] = True
                    else:
                        out.update(status=f"rejected_affine_rescue_{why_aff}_{aff_inliers}inl",
                                   rmse_px=float("nan"), rmse_gt_px=float("nan"),
                                   ssim=float("nan"), ncc=float("nan"), confidence=0.0,
                                   runtime_s=time.perf_counter() - t0, timings=timings)
                        self.runs.append(out)
                        return out
                else:
                    out.update(status=(f"rejected_{why}" if not ok_geom
                                       else f"rejected_inliers_{int(inliers)}"),
                               rmse_px=float("nan"), rmse_gt_px=float("nan"),
                               ssim=float("nan"), ncc=float("nan"), confidence=0.0,
                               runtime_s=time.perf_counter() - t0, timings=timings)
                    self.runs.append(out)
                    return out

            size = (ref.shape[1], ref.shape[0])
            warped = warp_image(src, H, size)
            H_final = H
            out["ecc_rejected"] = ""
            if variant != "no_ecc":
                t = time.perf_counter()
                W, cc, ok = ecc_refine(ref, warped, np.eye(3, dtype=np.float32),
                                       iterations=int(self.cfg.MATCHING.get("ecc_iterations", 100)),
                                       eps=float(self.cfg.MATCHING.get("ecc_epsilon", 1e-4)))
                timings["ecc"] = time.perf_counter() - t
                if ok:
                    cand = (W @ H).astype(np.float32)
                    ok_cc, why_cc = self._h_sanity(cand, src.shape)
                    rp_before = self._rmse_reproj(kp0, kp1, H)
                    rp_after = self._rmse_reproj(kp0, kp1, cand)
                    # ECC maximises photometric correlation, so on ill-posed pairs it can
                    # drift off the correspondence geometry. Keep it only when it does not.
                    if not ok_cc:
                        out["ecc_rejected"] = f"degenerate_after_ecc:{why_cc}"
                    elif rp_after > 1.5 * rp_before + 1.0:
                        out["ecc_rejected"] = (f"reproj_regression_{rp_before:.2f}"
                                               f"_to_{rp_after:.2f}")
                    else:
                        H_final = cand
                        warped = warp_image(src, H_final, size)
                    out["ecc_correlation"] = float(cc)
                    out["ecc_applied"] = H_final is not H
                else:
                    out["ecc_correlation"] = 0.0
                    out["ecc_applied"] = False
                    out["ecc_rejected"] = "ecc_did_not_converge"
            else:
                out["ecc_applied"] = False
                out["ecc_correlation"] = 0.0

            ssim, ncc = compute_ssim_ncc(warped, ref)
            out["ssim"], out["ncc"] = float(ssim), float(ncc)
            out["rmse_px"] = self._rmse_reproj(kp0, kp1, H_final)
            out["rmse_gt_px"] = (self._rmse_vs_gt(H_final, pair.H_gt, src.shape)
                                 if pair.H_gt is not None else float("nan"))
            out["confidence"] = confidence_score(
                out["rmse_gt_px"] if pair.H_gt is not None else out["rmse_px"],
                out["inlier_ratio"], out["coverage_score"], out["inlier_count"],
                out.get("ecc_correlation"))
            out["status"] = "ok"
            out["runtime_s"] = time.perf_counter() - t0
            out["timings"] = timings

            if save_dir is not None and variant == "full":
                save_dir.mkdir(parents=True, exist_ok=True)
                safe = f"{tag or pair.name}".replace("/", "_")[:60]
                cv2.imwrite(str(save_dir / f"{safe}_src.png"), src)
                cv2.imwrite(str(save_dir / f"{safe}_ref.png"), ref)
                cv2.imwrite(str(save_dir / f"{safe}_registered.png"), warped)
                if pair.H_gt is not None:
                    gt_warp = warp_image(src, pair.H_gt, size)
                    cv2.imwrite(str(save_dir / f"{safe}_gt_warp.png"), gt_warp)
                np.save(save_dir / f"{safe}_H.npy", H_final)
        except Exception as exc:  # noqa: BLE001
            out.update(status=f"error: {exc}", rmse_px=float("nan"), rmse_gt_px=float("nan"),
                       ssim=float("nan"), ncc=float("nan"), confidence=0.0,
                       runtime_s=time.perf_counter() - t0, timings=timings)
        self.runs.append(out)
        return out

    # --------------------------- quality guards ----------------------------- #
    @staticmethod
    def _h_sanity(H: np.ndarray, shape: tuple[int, int]) -> tuple[bool, str]:
        """Reject degenerate homographies before they can be reported as a success.

        A homography fitted to four correspondences reproduces those four exactly, so
        `inlier_ratio == 1.0` on a handful of matches is an artifact. We require the
        mapped frame quad to stay finite, convex and inside a plausible area/offset
        envelope; anything else is reported as a rejected run instead of a number.
        """
        if H is None:
            return False, "no_homography"
        h, w = shape
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1],
                              [0, h - 1]]).reshape(-1, 1, 2)
        try:
            q = cv2.perspectiveTransform(corners, H.astype(np.float32)).reshape(-1, 2)
        except Exception as exc:  # noqa: BLE001
            return False, f"transform_error"
        if not np.all(np.isfinite(q)):
            return False, "non_finite_quad"
        area_src = float(max(w - 1, 1) * max(h - 1, 1))
        area = 0.5 * abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) -
                                       np.roll(q[:, 0], -1) * q[:, 1])))
        ratio = area / area_src
        if not (0.05 <= ratio <= 20.0):
            return False, f"area_ratio_{ratio:.3g}"
        diag = float(np.hypot(w, h))
        if float(np.max(np.linalg.norm(q - q.mean(axis=0), axis=1))) > 3.0 * diag:
            return False, "quad_blowup"
        signs = {float(np.sign((q[(i + 1) % 4, 0] - q[i, 0]) *
                               (q[(i + 2) % 4, 1] - q[(i + 1) % 4, 1]) -
                               (q[(i + 1) % 4, 1] - q[i, 1]) *
                               (q[(i + 2) % 4, 0] - q[(i + 1) % 4, 0]))) for i in range(4)}
        if len(signs) > 1:
            return False, "non_convex_quad"
        return True, "ok"

    @staticmethod
    def _residual_ratio(kp0: np.ndarray, kp1: np.ndarray, H: np.ndarray,
                        tol: float = 3.0) -> tuple[float, int]:
        """Share of correspondences whose residual under H is <= tol px.

        Measured identically for every variant and baseline, so components stay
        comparable even when the algorithm's own inlier flag is trivially satisfied
        (a least-squares fit uses every point, hence "100% inliers").
        """
        if H is None or len(kp0) == 0:
            return 0.0, 0
        pts = cv2.perspectiveTransform(kp0.reshape(-1, 1, 2).astype(np.float32),
                                       H.astype(np.float32)).reshape(-1, 2)
        d = np.linalg.norm(pts - kp1.reshape(-1, 2).astype(np.float32), axis=1)
        keep = d <= tol
        return float(keep.mean()), int(keep.sum())

    @staticmethod
    def _plain_least_squares_h(kp0: np.ndarray, kp1: np.ndarray):
        """Homography via least squares on ALL matches — no robust rejection."""
        if len(kp0) < 4:
            return None, None, 0.0, 0
        src = np.ascontiguousarray(kp0.reshape(-1, 1, 2).astype(np.float32))
        dst = np.ascontiguousarray(kp1.reshape(-1, 1, 2).astype(np.float32))
        try:
            H, mask = cv2.findHomography(src, dst, 0)  # method=0 -> least squares
        except Exception:
            H, mask = None, None
        if H is None:
            return None, None, 0.0, 0
        return H, np.ones(len(kp0)), 1.0, int(len(kp0))

    @staticmethod
    def _rmse_reproj(kp0: np.ndarray, kp1: np.ndarray, H: np.ndarray) -> float:
        if H is None or len(kp0) == 0:
            return float("nan")
        pts = cv2.perspectiveTransform(kp0.reshape(-1, 1, 2).astype(np.float32),
                                       H.astype(np.float32))
        err = pts.reshape(-1, 2) - kp1.reshape(-1, 2).astype(np.float32)
        return float(np.sqrt((err ** 2).sum(axis=1).mean()))

    @staticmethod
    def _rmse_vs_gt(H: np.ndarray, H_gt: np.ndarray, shape: tuple[int, int]) -> float:
        """Pixel RMSE of the estimated transform vs ground truth over a
        uniform grid of source points (units: reference pixels)."""
        h, w = shape
        ys, xs = np.mgrid[0:h:6, 0:w:6]
        pts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=1)
        p = pts.reshape(-1, 1, 2)
        a = cv2.perspectiveTransform(p, H.astype(np.float32)).reshape(-1, 2)
        b = cv2.perspectiveTransform(p, H_gt.astype(np.float32)).reshape(-1, 2)
        return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))

    # ------------------------ classical baselines -------------------------- #
    def run_sift_ransac(self, pair: Pair) -> dict:
        return self._classical(pair, method="sift")

    def run_orb_ransac(self, pair: Pair) -> dict:
        return self._classical(pair, method="orb")

    def _classical(self, pair: Pair, method: str) -> dict:
        src, ref = self._prep(pair.src), self._prep(pair.ref)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src, ref = clahe.apply(src), clahe.apply(ref)   # same preprocessing as LunarAI
        t0 = time.perf_counter()
        out = {"pair": pair.name, "protocol": pair.protocol, "method": method,
               "sensor_a": pair.sensor_a, "sensor_b": pair.sensor_b,
               "scale": pair.meta.get("scale"), "illum_group": pair.meta.get("illum_group")}
        try:
            if method == "sift":
                det = cv2.SIFT_create(nfeatures=2048)
                norm = cv2.NORM_L2
            else:
                det = cv2.ORB_create(nfeatures=2048)
                norm = cv2.NORM_HAMMING
            k0, d0 = det.detectAndCompute(src, None)
            k1, d1 = det.detectAndCompute(ref, None)
            if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
                out.update(status="no_features", match_count=0, inlier_count=0,
                           inlier_ratio=0.0, coverage_score=0.0, rmse_px=float("nan"),
                           rmse_gt_px=float("nan"), ssim=float("nan"), ncc=float("nan"),
                           runtime_s=time.perf_counter() - t0)
                return out
            bf = cv2.BFMatcher(norm, crossCheck=True)
            matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:1500]
            p0 = np.float32([k0[m.queryIdx].pt for m in matches])
            p1 = np.float32([k1[m.trainIdx].pt for m in matches])
            out["match_count"] = len(matches)
            min_matches = int(self.cfg.MATCHING.get("min_matches", 10))
            if len(p0) < min_matches:
                out.update(status="insufficient_matches", inlier_count=0, inlier_ratio=0.0,
                           inlier_count_3px=0, inlier_ratio_3px=0.0, coverage_score=0.0,
                           rmse_px=float("nan"), rmse_gt_px=float("nan"),
                           ssim=float("nan"), ncc=float("nan"),
                           runtime_s=time.perf_counter() - t0)
                return out
            H, mask = cv2.findHomography(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                         cv2.RANSAC, 4.0)
            ok_geom, why = self._h_sanity(H, src.shape)
            inl_n = int(mask.sum()) if (mask is not None and H is not None) else 0
            if H is None or not ok_geom or inl_n < int(self.cfg.MATCHING.get("min_inliers", 8)):
                out.update(status=("failed_homography" if H is None else
                                   (f"rejected_{why}" if not ok_geom
                                    else f"rejected_inliers_{inl_n}")),
                           inlier_count=0, inlier_ratio=0.0, inlier_count_3px=0,
                           inlier_ratio_3px=0.0,
                           coverage_score=coverage_score(p0, src.shape, GRID) if len(p0) else 0.0,
                           rmse_px=float("nan"), rmse_gt_px=float("nan"),
                           ssim=float("nan"), ncc=float("nan"),
                           runtime_s=time.perf_counter() - t0)
                return out
            mask = mask.reshape(-1).astype(bool)
            inliers = int(mask.sum())
            out["inlier_count"] = inliers
            out["inlier_ratio"] = inliers / max(len(p0), 1)
            out["inlier_ratio_3px"], out["inlier_count_3px"] = self._residual_ratio(p0, p1, H)
            out["coverage_score"] = coverage_score(p0[mask], src.shape, GRID)
            size = (ref.shape[1], ref.shape[0])
            warped = warp_image(src, H, size)
            ssim, ncc = compute_ssim_ncc(warped, ref)
            out["ssim"], out["ncc"] = float(ssim), float(ncc)
            out["rmse_px"] = self._rmse_reproj(p0[mask], p1[mask], H)
            out["rmse_gt_px"] = (self._rmse_vs_gt(H, pair.H_gt, src.shape)
                                 if pair.H_gt is not None else float("nan"))
            out["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out.update(status=f"error: {exc}")
        out["runtime_s"] = time.perf_counter() - t0
        return out

    def run_splg_ransac(self, pair: Pair) -> dict:
        """SuperPoint + LightGlue matches with plain RANSAC — no ANMS / MAGSAC / ECC."""
        src, ref = self._prep(pair.src), self._prep(pair.ref)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src, ref = clahe.apply(src), clahe.apply(ref)   # same preprocessing as LunarAI
        t0 = time.perf_counter()
        out = {"pair": pair.name, "protocol": pair.protocol, "method": "splg_ransac",
               "sensor_a": pair.sensor_a, "sensor_b": pair.sensor_b,
               "scale": pair.meta.get("scale"), "illum_group": pair.meta.get("illum_group")}
        try:
            m = self.matcher.match(src, ref)
            out["match_count"] = m.n_matches
            min_matches = int(self.cfg.MATCHING.get("min_matches", 10))
            if m.n_matches >= max(min_matches, 4):
                H, mask, ratio, inliers = magsac_homography(
                    m.kp0, m.kp1, threshold=float(self.cfg.MATCHING.get("magsac_threshold", 4.0)))
            else:
                H, mask, ratio, inliers = None, None, 0.0, 0
            if H is not None:
                out["inlier_ratio_3px"], out["inlier_count_3px"] = self._residual_ratio(
                    m.kp0, m.kp1, H)
            ok_geom, why = self._h_sanity(H, src.shape)
            if H is None or not ok_geom or int(inliers) < int(self.cfg.MATCHING.get("min_inliers", 8)):
                out.update(status=("failed_homography" if H is None else
                                   (f"rejected_{why}" if not ok_geom
                                    else f"rejected_inliers_{int(inliers)}")),
                           inlier_count=0, inlier_ratio=0.0, inlier_count_3px=0,
                           inlier_ratio_3px=0.0, coverage_score=0.0, rmse_px=float("nan"),
                           rmse_gt_px=float("nan"), ssim=float("nan"), ncc=float("nan"))
            else:
                msk = mask.astype(bool) if mask is not None else np.ones(len(m.kp0), bool)
                out["inlier_count"] = int(inliers)
                out["inlier_ratio"] = float(ratio)
                out["coverage_score"] = coverage_score(m.kp0[msk], src.shape, GRID)
                size = (ref.shape[1], ref.shape[0])
                warped = warp_image(src, H, size)
                ssim, ncc = compute_ssim_ncc(warped, ref)
                out["ssim"], out["ncc"] = float(ssim), float(ncc)
                out["rmse_px"] = self._rmse_reproj(m.kp0[msk], m.kp1[msk], H)
                out["rmse_gt_px"] = (self._rmse_vs_gt(H, pair.H_gt, src.shape)
                                     if pair.H_gt is not None else float("nan"))
                out["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            out.update(status=f"error: {exc}")
        out["runtime_s"] = time.perf_counter() - t0
        return out

    # ---------------------------- retrieval -------------------------------- #
    def evaluate_retrieval(self, n_queries: int = 60, top_k: int = 10,
                           seed: int = 11) -> dict:
        """Top-K accuracy / Recall@K / mAP with two ground-truth definitions."""
        df = self.df.reset_index(drop=True)
        rng = np.random.default_rng(seed)
        qidx = rng.choice(len(df), size=min(n_queries, len(df)), replace=False)
        qdf = df.iloc[qidx].reset_index(drop=True)
        tmp_dir = Path(self.cfg.MATCHING_DIR) / "_retrieval_queries"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, p in enumerate(qdf.patch_path):
            q = Path(tmp_dir) / f"q{i:03d}.png"
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                paths.append(p)
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(q), img)
            paths.append(str(q))
        embs = compute_embeddings(self.model, paths, device=self.device,
                                  size=int(self.cfg.LUNADNA.get("input_size", 224)))
        scores, idxs = self.index.search(embs, top_k + 1)

        meta = df.set_index("patch_id")
        agg = {"top1": 0, "top5": 0, "top10": 0, "top1_cross": 0, "top5_cross": 0,
               "top10_cross": 0, "n": 0}
        aps, rr = [], []
        per_query = []
        for qi in range(len(qdf)):
            query_row = qdf.iloc[qi]
            ranked = []
            for s, gi in zip(scores[qi], idxs[qi]):
                if not (0 <= gi < len(self.mapping)):
                    continue
                cpath = self.mapping[int(gi)]
                cid = Path(cpath).stem
                if cid == query_row.patch_id:
                    continue
                ranked.append((float(s), cid))
                if len(ranked) >= top_k:
                    break
            if not ranked:
                continue
            agg["n"] += 1
            hits_any, hits_cross = [], []
            geo_top1 = float("nan")
            for rank, (s, cid) in enumerate(ranked, start=1):
                if cid not in meta.index:
                    hits_any.append(False); hits_cross.append(False); continue
                c = meta.loc[cid]
                if rank == 1:
                    geo_top1 = float(np.hypot(float(c["latitude"]) - float(query_row["latitude"]),
                                              float(c["longitude"]) - float(query_row["longitude"])))
                same_source = (c.source_image == query_row.source_image)
                geo_near_cross = (c.dataset_name != query_row.dataset_name and
                                  np.hypot(c.latitude - query_row.latitude,
                                           c.longitude - query_row.longitude) < 0.5)
                hits_any.append(bool(same_source))
                hits_cross.append(bool(geo_near_cross))
            for k, key in ((1, "top1"), (5, "top5"), (10, "top10")):
                if any(hits_any[:k]):
                    agg[key] += 1
                if any(hits_cross[:k]):
                    agg[key.replace("top", "top") + "_cross"] += 1
            # AP / RR for same-source relevance
            if any(hits_any):
                precisions = []
                n_rel = 0
                for rank, hit in enumerate(hits_any, start=1):
                    if hit:
                        n_rel += 1
                        precisions.append(n_rel / rank)
                aps.append(float(np.mean(precisions)))
                first = next((i for i, h in enumerate(hits_any, start=1) if h), None)
                rr.append(1.0 / first if first else 0.0)
            per_query.append({"query": query_row.patch_id, "sensor": query_row.dataset_name,
                              "top1_sim": ranked[0][0],
                              "top1_hit_same_source": bool(hits_any[0]) if hits_any else False,
                              "top1_hit_cross_sensor": bool(hits_cross[0]) if hits_cross else False,
                              "geo_top1_deg": geo_top1})
        n = max(agg["n"], 1)

        # geo-localization evidence: absolute coordinate distance of the retrieved
        # candidate vs what a random database patch would give.
        rng2 = np.random.default_rng(seed + 977)
        lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=float)
        ai = rng2.integers(0, len(df), 200)
        bi = rng2.integers(0, len(df), 200)
        geo_rand_vals = np.hypot(lat[ai] - lat[bi], lon[ai] - lon[bi])
        geo_rand = (float(np.nanmedian(geo_rand_vals))
                    if np.isfinite(geo_rand_vals).any() else float("nan"))
        geo_vals = np.asarray([q["geo_top1_deg"] for q in per_query], dtype=float)
        geo_top1_med = (float(np.nanmedian(geo_vals))
                        if len(geo_vals) and np.isfinite(geo_vals).any() else float("nan"))
        geo_same_vals = np.asarray([q["geo_top1_deg"] for q in per_query
                                    if q["top1_hit_same_source"]], dtype=float)
        geo_same_med = (float(np.nanmedian(geo_same_vals))
                        if len(geo_same_vals) and np.isfinite(geo_same_vals).any()
                        else float("nan"))
        gain = (geo_rand / geo_top1_med
                if (np.isfinite(geo_rand) and np.isfinite(geo_top1_med) and geo_top1_med > 0)
                else float("nan"))
        return {
            "n_queries": agg["n"],
            "top1_accuracy": agg["top1"] / n, "top5_accuracy": agg["top5"] / n,
            "top10_accuracy": agg["top10"] / n,
            "top1_accuracy_cross_sensor": agg["top1_cross"] / n,
            "top5_accuracy_cross_sensor": agg["top5_cross"] / n,
            "top10_accuracy_cross_sensor": agg["top10_cross"] / n,
            "mAP": float(np.mean(aps)) if aps else 0.0,
            "mrr": float(np.mean(rr)) if rr else 0.0,
            "geo_top1_median_deg": geo_top1_med,
            "geo_same_source_top1_median_deg": geo_same_med,
            "geo_random_pair_median_deg": geo_rand,
            "geo_localization_gain": gain,
            "per_query": per_query,
        }


def ncf_tag(patch_id: str) -> str:
    parts = str(patch_id).split("_")
    return "_".join(parts[-2:])
