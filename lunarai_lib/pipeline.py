"""End-to-end pipeline orchestrator: query -> retrieval -> matching -> ANMS ->
MAGSAC++ -> homography -> ECC -> metrics. Returns a single result dict consumed
by notebooks, exports and the dashboard."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import geometry, matching
from .geometry import Timer
from .faiss_db import VectorIndex
from .lunadna import LunaDNA, compute_embeddings


class LunarAIPipeline:
    def __init__(self, model: LunaDNA, index: VectorIndex, patch_paths: list[str],
                 matcher: matching.SuperPointLightGlue, cfg: dict, device: str = "cpu"):
        self.model = model
        self.index = index
        self.patch_paths = patch_paths
        self.matcher = matcher
        self.cfg = cfg
        self.device = device

    def load_gray(self, path: str | Path, size: int | None = None) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if size:
            img = cv2.resize(img, (size, size))
        return img

    def retrieve(self, query_img: np.ndarray, k: int | None = None) -> list[dict[str, Any]]:
        k = k or int(self.cfg.get("top_k", 10))
        emb = compute_embeddings(self.model, [self._tmp_for(query_img)],
                                 device=self.device, size=int(self.cfg.get("input_size", 224)))
        scores, idx = self.index.search(emb[0], k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if 0 <= i < len(self.patch_paths):
                out.append({"patch_path": self.patch_paths[int(i)], "score": float(s),
                            "rank": len(out) + 1})
        return out

    def _tmp_for(self, img: np.ndarray) -> str:
        import tempfile
        p = Path(tempfile.gettempdir()) / "lunarai_query.png"
        cv2.imwrite(str(p), img)
        return str(p)

    def run(self, query_img: np.ndarray, reference_img: np.ndarray,
            save_dir: Path | None = None, run_name: str = "run") -> dict[str, Any]:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        result: dict[str, Any] = {"run_name": run_name}

        # --- matching ---
        with Timer() as t:
            m = self.matcher.match(query_img, reference_img)
        timings["superpoint_lightglue"] = t.elapsed
        result["matcher"] = m.matcher
        result["total_matches"] = m.n_matches
        result["n_kp_query"] = len(m.all_kp0) if m.all_kp0 is not None else 0
        result["n_kp_ref"] = len(m.all_kp1) if m.all_kp1 is not None else 0

        # --- ANMS on matched query keypoints ---
        with Timer() as t:
            if m.n_matches:
                scores = np.ones(m.n_matches)
                kp0_sel, sel = matching.anms(m.kp0, scores,
                                             target=int(self.cfg.get("anms_target", 300)),
                                             gamma=float(self.cfg.get("anms_gamma", 1.6)))
                kp1_sel = m.kp1[sel]
            else:
                kp0_sel = kp1_sel = np.zeros((0, 2))
        timings["anms"] = t.elapsed
        result["anms_matches"] = len(kp0_sel)
        result["coverage_score"] = matching.coverage_score(kp0_sel, query_img.shape)
        result["uniformity_score"] = matching.uniformity_score(kp0_sel, query_img.shape)

        # --- MAGSAC++ ---
        with Timer() as t:
            H, mask, inlier_ratio, inlier_count = geometry.magsac_homography(
                kp0_sel, kp1_sel, threshold=float(self.cfg.get("magsac_threshold", 4.0)))
        timings["magsac"] = t.elapsed
        result["inlier_matches"] = inlier_count
        result["inlier_ratio"] = inlier_ratio

        # --- homography + warp ---
        with Timer() as t:
            if H is None:
                result["status"] = "failed: homography estimation"
                result["rmse"] = float("nan")
                result["ssim"] = result["ncc"] = float("nan")
                result["runtime_s"] = time.perf_counter() - t_start
                result["timings"] = timings
                return result
            size = (reference_img.shape[1], reference_img.shape[0])
            warped = geometry.warp_image(query_img, H, size)
        timings["homography_warp"] = t.elapsed
        rmse_before = geometry.reprojection_rmse(kp0_sel, kp1_sel, H)

        # --- ECC refinement ---
        with Timer() as t:
            W_ecc, ecc_cc, ecc_ok = geometry.ecc_refine(
                reference_img, warped, np.eye(3, dtype=np.float32),
                iterations=int(self.cfg.get("ecc_iterations", 100)),
                eps=float(self.cfg.get("ecc_epsilon", 1e-4)))
        timings["ecc"] = t.elapsed
        if ecc_ok:
            H_refined = (W_ecc @ H).astype(np.float32)
            warped_final = geometry.warp_image(query_img, H_refined, size)
            rmse_after = geometry.reprojection_rmse(kp0_sel, kp1_sel, H_refined)
        else:
            H_refined = H
            warped_final = warped
            rmse_after = rmse_before
        result["ecc_applied"] = bool(ecc_ok)
        result["ecc_correlation"] = ecc_cc
        result["rmse_homography"] = rmse_before
        result["rmse"] = rmse_after
        result["homography"] = H.tolist()
        result["homography_refined"] = H_refined.tolist()

        # --- similarity ---
        ssim, ncc = geometry.compute_ssim_ncc(warped_final, reference_img)
        result["ssim"] = ssim
        result["ncc"] = ncc

        # --- confidence (documentated 25/30/30/15 engine) ---
        # embedding-similarity axis is unavailable in this standalone demo path
        # (no index here), so it is neutral; the residual 3 px ratio is measured.
        from .validation_ext import confidence_engine
        from .models import residual_ratio_3px
        _rmse = rmse_after if np.isfinite(rmse_after) else rmse_before
        _inl3, _ = residual_ratio_3px(kp0_sel, kp1_sel, H_refined)
        _ce = confidence_engine(0.0, _inl3, _rmse, result["coverage_score"])
        result["confidence"] = _ce["confidence_score"]
        result["confidence_axes"] = _ce["axes"]

        result["status"] = "ok"
        result["runtime_s"] = time.perf_counter() - t_start
        result["timings"] = timings

        # --- artifacts ---
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"{run_name}_registered.png"), warped_final)
            if m.n_matches:
                vis = geometry.draw_matches(query_img, reference_img, kp0_sel, kp1_sel)
                cv2.imwrite(str(save_dir / f"{run_name}_matches.png"), vis)
            hm = geometry.coverage_heatmap(kp0_sel, query_img.shape)
            hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            hm_color = cv2.applyColorMap(255 - hm_norm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(
                cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR)
                if query_img.ndim == 2 else query_img, 0.55, hm_color, 0.45, 0)
            cv2.imwrite(str(save_dir / f"{run_name}_coverage.png"), overlay)
            np.save(save_dir / f"{run_name}_homography.npy", H_refined)
            np.savez(save_dir / f"{run_name}_inliers.npz", kp0=kp0_sel, kp1=kp1_sel)
            result["artifact_dir"] = str(save_dir)
        return result
