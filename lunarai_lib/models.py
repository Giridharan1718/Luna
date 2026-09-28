"""Benchmark arms: AKAZE+RANSAC and SuperPoint+SuperGlue — SIH Phase 4.

Both runners mirror the acceptance guard and metrics of
ValidationSuite._classical() so results are directly comparable with
sift_ransac / orb_ransac / splg_ransac / lunarai_full rows:

- identical pair set, identical CLAHE preprocessing,
- identical H sanity check, min_matches / min_inliers gates,
- identical metrics: match/inlier counts, inlier_ratio(_3px), coverage,
  rmse_px (reprojection), rmse_gt_px (corner-vs-GT), ssim, ncc, runtime.

SuperGlue notes: torch.hub downloads SuperGlue weights on first use and caches
them (~/.cache/torch/hub). If the machine is offline, construction fails
gracefully (self.available = False) and the row is reported as unavailable —
the benchmark degrades, it never crashes.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class BenchmarkFailure(Exception):
    pass


def create_akaze() -> "cv2.AKAZE":
    """AKAZE factory with a helpful error when OpenCV lacks it.

    The plain `opencv-python` wheels ship AKAZE, but some Anaconda main-package
    builds do not (it lives in contrib). `pip install opencv-contrib-python`
    fixes it; we surface that instead of a bare AttributeError.
    """
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create()
    raise RuntimeError(
        "cv2.AKAZE_create unavailable in this OpenCV build - "
        "install opencv-contrib-python (pip install opencv-contrib-python)")


def _grid_coverage(kp, shape, grid: int = 4) -> float:
    h, w = shape[:2]
    cells = set()
    for k in kp:
        cx = min(grid - 1, int(k[0] / max(w, 1) * grid))
        cy = min(grid - 1, int(k[1] / max(h, 1) * grid))
        cells.add((cy, cx))
    return len(cells) / (grid * grid)


def warp_image(src, H, size):
    return cv2.warpPerspective(src, H, size)


def compute_ssim_ncc(a, b):
    try:
        from skimage.metrics import structural_similarity
        ssim = structural_similarity(a, b)
    except Exception:
        ssim = float("nan")
    af, bf = a.astype(np.float32), b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.sqrt((af ** 2).sum() * (bf ** 2).sum()))
    ncc = float((af * bf).sum() / denom) if denom > 0 else 0.0
    return float(ssim), ncc


def reproj_rmse(p0, p1, H) -> float:
    """RMSE (px) of the forward mapping H(p0) vs p1."""
    if H is None or len(p0) == 0:
        return float("nan")
    ph = np.hstack([p0, np.ones((len(p0), 1), dtype=np.float32)]) @ H.T
    ph = ph[:, :2] / np.maximum(ph[:, 2:3], 1e-9)
    res = np.linalg.norm(ph - p1, axis=1)
    return float(np.sqrt((res ** 2).mean()))


def corner_rmse_vs_gt(H, H_gt, shape) -> float:
    """RMSE (px, reference frame) between H and H_gt over the 4 frame corners."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float32)

    def xform(M):
        ph = np.hstack([corners, np.ones((4, 1), dtype=np.float32)]) @ M.T
        return ph[:, :2] / np.maximum(ph[:, 2:3], 1e-9)

    return float(np.sqrt(((xform(H) - xform(H_gt)) ** 2).sum(axis=1).mean()))


def residual_ratio_3px(p0, p1, H, tol_px: float = 3.0) -> tuple[float, int]:
    """Fraction (and count) of matches whose residual under H is < tol_px."""
    if H is None or len(p0) == 0:
        return 0.0, 0
    ph = np.hstack([p0, np.ones((len(p0), 1), dtype=np.float32)]) @ H.T
    ph = ph[:, :2] / np.maximum(ph[:, 2:3], 1e-9)
    res = np.linalg.norm(ph - p1, axis=1)
    n = int((res < tol_px).sum())
    return n / len(p0), n


def h_sanity(H, shape, area_ratio: tuple[float, float] = (0.05, 20.0)) -> tuple[bool, str]:
    """Same guard as the suite: finite H, no blow-up/flip, bounded area change."""
    if H is None or not np.isfinite(H).all():
        return False, "invalid"
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float32)
    ph = np.hstack([corners, np.ones((4, 1), dtype=np.float32)]) @ H.T
    if not np.isfinite(ph).all() or (ph[:, 2:3] <= 0).any():
        return False, "invalid"
    mapped = ph[:, :2] / np.maximum(ph[:, 2:3], 1e-9)
    if not np.isfinite(mapped).all():
        return False, "invalid"
    area = 0.5 * abs(sum((mapped[i][0] * mapped[(i + 1) % 4][1]
                          - mapped[(i + 1) % 4][0] * mapped[i][1])
                         for i in range(4)))
    ratio = area / float(w * h)
    if not (area_ratio[0] <= ratio <= area_ratio[1]):
        return False, f"quad_blowup_{ratio:.2f}"
    return True, "ok"


def _empty_row(method: str, status: str, t0: float) -> dict:
    return {"method": method, "status": status, "match_count": 0,
            "inlier_count": 0, "inlier_ratio": 0.0, "inlier_count_3px": 0,
            "inlier_ratio_3px": 0.0, "coverage_score": 0.0,
            "rmse_px": float("nan"), "rmse_gt_px": float("nan"),
            "ssim": float("nan"), "ncc": float("nan"),
            "runtime_s": time.perf_counter() - t0}


def _base_row(pair, method: str) -> dict:
    return {"pair": pair.name, "protocol": pair.protocol, "method": method,
            "sensor_a": pair.sensor_a, "sensor_b": pair.sensor_b,
            "scale": pair.meta.get("scale"), "illum_group": pair.meta.get("illum_group")}


def _finalize_ok(out: dict, p0, p1, H, mask, pair, src, ref, t0) -> dict:
    mask_b = mask.reshape(-1).astype(bool)
    out.update(inlier_count=int(mask_b.sum()),
               inlier_ratio=float(mask_b.sum()) / max(len(p0), 1))
    r3, c3 = residual_ratio_3px(p0, p1, H)
    out["inlier_ratio_3px"], out["inlier_count_3px"] = r3, c3
    out["coverage_score"] = _grid_coverage(p0[mask_b], src.shape)
    size = (ref.shape[1], ref.shape[0])
    warped = warp_image(src, H, size)
    ssim, ncc = compute_ssim_ncc(warped, ref)
    out["ssim"], out["ncc"] = ssim, ncc
    out["rmse_px"] = reproj_rmse(p0[mask_b], p1[mask_b], H)
    out["rmse_gt_px"] = (corner_rmse_vs_gt(H, pair.H_gt, src.shape)
                         if pair.H_gt is not None else float("nan"))
    out["status"] = "ok"
    out["runtime_s"] = time.perf_counter() - t0
    return out


class AkazeRunner:
    """AKAZE features + BFMatcher(Hamming) + RANSAC homography."""

    method = "akaze_ransac"

    def run(self, pair, prep, cfg) -> dict:
        src, ref = prep(pair.src), prep(pair.ref)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src, ref = clahe.apply(src), clahe.apply(ref)
        t0 = time.perf_counter()
        out = _base_row(pair, self.method)
        try:
            det = create_akaze()
            k0, d0 = det.detectAndCompute(src, None)
            k1, d1 = det.detectAndCompute(ref, None)
            if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
                out.update(_empty_row(self.method, "no_features", t0))
                return out
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(d0, d1), key=lambda m: m.distance)[:1500]
            p0 = np.float32([k0[m.queryIdx].pt for m in matches])
            p1 = np.float32([k1[m.trainIdx].pt for m in matches])
            out["match_count"] = len(matches)
            if len(p0) < int(cfg.MATCHING.get("min_matches", 10)):
                out.update(_empty_row(self.method, "insufficient_matches", t0))
                return out
            H, mask = cv2.findHomography(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                         cv2.RANSAC, 4.0)
            inl_n = int(mask.sum()) if (mask is not None and H is not None) else 0
            ok_geom, why = h_sanity(H, src.shape)
            if H is None or not ok_geom or inl_n < int(cfg.MATCHING.get("min_inliers", 8)):
                out.update(_empty_row(
                    self.method,
                    "failed_homography" if H is None else
                    (f"rejected_{why}" if not ok_geom else f"rejected_inliers_{inl_n}"),
                    t0))
                out["match_count"] = len(matches)
                return out
            return _finalize_ok(out, p0, p1, H, mask, pair, src, ref, t0)
        except Exception as exc:  # noqa: BLE001
            out.update(_empty_row(self.method, f"error: {exc}", t0))
            return out


class SuperglueRunner:
    """SuperPoint + SuperGlue (torch.hub cog-cvg) with graceful offline failure."""

    method = "superglue_ransac"

    def __init__(self, max_keypoints: int = 2048, match_threshold: float = 0.2,
                 device: str | None = None) -> None:
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.available = False
        self.error = ""
        self.max_keypoints = max_keypoints
        try:
            # trust_repo=True avoids the interactive prompt that hangs headless runs
            self._load_via_hub(max_keypoints)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            try:
                # offline fallback: fetch the model defs + weights directly
                # (torch.hub can fail with a bare 'Authorization' error when the
                # github API is rate-limited or blocked on this machine)
                self._load_via_direct_download(max_keypoints)
                self.available = True
            except Exception as exc2:  # noqa: BLE001
                self.error = f"hub: {exc} | direct: {exc2}"[:500]

    def _load_via_hub(self, max_keypoints: int) -> None:
        self.sg = self.torch.hub.load("cog-cvg/SuperGlue", "superglue",
                                      weights="outdoor", device=self.device,
                                      trust_repo=True)
        try:
            self.sp = self.torch.hub.load("cog-cvg/SuperGlue", "superpoint",
                                          nms_radius=4,
                                          max_keypoints_destination=max_keypoints,
                                          device=self.device, trust_repo=True)
        except TypeError:
            self.sp = self.torch.hub.load("cog-cvg/SuperGlue", "superpoint",
                                          nms_radius=4, max_keypoints=max_keypoints,
                                          device=self.device, trust_repo=True)

    def _load_via_direct_download(self, max_keypoints: int) -> None:
        """Pull superglue.py/superpoint.py from the hub cache (or GitHub), fetch
        the .pth weights into torch's hub cache, then build the models directly."""
        import urllib.request
        from pathlib import Path as _P
        cache = _P(self.torch.hub.get_dir()) / "checkpoints"
        cache.mkdir(parents=True, exist_ok=True)
        src = _P(self.torch.hub.get_dir())
        mod_dir = None
        for cand in sorted(src.glob("**/SuperGlue*/")):
            if (cand / "models").is_dir():
                mod_dir = cand / "models"
                break
        if mod_dir is None:
            # last resort: download the model files from GitHub raw
            mod_dir = cache / "superglue_models"
            mod_dir.mkdir(parents=True, exist_ok=True)
            base = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/"
            for fn in ("superglue.py", "superpoint.py", "utils.py"):
                urllib.request.urlretrieve(base + fn, mod_dir / fn)
            (mod_dir / "__init__.py").write_text("", encoding="utf-8")
        # the magicleap model classes load their weights relative to their own
        # file: <module dir>/weights/<weights>.pth — put them exactly there
        wdir = mod_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        base_w = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/"
        for fn, url in (("superglue_outdoor.pth", base_w + "superglue_outdoor.pth"),
                        ("superpoint_v1.pth", base_w + "superpoint_v1.pth")):
            dst = wdir / fn
            if not dst.exists():
                urllib.request.urlretrieve(url, dst)
        import importlib.util
        for fn, cls in (("superpoint.py", "SuperPoint"), ("superglue.py", "SuperGlue")):
            spec = importlib.util.spec_from_file_location(fn[:-3], mod_dir / fn)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            setattr(self, "_mod_" + cls, mod)
        self.sp = self._mod_SuperPoint.SuperPoint(
            dict(nms_radius=4, max_keypoints=max_keypoints))
        self.sg = self._mod_SuperGlue.SuperGlue(
            dict(weights="outdoor", sinkhorn_iterations=100, match_threshold=0.2))

    def _prep_feats(self, img, img_id: int) -> dict:
        t = self.torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
        pred = self.sp({"image": t.to(self.device)})
        # The magicleap SuperPoint returns per-image LISTS (e.g. [tensor(N,2)]);
        # unwrap and add the batch dim SuperGlue expects (1, N, 2) / (1, D, N).
        out = {}
        for k, v in pred.items():
            if isinstance(v, (list, tuple)) and len(v):
                v = v[0]
            v = v.to(self.device)
            # per-image tensors: kps (N,2), scores (N,), desc (D,N)
            # -> batched: (1,N,2), (1,N), (1,D,N)
            if v.dim() in (1, 2):
                v = v.unsqueeze(0)
            out[f"{k}{img_id}"] = v
        return out

    def match(self, img0: np.ndarray, img1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if not self.available:
            raise BenchmarkFailure(f"superglue unavailable: {self.error}")
        # SuperGlue expects a FLAT dict: keypoints0/descriptors0/keypoints1/...
        # plus image0/image1 (used for keypoint normalization in the magicleap
        # model; harmless for the cog-cvg variant).
        t0 = self.torch.from_numpy(img0.astype(np.float32) / 255.0)[None, None]
        t1 = self.torch.from_numpy(img1.astype(np.float32) / 255.0)[None, None]
        d = {**self._prep_feats(img0, 0), **self._prep_feats(img1, 1),
             "image0": t0.to(self.device), "image1": t1.to(self.device)}
        with self.torch.no_grad():
            pred = self.sg(d)
        kpts0 = d["keypoints0"][0].cpu().numpy()
        kpts1 = d["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        conf = pred["matching_scores0"][0].cpu().numpy()
        valid = matches > -1
        if not valid.any():
            return np.zeros((0, 2)), np.zeros((0, 2)), 0.0
        return kpts0[valid], kpts1[matches[valid]], float(conf[valid].mean())

    def run(self, pair, prep, cfg) -> dict:
        src, ref = prep(pair.src), prep(pair.ref)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src, ref = clahe.apply(src), clahe.apply(ref)
        t0 = time.perf_counter()
        out = _base_row(pair, self.method)
        if not self.available:
            out.update(_empty_row(self.method, "unavailable", t0))
            out["error"] = self.error
            return out
        try:
            p0, p1, conf = self.match(src, ref)
            out["match_count"] = len(p0)
            out["mean_confidence"] = conf
            if len(p0) < int(cfg.MATCHING.get("min_matches", 10)):
                out.update(_empty_row(self.method, "insufficient_matches", t0))
                return out
            H, mask = cv2.findHomography(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                         cv2.RANSAC, 4.0)
            inl_n = int(mask.sum()) if (mask is not None and H is not None) else 0
            ok_geom, why = h_sanity(H, src.shape)
            if H is None or not ok_geom or inl_n < int(cfg.MATCHING.get("min_inliers", 8)):
                out.update(_empty_row(
                    self.method,
                    "failed_homography" if H is None else
                    (f"rejected_{why}" if not ok_geom else f"rejected_inliers_{inl_n}"),
                    t0))
                return out
            return _finalize_ok(out, p0, p1, H, mask, pair, src, ref, t0)
        except Exception as exc:  # noqa: BLE001
            out.update(_empty_row(self.method, f"error: {exc}", t0))
            return out
