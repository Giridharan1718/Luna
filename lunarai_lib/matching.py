"""SuperPoint + LightGlue correspondence generation with SIFT/FLANN fallback.

LightGlue is vendored in third_party/lightglue (github.com/cvg/LightGlue).
Weights download on first use and are cached by torch hub.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"
if _THIRD_PARTY.exists() and str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))


@dataclass
class MatchResult:
    kp0: np.ndarray                      # (N,2) query keypoints
    kp1: np.ndarray                      # (N,2) reference keypoints
    descriptors0: np.ndarray | None = None
    descriptors1: np.ndarray | None = None
    all_kp0: np.ndarray | None = None
    all_kp1: np.ndarray | None = None
    matcher: str = "superpoint+lightglue"
    extras: dict = field(default_factory=dict)

    @property
    def n_matches(self) -> int:
        return 0 if self.kp0 is None else len(self.kp0)


class SuperPointLightGlue:
    def __init__(self, max_keypoints: int = 2048, match_threshold: float = 0.2,
                 device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_keypoints = max_keypoints
        self.match_threshold = match_threshold
        self.available = False
        self.extractor = None
        self.matcher = None
        try:
            from lightglue import LightGlue, SuperPoint
            self.extractor = SuperPoint(
                nms_radius=4, max_num_keypoints=max_keypoints,
                detection_threshold=0.0005,      # faint lunar features need the default
            ).eval().to(self.device)
            # keep LightGlue early stopping + point pruning enabled (CPU speed)
            self.matcher = LightGlue(features="superpoint",
                                     filter_threshold=match_threshold).eval().to(self.device)
            self.available = True
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    def _prep(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
        return t.to(self.device)

    def _extract_feats(self, img: np.ndarray) -> dict:
        """Vendored LightGlue API: extractor.extract(tensor[B,1,H,W]) -> feats dict."""
        return self.extractor.extract(self._prep(img))

    def _feats_to_dict(self, feats: dict) -> dict:
        kps = feats["keypoints"][0].detach().cpu().numpy()
        desc = feats["descriptors"][0].detach().cpu().numpy()   # (N, 256)
        sc = feats.get("keypoint_scores")
        scores = sc[0].detach().cpu().numpy() if sc is not None else np.ones(len(kps))
        return {"keypoints": kps, "descriptors": desc, "scores": scores,
                "backend": "superpoint"}

    @torch.no_grad()
    def extract(self, img: np.ndarray) -> dict:
        if not self.available:
            return self._sift_extract(img)
        return self._feats_to_dict(self._extract_feats(img))

    @torch.no_grad()
    def match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        if not self.available:
            return self._sift_match(img0, img1)
        f0 = self._extract_feats(img0)
        f1 = self._extract_feats(img1)
        pred = self.matcher({"image0": f0, "image1": f1})
        matches = torch.as_tensor(pred["matches"][0])            # (M, 2) index pairs
        kp0_all = f0["keypoints"][0].detach().cpu().numpy()
        kp1_all = f1["keypoints"][0].detach().cpu().numpy()
        desc0_all = f0["descriptors"][0].detach().cpu().numpy()
        desc1_all = f1["descriptors"][0].detach().cpu().numpy()
        if matches.numel() == 0:
            return MatchResult(kp0=np.zeros((0, 2)), kp1=np.zeros((0, 2)),
                               all_kp0=kp0_all, all_kp1=kp1_all,
                               matcher="superpoint+lightglue",
                               extras={"n_kp0": len(kp0_all), "n_kp1": len(kp1_all)})
        i0 = matches[:, 0].detach().cpu().numpy()
        i1 = matches[:, 1].detach().cpu().numpy()
        return MatchResult(
            kp0=kp0_all[i0], kp1=kp1_all[i1],
            descriptors0=desc0_all[i0], descriptors1=desc1_all[i1],
            all_kp0=kp0_all, all_kp1=kp1_all,
            matcher="superpoint+lightglue",
            extras={"n_kp0": len(kp0_all), "n_kp1": len(kp1_all)},
        )

    # ---------------- SIFT fallback (fully offline) ----------------
    def _sift_extract(self, img: np.ndarray) -> dict:
        sift = cv2.SIFT_create(nfeatures=self.max_keypoints)
        kps, desc = sift.detectAndCompute(img, None)
        kps = np.array([k.pt for k in kps]) if kps else np.zeros((0, 2))
        return {"keypoints": kps, "descriptors": desc, "scores": np.ones(len(kps)),
                "backend": "sift"}

    def _sift_match(self, img0: np.ndarray, img1: np.ndarray) -> MatchResult:
        e0, e1 = self._sift_extract(img0), self._sift_extract(img1)
        if e0["descriptors"] is None or e1["descriptors"] is None or \
           len(e0["keypoints"]) < 4 or len(e1["keypoints"]) < 4:
            return MatchResult(kp0=np.zeros((0, 2)), kp1=np.zeros((0, 2)),
                               matcher="sift+flann",
                               extras={"reason": "insufficient keypoints"})
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        raw = flann.knnMatch(e0["descriptors"].astype(np.float32),
                             e1["descriptors"].astype(np.float32), k=2)
        good = [m for m, n in (pair for pair in raw if len(pair) == 2)
                if m.distance < 0.75 * n.distance]
        kp0 = e0["keypoints"][np.array([m.queryIdx for m in good])] if good else np.zeros((0, 2))
        kp1 = e1["keypoints"][np.array([m.trainIdx for m in good])] if good else np.zeros((0, 2))
        return MatchResult(
            kp0=kp0, kp1=kp1, descriptors0=None, descriptors1=None,
            all_kp0=e0["keypoints"], all_kp1=e1["keypoints"], matcher="sift+flann",
            extras={"n_kp0": len(e0["keypoints"]), "n_kp1": len(e1["keypoints"])},
        )


# ------------------------- ANMS -------------------------

def anms(keypoints: np.ndarray, scores: np.ndarray | None, target: int = 300,
         gamma: float = 1.6, min_radius: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive Non-Maximal Suppression for uniform spatial distribution.

    `target` caps how many correspondences survive; `min_radius` (px) enforces an
    actual suppression radius *even when fewer than `target` matches exist*, which is
    what turns ANMS from a no-op into a spatially spreading filter on sparse match
    sets: greedy score-ordered selection that blocks every neighbour within
    `min_radius` of an already-kept point.

    Returns selected keypoint subset (target or fewer) and their indices.
    """
    kps = np.asarray(keypoints, dtype=float)
    if len(kps) == 0:
        return kps, np.zeros(0, dtype=int)
    if scores is None:
        scores = np.ones(len(kps))
    scores = np.asarray(scores, dtype=float)
    n = len(kps)
    if min_radius and min_radius > 0:
        order = np.argsort(-scores)
        r2 = float(min_radius) ** 2
        cap = max(int(target), 1)
        blocked = np.zeros(n, dtype=bool)
        kept: list[int] = []
        for i in order:
            if blocked[i]:
                continue
            kept.append(int(i))
            if len(kept) >= cap:
                break
            d = ((kps - kps[i]) ** 2).sum(axis=1)
            blocked |= d <= r2
        keep = np.sort(np.asarray(kept, dtype=int))
        return kps[keep], keep
    if n <= target:
        order = np.argsort(-scores)
        return kps[order], order

    # rank by score, strongest = r=0
    order = np.argsort(-scores)
    radius = np.full(n, np.inf)
    pts = kps[order]
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    for i in range(n):
        # neighbors with strictly higher score (earlier in order)
        radius[order[i]] = np.min(dist[i, :i]) if i > 0 else np.inf
    radius[np.isinf(radius)] = radius[np.isfinite(radius)].max() if np.isfinite(radius).any() else 1e6
    gamma_r = radius ** gamma
    keep_order = np.argsort(-gamma_r)[:target]
    keep = np.sort(keep_order)
    return kps[keep], keep


def coverage_score(points: np.ndarray, shape: tuple[int, int], grid: tuple[int, int] = (8, 8)) -> float:
    """Fraction of image grid cells containing >=1 point."""
    if len(points) == 0:
        return 0.0
    h, w = shape
    gh, gw = grid
    cells = set()
    for x, y in points:
        cx = min(int(x / w * gw), gw - 1)
        cy = min(int(y / h * gh), gh - 1)
        cells.add((cy, cx))
    return len(cells) / (gh * gw)


def uniformity_score(points: np.ndarray, shape: tuple[int, int], grid: tuple[int, int] = (8, 8)) -> float:
    """1 - Gini coefficient of per-cell point counts (higher = more uniform)."""
    if len(points) == 0:
        return 0.0
    h, w = shape
    gh, gw = grid
    counts = np.zeros(gh * gw)
    for x, y in points:
        cx = min(int(x / w * gw), gw - 1)
        cy = min(int(y / h * gh), gh - 1)
        counts[cy * gw + cx] += 1
    c = np.sort(counts)
    n = len(c)
    if c.sum() == 0:
        return 0.0
    cum = np.cumsum(c)
    gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(1.0 - gini)
