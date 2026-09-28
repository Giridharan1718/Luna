"""PCA embedding compression + retrieval-quality ablation — SIH Phase 2.4.

The SIH architecture specifies a PCA stage between ResNet18/LunaDNA embeddings
and the FAISS index. This module:

- fits PCA (512 -> 256 / 128) on *training-split* embeddings only (no test
  leakage into the transform),
- L2-renormalizes projected vectors so IndexFlatIP keeps meaning cosine,
- evaluates retrieval quality against geo/sensor ground truth for every
  variant: Recall@1, Recall@5, Recall@10, mAP, query latency, and memory,
- persists fitted transforms under database/pca_<dim>.npz for the pipeline.

Ground truth mirrors the validation suite convention: a candidate is relevant
for query q when it comes from the same source terrain — same sensor AND
great-circle distance < 0.5 deg (lunar surface meters), matching
evaluate_retrieval() in validation.py / train_lunadna.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover - sklearn is a hard dep, but stay safe
    PCA = None

MOON_RADIUS_M = 1_737_400.0


class EmbeddingPCA:
    """Fit/apply/save/load a PCA transform on L2-normalized embeddings."""

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 dim_in: int, dim_out: int, explained: float) -> None:
        self.mean = mean
        self.components = components
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.explained_variance_ratio = explained

    @classmethod
    def fit(cls, embeddings: np.ndarray, dim_out: int) -> "EmbeddingPCA":
        if PCA is None:
            raise RuntimeError("sklearn unavailable")
        p = PCA(n_components=dim_out, svd_solver="full")
        p.fit(np.asarray(embeddings, dtype=np.float64))
        return cls(mean=p.mean_.astype(np.float32),
                   components=p.components_.astype(np.float32),
                   dim_in=int(p.mean_.shape[0]), dim_out=dim_out,
                   explained=float(p.explained_variance_ratio_.sum()))

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32) - self.mean
        z = x @ self.components.T
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (z / norms).astype(np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean, components=self.components,
                            dim_in=self.dim_in, dim_out=self.dim_out,
                            explained=self.explained_variance_ratio)
        return path

    @classmethod
    def load(cls, path: Path) -> "EmbeddingPCA":
        d = np.load(path)
        return cls(mean=d["mean"], components=d["components"],
                   dim_in=int(d["dim_in"]), dim_out=int(d["dim_out"]),
                   explained=float(d["explained"]))


def _relevance_matrix(sensors: np.ndarray, coords: np.ndarray,
                      geo_thresh_deg: float = 0.5) -> np.ndarray:
    """rel[i, j] = 1 iff j is a relevant neighbor of i (same sensor, geo-close)."""
    n = len(sensors)
    rel = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if sensors[i] != sensors[j]:
                continue
            d = np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            if d < geo_thresh_deg:
                rel[i, j] = True
    return rel


def evaluate_retrieval_quality(embs: np.ndarray, sensors: np.ndarray,
                               coords: np.ndarray, top_k: int = 10,
                               max_queries: int = 200,
                               seed: int = 7) -> dict:
    """Recall@{1,5,10}, mAP, query latency (ms) for an embedding matrix."""
    n = len(embs)
    if n < 2:
        return {"recall@1": float("nan"), "recall@5": float("nan"),
                "recall@10": float("nan"), "mAP": float("nan"),
                "latency_ms": float("nan"), "n_queries": 0}
    rel = _relevance_matrix(sensors, coords)
    has_rel = rel.any(axis=1)
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(has_rel)
    if len(pool) == 0:
        return {"recall@1": float("nan"), "recall@5": float("nan"),
                "recall@10": float("nan"), "mAP": float("nan"),
                "latency_ms": float("nan"), "n_queries": 0}
    q_idx = pool if len(pool) <= max_queries else rng.choice(
        pool, size=max_queries, replace=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs_n = embs / norms
    import time
    r1 = r5 = r10 = 0
    ap_sum, ap_n = 0.0, 0
    lat = []
    for qi in q_idx:
        t0 = time.perf_counter()
        sims = embs_n @ embs_n[qi]
        order = np.argsort(-sims)
        order = order[order != qi]
        lat.append((time.perf_counter() - t0) * 1000.0)
        top = order[:top_k]
        rlist = rel[qi, order]
        if rlist[0]:
            r1 += 1
        if rlist[:5].any():
            r5 += 1
        if rlist[:10].any():
            r10 += 1
        # AP over the full ranking, capped at top_k for tractability
        hits = rlist[:top_k]
        if hits.any():
            cum = np.cumsum(hits)
            prec = cum / (np.arange(len(hits)) + 1)
            ap_sum += float(prec[hits].mean())
            ap_n += 1
    nq = len(q_idx)
    return {"recall@1": r1 / nq, "recall@5": r5 / nq, "recall@10": r10 / nq,
            "mAP": (ap_sum / ap_n) if ap_n else float("nan"),
            "latency_ms": float(np.median(lat)), "n_queries": int(nq)}


def memory_mb(embs: np.ndarray) -> float:
    return float(embs.nbytes / (1024 * 1024))


def pca_ablation(train_embs: np.ndarray, eval_embs: np.ndarray,
                 sensors: np.ndarray, coords: np.ndarray,
                 dims: tuple[int, ...] = (512, 256, 128),
                 out_dir: Path | None = None) -> tuple[list[dict], list[EmbeddingPCA]]:
    """Full no-PCA vs 256 vs 128 comparison table.

    train_embs: embeddings used to FIT PCA (train split, no leakage).
    eval_embs : embeddings used for retrieval evaluation (held-out queries).
    """
    rows: list[dict] = []
    fitted: list[EmbeddingPCA] = []
    for d in dims:
        if d >= train_embs.shape[1]:
            embs_eval, embs_train = eval_embs, train_embs
            label = f"no_pca_{embs_eval.shape[1]}"
            pca = None
        else:
            pca = EmbeddingPCA.fit(train_embs, d)
            embs_eval = pca.transform(eval_embs)
            embs_train = pca.transform(train_embs)
            label = f"pca_{d}"
            if out_dir is not None:
                pca.save(out_dir / f"pca_{d}.npz")
        m = evaluate_retrieval_quality(embs_eval, sensors, coords)
        rows.append({
            "variant": label,
            "dim": int(embs_eval.shape[1]),
            "explained_variance": round(pca.explained_variance_ratio, 4) if pca else 1.0,
            "recall@1": round(m["recall@1"], 4),
            "recall@5": round(m["recall@5"], 4),
            "recall@10": round(m["recall@10"], 4),
            "mAP": round(m["mAP"], 4),
            "query_latency_ms": round(m["latency_ms"], 3),
            "index_memory_mb": round(memory_mb(embs_train), 3),
        })
        fitted.append(pca)
    return rows, fitted
