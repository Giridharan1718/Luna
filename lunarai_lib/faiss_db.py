"""FAISS IndexFlatIP database with graceful numpy-cosine fallback.

Stores database/faiss_index.bin, embeddings.npy and patch_mapping.pkl —
per 02_TRD Module 4 and the build spec (K=10 retrieval).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class VectorIndex:
    """Unified Top-K search API over FAISS (preferred) or numpy cosine (fallback)."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self.backend = "faiss"
        self.index = None
        self._embs: np.ndarray | None = None
        try:
            import faiss  # noqa: PLC0415
            self.index = faiss.IndexFlatIP(dim)
            self._faiss = faiss
        except Exception:
            self.backend = "numpy"
            self._faiss = None
            if not use_faiss:
                raise

    def add(self, embeddings: np.ndarray) -> None:
        embs = np.ascontiguousarray(embeddings.astype(np.float32))
        if self.backend == "faiss":
            self.index.add(embs)
        else:
            self._embs = embs if self._embs is None else np.vstack([self._embs, embs])

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.backend == "faiss" else (
            0 if self._embs is None else len(self._embs))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(query).astype(np.float32))
        if self.backend == "faiss":
            scores, idx = self.index.search(q, k)
            return scores, idx
        if self._embs is None or len(self._embs) == 0:
            return np.zeros((q.shape[0], 0)), np.zeros((q.shape[0], 0), dtype=int)
        k = min(k, len(self._embs))
        sims = self._embs @ q.T  # (N, Q) for normalized vectors
        order = np.argsort(-sims, axis=0)[:k].T
        scores = np.take_along_axis(sims.T, order, axis=1)
        return scores.astype(np.float32), order.astype(int)

    def save(self, dir_path: Path, embeddings: np.ndarray, mapping: list[str]) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "embeddings.npy", embeddings.astype(np.float32))
        with open(dir_path / "patch_mapping.pkl", "wb") as fh:
            pickle.dump(mapping, fh)
        if self.backend == "faiss":
            self._faiss.write_index(self.index, str(dir_path / "faiss_index.bin"))
        else:
            with open(dir_path / "faiss_index.bin", "wb") as fh:
                pickle.dump({"backend": "numpy", "dim": self.dim}, fh)

    @classmethod
    def load(cls, dir_path: Path) -> tuple["VectorIndex", np.ndarray, list[str]]:
        embeddings = np.load(dir_path / "embeddings.npy")
        with open(dir_path / "patch_mapping.pkl", "rb") as fh:
            mapping = pickle.load(fh)
        idx_file = dir_path / "faiss_index.bin"
        dim = embeddings.shape[1]
        try:
            import faiss  # noqa: PLC0415
            if idx_file.exists() and idx_file.stat().st_size > 64:
                obj = cls.__new__(cls)
                obj.dim = dim
                obj.backend = "faiss"
                obj._faiss = faiss
                obj.index = faiss.read_index(str(idx_file))
                obj._embs = None
                return obj, embeddings, mapping
        except Exception:
            pass
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.backend = "numpy"
        obj._faiss = None
        obj.index = None
        obj._embs = np.ascontiguousarray(embeddings.astype(np.float32))
        return obj, embeddings, mapping


def describe_backend() -> str:
    try:
        import faiss  # noqa: PLC0415
        return f"faiss {getattr(faiss, '__version__', 'ok')}"
    except Exception:
        return "numpy-cosine fallback"
