"""SQLite persistence per 05_Backend_Schema (images, patches, embeddings,
retrieval_results, correspondences, registration_results, experiments, reports)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
    image_id TEXT PRIMARY KEY,
    filename TEXT,
    mission TEXT,
    sensor TEXT,
    latitude REAL,
    longitude REAL,
    resolution REAL,
    sun_angle REAL,
    acquisition_date TEXT
);
CREATE TABLE IF NOT EXISTS patches(
    patch_id TEXT PRIMARY KEY,
    image_id TEXT,
    patch_path TEXT,
    latitude REAL,
    longitude REAL,
    patch_size INTEGER,
    sensor TEXT
);
CREATE TABLE IF NOT EXISTS embeddings(
    embedding_id TEXT PRIMARY KEY,
    patch_id TEXT,
    vector_path TEXT,
    model_version TEXT
);
CREATE TABLE IF NOT EXISTS retrieval_results(
    retrieval_id TEXT PRIMARY KEY,
    query_patch TEXT,
    retrieved_patch TEXT,
    similarity_score REAL
);
CREATE TABLE IF NOT EXISTS correspondences(
    match_id TEXT PRIMARY KEY,
    query_image TEXT,
    reference_image TEXT,
    total_matches INTEGER,
    inlier_matches INTEGER
);
CREATE TABLE IF NOT EXISTS registration_results(
    registration_id TEXT PRIMARY KEY,
    source_image TEXT,
    reference_image TEXT,
    rmse REAL,
    inlier_ratio REAL,
    coverage REAL
);
CREATE TABLE IF NOT EXISTS experiments(
    experiment_id TEXT PRIMARY KEY,
    model_name TEXT,
    epoch INTEGER,
    loss REAL,
    accuracy REAL,
    timestamp TEXT
);
CREATE TABLE IF NOT EXISTS reports(
    report_id TEXT PRIMARY KEY,
    report_path TEXT,
    generated_at TEXT
);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def executemany(self, sql: str, seq: list[tuple]) -> None:
        self.conn.executemany(sql, seq)
        self.conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self.conn.execute(sql, params).fetchall()

    def query_dicts(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
