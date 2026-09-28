"""LunaDNA training — SIH Phase 2.5 upgrade.

Upgrades over the pre-Finale trainer:
- 70/15/15 geographic train/val/test split (lunarai_lib.splits): the test
  split is never touched by training, triplet building, or early stopping.
- time-budgeted long schedules: --epochs 100 with --time-budget-min; the loop
  stops cleanly when the budget expires (CPU-only machine friendly).
- richer augmentations (in lunarai_lib.lunadna.PatchDataset): flips, rotation,
  scale jitter (random resized crop), gamma/brightness/contrast, noise.
- triplet modes: adjacent (texture), geo (cross-sensor proximity), sun
  (sun-aware negatives via lunarai_lib.sunangle).
- optional semi-hard negative mining: every --mine-every epochs, embeddings
  are recomputed on the train split and negatives are re-picked as the most
  similar cross-sensor geo-far patches.

SIH schedule:
  python scripts/train_lunadna.py --epochs 100 --triplet-mode sun \
      --hard-negative-mining --time-budget-min 300
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lunarai_lib.config import load_config
from lunarai_lib.lunadna import (LunaDNA, compute_embeddings, make_adjacent_triplets,
                                 make_geo_triplets, train_lunadna)
from lunarai_lib.splits import export_splits
from lunarai_lib.sunangle import make_sun_aware_triplets


def pick_triplets(df: pd.DataFrame, mode: str, seed: int) -> list[tuple[str, str, str]]:
    if mode == "geo":
        return make_geo_triplets(df, seed=seed)
    if mode == "sun":
        return make_sun_aware_triplets(df, seed=seed)
    return make_adjacent_triplets(df, seed=seed)


def mine_hard_negatives(model: LunaDNA, train_df: pd.DataFrame, seed: int,
                        max_neg_pool: int = 2000, device: str = "cpu"
                        ) -> list[tuple[str, str, str]]:
    """Semi-hard mining: negatives = most-similar cross-sensor geo-far patches."""
    paths = train_df.patch_path.tolist()
    embs = compute_embeddings(model, paths, device=device, size=224)
    sensors = train_df.dataset_name.to_numpy()
    coords = train_df[["latitude", "longitude"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    n = len(paths)
    pool = rng.choice(n, size=min(n, max_neg_pool), replace=False)
    triplets: list[tuple[str, str, str]] = []
    for i in pool:
        cross = np.flatnonzero(sensors != sensors[i])
        if not len(cross):
            continue
        d = np.hypot(*(coords[cross] - coords[i]).T)
        far = cross[d > 10.0]
        if not len(far):
            far = cross[d > 3.0]
        if not len(far):
            continue
        sims = embs[far] @ embs[i]
        hardest = far[np.argsort(-sims)[:8]]
        n_idx = int(rng.choice(hardest))
        # positive: same as geo mode (nearest cross-sensor neighbor)
        pos_pool = cross[d < 3.0]
        if not len(pos_pool):
            pos_pool = cross[np.argsort(d)[:1]]
        p_idx = int(rng.choice(pos_pool))
        triplets.append((paths[i], paths[p_idx], paths[n_idx]))
    return triplets


def recall_at_k(model: LunaDNA, df: pd.DataFrame, seed: int = 42,
                max_queries: int = 200, device: str = "cpu") -> dict:
    """Recall@{1,5,10} on a held-out split (same-source + geo-close = relevant)."""
    if len(df) < 8:
        return {}
    embs = compute_embeddings(model, df.patch_path.tolist(), device=device, size=224)
    sensors = df.dataset_name.to_numpy()
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    r = np.random.default_rng(seed)
    qidx = r.choice(len(embs), size=min(max_queries, len(embs)), replace=False)
    sims = embs @ embs.T
    hits = {1: 0, 5: 0, 10: 0}

    def match(j, qi):
        return sensors[j] == sensors[qi] and np.hypot(*(coords[qi] - coords[j])) < 0.5

    for qi in qidx:
        order = np.argsort(-sims[qi])
        order = order[order != qi]
        for k in hits:
            if any(match(j, qi) for j in order[:k]):
                hits[k] += 1
    n = len(qidx)
    return {"recall@1": hits[1] / n, "recall@5": hits[5] / n, "recall@10": hits[10] / n,
            "n_queries": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max-triplets", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triplet-mode", choices=["adjacent", "geo", "sun"], default="sun")
    ap.add_argument("--hard-negative-mining", action="store_true")
    ap.add_argument("--mine-every", type=int, default=10)
    ap.add_argument("--time-budget-min", type=int, default=300,
                    help="wall-clock budget; training stops cleanly when exceeded")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.perf_counter()
    deadline = t_start + args.time_budget_min * 60.0
    print(f"[train] device={device} epochs={args.epochs} mode={args.triplet_mode} "
          f"budget={args.time_budget_min}min", flush=True)

    # ---- 70/15/15 geographic split (test never touched) ----
    patch_csv = cfg.OUTPUTS_ROOT / "patches" / "patch_index.csv"
    df_all = pd.read_csv(patch_csv).dropna(subset=["patch_path"])
    meta = export_splits(df_all, cfg.OUTPUTS_ROOT / "splits", seed=args.split_seed)
    # reload with the assigned split column (geo_cell is NaN-safe in splits.py)
    df = pd.read_csv(cfg.OUTPUTS_ROOT / "splits" / "all_patches_with_splits.csv")
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    print(f"[train] splits: train {len(train_df)} / val {len(val_df)} / "
          f"test {len(test_df)} (cells {meta['total_geo_cells']})", flush=True)

    train_triplets = pick_triplets(train_df, args.triplet_mode, args.seed)
    val_triplets = pick_triplets(val_df, args.triplet_mode, args.seed + 1)
    if not val_triplets:
        borrow = val_df if len(val_df) else train_df.sample(min(150, len(train_df)),
                                                            random_state=args.seed)
        val_triplets = pick_triplets(borrow, args.triplet_mode, args.seed + 2)
        print("[train] note: val triplets borrowed from train regions", flush=True)
    if len(train_triplets) > args.max_triplets:
        sel = np.random.default_rng(args.seed).choice(
            len(train_triplets), args.max_triplets, replace=False)
        train_triplets = [train_triplets[i] for i in sorted(sel)]
    if not train_triplets or not val_triplets:
        print("[train] ERROR: no triplets", flush=True)
        return 2
    print(f"[train] triplets: train {len(train_triplets)} val {len(val_triplets)}", flush=True)

    model = LunaDNA(pretrained=True, grayscale=True).to(device)
    ckpt_path = cfg.MODELS_DIR / "lunadna.pt"
    backup = cfg.MODELS_DIR / "lunadna_pre_upgrade.bak"
    if ckpt_path.exists() and not backup.exists():
        shutil.copy2(ckpt_path, backup)
        print("[train] previous checkpoint archived to lunadna_pre_upgrade.bak", flush=True)
    if ckpt_path.exists() and args.resume:
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["state_dict"])
            print(f"[train] resumed from checkpoint epoch {ck.get('epoch')}", flush=True)
        except Exception as exc:
            print(f"[train] checkpoint load failed ({exc}); starting fresh", flush=True)

    budget_flag = {"expired": False}

    def log_train(msg: str) -> None:
        print(f"[train] {msg}", flush=True)

    info = train_lunadna(model, train_triplets, val_triplets,
                         cfg={"epochs": args.epochs, "batch_size": 32, "lr": 1e-4,
                              "weight_decay": 1e-5, "margin": 0.3,
                              "patience": args.patience, "input_size": 224,
                              "num_workers": 0,
                              "time_budget_s": max(60.0, deadline - time.perf_counter()),
                              "stop_flag": budget_flag},
                         device=device, out_dir=cfg.MODELS_DIR,
                         log_cb=log_train)

    # ---- hard-negative mining rounds (budget permitting) ----
    round_no = 0
    while args.hard_negative_mining and not budget_flag["expired"] \
            and time.perf_counter() < deadline and round_no < 3:
        round_no += 1
        print(f"[train] hard-negative mining round {round_no}", flush=True)
        mined = mine_hard_negatives(model, train_df, seed=args.seed + round_no,
                                    device=device)
        if len(mined) > args.max_triplets:
            sel = np.random.default_rng(args.seed).choice(
                len(mined), args.max_triplets, replace=False)
            mined = [mined[i] for i in sorted(sel)]
        if mined:
            info = train_lunadna(model, mined, val_triplets,
                                 cfg={"epochs": max(4, args.mine_every),
                                      "batch_size": 32, "lr": 3e-5, "weight_decay": 1e-5,
                                      "margin": 0.3, "patience": 4, "input_size": 224,
                                      "num_workers": 0},
                                 device=device, out_dir=cfg.MODELS_DIR,
                                 log_cb=lambda m: print(f"[train] {m}", flush=True))

    # ---- evaluation: val for reference, TEST for the record ----
    metrics: dict = {"best_val_loss": info.get("best_val_loss"),
                     "best_epoch": info.get("best_epoch"), "embedding_dim": 512,
                     "triplet_mode": args.triplet_mode,
                     "hard_negative_mining": bool(args.hard_negative_mining),
                     "epochs_requested": args.epochs,
                     "time_budget_min": args.time_budget_min,
                     "elapsed_s": round(time.perf_counter() - t_start, 1)}
    val_metrics = recall_at_k(model, val_df, seed=args.seed, device=device)
    test_metrics = recall_at_k(model, test_df, seed=args.seed, device=device)
    if val_metrics:
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
    if test_metrics:
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    report = {"metrics": metrics, "history": info.get("history", []),
              "n_train_triplets": len(train_triplets), "n_val_triplets": len(val_triplets),
              "device": device, "model_path": info.get("model_path"),
              "training_seconds": round(time.perf_counter() - t_start, 1),
              "split_metadata": meta,
              "split_files": str(cfg.OUTPUTS_ROOT / "splits")}
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICS_DIR / "lunadna_training_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[train] done in {time.perf_counter() - t_start:.0f}s | "
          f"best_epoch={info.get('best_epoch')}", flush=True)
    print(f"[train] metrics: {json.dumps(metrics, default=str)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
