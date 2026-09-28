# %% [markdown]
# ## 1. Objective
# SIH Phase 2 — model architecture upgrades: PCA embedding compression
# (no-PCA vs 256 vs 128), evaluated with Recall@1/5/10, mAP, query latency and
# index memory on the held-out TEST split; plus LunaDNA training-curvature
# review from the upgraded 100-epoch schedule.

# %% [markdown]
# ## 2. Prerequisites

# %%
import sys
from pathlib import Path

def _find_project_root() -> Path:
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "lunarai_lib").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

import json

from lunarai_lib.config import load_config

cfg = load_config()
SIH = cfg.OUTPUTS_ROOT / "sih_upgrade"
SIH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Load suite (model + FAISS + catalog)

# %%
import pandas as pd
import torch

from lunarai_lib.lunadna import LunaDNA, compute_embeddings
from lunarai_lib.splits import assign_geo_splits
from lunarai_lib.validation import ValidationSuite

suite = ValidationSuite(cfg)
df = assign_geo_splits(suite.df.copy(), seed=42)
train_df = df[df.split == "train"]
test_df = df[df.split == "test"]
print(f"train {len(train_df)} / test {len(test_df)} patches")

# %% [markdown]
# ## 4. PCA ablation — fit on TRAIN, evaluate on TEST (no leakage)

# %%
from lunarai_lib.pca import pca_ablation

train_embs = compute_embeddings(suite.model, train_df.patch_path.tolist(),
                                device=suite.device, size=224)
test_embs = compute_embeddings(suite.model, test_df.patch_path.tolist(),
                               device=suite.device, size=224)
rows, fitted = pca_ablation(train_embs, test_embs,
                            test_df.dataset_name.to_numpy(),
                            test_df[["latitude", "longitude"]].to_numpy(dtype=float),
                            dims=(512, 256, 128), out_dir=cfg.DATABASE_DIR)
pca_df = pd.DataFrame(rows)
pca_df.to_csv(SIH / "pca_ablation.csv", index=False)
display(pca_df)
for f in fitted:
    if f is not None:
        print(f"saved {cfg.DATABASE_DIR / f'pca_{f.dim_out}.npz'} "
              f"(explained variance {f.explained_variance_ratio:.4f})")

# %% [markdown]
# ## 5. Visualization — quality vs memory frontier

# %%
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(7, 4))
x = pca_df["dim"].astype(str)
ax1.bar(x, pca_df["recall@5"], color="#38BDF8", alpha=0.85, label="Recall@5")
ax1.set_ylim(0, 1.05)
ax1.set_xlabel("embedding dim (512 = no PCA)")
ax1.set_ylabel("Recall@5")
ax2 = ax1.twinx()
ax2.plot(x, pca_df["index_memory_mb"], "o-", color="#F59E0B", label="index memory (MB)")
ax2.set_ylabel("index memory (MB)", color="#F59E0B")
ax1.set_title("PCA compression: retrieval quality vs memory (test split)")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8)
plt.tight_layout()
plt.savefig(SIH / "pca_frontier.png", dpi=130)
plt.show()

# %% [markdown]
# ## 6. LunaDNA training curvature (upgraded schedule)

# %%
train_p = cfg.METRICS_DIR / "lunadna_training_report.json"
if train_p.exists():
    tr = json.loads(train_p.read_text(encoding="utf-8"))
    hist = pd.DataFrame(tr.get("history", []))
    if len(hist):
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(hist.epoch, hist.train_loss, label="train loss")
        ax.plot(hist.epoch, hist.val_loss, label="val loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("triplet loss")
        ax.set_title(f"LunaDNA training (best epoch {tr['metrics'].get('best_epoch')}, "
                     f"mode {tr['metrics'].get('triplet_mode')})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(SIH / "lunadna_training_curve.png", dpi=130)
        plt.show()
    print(json.dumps(tr.get("metrics", {}), indent=1, default=str))
else:
    print("training report not found yet — run scripts/train_lunadna.py")

# %% [markdown]
# ## 7. Summary
# - PCA stage restored to the SIH architecture, with fitted transforms persisted
#   to database/pca_256.npz / pca_128.npz for pipeline use.
# - The no-PCA vs 256 vs 128 trade-off is measured on held-out test cells —
#   cite the table row that keeps Recall@5 within ~1% of 512-D for deployment.
print("nb18 model upgrades: OK")
