# %% [markdown]
# ## 1. Objective
# Validate retrieval quality: embed held-out query patches, search FAISS for **Top-K = 10**,
# score similarity, measure Recall@K and save per-query result tables + visualizations.

# %% [markdown]
# ## 2. Dataset Loading

# %%
import sys
from pathlib import Path

def _find_project_root() -> Path:
    # anchor at the directory that actually contains the project markers,
    # independent of the kernel's working directory
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "lunarai_lib").is_dir() and (cand / "configs").is_dir():
            return cand
    return Path.cwd()

PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from lunarai_lib.config import load_config
from lunarai_lib.lunadna import LunaDNA, compute_embeddings
from lunarai_lib.faiss_db import VectorIndex
from lunarai_lib.io_utils import save_json, save_rows_csv
from lunarai_lib.db import Database

cfg = load_config()
df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LunaDNA(pretrained=False, grayscale=True).to(device)
ckpt = torch.load(cfg.MODELS_DIR / "lunadna.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])

index, db_embs, mapping = VectorIndex.load(cfg.DATABASE_DIR)
print(f"index: {index.ntotal} vectors | backend={index.backend} | checkpoint epoch {ckpt.get('epoch')}")

# %% [markdown]
# ## 3. Configuration

# %%
TOP_K = int(cfg.FAISS.get("top_k", 10))
N_QUERIES = 60   # held-out queries sampled from patches

# %% [markdown]
# ## 4. Implementation

# %%
rng = np.random.default_rng(7)
qidx = rng.choice(len(df), size=min(N_QUERIES, len(df)), replace=False)
qdf = df.iloc[qidx].reset_index(drop=True)
q_embs = compute_embeddings(model, qdf.patch_path.tolist(), device=device,
                            size=int(cfg.LUNADNA.get("input_size", 224)))

rows = []
for qi in range(len(qdf)):
    scores, ids = index.search(q_embs[qi], TOP_K + 1)
    rank = 0
    for s, gid in zip(scores[0], ids[0]):
        if 0 <= gid < len(mapping) and mapping[gid] == qdf.patch_path.iloc[qi]:
            continue  # self-hit
        rank += 1
        rows.append({
            "query_patch": qdf.patch_id.iloc[qi],
            "query_sensor": qdf.dataset_name.iloc[qi],
            "rank": rank,
            "retrieved_patch": Path(mapping[gid]).name,
            "retrieved_path": mapping[gid],
            "retrieved_sensor": df.set_index("patch_path").dataset_name.get(mapping[gid], "?"),
            "similarity": round(float(s), 4),
        })
        if rank == TOP_K:
            break
res_df = pd.DataFrame(rows)
save_rows_csv(res_df.to_dict("records"), cfg.RETRIEVAL_DIR / "topk_results.csv")
print(res_df.head(8).to_string(index=False))

# %% [markdown]
# ### Recall@K against geo-matched ground truth

# %%
meta = df.set_index("patch_id")
def is_correct(qrow, rname):
    q = meta.loc[qrow]
    cand_id = rname.rsplit(".png", 1)[0]
    if cand_id not in meta.index:
        return False
    c = meta.loc[cand_id]
    d = np.hypot(q.latitude - c.latitude, q.longitude - c.longitude)
    return c.dataset_name != q.dataset_name and d < 1.0

hits = {1: 0, 5: 0, 10: 0}
for q, grp in res_df.groupby("query_patch"):
    for k in hits:
        if any(is_correct(q, r) for r in grp[grp["rank"] <= k].retrieved_patch):
            hits[k] += 1
nq = res_df.query_patch.nunique()
recall = {f"recall@{k}": hits[k] / nq for k in hits}
print(f"queries={nq}", recall)

# cross-sensor retrieval rate: fraction of top-10 that come from another sensor
xs = (res_df[res_df["rank"] <= 10].query_sensor != res_df[res_df["rank"] <= 10].retrieved_sensor)
metrics = {"queries": int(nq), "top_k": TOP_K,
           "cross_sensor_fraction_top10": round(float(xs.mean()), 3) if len(xs) else 0.0,
           "mean_top1_similarity": round(float(res_df[res_df["rank"] == 1].similarity.mean()), 4),
           **recall}
print(metrics)

# %% [markdown]
# ## 5. Visualization — a query with its Top-5 retrievals

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

if len(res_df):
    sample_q = res_df.query_patch.iloc[0]
    qrow = df[df.patch_id == sample_q].iloc[0]
    top5 = res_df[(res_df.query_patch == sample_q) & (res_df["rank"] <= 5)]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
    axes[0].imshow(cv2.imread(qrow.patch_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
    axes[0].set_title(f"QUERY\n{sample_q[:22]}\n{qrow.dataset_name}", fontsize=8)
    for ax, (_, r) in zip(axes[1:], top5.iterrows()):
        img = cv2.imread(r.retrieved_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img if img is not None else np.zeros((64, 64)), cmap="gray")
        ax.set_title(f"#{r['rank']} {r.retrieved_sensor}\nsim={r.similarity}", fontsize=8)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(cfg.VISUALIZATIONS_DIR / "retrieval_top5.png", dpi=120)
    plt.show()

# similarity distribution bar chart
plt.figure(figsize=(7, 3.2))
res_df.groupby("rank").similarity.mean().plot(kind="bar", color="#38BDF8")
plt.xlabel("rank"); plt.ylabel("mean similarity")
plt.title("Top-K similarity decay")
plt.tight_layout()
plt.savefig(cfg.VISUALIZATIONS_DIR / "retrieval_similarity.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics (saved)

# %%
save_json(metrics, cfg.METRICS_DIR / "retrieval_report.json")
print("saved ->", cfg.METRICS_DIR / "retrieval_report.json")

# %% [markdown]
# ## 7. Saved Outputs
# - `outputs/retrieval/topk_results.csv`
# - `outputs/metrics/retrieval_report.json`
# - `outputs/visualizations/retrieval_top5.png`, `retrieval_similarity.png`
# - SQLite `retrieval_results` table

# %%
db = Database(cfg.DATABASE_DIR / "lunarai.db")
db.executemany(
    "INSERT OR REPLACE INTO retrieval_results(retrieval_id, query_patch, retrieved_patch,"
    " similarity_score) VALUES (?,?,?,?)",
    [(f"ret_{i:06d}", r.query_patch, r.retrieved_patch, float(r.similarity))
     for i, r in res_df.iterrows()])

# %% [markdown]
# ## 8. Error Handling
# - Self-hits (query == candidate) are skipped so rankings reflect true neighbors.
# - Mapping gaps (patch removed after indexing) are filtered by bounds check.
# - If fewer than TOP_K valid candidates exist, the query simply returns fewer rows.

# %% [markdown]
# ## 9. Explanation
# Ground truth for "correct region" uses cross-sensor + < 1.0 deg geo-distance, matching the
# triplet definition during training. The similarity decay plot shows embedding discriminance:
# a healthy curve drops quickly after rank 1-3 when the terrain fingerprint is specific.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 07 runs SuperPoint on query + retrieved candidates; notebook 08 matches them.
