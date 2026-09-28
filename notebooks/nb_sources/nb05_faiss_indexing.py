# %% [markdown]
# ## 1. Objective
# Embed every patch with LunaDNA and build the **FAISS IndexFlatIP** database:
# `database/faiss_index.bin`, `database/embeddings.npy`, `database/patch_mapping.pkl`.

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
from lunarai_lib.faiss_db import VectorIndex, describe_backend
from lunarai_lib.io_utils import save_json
from lunarai_lib.db import Database

cfg = load_config()
df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{len(df)} patches | device={device} | backend={describe_backend()}")

# %% [markdown]
# ## 3. Configuration

# %%
BATCH = 64
TOP_K = int(cfg.FAISS.get("top_k", 10))

# %% [markdown]
# ## 4. Implementation

# %%
model = LunaDNA(pretrained=False, grayscale=True).to(device)
ckpt_path = cfg.MODELS_DIR / "lunadna.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print("loaded checkpoint from", ckpt_path, "| epoch", ckpt.get("epoch"))
else:
    print("WARNING: no trained checkpoint found - using random weights (run NB04 first)")

paths = df.patch_path.tolist()
embeddings = compute_embeddings(model, paths, device=device, batch_size=BATCH,
                                size=int(cfg.LUNADNA.get("input_size", 224)))
print("embedding matrix:", embeddings.shape)

# %%
index = VectorIndex(dim=embeddings.shape[1], use_faiss=True)
index.add(embeddings)
mapping = paths  # row i of embeddings corresponds to mapping[i]
index.save(cfg.DATABASE_DIR, embeddings, mapping)
print(f"index backend={index.backend} ntotal={index.ntotal}")
print("saved ->", cfg.DATABASE_DIR / "faiss_index.bin")

# sanity round-trip
index2, embs2, map2 = VectorIndex.load(cfg.DATABASE_DIR)
s, i = index2.search(embs2[0], 3)
print("self-query top-3:", [(round(float(x), 4), Path(map2[j]).name[:34]) for x, j in zip(s[0], i[0])])

# %% [markdown]
# ### SQLite embeddings table

# %%
db = Database(cfg.DATABASE_DIR / "lunarai.db")
db.executemany(
    "INSERT OR REPLACE INTO embeddings(embedding_id, patch_id, vector_path, model_version)"
    " VALUES (?,?,?,?)",
    [(f"emb_{i:06d}", str(df.patch_id.iloc[i]),
      str(cfg.DATABASE_DIR / "embeddings.npy"), "lunadna_v1") for i in range(len(df))])
print("embeddings rows:", db.query("SELECT COUNT(*) FROM embeddings"))

# %% [markdown]
# ## 5. Visualization — embedding norm/similarity sanity

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

self_sim = np.einsum("ij,ij->i", embeddings, embeddings)
# off-diagonal pairwise similarity sample (shows embedding discriminance)
rng_v = np.random.default_rng(0)
i_idx = rng_v.integers(0, len(embeddings), 4000)
j_idx = rng_v.integers(0, len(embeddings), 4000)
mask = i_idx != j_idx
pair_sim = np.einsum("ij,ij->i", embeddings[i_idx[mask]], embeddings[j_idx[mask]])
plt.figure(figsize=(7, 3.2))
plt.hist(pair_sim, bins=40, color="#38BDF8")
plt.title("Pairwise cosine similarity between random patches (L2-normalized)")
plt.xlabel("cosine"); plt.ylabel("pairs")
plt.tight_layout()
plt.savefig(cfg.VISUALIZATIONS_DIR / "embedding_selfsim.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics

# %%
metrics = {"n_vectors": int(embeddings.shape[0]), "dim": int(embeddings.shape[1]),
           "backend": index.backend, "top_k": TOP_K,
           "mean_self_similarity": round(float(self_sim.mean()), 4)}
print(metrics)

# %% [markdown]
# ## 7. Saved Outputs
# - `database/faiss_index.bin`, `database/embeddings.npy`, `database/patch_mapping.pkl`
# - `database/lunarai.db` embeddings table
# - `outputs/visualizations/embedding_selfsim.png`

# %%
save_json(metrics, cfg.METRICS_DIR / "faiss_index_report.json")

# %% [markdown]
# ## 8. Error Handling
# - If faiss-cpu is unavailable on this Python, VectorIndex transparently falls back to a
#   numpy cosine index with identical search semantics and still writes faiss_index.bin
#   (as a marker) plus embeddings.npy / patch_mapping.pkl.
# - Missing checkpoint -> explicit warning; a random-weight index is still built so the
#   pipeline remains executable end-to-end for integration testing.

# %% [markdown]
# ## 9. Explanation
# IndexFlatIP performs exact inner-product search; with L2-normalized vectors inner product
# equals cosine similarity, which is the ranking we want for terrain fingerprints. Exact
# search is the right SIH-scale choice (sub-ms for tens of thousands of vectors); IVF-PQ is
# the documented future scale-up path.

# %% [markdown]
# ## 10. Next-Step Integration
# Notebook 06 loads the saved index and runs Top-K retrieval + Recall@K evaluation.
