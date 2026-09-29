# %% [markdown]
# ## 1. Objective
# Assemble and launch the **Streamlit application** — navigation is exactly nine
# destinations: Mission Dashboard, Dataset Manager, LunaDNA Retrieval, Image Registration,
# Validation Analytics, Reports & Exports, Explainability, System Health, About LunarAI —
# backed by all pipeline artifacts, and verify the end-to-end chain is executable.
#
# `app_build.py` is the source of truth: this notebook calls `write_dashboard()` which
# regenerates `app/streamlit_app.py`, so hand-edits to the generated file are never kept.

# %% [markdown]
# ## 2. Dataset Loading — artifact inventory

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

import json

from lunarai_lib.config import load_config
from lunarai_lib.io_utils import load_json, save_json

cfg = load_config()
inventory = {
    "dataset_report": (cfg.DATASET_ANALYSIS_DIR / "dataset_report.json").exists(),
    "patch_index": (cfg.PATCHES_DIR / "patch_index.csv").exists(),
    "lunadna_model": (cfg.MODELS_DIR / "lunadna.pt").exists(),
    "faiss_index": (cfg.DATABASE_DIR / "faiss_index.bin").exists(),
    "embeddings": (cfg.DATABASE_DIR / "embeddings.npy").exists(),
    "topk_results": (cfg.RETRIEVAL_DIR / "topk_results.csv").exists(),
    "superpoint_features": len(list(cfg.MATCHING_DIR.glob("feat_*.npz"))) > 0,
    "lightglue_matches": len(list(cfg.MATCHING_DIR.glob("matches_*.npz"))) > 0,
    "anms": len(list(cfg.MATCHING_DIR.glob("anms_*.npz"))) > 0,
    "registered_images": len(list(cfg.REGISTRATION_DIR.glob("*_registered.png"))) > 0,
    "evaluation_report": (cfg.METRICS_DIR / "evaluation_report.json").exists(),
    "pdf_report": (cfg.PDF_DIR / "LunarAI_Evaluation_Report.pdf").exists(),
}
print(json.dumps(inventory, indent=2))

# %% [markdown]
# ## 3. Configuration

# %%
APP_DIR = PROJECT_ROOT / "app"
APP_DIR.mkdir(exist_ok=True)
APP_PATH = APP_DIR / "streamlit_app.py"
DASH_PORT = 8501

# %% [markdown]
# ## 4. Implementation — dashboard is generated from a dedicated module

# %%
# The dashboard source lives in app/build_dashboard.py (kept as real .py for import-safety
# in Streamlit's runtime). This notebook writes it, verifies it compiles, and boots it.
from app_build import write_dashboard

write_dashboard(APP_PATH, PROJECT_ROOT)
print("dashboard written ->", APP_PATH)

import py_compile
py_compile.compile(str(APP_PATH), doraise=True)
print("dashboard compiles OK")

# %% [markdown]
# ### End-to-end pipeline verification (single run through LunarAIPipeline)

# %%
import cv2
import numpy as np
import pandas as pd
import torch

from lunarai_lib.lunadna import LunaDNA, compute_embeddings
from lunarai_lib.faiss_db import VectorIndex
from lunarai_lib.matching import SuperPointLightGlue
from lunarai_lib.pipeline import LunarAIPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
model = LunaDNA(pretrained=False, grayscale=True).to(device)
ckpt = torch.load(cfg.MODELS_DIR / "lunadna.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
index, embs, mapping = VectorIndex.load(cfg.DATABASE_DIR)
matcher = SuperPointLightGlue(max_keypoints=int(cfg.MATCHING.get("superpoint_max_kp", 2048)),
                              match_threshold=float(cfg.MATCHING.get("lightglue_threshold", 0.2)),
                              device=device)
pipe = LunarAIPipeline(model, index, mapping, matcher,
                       cfg=dict(cfg.FAISS, **cfg.MATCHING, input_size=int(cfg.LUNADNA.get("input_size", 224))),
                       device=device)

# pick a query + top-1 retrieval as reference
qrow = df.iloc[10]
q_img = cv2.imread(qrow.patch_path, cv2.IMREAD_GRAYSCALE)
cands = pipe.retrieve(q_img, k=3)
ref_path = cands[0]["patch_path"] if cands else qrow.patch_path
ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
print("query:", qrow.patch_id, "| reference:", Path(ref_path).stem,
      "| score:", round(cands[0]["score"], 3) if cands else None)

result = pipe.run(q_img, ref_img, save_dir=cfg.REGISTRATION_DIR / "pipeline_demo",
                  run_name="demo_end_to_end")
print(json.dumps({k: v for k, v in result.items()
                  if k not in ("homography", "homography_refined")}, indent=2, default=str))

# %% [markdown]
# ## 5. Visualization — end-to-end demo artifacts

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

demo_dir = cfg.REGISTRATION_DIR / "pipeline_demo"
panels = ["demo_end_to_end_registered.png", "demo_end_to_end_matches.png",
          "demo_end_to_end_coverage.png"]
fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
for ax, name in zip(np.atleast_1d(axes), panels):
    p = demo_dir / name
    if p.exists():
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        ax.set_title(name.replace("demo_end_to_end_", "").replace(".png", ""))
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(cfg.VISUALIZATIONS_DIR / "pipeline_demo.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. Metrics — system readiness

# %%
runtime = result.get("runtime_s", None)
metrics = {
    "artifacts_present": sum(1 for v in inventory.values() if v),
    "artifacts_total": len(inventory),
    "end_to_end_status": result.get("status"),
    "end_to_end_runtime_s": round(runtime, 2) if runtime else None,
    "end_to_end_rmse_px": result.get("rmse"),
    "end_to_end_inlier_ratio": result.get("inlier_ratio"),
    "end_to_end_coverage": result.get("coverage_score"),
}
print(json.dumps(metrics, indent=2, default=str))
save_json(metrics, cfg.METRICS_DIR / "system_readiness.json")

# %% [markdown]
# ## 7. Saved Outputs
# - `app/streamlit_app.py` (the nine-destination application, generated by `app_build.py`)
# - `outputs/registration/pipeline_demo/*` (registered image, matches, coverage)
# - `outputs/metrics/system_readiness.json`
# - `outputs/visualizations/pipeline_demo.png`

# %% [markdown]
# ## 8. Error Handling
# - Artifact inventory reports exactly which stage outputs are missing.
# - The demo pipeline run exercises every module live: failures surface with explicit
#   status strings ("failed: homography estimation") instead of exceptions.
# - Dashboard launch degrades gracefully: each page checks its artifacts and shows a
#   "run notebook XX first" hint.

# %% [markdown]
# ## 9. Explanation
# The app is a single-file Streamlit program reading saved artifacts + the SQLite DB
# (`lunarai_lib.appdata`) and, for the live pages, the deployed checkpoint and FAISS
# index (`lunarai_lib.live`). Pages are self-contained: no page reads a variable bound
# in another page (enforced by `tests/test_app.py`, which also renders all nine destinations
# headlessly with `streamlit.testing.v1.AppTest`).

# %% [markdown]
# ## 10. Launch / Next-Step Integration
# Run: `streamlit run app/streamlit_app.py --server.port 8501`
# This is the final stage; the dashboard links back to every notebook's outputs.
