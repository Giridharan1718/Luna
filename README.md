# LunarAI — Multi-Modal Lunar Image Correspondence Platform

Sun-angle- and scale-invariant **correspondence and registration** between modern lunar
orbiter imagery (Chandrayaan-2 **OHRC / TMC-2 / IIRS**) and the historical reference
frameworks (**LRO NAC**, **KAGUYA SELENE TC**).

Built for **SIH 2026 · Problem Statement 26166 · ISRO**.

The deliverable is not a notebook and not a dashboard of charts — it is a **mission
operations console**: a nine-destination Streamlit application that reads real pipeline
artifacts, runs the deployed model live, and reports every number with its provenance.

---

## Quickstart

```bash
# the two OpenMP runtimes in this stack collide on Windows — set this first
export KMP_DUPLICATE_LIB_OK=TRUE          # PowerShell: $env:KMP_DUPLICATE_LIB_OK="TRUE"

python -m streamlit run app/streamlit_app.py --server.port 8501
```

Then open <http://localhost:8501>. Nothing else is required: the console ships with its
artifacts (`outputs/`), its index (`database/lunarai.db`) and its checkpoint
(`models/lunadna.pt`).

Run the test suite:

```bash
python -m pytest tests/ -q
```

---

## The nine destinations

| # | Destination | What it answers |
|---|-------------|-----------------|
| 1 | 🌕 Mission Dashboard | mission headline, operational status, artifact freshness |
| 2 | 📤 Dataset Manager | catalogue composition, quality scoring, splits, uploads |
| 3 | 🧠 LunaDNA Retrieval | live image → embedding → FAISS → ranked matches |
| 4 | 📍 Image Registration | ECC / homography refinement, match overlays, coverage |
| 5 | 📊 Validation Analytics | sub-pixel error, localization, robustness, ablation |
| 6 | 📑 Reports & Exports | generated mission reports and downloadable tables |
| 7 | 🛰 Explainability | why the model matched — correspondence and attention evidence |
| 8 | ⚙ System Health | checkpoints, index, DB, environment, dependency readiness |
| 9 | ℹ About LunarAI | problem statement, architecture, limitations, team |

A **judge mode** (presentation mode) collapses the console into a ten-step narrative and
hides the operator controls — the sidebar and the top header are replaced by the pitch.

---

## Architecture

```
dataset/ ─► patch extraction ─► LunaDNA (CNN + descriptor) ─► FAISS index
                                                                    │
 image upload ─► embed ─► retrieve ─► match (SuperPoint + LightGlue) ─► ECC / homography
                                                                    │
                          outputs/ + SQLite ─► lunarai_lib.appdata ─► nine-page console
```

* `app_build.py` generates `app/streamlit_app.py` — **never hand-edit the generated file**.
* `app/lib/theme.py` injects the entire console stylesheet; `app/lib/design.py` holds the
  design tokens; `app/lib/pages/*.py` are one module per destination.
* `lunarai_lib/appdata.py` reads artifacts; `lunarai_lib/live.py` runs real inference.
* `.streamlit/config.toml` pins the base palette because `st.dataframe` paints its
  interior on a `<canvas>` that injected CSS cannot reach.

Deployed state (live-read, not hard-coded): **940** indexed patches at **512**-d,
FAISS `ntotal = 940`, matcher `superpoint+lightglue`, checkpoint epoch read from
`models/lunadna.pt` at runtime.

---

## Honest metrics

These are reported as measured, including the places where a different method wins.

| Metric | Value | Note |
|--------|-------|------|
| Sub-pixel RMSE after ECC (median) | **0.256 px** | 29 validated pairs |
| Pairs under 1 px | **86.2 %** | same 29 pairs |
| Localization (median) | **466.4 m** | vs 87 773.7 m chance → **188×** |
| Patches retrieved | **940** | 512-d embeddings |
| Robustness cases passed | **6 / 6** | degraded-condition suite |

**Caveats we state up front, every time:**

* **SIFT leads single-pair recall (0.88 vs LunarAI 0.68).** LunarAI leads on
  `inlier_ratio_3px` (0.895) and is the only system here that produces a *localization*.
* The **confidence score is a weighted blend** (25 % similarity + 30 % inliers@3px
  + 30 % RMSE + 15 % coverage) — it is **not** a probability and is never presented as one.
* `outputs/reports/ecc_report.csv` rows use **synthetic `feat_*` pairs** and are
  quarantined inside a caveat expander — they are **never** quoted as accuracy.
* Catalogue coordinates carry **±0.25°** uncertainty.
* **AKAZE is unavailable** in this environment (no `opencv-contrib-python`); the console
  says so instead of silently omitting the row.
* No **KAGUYA cross-mission registration** claim is made — only retrieval-level evidence.

---

## Repository layout

```
app/                  console (generated shell + theme + pages + components)
app_build.py          shell generator — source of truth for navigation
lunarai_lib/          artifact readers, live inference, config
models/               lunadna.pt checkpoint
database/             lunarai.db (FAISS index + metadata)
outputs/              reports, metrics, splits, patches, explainability
notebooks/            the pipeline, stage by stage
docs/                 UI_UX_DESIGN_SYSTEM.md, SIH Judge Q&A, status reports
tests/                architecture, isolation, design-system and render guards
```

See **`docs/UI_UX_DESIGN_SYSTEM.md`** for the full design system, information
architecture, component library and page wireframes, and **`docs/SIH_JUDGE_QA.md`**
for the judge briefing.
