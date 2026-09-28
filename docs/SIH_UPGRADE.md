# LunarAI — SIH Grand Finale Upgrade

**Date:** 2026-09-27 · **Scope:** ISRO PS 26166 Phase 1–6 upgrade program
(`docs/SIH_UPGRADE.md` documents every design decision; `outputs/sih_upgrade/`
holds the machine-readable evidence).

## What was added

| Phase | Upgrade | Status | Evidence |
|---|---|---|---|
| 1.1 | 70/15/15 **geographic train/val/test split** (no location leakage) | ✅ done | `outputs/splits/split_metadata.json` (+ per-split CSVs, nb17) |
| 1.2 | **ElevationProfile** LOLA Tycho-track corridor mapping | ✅ done | `outputs/sih_upgrade/patch_index_with_elevation.csv` |
| 1.3 | **SunAngle** training/retrieval integration | ✅ done | `lunarai_lib/sunangle.py`; sun-aware triplets + re-ranking |
| 2.4 | **PCA compression** 512→256/128 + ablation | ✅ done | `outputs/sih_upgrade/pca_ablation.csv` |
| 2.5 | **LunaDNA retrain**: 100 epochs, sun-aware triplets, hard-negative mining, augs | ✅ done | `logs/sih_training.log`, `outputs/metrics/lunadna_training_report.json` |
| 3.6 | **TMC-2↔IIRS** pair (was missing from all pair sets) | ✅ done | `outputs/sih_upgrade/tmc2_iirs_validation.csv` |
| 3.7 | **Full 5-bin sun validation** (measured + labelled synthetic-illumination protocol) | ✅ done | `outputs/sih_upgrade/sun_bins_full.csv` |
| 3.8 | **Scale 0.5–8×** with equalized-pyramid retry + retrieval@scale | ✅ done | `outputs/sih_upgrade/scale_extended.csv` |
| 4.9a | **AKAZE + RANSAC** benchmark | ✅ done | `outputs/sih_upgrade/benchmarks_extended.csv` |
| 4.9b | **SuperPoint+SuperGlue** benchmark (offline fallback) | ✅ done | same CSV; weights auto-cached |
| 5 | ISRO requirement fixes: viewpoint escalation ladder, multiscale retries | ✅ done | `lunarai_lib/validation.py` |
| 6 | nb17–nb19, the 📊 Analytics page tabs for every upgrade, 36 unit tests | ✅ done | `notebooks/`, `app/streamlit_app.py`, `tests/` |

## Key results (upgrade runs)

- **TMC-2↔IIRS**: `ok` — the fifth sensor pair now validates (guard-passed).
- **Sun bins**: all five now report numbers; empty bins use the labelled
  synthetic-illumination protocol (flagged in-table, not passed off as measured).
- **Scale**: 0.5× rescued (`ok`), 1×/2× ok; 4×/8× remain honestly
  `insufficient_matches` — the equalized patch is feature-free regolith.
- **Viewpoint (real ncf↔ncn)**: escalates to affine rescue with a physically
  correct compression (~1.56×) but caps at 7/8 inliers → still guard-rejected;
  documented as future work (stereo-aware matching), no number fabricated.
- **SuperGlue**: `available=True` via the direct-download fallback; runs on the
  identical pair set and guard as every other method.

## Run order (full reproduction)

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python -m pytest tests/ -q                          # unit tests
python scripts/train_lunadna.py --epochs 100 --triplet-mode sun \
    --hard-negative-mining --mine-every 8 --time-budget-min 210
python scripts/run_sih_upgrade.py                   # ~3 min: all Phase 1–5 sections
python scripts/run_validation.py                    # base suite (exp1–8)
python scripts/run_extended_validation.py           # extended suite
python notebooks/build_notebooks.py && python notebooks/execute_notebooks.py
streamlit run app/streamlit_app.py                  # 📊 Analytics tabs
```

## Honest-claims ledger

- SuperGlue outdoor weights trained on internet photos — lunar deltas are
  directional; treat its row as a reference, not a ceiling.
- 4×/8× scale rejections and the ncf↔ncn rejection are documented limits, not
  hidden failures; both have a named, concrete remedy in future work.
- Synthetic-illumination sun-bin rows are labelled `synthetic_illumination`
  in `sun_bins_full.csv`.
