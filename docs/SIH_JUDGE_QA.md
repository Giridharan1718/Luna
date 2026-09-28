# LunarAI — SIH Grand Finale Judge Q&A (upgrade supplement)

Companion to the Q&A embedded in the validation reports. These cover the
Phase 1–5 upgrades specifically.

**Q1. Where is your test split?**
70/15/15 geographic cell split — the split unit is a lat/lon cell, never a
patch, so no terrain is shared across splits (`outputs/splits/split_metadata.json`).
A leakage assertion runs in nb17: every geo cell lives in exactly one split.
Retrieval/PCA numbers are evaluated on held-out test cells.

**Q2. Why PCA, and what does it cost you?**
The PS architecture specifies a PCA stage between embeddings and FAISS. We fit
on train cells only, persist `pca_256.npz`/`pca_128.npz`, and report the full
trade-off (Recall@1/5/10, mAP, latency, memory) in `pca_ablation.csv`. The
deployment dim is whichever keeps Recall@5 within ~1% of 512-D.

**Q3. How does SunAngle.csv actually affect the system?**
Three ways: (1) sun-aware triplet sampling — negatives preferentially come from
a different sun regime, so illumination invariance is trained; (2) per-patch
sun-bin labels; (3) illumination-aware re-ranking subtracts a λ·|Δsun| penalty
at retrieval time.

**Q4. ElevationProfile.csv has 100 rows — what can it possibly do?**
It is a LOLA ground track over Tycho, not a global DEM. We map it as a corridor:
patches within 1° of the track receive track elevation and a terrain band used
as retrieval metadata; coverage is reported honestly and the rest stays NaN
(see `patch_index_with_elevation.csv`). Claiming global elevation coverage
would be false.

**Q5. TMC-2↔IIRS — why was it missing and what does it show?**
It was a pair-set gap; it now runs in the base suite, the extended driver and
the upgrade driver, all guard-passed. Both sensors are mid-latitude with ~37°
sun — it is the pair an evaluator checks first against the PS.

**Q6. Your benchmark ranks SIFT #1 — so is LunarAI worse?**
No. SIFT wins on single-pair reliability at this scale; LunarAI is at parity
there and is the only system on the bench that also solves *where a patch comes
from* (retrieval, 286× better than chance) and *spatial uniformity*. The bench
now also includes AKAZE and SuperPoint+SuperGlue on the identical pairs and guard.

**Q7. What did SuperGlue show?**
Its row is in the same table. Outdoor-trained SuperGlue transfers imperfectly
to lunar imagery (domain gap is directional — internet photos vs orbital
grayscale), which is exactly why we benchmark rather than assume.

**Q8. What happens at 4×/8× scale?**
The equalized-pyramid retry rescues the coarse side (0.5× now solves; 4×/8×
find no features after equalization — 96-px feature-free regolith). The honest
envelope is 0.5–2× natively, 0.5–4× with retrieval-assisted scale normalization;
beyond that the guard rejects rather than inventing a warp.

**Q9. Why is the real forward-vs-nadir pair still rejected?**
The escalation ladder (pyramid retry → affine rescue → loose-matcher affine)
produces a physically correct compression (~1.56× in x — the epipolar
compression you expect for forward-looking vs nadir) but caps at 7/8 inliers,
below the minimum-inlier guard. We will not lower the guard to manufacture a
registration; the remedy is stereo-aware matching (epipolar-constrained
descriptors), which is future work. The rejection is documented, not hidden.

**Q10. What is calibrated about your confidence score?**
It is a documented weighted blend (similarity 25, inliers@3px 30, RMSE 30,
coverage 15) with similarity-band calibration on retrieval error; it is not a
probability. The calibration table ships in `localization_report.csv`.
