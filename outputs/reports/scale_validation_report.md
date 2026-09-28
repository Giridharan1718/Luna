# Experiment 3 — Scale-Invariant Validation Report

**Generated:** 2026-09-27 19:43 · Matcher superpoint+lightglue

Scale factors 0.5x / 1x / 2x / 4x are applied to the **reference** image via the
known ground-truth homography (a 2x reference renders the terrain at twice the
resolution of the query, 0.5x at half). `rmse_gt_px` = RMSE of the estimated warp
against that ground truth, in reference pixels.

| combo | scale_factor | runs | success_runs | success_rate | match_count | inlier_ratio | inlier_ratio_3px | coverage_score | rmse_gt_px_median | subpixel_share | ssim | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OHRC vs TMC-2 | 0.500 | 1 | 1 | 1.000 | 255.000 | 0.970 | 0.866 | 0.594 | 1.304 | 0.000 | 0.083 | 10.325 |
| OHRC vs TMC-2 | 1.000 | 1 | 1 | 1.000 | 139.000 | 0.855 | 0.771 | 0.562 | 0.782 | 1.000 | 0.301 | 5.038 |
| OHRC vs TMC-2 | 2.000 | 1 | 1 | 1.000 | 134.000 | 0.933 | 0.800 | 0.266 | 0.294 | 1.000 | 0.584 | 4.986 |
| OHRC vs TMC-2 | 4.000 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | None | - | 35.097 |
| OHRC vs LRO NAC | 0.500 | 1 | 1 | 1.000 | 106.000 | 0.939 | 0.879 | 0.500 | 1.127 | 0.000 | 0.073 | 5.049 |
| OHRC vs LRO NAC | 1.000 | 1 | 1 | 1.000 | 353.000 | 0.988 | 0.957 | 0.688 | 0.257 | 1.000 | 0.601 | 5.061 |
| OHRC vs LRO NAC | 2.000 | 1 | 1 | 1.000 | 180.000 | 0.970 | 0.881 | 0.266 | 0.312 | 1.000 | 0.683 | 5.131 |
| OHRC vs LRO NAC | 4.000 | 1 | 0 | 0.000 | 3.000 | 0.000 | 0.000 | 0.000 | - | None | - | 35.716 |
| TMC-2 vs LRO NAC | 0.500 | 1 | 1 | 1.000 | 201.000 | 1.000 | 1.000 | 0.906 | 0.136 | 1.000 | 0.185 | 5.321 |
| TMC-2 vs LRO NAC | 1.000 | 1 | 1 | 1.000 | 780.000 | 0.996 | 0.987 | 0.984 | 0.231 | 1.000 | 0.817 | 5.247 |
| TMC-2 vs LRO NAC | 2.000 | 1 | 1 | 1.000 | 33.000 | 0.588 | 0.529 | 0.125 | 4.115 | 0.000 | 0.798 | 5.277 |
| TMC-2 vs LRO NAC | 4.000 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | None | - | 35.088 |
| TMC-2 vs IIRS | 0.500 | 1 | 1 | 1.000 | 182.000 | 1.000 | 1.000 | 0.859 | 0.251 | 1.000 | 0.213 | 5.409 |
| TMC-2 vs IIRS | 1.000 | 1 | 1 | 1.000 | 708.000 | 0.986 | 0.941 | 0.969 | 0.192 | 1.000 | 0.850 | 5.441 |
| TMC-2 vs IIRS | 2.000 | 1 | 0 | 0.000 | 2.000 | 0.000 | 0.000 | 0.000 | - | None | - | 34.946 |
| TMC-2 vs IIRS | 4.000 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | None | - | 34.452 |


![scale rmse](figures/scale_rmse.png)
![scale matches](figures/scale_matches.png)
![scale inlier](figures/scale_inlier.png)

## Interpretation

Scale is applied to the **reference** through the ground-truth homography, so each row
is the residual scale budget LunarAI must absorb *after* both sides were resampled to a
common working GSD (see the GSD protocol in Experiment 1). Registration error is the
median over runs that passed the acceptance guard; `success_rate` reports how many
pairs produced an accepted warp at that scale, which is the honest envelope of a
single-scale SuperPoint/LightGlue front end.

GSD itself is not left to the matcher: the pipeline resamples every product to the
working GSD from its PDS label before this stage (OHRC 0.24 m/px vs TMC-2 ~5 m/px is
~20x), and LunaDNA is scale-robust by construction because embeddings are computed on
resized 224x224 inputs — that is what makes retrieval across sensors possible at all.
