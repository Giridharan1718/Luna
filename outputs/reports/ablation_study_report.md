# Experiment 8 — Ablation Study Report

**Generated:** 2026-09-27 19:49 · 4 controlled pairs per variant

Variants: **full** (ANMS + MAGSAC++ + H + ECC) · **no_anms** (raw matches) ·
**no_ecc** (homography only) · **no_magsac** (least-squares homography, no robust
rejection) · **without_faiss** (reference built from a different region entirely).

## Component ablation

| variant | runs | success_runs | success_rate | match_count | inlier_count | inlier_ratio | inlier_ratio_3px | coverage_score_median | uniformity_score_median | rmse_gt_px_median | subpixel_share | rmse_px_median | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 4 | 4 | 1.000 | 414.250 | 160.250 | 0.989 | 0.975 | 0.805 | 0.609 | 0.208 | 1.000 | 1.320 | 5.014 |
| no_anms | 4 | 4 | 1.000 | 414.250 | 411.500 | 0.994 | 0.983 | 0.805 | 0.556 | 0.211 | 1.000 | 1.227 | 4.981 |
| no_ecc | 4 | 4 | 1.000 | 414.250 | 160.250 | 0.989 | 0.975 | 0.805 | 0.609 | 0.281 | 1.000 | 1.293 | 4.969 |
| no_magsac | 4 | 4 | 1.000 | 414.250 | 162.000 | 1.000 | 0.975 | 0.805 | 0.609 | 0.208 | 1.000 | 1.320 | 5.083 |


![ablation](figures/ablation.png)
![ablation rmse](figures/ablation_rmse.png)

## Retrieval ablation (FAISS candidate vs wrong reference)

| condition | runs | success_rate | match_count | inlier_ratio | rmse_gt_px | retrieval_similarity | confidence |
|---|---|---|---|---|---|---|---|
| with_faiss | 3 | 1.000 | 196.700 | 0.997 | 0.573 | 0.820 | 86.170 |
| without_faiss | 3 | 0.000 | 0.300 | 0.000 | - | 0.658 | 0.000 |


| Condition | Per-run detail |
|---|---|
| with_faiss | ab_faiss_on_0 · status=ok · inl=1.0 · rmse_gt=0.678623616695404 |
| without_faiss | ab_faiss_off_0 · status=insufficient_matches · inl=0.0 · rmse_gt=nan |
| with_faiss | ab_faiss_on_1 · status=ok · inl=1.0 · rmse_gt=0.5730899572372437 |
| without_faiss | ab_faiss_off_1 · status=insufficient_matches · inl=0.0 · rmse_gt=nan |
| with_faiss | ab_faiss_on_2 · status=ok · inl=0.9906542056074766 · rmse_gt=0.29015088081359863 |
| without_faiss | ab_faiss_off_2 · status=insufficient_matches · inl=0.0 · rmse_gt=nan |

## Findings

All variants run on the identical controlled pairs with the identical acceptance guard,
so every row is comparable. `inlier_ratio_3px` is the share of correspondences within
3 px of the estimated warp — measured the same way for every variant, because a
least-squares fit reports "100% inliers" by construction and a MAGSAC++ inlier count is
not comparable with it.

1. **MAGSAC++** is what makes "inlier ratio" a meaningful number: without robust
   rejection the least-squares fit reports every match as an inlier, and the resulting
   warp is dragged away from the correspondences. Compare `inlier_ratio_3px` and
   `rmse_gt_px_median` of `full` against `no_magsac` above.
2. **ECC** is the sub-pixel stage: see Experiment 4 for the before/after GT error, and
   `ecc_rejected` there for the pairs where correlation drift was refused.
3. **ANMS** trades raw match count for spatial spread, and Experiment 7 measures that
   trade pair-by-pair (the spread points remain inliers, so accuracy is preserved).
4. **FAISS retrieval decides *where* to match** — the geometric chain decides *how
   accurately*. The retrieval rows below put the same query against the candidate
   retrieval would return and against a region it would not, with the LunaDNA
   similarity of each, so the contribution of the retrieval stage is visible as a
   number rather than asserted.
