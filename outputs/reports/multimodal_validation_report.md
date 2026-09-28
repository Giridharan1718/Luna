# Experiment 1 — Multi-modal Validation Report

**Project:** LunarAI — SIH 26166 · **Generated:** 2026-09-27 19:38
**Matcher:** superpoint+lightglue · **Runtime measured on:** CPU

**Data-availability note.** The provided distribution contains no spatially overlapping OHRC/TMC/IIRS/LRO image pairs (OHRC images are south-polar, TMC mid-latitude 2021/2024/2026 products, IIRS is browse-only, LRO is a QuickMap reference export). Cross-sensor behaviour is therefore reported twice: *controlled* pairs (terrain warps with a known ground-truth homography plus measured sensor-appearance transfer — sharpness, contrast, brightness, noise) give precise error figures; *retrieved* pairs are the real cross-sensor candidates FAISS selects, showing raw behaviour on this distribution.

**GSD protocol (why the pairs are GSD-normalized).** Product GSDs span 0.24 m/px (OHRC) to 80 m/px (IIRS), a 300x+ gap that no single-scale keypoint matcher can bridge and that the pipeline never asks it to: both sides are resampled to a common working GSD using the per-product GSD from the PDS labels before matching. The controlled pairs model exactly that regime — shared geometry at a common GSD, with the measured sensor appearance differences (sharpness/contrast/noise) and illumination on top. Residual scale tolerance after that resample is what Experiment 3 measures (±0.5x … 4x).

**Acceptance guard.** A homography fitted to four correspondences reproduces them exactly, so `inlier_ratio = 1.0` on a handful of matches is an artifact of the fit, not a registration. Runs with fewer than 10 correspondences or with a geometrically implausible warp (degenerate/flipped/exploding quad) are therefore reported as rejected, not scored; the tables show `success_runs` next to the medians so both are visible.

Score in each pair: `inlier_ratio` measured after MAGSAC++ on ANMS-filtered matches;
`rmse_gt_px` is the pixel RMSE of the estimated warp against the known ground truth
(controlled pairs only); `ssim`/`ncc` compare the ECC-registered image to the reference.

## Controlled pairs (ground-truth protocol)

| group | runs | success_runs | success_rate | match_count | inlier_count | inlier_ratio | inlier_ratio_3px | coverage_score | rmse_px_median | rmse_gt_px_median | subpixel_share | ssim | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OHRC <-> LRO NAC | 3 | 3 | 1.000 | 381.667 | 164.667 | 0.995 | 0.973 | 0.646 | 1.255 | 0.295 | 1.000 | 0.583 | 5.210 |
| OHRC <-> TMC-2 | 3 | 3 | 1.000 | 221.333 | 116.333 | 0.994 | 0.949 | 0.625 | 1.593 | 0.225 | 1.000 | 0.383 | 5.056 |
| OHRC <-> IIRS | 3 | 3 | 1.000 | 248.667 | 126.667 | 0.982 | 0.953 | 0.651 | 1.451 | 0.283 | 1.000 | 0.405 | 5.067 |
| TMC-2 <-> LRO NAC | 3 | 3 | 1.000 | 712.333 | 224.333 | 0.975 | 0.932 | 0.995 | 1.799 | 0.239 | 1.000 | 0.745 | 5.123 |
| TMC-2 <-> IIRS | 3 | 3 | 1.000 | 576.667 | 183.000 | 0.970 | 0.929 | 0.969 | 1.905 | 0.270 | 1.000 | 0.777 | 5.111 |


![multimodal summary](figures/multimodal_summary.png)

## Retrieved pairs (real FAISS candidates, non-overlapping imagery)

| group | runs | success_runs | success_rate | match_count | inlier_count | inlier_ratio | inlier_ratio_3px | coverage_score | rmse_px_median | rmse_gt_px_median | subpixel_share | ssim | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OHRC <-> LRO NAC | 2 | 0 | 0.000 | 4.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | None | - | 35.541 |
| OHRC <-> TMC-2 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | - | None | - | 35.398 |


`rmse_*_median` are medians over the runs that passed the acceptance guard, so a
single degenerate pair cannot dominate the group.

Per-run detail (all requested metrics)

| group | protocol | pair | status | match_count | inlier_count | inlier_ratio | inlier_ratio_3px | coverage_score | rmse_px | rmse_gt_px | ssim | ncc | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OHRC <-> LRO NAC | controlled | ctl_OHRC_to_LRO NAC_s1_i0 | ok | 355 | 156 | 1.000 | 0.962 | 0.688 | 1.468 | 0.320 | 0.607 | 0.976 | 5.658 |
| OHRC <-> LRO NAC | controlled | ctl_OHRC_to_LRO NAC_s1_i1 | ok | 476 | 196 | 1.000 | 0.985 | 0.672 | 1.255 | 0.146 | 0.643 | 0.979 | 5.006 |
| OHRC <-> LRO NAC | controlled | ctl_OHRC_to_LRO NAC_s1_i2 | ok | 314 | 142 | 0.986 | 0.972 | 0.578 | 1.237 | 0.295 | 0.498 | 0.966 | 4.966 |
| OHRC <-> TMC-2 | controlled | ctl_OHRC_to_TMC-2_s1_i0 | ok | 206 | 108 | 0.982 | 0.918 | 0.656 | 1.843 | 0.641 | 0.382 | 0.956 | 5.087 |
| OHRC <-> TMC-2 | controlled | ctl_OHRC_to_TMC-2_s1_i1 | ok | 259 | 132 | 1.000 | 0.992 | 0.672 | 1.501 | 0.225 | 0.418 | 0.967 | 5.035 |
| OHRC <-> TMC-2 | controlled | ctl_OHRC_to_TMC-2_s1_i2 | ok | 199 | 109 | 1.000 | 0.936 | 0.547 | 1.593 | 0.218 | 0.350 | 0.960 | 5.045 |
| OHRC <-> IIRS | controlled | ctl_OHRC_to_IIRS_s1_i0 | ok | 195 | 105 | 0.946 | 0.892 | 0.688 | 2.091 | 0.810 | 0.347 | 0.948 | 4.967 |
| OHRC <-> IIRS | controlled | ctl_OHRC_to_IIRS_s1_i1 | ok | 326 | 161 | 1.000 | 0.994 | 0.672 | 1.451 | 0.254 | 0.479 | 0.969 | 5.117 |
| OHRC <-> IIRS | controlled | ctl_OHRC_to_IIRS_s1_i2 | ok | 225 | 114 | 1.000 | 0.974 | 0.594 | 1.369 | 0.283 | 0.389 | 0.960 | 5.116 |
| TMC-2 <-> LRO NAC | controlled | ctl_TMC-2_to_LRO NAC_s1_i0 | ok | 743 | 228 | 0.983 | 0.918 | 1.000 | 1.799 | 0.239 | 0.821 | 0.976 | 5.082 |
| TMC-2 <-> LRO NAC | controlled | ctl_TMC-2_to_LRO NAC_s1_i1 | ok | 683 | 215 | 0.973 | 0.941 | 1.000 | 1.712 | 0.184 | 0.719 | 0.970 | 5.118 |
| TMC-2 <-> LRO NAC | controlled | ctl_TMC-2_to_LRO NAC_s1_i2 | ok | 711 | 230 | 0.970 | 0.937 | 0.984 | 1.923 | 0.295 | 0.695 | 0.959 | 5.169 |
| TMC-2 <-> IIRS | controlled | ctl_TMC-2_to_IIRS_s1_i0 | ok | 601 | 180 | 0.989 | 0.973 | 1.000 | 1.598 | 0.270 | 0.842 | 0.976 | 5.157 |
| TMC-2 <-> IIRS | controlled | ctl_TMC-2_to_IIRS_s1_i1 | ok | 522 | 174 | 0.956 | 0.934 | 0.938 | 1.905 | 0.272 | 0.755 | 0.975 | 5.020 |
| TMC-2 <-> IIRS | controlled | ctl_TMC-2_to_IIRS_s1_i2 | ok | 607 | 195 | 0.965 | 0.881 | 0.969 | 2.140 | 0.267 | 0.734 | 0.971 | 5.155 |
| OHRC <-> LRO NAC | retrieved | retrieved_OHRC_to_LRO NAC | insufficient_matches | 4 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 35.541 |
| OHRC <-> LRO NAC | retrieved | retrieved_OHRC_to_LRO NAC | insufficient_matches | 4 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 35.541 |
| OHRC <-> TMC-2 | retrieved | retrieved_OHRC_to_TMC-2 | insufficient_matches | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 35.397 |

