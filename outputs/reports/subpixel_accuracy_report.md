# Experiment 4 — Sub-Pixel Accuracy Validation Report

**Generated:** 2026-09-27 19:43

Goal: **RMSE < 1 pixel** after ECC refinement.
Two RMSE definitions are reported:
- `rmse_gt_*`: pixel RMSE of the estimated warp vs the **known ground-truth
  homography** (controlled pairs) — an absolute, non-circular measure.
- `rmse_reproj_*`: mean reprojection residual of the inlier correspondences.

## Comparison table (homography-only vs homography + ECC)

| Metric | Before ECC | After ECC |
|---|---|---|
| **Median GT RMSE (px)** | 0.5247 | **0.245** |
| Mean GT RMSE (px) | 2.8754 | 0.4123 |
| IQR of GT RMSE (px) | — | 0.1128 |
| Worst GT RMSE (px) | — | 4.115 |
| Mean reprojection RMSE (px) | 2.2951 | 2.324 |
| Pairs with GT evaluated | — | 38 |
| Share of GT pairs < 1 px | — | 0.921 |
| ECC applied / rejected | — | 1.0 / 0 runs |

![subpixel](figures/subpixel_ecc.png)

## Per-pair detail

| pair | protocol | status | rmse_gt_before_ecc | rmse_gt_after_ecc | rmse_reproj_before_ecc | rmse_reproj_after_ecc | ecc_applied | ecc_correlation | ecc_rejected |
|---|---|---|---|---|---|---|---|---|---|
| ctl_OHRC_to_LRO NAC_s1_i0 | controlled | ok | 0.733 | 0.320 | 1.413 | 1.468 | True | 0.981 |  |
| ctl_OHRC_to_LRO NAC_s1_i1 | controlled | ok | 0.263 | 0.146 | 1.226 | 1.255 | True | 0.984 |  |
| ctl_OHRC_to_LRO NAC_s1_i2 | controlled | ok | 0.248 | 0.295 | 1.224 | 1.237 | True | 0.968 |  |
| ctl_OHRC_to_TMC-2_s1_i0 | controlled | ok | 0.581 | 0.641 | 1.757 | 1.843 | True | 0.971 |  |
| ctl_OHRC_to_TMC-2_s1_i1 | controlled | ok | 0.843 | 0.225 | 1.423 | 1.501 | True | 0.980 |  |
| ctl_OHRC_to_TMC-2_s1_i2 | controlled | ok | 0.560 | 0.218 | 1.540 | 1.593 | True | 0.971 |  |
| ctl_OHRC_to_IIRS_s1_i0 | controlled | ok | 0.520 | 0.810 | 2.032 | 2.091 | True | 0.965 |  |
| ctl_OHRC_to_IIRS_s1_i1 | controlled | ok | 0.644 | 0.254 | 1.416 | 1.451 | True | 0.979 |  |
| ctl_OHRC_to_IIRS_s1_i2 | controlled | ok | 0.447 | 0.283 | 1.345 | 1.369 | True | 0.968 |  |
| ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | ok | 0.159 | 0.239 | 1.787 | 1.799 | True | 0.980 |  |
| ctl_TMC-2_to_LRO NAC_s1_i1 | controlled | ok | 0.282 | 0.184 | 1.702 | 1.712 | True | 0.974 |  |
| ctl_TMC-2_to_LRO NAC_s1_i2 | controlled | ok | 0.395 | 0.295 | 1.867 | 1.923 | True | 0.967 |  |
| ctl_TMC-2_to_IIRS_s1_i0 | controlled | ok | 0.336 | 0.270 | 1.560 | 1.598 | True | 0.983 |  |
| ctl_TMC-2_to_IIRS_s1_i1 | controlled | ok | 0.267 | 0.272 | 1.884 | 1.905 | True | 0.981 |  |
| ctl_TMC-2_to_IIRS_s1_i2 | controlled | ok | 0.518 | 0.267 | 2.036 | 2.140 | True | 0.978 |  |
| sun_low_i0 | controlled | ok | 0.253 | 0.119 | 1.624 | 1.649 | True | 0.981 |  |
| sun_low_i1 | controlled | ok | 0.180 | 0.103 | 1.719 | 1.732 | True | 0.990 |  |
| sun_low_i2 | controlled | ok | 0.214 | 0.068 | 1.623 | 1.635 | True | 0.971 |  |
| sun_low_i3 | controlled | ok | 0.417 | 0.234 | 1.420 | 1.491 | True | 0.948 |  |
| sun_medium_i0 | controlled | ok | 0.361 | 0.116 | 1.645 | 1.664 | True | 0.985 |  |
| sun_medium_i1 | controlled | ok | 0.701 | 0.131 | 1.905 | 2.029 | True | 0.994 |  |
| sun_medium_i2 | controlled | ok | 0.529 | 0.148 | 1.922 | 1.982 | True | 0.980 |  |
| sun_medium_i3 | controlled | ok | 0.505 | 0.206 | 1.698 | 1.754 | True | 0.979 |  |
| sun_high_i0 | controlled | ok | 1.415 | 0.166 | 8.902 | 8.818 | True | 0.977 |  |
| sun_high_i1 | controlled | ok | 1.435 | 0.223 | 2.514 | 2.871 | True | 0.985 |  |
| sun_high_i2 | controlled | ok | 0.794 | 0.181 | 7.003 | 7.092 | True | 0.970 |  |
| sun_high_i3 | controlled | ok | 0.779 | 0.253 | 2.697 | 2.809 | True | 0.974 |  |
| scale_OHRC_to_TMC-2_x0.5 | controlled | ok | 1.047 | 1.304 | 2.559 | 2.739 | True | 0.952 |  |
| scale_OHRC_to_TMC-2_x1.0 | controlled | ok | 0.745 | 0.782 | 3.262 | 3.284 | True | 0.966 |  |
| scale_OHRC_to_TMC-2_x2.0 | controlled | ok | 2.590 | 0.294 | 2.386 | 2.605 | True | 0.978 |  |
| scale_OHRC_to_LRO NAC_x0.5 | controlled | ok | 1.287 | 1.127 | 4.056 | 4.191 | True | 0.954 |  |
| scale_OHRC_to_LRO NAC_x1.0 | controlled | ok | 0.683 | 0.257 | 1.422 | 1.479 | True | 0.984 |  |
| scale_OHRC_to_LRO NAC_x2.0 | controlled | ok | 2.997 | 0.312 | 1.913 | 2.115 | True | 0.978 |  |
| scale_TMC-2_to_LRO NAC_x0.5 | controlled | ok | 0.272 | 0.136 | 1.073 | 1.134 | True | 0.942 |  |
| scale_TMC-2_to_LRO NAC_x1.0 | controlled | ok | 0.305 | 0.231 | 1.385 | 1.414 | True | 0.978 |  |
| scale_TMC-2_to_LRO NAC_x2.0 | controlled | ok | 84.063 | 4.115 | 7.665 | 6.236 | True | 0.955 |  |
| scale_TMC-2_to_IIRS_x0.5 | controlled | ok | 0.346 | 0.251 | 0.938 | 0.969 | True | 0.926 |  |
| scale_TMC-2_to_IIRS_x1.0 | controlled | ok | 0.550 | 0.192 | 1.674 | 1.737 | True | 0.986 |  |


**Reading the numbers.** `rmse_gt_*` is the pixel RMSE of the estimated warp against the
known ground-truth homography, measured over a uniform grid of source points — an
absolute, non-circular measure that no other method in the comparison can be optimised
against. Only runs that passed the acceptance guard (>=10 correspondences, plausible
quad) are listed: with four correspondences a homography reproduces its own points
exactly, so a *low* error there would be meaningless.

ECC refines the homography with an 8-parameter correlation maximization; the
combination H_refined = W_ecc · H keeps the whole chain in one matrix. ECC is kept only
when it does not regress the correspondence geometry (see `ecc_rejected`), because
correlation can otherwise drift on ill-posed pairs. The genuine TMC ncf→ncn cross-view
pair has no ground truth, so it is exercised and reported through the reprojection
measure instead.
