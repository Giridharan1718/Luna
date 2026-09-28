# Experiment 7 — Uniform Distribution Validation Report

**Generated:** 2026-09-27 19:46

ANMS (adaptive non-maximal suppression, γ=1.6, target 300, minimum suppression radius
14 px) redistributes matched keypoints so no region of the image is over-represented.
Coverage Score = fraction of 8×8 image cells containing ≥1 correspondence; Spatial
Distribution Score (uniformity) = 1 − Gini of per-cell counts; both are computed on the
query frame.

The minimum suppression radius is what makes the effect measurable: a plain "keep the
best 300" rule is a no-op whenever the matcher already returns ~300 matches, while a
greedy radius filter removes clusters regardless of how many matches exist. Because
both arms use the *same* matcher output, any difference below is attributable to the
filter alone.

## Comparison: without ANMS vs with ANMS

| protocol | runs | success_rate | match_count | uniformity_score | uniformity_score_median | coverage_score | coverage_score_median | inlier_ratio | rmse_gt_px_median |
|---|---|---|---|---|---|---|---|---|---|
| with_anms | 7 | 1.000 | 490.714 | 0.640 | 0.639 | 0.848 | 0.859 | 0.993 | 0.218 |
| without_anms | 7 | 1.000 | 490.714 | 0.580 | 0.544 | 0.853 | 0.875 | 0.993 | 0.219 |


Mean per-pair uniformity gain from ANMS: **0.0606**
(7/7 pairs improve).

![uniformity bars](figures/uniformity_bars.png)
![uniform distribution](figures/uniform_distribution.png)

## Per-pair effect (same matches, only the filter differs)

| pair | matches | kept_by_anms | coverage_without | coverage_with | uniformity_without | uniformity_with | uniformity_gain | rmse_gt_with | rmse_gt_without |
|---|---|---|---|---|---|---|---|---|---|
| uni_OHRC_to_IIRS_503 | 248 | 129 | 0.750 | 0.734 | 0.470 | 0.539 | 0.069 | 0.265 | 0.259 |
| uni_OHRC_to_LRO NAC_504 | 343 | 161 | 0.703 | 0.703 | 0.490 | 0.529 | 0.039 | 0.218 | 0.219 |
| uni_OHRC_to_TMC-2_501 | 210 | 110 | 0.656 | 0.656 | 0.424 | 0.467 | 0.043 | 0.451 | 0.453 |
| uni_TMC-2_to_IIRS_506 | 848 | 244 | 1.000 | 1.000 | 0.753 | 0.823 | 0.070 | 0.123 | 0.112 |
| uni_TMC-2_to_LRO NAC_502 | 694 | 216 | 1.000 | 1.000 | 0.714 | 0.775 | 0.061 | 0.170 | 0.171 |
| uni_TMC-2_to_TMC-2_500 | 618 | 195 | 0.984 | 0.984 | 0.664 | 0.712 | 0.048 | 0.180 | 0.187 |
| uni_TMC-2_to_TMC-2_505 | 474 | 153 | 0.875 | 0.859 | 0.544 | 0.639 | 0.095 | 0.390 | 0.387 |


## Per-run detail

| pair | protocol | status | coverage_score | uniformity_score | match_count | anms_matches | inlier_ratio | rmse_gt_px |
|---|---|---|---|---|---|---|---|---|
| uni_TMC-2_to_TMC-2_500 | with_anms | ok | 0.984 | 0.712 | 618 | 195 | 0.985 | 0.180 |
| uni_TMC-2_to_TMC-2_500 | without_anms | ok | 0.984 | 0.664 | 618 | 618 | 0.994 | 0.187 |
| uni_OHRC_to_TMC-2_501 | with_anms | ok | 0.656 | 0.467 | 210 | 110 | 1.000 | 0.451 |
| uni_OHRC_to_TMC-2_501 | without_anms | ok | 0.656 | 0.424 | 210 | 210 | 1.000 | 0.453 |
| uni_TMC-2_to_LRO NAC_502 | with_anms | ok | 1.000 | 0.775 | 694 | 216 | 0.986 | 0.170 |
| uni_TMC-2_to_LRO NAC_502 | without_anms | ok | 1.000 | 0.714 | 694 | 694 | 0.991 | 0.171 |
| uni_OHRC_to_IIRS_503 | with_anms | ok | 0.734 | 0.539 | 248 | 129 | 1.000 | 0.265 |
| uni_OHRC_to_IIRS_503 | without_anms | ok | 0.750 | 0.470 | 248 | 248 | 0.996 | 0.259 |
| uni_OHRC_to_LRO NAC_504 | with_anms | ok | 0.703 | 0.529 | 343 | 161 | 0.994 | 0.218 |
| uni_OHRC_to_LRO NAC_504 | without_anms | ok | 0.703 | 0.490 | 343 | 343 | 0.994 | 0.219 |
| uni_TMC-2_to_TMC-2_505 | with_anms | ok | 0.859 | 0.639 | 474 | 153 | 0.993 | 0.390 |
| uni_TMC-2_to_TMC-2_505 | without_anms | ok | 0.875 | 0.544 | 474 | 474 | 0.987 | 0.387 |
| uni_TMC-2_to_IIRS_506 | with_anms | ok | 1.000 | 0.823 | 848 | 244 | 0.992 | 0.123 |
| uni_TMC-2_to_IIRS_506 | without_anms | ok | 1.000 | 0.753 | 848 | 848 | 0.989 | 0.112 |


**Why it matters for registration.** A clustered correspondence set biases the
homography toward that cluster and leaves the rest of the frame unconstrained, so the
estimated 3×3 matrix is only valid where the matches were — the opposite of what a
rover or an ortho-rectification product needs. ANMS buys spatial spread at the cost of
raw match count, and because the spread points are still inliers (see `inlier_ratio`)
the accuracy of the warp does not suffer: that trade is the ISRO uniform-distribution
requirement, measured above pair-by-pair.
