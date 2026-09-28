# Experiment 5 — Benchmark Comparison Report

**Generated:** 2026-09-27 19:44 · Pair set: 9 pairs
(controlled GT pairs, scale variants, the genuine TMC ncf↔ncn cross-view pair and one
retrieved cross-sensor pair) — identical pairs for every method.

All four methods run on the identical pair set with the identical acceptance guard
(>=10 correspondences and a geometrically plausible warp), so `success_rate` and the
medians are directly comparable. `median_rmse_gt_px` is the median error of the
estimated warp against the known ground-truth homography over the pairs that method
evaluated — the one number none of the methods can be tuned against.

## Ranking (reliability first, then GT error)

| method | pairs | success_rate | gt_evaluated | median_rmse_gt_px | median_rmse_px | inlier_ratio | inlier_ratio_3px | coverage_score | match_count | runtime_s | rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_ransac | 9 | 0.778 | 7 | 0.243 | 0.803 | 0.774 | 0.763 | 0.623 | 205.700 | 0.063 | 1 |
| lunarai_full | 9 | 0.778 | 7 | 0.294 | 1.843 | 0.915 | 0.853 | 0.656 | 334.100 | 10.153 | 2 |
| splg_ransac | 9 | 0.778 | 7 | 0.600 | 1.487 | 0.924 | 0.871 | 0.656 | 334.100 | 5.116 | 3 |
| orb_ransac | 9 | 0.667 | 6 | 1.246 | 1.803 | 0.692 | 0.647 | 0.438 | 238.000 | 0.054 | 4 |
| akaze_ransac | 9 | 0.000 | 0 | - | - | - | - | - | - | 0.000 | 5 |
| superglue_ransac | 9 | 0.000 | 0 | - | - | - | - | - | - | 0.725 | 6 |


## Head-to-head vs LunarAI (pairs both methods solved)

| comparison | pairs_both_solved | lunarai_lower_gt_rmse | baseline_lower_gt_rmse |
|---|---|---|---|
| lunarai_full vs sift_ransac | 7 | 3 | 4 |
| lunarai_full vs orb_ransac | 6 | 5 | 1 |
| lunarai_full vs splg_ransac | 7 | 2 | 5 |
| lunarai_full vs akaze_ransac | 0 | 0 | 0 |
| lunarai_full vs superglue_ransac | 0 | 0 | 0 |


## Full metrics per method

| method | pairs | success_rate | gt_evaluated | median_rmse_gt_px | median_rmse_px | inlier_ratio | inlier_ratio_3px | coverage_score | match_count | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|
| lunarai_full | 9 | 0.778 | 7 | 0.294 | 1.843 | 0.915 | 0.853 | 0.656 | 334.100 | 10.153 |
| sift_ransac | 9 | 0.778 | 7 | 0.243 | 0.803 | 0.774 | 0.763 | 0.623 | 205.700 | 0.063 |
| orb_ransac | 9 | 0.667 | 6 | 1.246 | 1.803 | 0.692 | 0.647 | 0.438 | 238.000 | 0.054 |
| splg_ransac | 9 | 0.778 | 7 | 0.600 | 1.487 | 0.924 | 0.871 | 0.656 | 334.100 | 5.116 |
| akaze_ransac | 9 | 0.000 | 0 | - | - | - | - | - | - | 0.000 |
| superglue_ransac | 9 | 0.000 | 0 | - | - | - | - | - | - | 0.725 |


![benchmark](figures/benchmark_comparison.png)
![runtime](figures/benchmark_runtime.png)

## Per-pair detail

| method | pair | protocol | status | match_count | inlier_count | inlier_ratio | inlier_ratio_3px | coverage_score | rmse_px | rmse_gt_px | ssim | ncc | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lunarai_full | ctl_OHRC_to_TMC-2_s1_i0 | controlled | ok | 206 | 108 | 0.982 | 0.918 | 0.656 | 1.843 | 0.641 | 0.382 | 0.956 | 5.087 |
| lunarai_full | ctl_OHRC_to_IIRS_s1_i0 | controlled | ok | 195 | 105 | 0.946 | 0.892 | 0.688 | 2.091 | 0.810 | 0.347 | 0.948 | 4.967 |
| lunarai_full | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | ok | 743 | 228 | 0.983 | 0.918 | 1.000 | 1.799 | 0.239 | 0.821 | 0.976 | 5.082 |
| lunarai_full | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | ok | 427 | 146 | 0.986 | 0.939 | 0.859 | 1.693 | 0.256 | 0.855 | 0.978 | 5.252 |
| lunarai_full | ctl_TMC-2_to_IIRS_s1_i0 | controlled | ok | 601 | 180 | 0.989 | 0.973 | 1.000 | 1.598 | 0.270 | 0.842 | 0.976 | 5.157 |
| lunarai_full | scale_OHRC_to_TMC-2_x2.0 | controlled | ok | 134 | 56 | 0.933 | 0.800 | 0.266 | 2.605 | 0.294 | 0.584 | 0.971 | 4.986 |
| lunarai_full | scale_TMC-2_to_LRO NAC_x2.0 | controlled | ok | 33 | 10 | 0.588 | 0.529 | 0.125 | 6.236 | 4.115 | 0.798 | 0.955 | 5.277 |
| lunarai_full | real_ncf00000_00000_vs_ncn12227_00000 | real | rejected_affine_rescue_ok_4inl | 19 | 4 | 0.286 | 0.286 | 0.203 | - | - | - | - | 20.174 |
| lunarai_full | retrieved_OHRC_to_TMC-2 | retrieved | insufficient_matches | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 35.397 |
| sift | ctl_OHRC_to_TMC-2_s1_i0 | controlled | ok | 247 | 188 | 0.761 | 0.761 | 0.641 | 0.635 | 0.157 | 0.384 | 0.955 | 0.069 |
| sift | ctl_OHRC_to_IIRS_s1_i0 | controlled | ok | 196 | 120 | 0.612 | 0.607 | 0.531 | 0.861 | 0.393 | 0.350 | 0.947 | 0.064 |
| sift | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | ok | 204 | 179 | 0.877 | 0.873 | 0.922 | 0.747 | 0.243 | 0.818 | 0.973 | 0.057 |
| sift | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | ok | 184 | 171 | 0.929 | 0.913 | 0.906 | 0.809 | 0.157 | 0.855 | 0.978 | 0.059 |
| sift | ctl_TMC-2_to_IIRS_s1_i0 | controlled | ok | 196 | 182 | 0.929 | 0.913 | 0.922 | 0.803 | 0.135 | 0.843 | 0.975 | 0.054 |
| sift | scale_OHRC_to_TMC-2_x2.0 | controlled | ok | 350 | 274 | 0.783 | 0.780 | 0.234 | 0.674 | 0.426 | 0.578 | 0.970 | 0.065 |
| sift | scale_TMC-2_to_LRO NAC_x2.0 | controlled | ok | 63 | 33 | 0.524 | 0.492 | 0.203 | 1.519 | 4.491 | 0.801 | 0.952 | 0.055 |
| sift | real_ncf00000_00000_vs_ncn12227_00000 | real | rejected_area_ratio_1.93e-05 | 54 | 0 | 0.000 | 0.000 | 0.500 | - | - | - | - | 0.071 |
| sift | retrieved_OHRC_to_TMC-2 | retrieved | rejected_area_ratio_0.000836 | 82 | 0 | 0.000 | 0.000 | 0.250 | - | - | - | - | 0.075 |
| orb | ctl_OHRC_to_TMC-2_s1_i0 | controlled | ok | 298 | 218 | 0.732 | 0.705 | 0.500 | 1.585 | 0.726 | 0.378 | 0.954 | 0.150 |
| orb | ctl_OHRC_to_IIRS_s1_i0 | controlled | ok | 249 | 158 | 0.635 | 0.606 | 0.453 | 1.838 | 0.576 | 0.350 | 0.946 | 0.031 |
| orb | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | ok | 220 | 181 | 0.823 | 0.759 | 0.516 | 1.798 | 1.698 | 0.781 | 0.949 | 0.029 |
| orb | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | ok | 164 | 147 | 0.896 | 0.817 | 0.484 | 1.891 | 2.057 | 0.794 | 0.959 | 0.026 |
| orb | ctl_TMC-2_to_IIRS_s1_i0 | controlled | ok | 204 | 185 | 0.907 | 0.833 | 0.500 | 1.808 | 0.794 | 0.833 | 0.974 | 0.026 |
| orb | scale_OHRC_to_TMC-2_x2.0 | controlled | ok | 293 | 47 | 0.160 | 0.160 | 0.172 | 1.482 | 3.784 | 0.522 | 0.959 | 0.072 |
| orb | scale_TMC-2_to_LRO NAC_x2.0 | controlled | rejected_quad_blowup | 63 | 0 | 0.000 | 0.000 | 0.328 | - | - | - | - | 0.050 |
| orb | real_ncf00000_00000_vs_ncn12227_00000 | real | rejected_area_ratio_0.00626 | 41 | 0 | 0.000 | 0.000 | 0.266 | - | - | - | - | 0.049 |
| orb | retrieved_OHRC_to_TMC-2 | retrieved | rejected_area_ratio_2.52e-06 | 97 | 0 | 0.000 | 0.000 | 0.234 | - | - | - | - | 0.051 |
| splg_ransac | ctl_OHRC_to_TMC-2_s1_i0 | controlled | ok | 206 | 203 | 0.985 | 0.951 | 0.672 | 1.445 | 0.600 | 0.372 | 0.952 | 4.969 |
| splg_ransac | ctl_OHRC_to_IIRS_s1_i0 | controlled | ok | 195 | 184 | 0.944 | 0.897 | 0.672 | 1.612 | 0.784 | 0.342 | 0.947 | 5.193 |
| splg_ransac | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | ok | 743 | 736 | 0.991 | 0.949 | 1.000 | 1.487 | 0.117 | 0.821 | 0.976 | 5.197 |
| splg_ransac | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | ok | 427 | 423 | 0.991 | 0.972 | 0.891 | 1.416 | 0.200 | 0.855 | 0.978 | 5.031 |
| splg_ransac | ctl_TMC-2_to_IIRS_s1_i0 | controlled | ok | 601 | 598 | 0.995 | 0.982 | 1.000 | 1.371 | 0.120 | 0.843 | 0.976 | 5.092 |
| splg_ransac | scale_OHRC_to_TMC-2_x2.0 | controlled | ok | 134 | 124 | 0.925 | 0.799 | 0.266 | 1.925 | 1.280 | 0.577 | 0.970 | 5.077 |
| splg_ransac | scale_TMC-2_to_LRO NAC_x2.0 | controlled | ok | 33 | 21 | 0.636 | 0.545 | 0.094 | 2.060 | 68.429 | 0.467 | 0.433 | 5.287 |
| splg_ransac | real_ncf00000_00000_vs_ncn12227_00000 | real | rejected_inliers_6 | 19 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 5.008 |
| splg_ransac | retrieved_OHRC_to_TMC-2 | retrieved | failed_homography | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 5.190 |
| akaze_ransac | ctl_OHRC_to_TMC-2_s1_i0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | ctl_OHRC_to_IIRS_s1_i0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | ctl_TMC-2_to_IIRS_s1_i0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | scale_OHRC_to_TMC-2_x2.0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | scale_TMC-2_to_LRO NAC_x2.0 | controlled | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | real_ncf00000_00000_vs_ncn12227_00000 | real | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| akaze_ransac | retrieved_OHRC_to_TMC-2 | retrieved | error: module 'cv2' has no attribute 'AKAZE_create' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.000 |
| superglue_ransac | ctl_OHRC_to_TMC-2_s1_i0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.715 |
| superglue_ransac | ctl_OHRC_to_IIRS_s1_i0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.729 |
| superglue_ransac | ctl_TMC-2_to_LRO NAC_s1_i0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.728 |
| superglue_ransac | ctl_TMC-2_to_TMC-2_s1_i0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.732 |
| superglue_ransac | ctl_TMC-2_to_IIRS_s1_i0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.725 |
| superglue_ransac | scale_OHRC_to_TMC-2_x2.0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.746 |
| superglue_ransac | scale_TMC-2_to_LRO NAC_x2.0 | controlled | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.716 |
| superglue_ransac | real_ncf00000_00000_vs_ncn12227_00000 | real | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.730 |
| superglue_ransac | retrieved_OHRC_to_TMC-2 | retrieved | error: 'descriptors0' | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - | - | - | 0.700 |


**Reading the table.** The comparison is deliberately uneven in one respect and fair in
every other: `lunarai_full` runs the complete chain
(CLAHE -> SuperPoint+LightGlue -> ANMS -> MAGSAC++ -> homography -> ECC) while the
baselines run their detector plus RANSAC, and all are judged by the same
non-circular GT criterion and the same acceptance guard. On these controlled pairs
SIFT is a strong single-pair matcher and essentially ties LunarAI (see head-to-head);
what SIFT/ORB cannot do is decide *where* on the Moon a patch comes from at database
scale (Experiments 6 and 8) or enforce a spatially uniform correspondence set
(Experiment 7) — the two requirements this pipeline exists for.

SuperGlue row marked 'unavailable': torch.hub weights could not be fetched on this machine at run time (offline demo box).
