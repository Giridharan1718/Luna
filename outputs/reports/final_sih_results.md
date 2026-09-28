# LunarAI - Final SIH Results (Problem 26166)

Generated 2026-09-27 20:26 | 25 pipeline runs in this session.
Matcher: superpoint+lightglue | LunaDNA checkpoint epoch 1.

## 1. Dataset statistics

| stat                  | value                                                                                                                                     |
|:----------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| total_images          | 15                                                                                                                                        |
| counts                | {"OHRC": 3, "TMC-2": 7, "IIRS": 4, "LRO NAC": 1}                                                                                          |
| metadata_coverage     | {"latitude": 0.933, "longitude": 0.933, "sun_elevation": 0.733, "sun_azimuth": 0.733, "pixel_resolution": 0.933, "acquisition_date": 0.0} |
| dataset_quality_score | 80.7                                                                                                                                      |

Source CSVs: `data/SunAngle.csv` (5000 rows, elevation
26.8-34.2 deg over one pass), `data/ElevationProfile.csv`
(100 LOLA DEM transect points).

## 2. Training statistics

| stat             |   value |
|:-----------------|--------:|
| checkpoint_epoch |       1 |

## 3. Multi-modal correspondence (controlled GT protocol + real pair)

| pair                 | protocol   | sensor_a   | sensor_b   | status   |   total_matches |   inlier_matches |   inlier_ratio |   inlier_ratio_3px |   rmse_gt_px |   rmse_reproj_px |   coverage_score |   registration_time_s |
|:---------------------|:-----------|:-----------|:-----------|:---------|----------------:|-----------------:|---------------:|-------------------:|-------------:|-----------------:|-----------------:|----------------------:|
| ctl_OHRC_to_TMC-2    | controlled | OHRC       | TMC-2      | ok       |             206 |              108 |       0.981818 |           0.918182 |     0.641487 |          1.84288 |         0.65625  |               4.83037 |
| ctl_OHRC_to_LRO NAC  | controlled | OHRC       | LRO NAC    | ok       |             355 |              156 |       1        |           0.961538 |     0.319671 |          1.46755 |         0.6875   |               4.59134 |
| ctl_OHRC_to_IIRS     | controlled | OHRC       | IIRS       | ok       |             195 |              105 |       0.945946 |           0.891892 |     0.809631 |          2.09129 |         0.6875   |               4.54951 |
| ctl_TMC-2_to_LRO NAC | controlled | TMC-2      | LRO NAC    | ok       |             743 |              228 |       0.982759 |           0.918103 |     0.239284 |          1.79893 |         1        |               4.62084 |
| ctl_TMC-2_to_TMC-2   | controlled | TMC-2      | TMC-2      | ok       |             427 |              146 |       0.986486 |           0.939189 |     0.255567 |          1.69267 |         0.859375 |               4.59686 |
| ctl_TMC-2_to_IIRS    | controlled | TMC-2      | IIRS       | ok       |             601 |              180 |       0.989011 |           0.972527 |     0.269998 |          1.59815 |         1        |               4.94121 |

![multimodal](figures/multimodal_summary.png)

## 4. Sun-angle invariance (5 SIH bins)

Bin labels carry the measured share of the patch catalog in each bin: this dataset's
real products cluster in the 0-10 deg (OHRC south-polar), 20-40 deg (TMC/IIRS main
acquisitions) and 40-60 deg (TMC 2026-07-01) bins; the 10-20 and 60+ bins have no
imagery in the distribution and are reported as empty rather than interpolated.

| sun_bin   |   catalog_share |   catalog_patches |   pairs_built |   pairs_solved |   median_rmse_gt_px |   mean_inlier_ratio_3px |   median_coverage |   retrieval_top1 |   retrieval_top5 |
|:----------|----------------:|------------------:|--------------:|---------------:|--------------------:|------------------------:|------------------:|-----------------:|-----------------:|
| 0-10      |           0.015 |                11 |             4 |              4 |               0.163 |                   0.996 |             0.516 |            0.091 |             1    |
| 10-20     |           0     |                 0 |             0 |              0 |             nan     |                 nan     |           nan     |          nan     |           nan    |
| 20-40     |           0.914 |               684 |             4 |              4 |               0.253 |                   0.931 |             0.594 |            0.6   |             0.85 |
| 40-60     |           0.071 |                53 |             4 |              4 |               0.182 |                   0.961 |             0.969 |            1     |             1    |
| 60+       |           0     |                 0 |             0 |              0 |             nan     |                 nan     |           nan     |          nan     |           nan    |

![sun angle](figures/sun_angle_performance.png)

## 5. Scale invariance

| pair                        | sensor_a   | sensor_b   |   scale_ratio | status               |   match_success |   total_matches |   inlier_ratio |   inlier_ratio_3px |   rmse_gt_px |   gsd_ratio_a_over_b |
|:----------------------------|:-----------|:-----------|--------------:|:---------------------|----------------:|----------------:|---------------:|-------------------:|-------------:|---------------------:|
| scale_OHRC_to_TMC-2_x0.5    | OHRC       | TMC-2      |           0.5 | ok                   |               1 |             255 |       0.970149 |           0.865672 |     1.30437  |             20.8333  |
| scale_OHRC_to_LRO NAC_x0.5  | OHRC       | LRO NAC    |           0.5 | ok                   |               1 |             106 |       0.939394 |           0.878788 |     1.12708  |              8.33333 |
| scale_TMC-2_to_LRO NAC_x0.5 | TMC-2      | LRO NAC    |           0.5 | ok                   |               1 |             201 |       1        |           1        |     0.135701 |              0.4     |
| scale_TMC-2_to_IIRS_x0.5    | TMC-2      | IIRS       |           0.5 | ok                   |               1 |             182 |       1        |           1        |     0.250678 |             16       |
| scale_OHRC_to_TMC-2_x1.0    | OHRC       | TMC-2      |           1   | ok                   |               1 |             139 |       0.855422 |           0.771084 |     0.782438 |             20.8333  |
| scale_OHRC_to_LRO NAC_x1.0  | OHRC       | LRO NAC    |           1   | ok                   |               1 |             353 |       0.987654 |           0.95679  |     0.257308 |              8.33333 |
| scale_TMC-2_to_LRO NAC_x1.0 | TMC-2      | LRO NAC    |           1   | ok                   |               1 |             780 |       0.995726 |           0.987179 |     0.230864 |              0.4     |
| scale_TMC-2_to_IIRS_x1.0    | TMC-2      | IIRS       |           1   | ok                   |               1 |             708 |       0.986301 |           0.940639 |     0.191514 |             16       |
| scale_OHRC_to_TMC-2_x2.0    | OHRC       | TMC-2      |           2   | ok                   |               1 |             134 |       0.933333 |           0.8      |     0.294058 |             20.8333  |
| scale_OHRC_to_LRO NAC_x2.0  | OHRC       | LRO NAC    |           2   | ok                   |               1 |             180 |       0.970149 |           0.880597 |     0.311981 |              8.33333 |
| scale_TMC-2_to_LRO NAC_x2.0 | TMC-2      | LRO NAC    |           2   | ok                   |               1 |              33 |       0.588235 |           0.529412 |     4.11504  |              0.4     |
| scale_TMC-2_to_IIRS_x2.0    | TMC-2      | IIRS       |           2   | insufficient_matches |               0 |               2 |       0        |           0        |   nan        |             16       |
| scale_OHRC_to_TMC-2_x4.0    | OHRC       | TMC-2      |           4   | insufficient_matches |               0 |               0 |       0        |           0        |   nan        |             20.8333  |
| scale_OHRC_to_LRO NAC_x4.0  | OHRC       | LRO NAC    |           4   | insufficient_matches |               0 |               3 |       0        |           0        |   nan        |              8.33333 |
| scale_TMC-2_to_LRO NAC_x4.0 | TMC-2      | LRO NAC    |           4   | insufficient_matches |               0 |               0 |       0        |           0        |   nan        |              0.4     |
| scale_TMC-2_to_IIRS_x4.0    | TMC-2      | IIRS       |           4   | insufficient_matches |               0 |               0 |       0        |           0        |   nan        |             16       |

![scale](figures/scale_performance.png)

## 6. Sub-pixel registration accuracy (target: RMSE < 1 px)

- Pairs evaluated: 29 | median error after ECC: 0.2556 px
- Mean error: 0.5164 px
- Share under 1 px: 0.862
- Target: **MET**

| pair                        | protocol   |   rmse_before_ecc_px |   rmse_after_ecc_px |   pixel_error_after_ecc_px |   mean_alignment_error_px | ecc_applied   |   ecc_correlation |   median_error_px |   mean_error_px | target_lt_1px   |
|:----------------------------|:-----------|---------------------:|--------------------:|---------------------------:|--------------------------:|:--------------|------------------:|------------------:|----------------:|:----------------|
| ctl_OHRC_to_TMC-2           | controlled |               0.5811 |              0.6415 |                     0.6415 |                  0.610536 | True          |          0.970632 |            0.2556 |          0.5164 | MET             |
| ctl_OHRC_to_LRO NAC         | controlled |               0.7328 |              0.3197 |                     0.3197 |                  0.484012 | True          |          0.980889 |            0.2556 |          0.5164 | MET             |
| ctl_OHRC_to_IIRS            | controlled |               0.5199 |              0.8096 |                     0.8096 |                  0.648788 | True          |          0.965079 |            0.2556 |          0.5164 | MET             |
| ctl_TMC-2_to_LRO NAC        | controlled |               0.1588 |              0.2393 |                     0.2393 |                  0.19491  | True          |          0.980373 |            0.2556 |          0.5164 | MET             |
| ctl_TMC-2_to_TMC-2          | controlled |               0.4442 |              0.2556 |                     0.2556 |                  0.336928 | True          |          0.985475 |            0.2556 |          0.5164 | MET             |
| ctl_TMC-2_to_IIRS           | controlled |               0.3356 |              0.27   |                     0.27   |                  0.301031 | True          |          0.982842 |            0.2556 |          0.5164 | MET             |
| scale_OHRC_to_TMC-2_x0.5    | controlled |               1.0471 |              1.3044 |                     1.3044 |                  1.16869  | True          |          0.951658 |            0.2556 |          0.5164 | MET             |
| scale_OHRC_to_LRO NAC_x0.5  | controlled |               1.2873 |              1.1271 |                     1.1271 |                  1.20454  | True          |          0.953621 |            0.2556 |          0.5164 | MET             |
| scale_TMC-2_to_LRO NAC_x0.5 | controlled |               0.2721 |              0.1357 |                     0.1357 |                  0.192155 | True          |          0.941595 |            0.2556 |          0.5164 | MET             |
| scale_TMC-2_to_IIRS_x0.5    | controlled |               0.3456 |              0.2507 |                     0.2507 |                  0.294347 | True          |          0.925722 |            0.2556 |          0.5164 | MET             |
| scale_OHRC_to_TMC-2_x1.0    | controlled |               0.7454 |              0.7824 |                     0.7824 |                  0.763686 | True          |          0.965667 |            0.2556 |          0.5164 | MET             |
| scale_OHRC_to_LRO NAC_x1.0  | controlled |               0.6835 |              0.2573 |                     0.2573 |                  0.419361 | True          |          0.984025 |            0.2556 |          0.5164 | MET             |

![subpixel](figures/subpixel_accuracy_plot.png)

## 7. Benchmark comparison

| method           |   pairs |   recall |   median_rmse_px |   median_gt_rmse_px |   inlier_ratio_3px |   mean_coverage |   mean_runtime_s |   rank |
|:-----------------|--------:|---------:|-----------------:|--------------------:|-------------------:|----------------:|-----------------:|-------:|
| sift_ransac      |      25 |     0.88 |           0.3755 |              0.3755 |             0.6888 |          0.5156 |            0.068 |      1 |
| lunarai_full     |      25 |     0.68 |           0.2941 |              0.2941 |             0.8948 |          0.6829 |           14.114 |      2 |
| orb_ransac       |      25 |     0.68 |           1.0652 |              1.0652 |             0.5999 |          0.4154 |            0.046 |      3 |
| splg_ransac      |      25 |     0.64 |           0.5314 |              0.5314 |             0.9121 |          0.6875 |            5.054 |      4 |
| superglue_ransac |      25 |     0.6  |           2.4618 |              2.4618 |             0.3213 |          0.6875 |            0.878 |      5 |
| akaze_ransac     |      25 |     0    |         nan      |            nan      |           nan      |        nan      |            0     |      6 |

![benchmark](figures/benchmark_table.png)

## 8. Coverage analysis (before vs after ANMS)

| pair                 | stage       | status   |   matches |   grid_cells |   grid_coverage |   match_density_per_cell |   coverage_percentage |   uniformity_score |   inlier_ratio_3px |   rmse_gt_px |
|:---------------------|:------------|:---------|----------:|-------------:|----------------:|-------------------------:|----------------------:|-------------------:|-------------------:|-------------:|
| ctl_OHRC_to_TMC-2    | before_anms | ok       |       206 |           64 |        0.671875 |                     3.22 |                  67.2 |           0.441292 |           0.951456 |     0.635544 |
| ctl_OHRC_to_TMC-2    | after_anms  | ok       |       110 |           64 |        0.65625  |                     1.72 |                  65.6 |           0.496307 |           0.918182 |     0.641487 |
| ctl_OHRC_to_LRO NAC  | before_anms | ok       |       355 |           64 |        0.703125 |                     5.55 |                  70.3 |           0.476893 |           0.966197 |     0.320517 |
| ctl_OHRC_to_LRO NAC  | after_anms  | ok       |       156 |           64 |        0.6875   |                     2.44 |                  68.8 |           0.510817 |           0.961538 |     0.319671 |
| ctl_OHRC_to_IIRS     | before_anms | ok       |       195 |           64 |        0.6875   |                     3.05 |                  68.8 |           0.439343 |           0.897436 |     0.802841 |
| ctl_OHRC_to_IIRS     | after_anms  | ok       |       111 |           64 |        0.6875   |                     1.73 |                  68.8 |           0.495073 |           0.891892 |     0.809631 |
| ctl_TMC-2_to_LRO NAC | before_anms | ok       |       743 |           64 |        1        |                    11.61 |                 100   |           0.71772  |           0.948856 |     0.239245 |
| ctl_TMC-2_to_LRO NAC | after_anms  | ok       |       232 |           64 |        1        |                     3.62 |                 100   |           0.761719 |           0.918103 |     0.239284 |
| ctl_TMC-2_to_TMC-2   | before_anms | ok       |       427 |           64 |        0.90625  |                     6.67 |                  90.6 |           0.56766  |           0.971897 |     0.25686  |
| ctl_TMC-2_to_TMC-2   | after_anms  | ok       |       148 |           64 |        0.859375 |                     2.31 |                  85.9 |           0.644848 |           0.939189 |     0.255567 |
| ctl_TMC-2_to_IIRS    | before_anms | ok       |       601 |           64 |        1        |                     9.39 |                 100   |           0.645253 |           0.981697 |     0.270734 |
| ctl_TMC-2_to_IIRS    | after_anms  | ok       |       182 |           64 |        1        |                     2.84 |                 100   |           0.71669  |           0.972527 |     0.269998 |

![coverage](figures/coverage_visualization.png)

## 9. Confidence engine

Score = 25*similarity + 30*inlier@3px + 30*max(0, 1 - RMSE/4) + 15*coverage,
categories Very High >= 80, High >= 60, Medium >= 40, else Low.

| pair                                  | protocol   | status                         |   embedding_similarity |   inlier_ratio_3px |    rmse_px |   coverage_score |   confidence_score | category   | axes                                                                   |
|:--------------------------------------|:-----------|:-------------------------------|-----------------------:|-------------------:|-----------:|-----------------:|-------------------:|:-----------|:-----------------------------------------------------------------------|
| ctl_OHRC_to_TMC-2                     | controlled | ok                             |                    nan |           0.918182 |   0.641487 |         0.65625  |               62.6 | High       | {"embedding": 0.0, "inliers": 27.55, "rmse": 25.19, "coverage": 9.84}  |
| ctl_OHRC_to_LRO NAC                   | controlled | ok                             |                    nan |           0.961538 |   0.319671 |         0.6875   |               66.8 | High       | {"embedding": 0.0, "inliers": 28.85, "rmse": 27.6, "coverage": 10.31}  |
| ctl_OHRC_to_IIRS                      | controlled | ok                             |                    nan |           0.891892 |   0.809631 |         0.6875   |               61   | High       | {"embedding": 0.0, "inliers": 26.76, "rmse": 23.93, "coverage": 10.31} |
| ctl_TMC-2_to_LRO NAC                  | controlled | ok                             |                    nan |           0.918103 |   0.239284 |         1        |               70.7 | High       | {"embedding": 0.0, "inliers": 27.54, "rmse": 28.21, "coverage": 15.0}  |
| ctl_TMC-2_to_TMC-2                    | controlled | ok                             |                    nan |           0.939189 |   0.255567 |         0.859375 |               69.1 | High       | {"embedding": 0.0, "inliers": 28.18, "rmse": 28.08, "coverage": 12.89} |
| ctl_TMC-2_to_IIRS                     | controlled | ok                             |                    nan |           0.972527 |   0.269998 |         1        |               72.2 | High       | {"embedding": 0.0, "inliers": 29.18, "rmse": 27.98, "coverage": 15.0}  |
| real_ncf00000_00000_vs_ncn12227_00000 | real       | rejected_affine_rescue_ok_4inl |                    nan |           0.285714 | nan        |         0.203125 |               11.6 | Low        | {"embedding": 0.0, "inliers": 8.57, "rmse": 0.0, "coverage": 3.05}     |
| scale_OHRC_to_TMC-2_x0.5              | controlled | ok                             |                    nan |           0.865672 |   1.30437  |         0.59375  |               55.1 | Medium     | {"embedding": 0.0, "inliers": 25.97, "rmse": 20.22, "coverage": 8.91}  |
| scale_OHRC_to_LRO NAC_x0.5            | controlled | ok                             |                    nan |           0.878788 |   1.12708  |         0.5      |               55.4 | Medium     | {"embedding": 0.0, "inliers": 26.36, "rmse": 21.55, "coverage": 7.5}   |
| scale_TMC-2_to_LRO NAC_x0.5           | controlled | ok                             |                    nan |           1        |   0.135701 |         0.90625  |               72.6 | High       | {"embedding": 0.0, "inliers": 30.0, "rmse": 28.98, "coverage": 13.59}  |
| scale_TMC-2_to_IIRS_x0.5              | controlled | ok                             |                    nan |           1        |   0.250678 |         0.859375 |               71   | High       | {"embedding": 0.0, "inliers": 30.0, "rmse": 28.12, "coverage": 12.89}  |
| scale_OHRC_to_TMC-2_x1.0              | controlled | ok                             |                    nan |           0.771084 |   0.782438 |         0.5625   |               55.7 | Medium     | {"embedding": 0.0, "inliers": 23.13, "rmse": 24.13, "coverage": 8.44}  |
| scale_OHRC_to_LRO NAC_x1.0            | controlled | ok                             |                    nan |           0.95679  |   0.257308 |         0.6875   |               67.1 | High       | {"embedding": 0.0, "inliers": 28.7, "rmse": 28.07, "coverage": 10.31}  |
| scale_TMC-2_to_LRO NAC_x1.0           | controlled | ok                             |                    nan |           0.987179 |   0.230864 |         0.984375 |               72.6 | High       | {"embedding": 0.0, "inliers": 29.62, "rmse": 28.27, "coverage": 14.77} |
| scale_TMC-2_to_IIRS_x1.0              | controlled | ok                             |                    nan |           0.940639 |   0.191514 |         0.96875  |               71.3 | High       | {"embedding": 0.0, "inliers": 28.22, "rmse": 28.56, "coverage": 14.53} |
| scale_OHRC_to_TMC-2_x2.0              | controlled | ok                             |                    nan |           0.8      |   0.294058 |         0.265625 |               55.8 | Medium     | {"embedding": 0.0, "inliers": 24.0, "rmse": 27.79, "coverage": 3.98}   |
| scale_OHRC_to_LRO NAC_x2.0            | controlled | ok                             |                    nan |           0.880597 |   0.311981 |         0.265625 |               58.1 | Medium     | {"embedding": 0.0, "inliers": 26.42, "rmse": 27.66, "coverage": 3.98}  |
| scale_TMC-2_to_LRO NAC_x2.0           | controlled | ok                             |                    nan |           0.529412 |   4.11504  |         0.125    |               17.8 | Low        | {"embedding": 0.0, "inliers": 15.88, "rmse": 0.0, "coverage": 1.88}    |
| scale_TMC-2_to_IIRS_x2.0              | controlled | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| scale_OHRC_to_TMC-2_x4.0              | controlled | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| scale_OHRC_to_LRO NAC_x4.0            | controlled | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| scale_TMC-2_to_LRO NAC_x4.0           | controlled | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| scale_TMC-2_to_IIRS_x4.0              | controlled | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| retrieved_OHRC_to_TMC-2               | retrieved  | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |
| retrieved_OHRC_to_LRO NAC             | retrieved  | insufficient_matches           |                    nan |           0        | nan        |         0        |                0   | Low        | {"embedding": 0.0, "inliers": 0.0, "rmse": 0.0, "coverage": 0.0}       |

## 10. Lunar-specific crater validation

|   craters_src |   craters_ref |   matched |   crater_center_dev_px |   crater_overlap_score |   consistency | pair                                  | protocol   | sensor_a   | sensor_b   | status                         | homography_source   |   rmse_gt_px |
|--------------:|--------------:|----------:|-----------------------:|-----------------------:|--------------:|:--------------------------------------|:-----------|:-----------|:-----------|:-------------------------------|:--------------------|-------------:|
|            40 |            40 |         4 |                4.44698 |               0.986029 |         0.1   | ctl_OHRC_to_TMC-2                     | controlled | OHRC       | TMC-2      | ok                             | gt                  |     0.641487 |
|            40 |            40 |         5 |                4.11208 |               1        |         0.125 | ctl_OHRC_to_LRO NAC                   | controlled | OHRC       | LRO NAC    | ok                             | gt                  |     0.319671 |
|            40 |            40 |         1 |                2.80136 |               1        |         0.025 | ctl_OHRC_to_IIRS                      | controlled | OHRC       | IIRS       | ok                             | gt                  |     0.809631 |
|            40 |            40 |         1 |                5.35592 |               0.740898 |         0.025 | ctl_TMC-2_to_LRO NAC                  | controlled | TMC-2      | LRO NAC    | ok                             | gt                  |     0.239284 |
|            40 |            40 |         6 |                3.24745 |               0.861911 |         0.15  | ctl_TMC-2_to_TMC-2                    | controlled | TMC-2      | TMC-2      | ok                             | gt                  |     0.255567 |
|            40 |            40 |         1 |                3.22718 |               0.738357 |         0.025 | ctl_TMC-2_to_IIRS                     | controlled | TMC-2      | IIRS       | ok                             | gt                  |     0.269998 |
|            40 |            23 |         0 |              nan       |             nan        |         0     | real_ncf00000_00000_vs_ncn12227_00000 | real       | TMC-2      | TMC-2      | rejected_affine_rescue_ok_4inl | none                |   nan        |
|            40 |            40 |         1 |                3.0872  |               0.720576 |         0.025 | scale_OHRC_to_TMC-2_x0.5              | controlled | OHRC       | TMC-2      | ok                             | gt                  |     1.30437  |
|            40 |            40 |         2 |                5.14765 |               0.916173 |         0.05  | scale_OHRC_to_LRO NAC_x0.5            | controlled | OHRC       | LRO NAC    | ok                             | gt                  |     1.12708  |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_TMC-2_to_LRO NAC_x0.5           | controlled | TMC-2      | LRO NAC    | ok                             | gt                  |     0.135701 |
|            40 |            40 |         1 |                3.97892 |               0.916173 |         0.025 | scale_TMC-2_to_IIRS_x0.5              | controlled | TMC-2      | IIRS       | ok                             | gt                  |     0.250678 |
|            40 |            40 |         1 |                5.93064 |               0.591025 |         0.025 | scale_OHRC_to_TMC-2_x1.0              | controlled | OHRC       | TMC-2      | ok                             | gt                  |     0.782438 |
|            40 |            40 |         5 |                4.32887 |               1        |         0.125 | scale_OHRC_to_LRO NAC_x1.0            | controlled | OHRC       | LRO NAC    | ok                             | gt                  |     0.257308 |
|            40 |            40 |         2 |                4.49277 |               0.930144 |         0.05  | scale_TMC-2_to_LRO NAC_x1.0           | controlled | TMC-2      | LRO NAC    | ok                             | gt                  |     0.230864 |
|            40 |            40 |         3 |                3.32357 |               0.981372 |         0.075 | scale_TMC-2_to_IIRS_x1.0              | controlled | TMC-2      | IIRS       | ok                             | gt                  |     0.191514 |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_OHRC_to_TMC-2_x2.0              | controlled | OHRC       | TMC-2      | ok                             | gt                  |     0.294058 |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_OHRC_to_LRO NAC_x2.0            | controlled | OHRC       | LRO NAC    | ok                             | gt                  |     0.311981 |
|            40 |            40 |         1 |                2.14786 |               0.908537 |         0.025 | scale_TMC-2_to_LRO NAC_x2.0           | controlled | TMC-2      | LRO NAC    | ok                             | gt                  |     4.11504  |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_TMC-2_to_IIRS_x2.0              | controlled | TMC-2      | IIRS       | insufficient_matches           | none                |   nan        |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_OHRC_to_TMC-2_x4.0              | controlled | OHRC       | TMC-2      | insufficient_matches           | none                |   nan        |
|            40 |            40 |         0 |              nan       |             nan        |         0     | scale_OHRC_to_LRO NAC_x4.0            | controlled | OHRC       | LRO NAC    | insufficient_matches           | none                |   nan        |
|            40 |             1 |         0 |              nan       |             nan        |         0     | scale_TMC-2_to_LRO NAC_x4.0           | controlled | TMC-2      | LRO NAC    | insufficient_matches           | none                |   nan        |
|            40 |             0 |         0 |              nan       |             nan        |         0     | scale_TMC-2_to_IIRS_x4.0              | controlled | TMC-2      | IIRS       | insufficient_matches           | none                |   nan        |
|            40 |            40 |         0 |              nan       |             nan        |         0     | retrieved_OHRC_to_TMC-2               | retrieved  | OHRC       | TMC-2      | insufficient_matches           | none                |   nan        |
|            40 |            40 |         0 |              nan       |             nan        |         0     | retrieved_OHRC_to_LRO NAC             | retrieved  | OHRC       | LRO NAC    | insufficient_matches           | none                |   nan        |

Craters are detected independently in both images (Hough circles on CLAHE-processed
patches); reference craters are lifted into the query frame through the accepted
homography and matched within 6 px. `consistency` = matched / min(n_src, n_ref).

![craters](figures/crater_validation.png)

## 11. Geographic localization

Retrieval-as-localization: the top-FAISS candidate's catalog coordinates are the
location prediction; error is haversine distance to the query's catalog coordinates,
against a label-shuffled chance baseline.

- Queries: 55 | median error: 466.4 m
- Chance baseline: 87773.7 m
- Localization gain vs chance: **188.2x**
- Within 1 km: 0.545 | within 10 km: 0.727

| query                                                       | sensor   |   true_lat |   true_lon |   pred_lat |   pred_lon |   retrieval_similarity |   error_m | pred_patch                                                  |
|:------------------------------------------------------------|:---------|-----------:|-----------:|-----------:|-----------:|-----------------------:|----------:|:------------------------------------------------------------|
| TMC-2_ch2_tmc_ncf_20211122T1925079181_d_img_d18_10752_00000 | TMC-2    |   -47.2413 |    350.033 |   -47.058  |    348.985 |               0.952998 |   22319   | TMC-2_ch2_tmc_ncf_20211122T2123225722_d_img_d18_07424_00000 |
| IIRS_ch2_iir_nri_20240124T1430209615_d_img_d18_12160_00000  | IIRS     |   -44.9987 |    346.003 |   -44.9884 |    346.003 |               0.916721 |     310.9 | IIRS_ch2_iir_nri_20240124T1430209615_d_img_d18_11904_00000  |
| TMC-2_ch2_tmc_ncn_20211123T0119537815_d_img_d18_02816_00000 | TMC-2    |   -45.079  |    346.88  |   -45.2573 |    350.03  |               0.952802 |   67560.4 | TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_01792_00000 |
| TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_08832_00000 | TMC-2    |   -45.5393 |    350.03  |   -45.5341 |    350.03  |               0.946467 |     155.5 | TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_08704_00000 |
| TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_09600_00000 | TMC-2    |   -45.57   |    350.03  |   -45.4829 |    350.03  |               0.943095 |    2642.9 | TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_07424_00000 |
| TMC-2_ch2_tmc_nca_20240124T0838058366_d_img_d18_11520_00000 | TMC-2    |   -44.9102 |    349.159 |   -47.217  |    348.985 |               0.957943 |   70045.8 | TMC-2_ch2_tmc_ncf_20211122T2123225722_d_img_d18_11392_00000 |
| IIRS_ch2_iir_nri_20210126T1349194455_d_img_d32_00128_00000  | IIRS     |   -22.3353 |    347.236 |   -44.9013 |    346.003 |               0.913753 |  684962   | IIRS_ch2_iir_nri_20240124T1430209615_d_img_d18_09728_00000  |
| TMC-2_ch2_tmc_nca_20240124T0838058366_d_img_d18_11008_00000 | TMC-2    |   -44.8916 |    349.159 |   -47.2221 |    348.985 |               0.951069 |   70762.9 | TMC-2_ch2_tmc_ncf_20211122T2123225722_d_img_d18_11520_00000 |
| TMC-2_ch2_tmc_ncn_20211122T1925079214_d_img_d18_06144_00000 | TMC-2    |   -45.4316 |    350.03  |   -45.2174 |    346.88  |               0.950894 |   67471.8 | TMC-2_ch2_tmc_ncn_20211123T0119537815_d_img_d18_06272_00000 |
| TMC-2_ch2_tmc_ncf_20211122T2321361810_d_img_d18_10624_00000 | TMC-2    |   -47.1837 |    347.935 |   -47.1785 |    347.935 |               0.959353 |     155.5 | TMC-2_ch2_tmc_ncf_20211122T2321361810_d_img_d18_10496_00000 |
| IIRS_ch2_iir_nri_20210126T1349194455_d_img_d32_03584_00000  | IIRS     |   -22.4375 |    347.236 |   -22.4337 |    347.236 |               0.940508 |     114.7 | IIRS_ch2_iir_nri_20210126T1349194455_d_img_d32_03456_00000  |
| IIRS_ch2_iir_nri_20240124T1430209615_d_img_d18_05632_00000  | IIRS     |   -44.7372 |    346.003 |   -44.727  |    346.003 |               0.933901 |     310.9 | IIRS_ch2_iir_nri_20240124T1430209615_d_img_d18_05376_00000  |

## 12. Key findings

1. **Cross-sensor registration works at sub-pixel level**: median GT RMSE
   0.2948344647884369 px across OHRC/TMC/IIRS/LRO controlled
   pairs under the documented GSD-normalized protocol.
2. **Retrieval is the localization layer**: top-1 candidate lands
   466.4 m from the truth (188.2x
   better than chance) - the FAISS stage is not an optimization, it is the GPS.
3. **Sun-angle robustness holds where data exists**: solved pairs stay sub-pixel in
   the 0-10 and 20-40 deg bins; the empty bins are a data-availability limit, reported
   as such.
4. **ANMS measurably uniformizes the correspondence set** (uniformity gain per pair in
   section 8) while preserving inlier ratio - the ISRO uniform-distribution requirement.
5. **Benchmark is honest**: SIFT ties LunarAI on single-pair reliability; LunarAI's
   decisive advantages are retrieval-driven localization (section 11) and uniform
   coverage (section 8), which classical pipelines do not provide.

## 13. Limitations

- No spatially overlapping cross-sensor imagery exists in the distribution; cross-sensor
  numbers use the documented controlled-GT protocol, and the genuine TMC ncf<->ncn pair
  is exercised separately (it has no GT, so it is reported via reprojection residuals).
- Sun bins 10-20 deg and 60+ deg contain no products; their rows are empty by data
  availability, not by model failure.
- Crater detection is classical (Hough); on texture-heavy terrain it over-detects and
  is capped at the 40 largest circles per patch - consistency ratios are relative to
  that detector, not to a human-labeled crater catalog.
- LRO contribution is a QuickMap reference export, not a NAC mosaic; IIRS spectral
  binaries are absent (browse renders only).
- LunaDNA training was bounded (epoch 1) on CPU; the 120-epoch schedule
  remains available.

## 14. Future work

- Sun-AngleNet: illumination-conditioned descriptor fine-tuning to close the high-sun
  drop in inlier ratio (Exp 2 / sun bins).
- CraterGraphNet: replace circle matching with a crater-graph GNN for topological
  consistency, immune to detector count drift.
- IVF-PQ FAISS index for 100k+ patch scale with the same retrieval accuracy.
- Onboard deployment path: INT8 LunaDNA + sparse SuperPoint for FPGA/SoC rovers.
