# Experiment 2 — Sun-Angle Validation Report

**Generated:** 2026-09-27 19:39 · Matcher superpoint+lightglue

## Illumination groups (measured from product labels + SunAngle.csv)

| Group | Real products | Sun elevation |
|---|---|---|
| Low | OHRC south-polar products | ≈ −0.8° … +0.8° (near-terminator) |
| Medium | TMC-2 2021-11-22 forward/nadir, IIRS 2024-01-24 | 37.1° – 38.3° |
| High | TMC-2 2026-07-01 forward | 51.7° |

SunAngle.csv pass statistics: elevation see
`outputs/dataset_analysis/dataset_report.json` (`sun_angle_statistics`).
Illumination transfer matches each group's measured mean/std (gamma 0.85 / 1.0 / 1.15),
then the pipeline's CLAHE normalization is applied to both images before matching.

## Correspondence quality per group (controlled illumination transfer)

| sun_group | runs | success_runs | match_count | inlier_count | inlier_ratio | coverage_score | rmse_px | rmse_gt_px | ssim | ncc | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| low | 4 | 4 | 525.250 | 174.500 | 0.992 | 0.945 | 1.627 | 0.131 | 0.772 | 0.972 | 5.095 |
| medium | 4 | 4 | 268.750 | 108.000 | 0.984 | 0.805 | 1.857 | 0.150 | 0.863 | 0.978 | 4.959 |
| high | 4 | 4 | 158.750 | 76.750 | 0.878 | 0.574 | 5.397 | 0.206 | 0.757 | 0.971 | 6.383 |


![sun angle quality](figures/sun_angle_validation.png)
![sun angle rmse](figures/sun_angle_rmse.png)

## Retrieval accuracy per group (LunaDNA + FAISS)

| sun_group | n | top1_acc | top5_acc |
|---|---|---|---|
| low | 60 | 0.350 | 0.567 |
| medium | 60 | 0.467 | 0.767 |
| high | 60 | 0.367 | 0.617 |


## Real per-product sun elevations

| product | sun_elevation_deg |
|---|---|
| OHRC:ch2_ohr_ncp_20251010T0942085687_d_img_d18.img | -0.770 |
| OHRC:ch2_ohr_ncp_20241115T1326321339_d_img_d18.img | -0.170 |
| OHRC:ch2_ohr_ncp_20241115T1525004388_d_img_d18.img | 0.790 |
| TMC-2:ch2_tmc_ncf_20211122T1925079181_d_img_d18.img | 37.080 |
| TMC-2:ch2_tmc_ncn_20211122T1925079214_d_img_d18.img | 37.080 |
| TMC-2:ch2_tmc_ncf_20211122T2123225722_d_img_d18.img | 37.150 |
| TMC-2:ch2_tmc_ncf_20211122T2321361810_d_img_d18.img | 37.200 |
| TMC-2:ch2_tmc_ncn_20211123T0119537815_d_img_d18.img | 37.330 |
| IIRS:ch2_iir_nri_20240124T1430209615_d_img_d18.hdr | 38.270 |
| TMC-2:ch2_tmc_nca_20240124T0838058366_d_img_d18.img | 39.380 |
| TMC-2:ch2_tmc_ncf_20260701T0716265753_d_img_d18.img | 51.680 |

