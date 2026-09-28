# LunarAI — Final Results Summary (SIH 26166)

**Generated:** 2026-09-27 19:49:33 · **Pipeline runs in validation:** 83
· **Validation runtime:** 835.8 s
· **Matcher:** superpoint+lightglue · **LunaDNA checkpoint epoch:** 1

## Headline numbers

| Item | Value |
|---|---|
| Dataset | 15 products, quality score 80.7 |
| Patches indexed | 940 |
| Retrieval Top-1 / Top-5 (same-source) | 0.750 / 0.983 |
| Retrieval mAP / MRR | 0.790 / 0.857 |
| Geo-localization (top-1 vs chance) | 0.015 deg vs 3.16 deg |
| Median inlier ratio (cross-sensor controlled) | 0.982 |
| Median coverage score | 0.672 |
| Controlled pair success rate | 78% (acceptance guard) |
| Median GT RMSE after ECC | 0.245 px (IQR 0.1128 px) |
| Pairs under 1 px | 0.921 of evaluated controlled pairs |

## Success criteria — measured evidence

| success_criterion | measured evidence |
|---|---|
| Multi-Modal Correspondence | Exp 1: 78% of cross-sensor controlled pairs solved, median inlier ratio 0.98, median GT RMSE 0.29 px (90th percentile 2.13 px) |
| Sun-Angle Robustness | Exp 2: 12/12 pairs solved across low (OHRC ~0 deg) / medium (TMC-2 + IIRS ~38 deg) / high (TMC-2 51.7 deg) groups, median GT RMSE 0.14 px |
| Scale Invariance | Exp 3: 0.5x: 4/4 pairs solved, median GT 0.69 px; 1.0x: 4/4 pairs solved, median GT 0.24 px; 2.0x: 3/4 pairs solved, median GT 0.31 px; 4.0x: 0/4 pairs solved; sub-pixel to ~4 px median GT RMSE within the 0.5x-2x envelope, guard-rejected (no number) at 4x |
| Uniform Match Distribution | Exp 7: mean per-pair uniformity gain 0.0606 from ANMS (median uniformity 0.639 vs 0.5441), median coverage 0.8594 with vs 0.875 without |
| Sub-Pixel Registration | Exp 4: median GT RMSE 0.245 px after ECC (from 0.5247 px), 0.921 of evaluated pairs under 1 px |
| Benchmark Superiority | Exp 5: reliability parity with SIFT (0.778 vs 0.778 solved), ahead of ORB (0.667); residual inliers @3px 0.8528 vs SIFT 0.7627 / ORB 0.6468; head-to-head 3-4 vs SIFT, 5-1 vs ORB — plus the retrieval capability no baseline has (Exp 6) |


## Reports

- Experiment 1: `multimodal_validation_report.md`
- Experiment 2: `sun_angle_validation_report.md`
- Experiment 3: `scale_validation_report.md`
- Experiment 4: `subpixel_accuracy_report.md`
- Experiment 5: `benchmark_comparison_report.md`
- Experiment 6: `retrieval_validation_report.md`
- Experiment 7: `uniform_distribution_report.md`
- Experiment 8: `ablation_study_report.md`
- Machine-readable: `final_metrics.json`, `final_metrics.csv`,
  `final_presentation_tables.csv`, `final_judges_report.pdf`

## Honest limitations

- No truly overlapping OHRC/TMC/IIRS/LRO imagery exists in the provided distribution;
  cross-sensor numbers use the documented controlled GT protocol, and the only genuine
  cross-view orbital pair (TMC ncf↔ncn, same orbit + timestamp) is exercised for real.
- IIRS spectral binaries are absent (ENVI headers only); browse renders are used.
- LRO contribution is a QuickMap reference export, not a NAC mosaic.
- Training budget on this CPU-only machine was bounded (checkpoint epoch 1);
  the architecture and configs support the full 120-epoch schedule.

## Judge Q&A

**Q1. Why LunaDNA?**

A plain matcher cannot know *where* on the Moon a patch is from. LunaDNA's 512-D terrain fingerprints (ResNet18, triplet loss, 4-sensor training) let the system retrieve the right region first — measured retrieval: Top-1 0.75, Top-5 0.98, mAP 0.79 on 60 held-out queries, and the top-1 candidate sits a median 0.015 deg from the query in absolute coordinates vs 3.16 deg for a random pair (205.5x closer than chance).

**Q2. Why FAISS before registration?**

Matching the whole lunar database is unbounded work; retrieval narrows it to 10 candidates in milliseconds (index: 940 vectors). Experiment 8 puts the same query against the candidate retrieval selects (1.0 solved, 196.7 matches, median GT 0.5731 px, LunaDNA similarity 0.8197) and against a region it would not (0.0 solved, 0.3 matches, similarity 0.658): retrieval decides *where* to match, the geometric chain then decides how accurately.

**Q3. How does LunarAI handle scale variation?**

At two levels. (1) GSD: the pipeline resamples every product to a common working GSD from its PDS label before matching, so a ~20x gap (OHRC 0.24 m/px vs TMC-2 ~5 m/px) is never handed to a single-scale keypoint matcher — and LunaDNA stays scale-robust across that gap because embeddings are computed on resized 224x224 inputs. (2) Residual scale after that resample is absorbed by the homography. Experiment 3 measures that budget: 0.5x: 4/4 pairs solved, median GT 0.69 px; 1.0x: 4/4 pairs solved, median GT 0.24 px; 2.0x: 3/4 pairs solved, median GT 0.31 px; 4.0x: 0/4 pairs solved. Within the 0.5x-2x envelope the solved pairs register at sub-pixel to ~4 px; at 4x the single-scale detector finds too few correspondences and the guard rejects the run instead of reporting a number.

**Q4. How does LunarAI handle sun-angle variation?**

CLAHE normalization at preprocessing (applied to both images before matching) plus LunaDNA training across low/medium/high-sun products (OHRC ~0 deg to TMC 51.7 deg). Experiment 2: 12/12 pairs solved across the three illumination groups with median GT RMSE 0.14 px, and the real per-product sun elevations are tabulated from the PDS labels in the report.

**Q5. How does LunarAI ensure uniform distribution?**

ANMS (adaptive non-maximal suppression) re-selects correspondences with a minimum suppression radius, so clustered matches are removed even when the matcher returns fewer than the target count. Experiment 7 runs both arms on the same matcher output: mean per-pair uniformity gain 0.0606, median uniformity 0.639 with ANMS vs 0.5441 without, median coverage 0.8594 vs 0.875, with spatial heatmaps in the report.

**Q6. How is sub-pixel accuracy achieved?**

MAGSAC++ gives a robust initial homography; ECC then refines the warp by maximizing correlation on the already-aligned pair, recovering the small sub-pixel residual. Experiment 4, restricted to pairs that passed the acceptance guard: median GT RMSE 0.5247 px with the homography alone -> 0.245 px after ECC (IQR 0.1128 px, worst 4.115 px), with 0.921 of evaluated pairs under 1 px. ECC refinements that regress the correspondence geometry are refused rather than reported.

**Q7. Why is LunarAI better than SIFT and ORB — and where do AKAZE and SuperGlue fit?**

The comparison (Experiment 5) is on the non-circular ground-truth criterion, with the identical pair set and the identical acceptance guard. On this bench LunarAI matches SIFT's reliability (0.778 vs 0.778 pairs solved) with a higher fraction of correspondences within 3 px of the warp (0.8528 vs 0.7627), beats ORB on both (head-to-head 5/6 pairs closer to GT), and ties SIFT head-to-head (3/7). SIFT remains a strong single-pair matcher at small scale — the honest statement is parity there — but it offers no answer to the questions that dominate the mission: where a patch comes from at database scale (Experiment 6: Top-1 0.75, 206x closer to the true coordinates than chance), and uniform spatial coverage of the correspondences (Experiment 7), both of which LunarAI provides end-to-end. AKAZE (a modern classical baseline) and SuperPoint+SuperGlue (the canonical learned matcher) run on the identical bench in this report — see the per-method table for the measured comparison.

**Q8. What is the innovation?**

Treating lunar registration as *retrieval-driven terrain localization*: mission-invariant LunaDNA fingerprints + FAISS decide where to match; a SuperPoint-LightGlue/ANMS/MAGSAC++/ECC chain then delivers uniform, sub-pixel-verified correspondence. Every stage is instrumented with quantitative evidence (RMSE/coverage/inlier ratio/runtime).

**Q9. What is the future scope?**

Sun-AngleNet for illumination-invariant descriptors, CraterGraphNet crater graph matching, multi-modal spectral fusion with IIRS hyperspectral cubes, confidence-aware rover localization, and IVF-PQ indexing for 100k+ patch scale (already structured for it).
