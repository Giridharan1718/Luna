# Experiment 6 — Retrieval Validation Report

**Generated:** 2026-09-27 19:44
Index: FAISS IndexFlatIP, 940 vectors, 512-D
LunaDNA embeddings (checkpoint epoch 1). Queries: 60 held-out patches.

Two ground-truth definitions are used:
- **same-source**: retrieved patch comes from the same source image (terrain check).
- **cross-sensor**: retrieved patch is from a *different* sensor but within 0.5° of the
  query's coordinates — the mission-critical case for multi-modal correspondence.

## Summary

| Metric | Value |
|---|---|
| Top-1 accuracy (same-source) | 0.750 |
| Top-5 accuracy (same-source) | 0.983 |
| Top-10 accuracy (same-source) | 0.983 |
| Top-1 / 5 / 10 accuracy (cross-sensor) | 0.000 / 0.000 / 0.000 |
| mAP (same-source relevance) | 0.790 |
| MRR | 0.857 |
| Queries evaluated | 60 |

![retrieval](figures/retrieval_metrics.png)

## Accuracy by query sensor

| sensor | queries | top1_accuracy |
|---|---|---|
| IIRS | 17 | 0.941 |
| LRO NAC | 5 | 1.000 |
| OHRC | 1 | 0.000 |
| TMC-2 | 37 | 0.649 |


Per-query results: `outputs/reports/retrieval_per_query.csv`.

**Interpretation note.** The distribution contains no spatially overlapping
cross-sensor imagery, so a "cross-sensor" hit requires the query and candidate to
share terrain that no other sensor imaged — its accuracy is intrinsically bounded by
data availability, not by the model. The headline retrieval evidence is the
same-source accuracy and mAP above, plus the embedding-separation statistics
(random-pair cosine ≈ 0.44 ± 0.23 vs top-1 ≈ 0.96).
