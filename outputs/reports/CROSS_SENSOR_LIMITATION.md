# Cross-Sensor Retrieval Limitation Documentation

**Document Purpose:** Official documentation of cross-sensor retrieval limitations and root cause analysis.

**Date:** 2026-09-28  
**Problem Statement:** SIH 2026 PS26166  
**Project:** LunarAI - Multi-Modal Lunar Image Correspondence & Registration

---

## ⚠️ CROSS-SENSOR RETRIEVAL LIMITATION

### Current Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cross-Sensor Recall@1** | 0.0% | Complete failure on cross-sensor queries |
| **Cross-Sensor Recall@5** | 0.0% | No cross-sensor matches in top-5 |
| **Cross-Sensor Recall@10** | 0.0% | No cross-sensor matches in top-10 |
| **Cross-Sensor Fraction in Top-10** | 0.7% | Nearly all top-10 results are same-sensor |
| **Same-Sensor Recall@1** | 73.3% | Good performance on same-sensor queries |

**Conclusion:** Cross-sensor retrieval fails completely (0% success). This is a known limitation.

---

## ROOT CAUSE ANALYSIS

### 1. Embedding Sensor Bias (Proven Quantitatively)

**Embedding Bias Analysis Results:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.1774 | Positive = clustering by sensor |
| **Avg Intra-Sensor Similarity** | 0.9068 | Same-sensor patches very similar |
| **Avg Inter-Sensor Similarity** | 0.5041 | Different-sensor patches less similar |
| **Sensor Bias Score** | 0.4027 | Positive = sensor-biased |
| **Is Sensor Biased** | **TRUE** | Embeddings cluster by sensor, not location |

**Interpretation:**
- Silhouette score > 0 indicates clustering (0.1774 = moderate sensor clustering)
- Intra-sensor similarity (0.9068) >> inter-sensor similarity (0.5041)
- 45% gap indicates sensor-specific patterns dominate over location patterns
- For location-invariant embeddings, should cluster by location, not sensor

**Visual Evidence:**
- PCA visualization shows sensor clustering (not shown in this document)
- Embeddings from same sensor are very similar
- Embeddings from different sensors are less similar
- This is the opposite of what location-invariant embeddings should do

### 2. Training Protocol Limitation

**Current Training Protocol (nb04_lunadna_training.py):**

**Triplet Formation:**
- Positives: Same-sensor, geo-proximity (<0.5° or <1.5°)
- Negatives: Different-sensor, geo-distant (>5°)
- Geo-split: 1° grid cells to reduce leakage

**Training Statistics:**
- Training triplets: 580
- Validation triplets: 132
- Total triplets: 712
- Sensors: OHRC, TMC-2, IIRS, LRO NAC (KAGUYA not included)

**Limitations:**
1. **No True Cross-Sensor Positives:** Positives are same-sensor only
2. **No Cross-Sensor Hard Negatives:** No different-sensor, similar-location negatives
3. **Sparse Data Constraint:** 712 triplets is extremely small
4. **Fallback Protocol:** Uses training-region patches for validation (potential leakage)

**Impact on Cross-Sensor Invariance:**
- Training objective: Optimize for same-sensor similarity
- No cross-sensor signal: Model never sees cross-sensor positive pairs
- Sensor bias emerges: Model learns sensor-specific features
- Result: Embeddings cluster by sensor, not location

### 3. Validation Protocol Limitation

**Current Validation (nb06_retrieval_testing.py):**

**Test Set Composition:**
- Total queries: 60
- Cross-sensor queries: 0
- Same-sensor queries: 60
- Test set limitation: No cross-sensor queries

**Why This Matters:**
- Cannot measure cross-sensor success without cross-sensor queries
- 0/0 cross-sensor results are meaningless
- Validation protocol does not test cross-sensor capability

---

## DISTINCTION: TRAINING VS REAL RETRIEVAL

### Training Recall Metrics (from nb04_lunadna_training.json)

| Metric | Value | Context |
|--------|-------|---------|
| **Recall@1** | 64.9% | Same-sensor, geo-proximity validation |
| **Recall@5** | 94.8% | Same-sensor, geo-proximity validation |
| **Recall@10** | 97.4% | Same-sensor, geo-proximity validation |

**Context:** These metrics are for same-sensor validation with geographic proximity. They do not represent cross-sensor retrieval performance.

### Real Cross-Sensor Retrieval Metrics (from cross_sensor_recall_analysis.json)

| Metric | Value | Context |
|--------|-------|---------|
| **Cross-Sensor Recall@1** | 0.0% | Real cross-sensor queries |
| **Same-Sensor Recall@1** | 73.3% | Same-sensor queries |

**Context:** These metrics represent real cross-sensor and same-sensor retrieval performance.

**Key Difference:**
- Training recall (64.9%) is for same-sensor, geo-proximity pairs
- Real cross-sensor recall (0.0%) is for actual multi-sensor queries
- These are fundamentally different metrics
- Do not conflate training recall with real retrieval performance

---

## WHAT THIS MEANS FOR SIH EVALUATION

### Claims That ARE Supported

✅ **Pipeline Completeness:** Complete end-to-end pipeline implemented  
✅ **Technical Implementation:** All components working correctly  
✅ **Same-Sensor Retrieval:** 73.3% success on same-sensor queries  
✅ **Controlled Protocol Validation:** 100% success on controlled pairs  
✅ **Sub-Pixel Accuracy:** Achieved on controlled protocol  
✅ **Engineering Quality:** Well-architected, modular, reproducible  

### Claims That Are NOT Supported

❌ **Cross-Sensor Correspondence:** 0% success on real cross-sensor queries  
❌ **Sensor-Invariant Embeddings:** Proven sensor-biased (0.4027 bias score)  
❌ **Multi-Sensor Invariance:** Not validated, current 0% success  
❌ **Real-World Performance:** Not tested on operational scenarios  
❌ **Operational Readiness:** Not ready for deployment  

---

## REQUIRED DISCLAIMERS

### For Retrieval Section

> **Cross-Sensor Retrieval Disclaimer:** Cross-sensor retrieval currently shows 0% success due to sensor-biased embeddings (silhouette score: 0.1774, bias score: 0.4027). This is a known limitation caused by training with same-sensor triplets only. Cross-sensor invariance requires expanded dataset and retraining with cross-sensor positive pairs.

### For Training Section

> **Training vs Retrieval Disclaimer:** Training recall metrics (64.9% @1) are for same-sensor, geo-proximity validation. Real cross-sensor retrieval shows 0% success. These are fundamentally different metrics and should not be conflated.

### For Embedding Section

> **Embedding Bias Disclaimer:** LunaDNA embeddings are sensor-biased (proven by quantitative analysis). Embeddings cluster by sensor (silhouette score: 0.1774) rather than by location. This is a known limitation caused by training protocol using same-sensor triplets only.

---

## FUTURE WORK FOR CROSS-SENSOR INVARIANCE

### Required Changes

1. **Expand Dataset:**
   - Target: 5000+ triplets
   - Include KAGUYA in training
   - Add overlapping sensor coverage
   - Acquire real cross-sensor correspondences

2. **Implement Cross-Sensor Training:**
   - Add cross-sensor positive pairs (same location, different sensors)
   - Add cross-sensor hard negatives (different location, similar appearance)
   - Use contrastive learning (InfoNCE) for better invariance
   - Implement proper geographic split

3. **Implement Real Cross-Sensor Validation:**
   - Test on real multi-sensor image pairs
   - Use ground truth correspondences
   - Report both controlled and real validation results
   - Add statistical significance testing

### Expected Timeline

- Dataset expansion: 1-2 months
- Cross-sensor training implementation: 2-3 months
- Real cross-sensor validation: 1 month
- **Total:** 4-6 months

---

## HONEST ASSESSMENT

### Current State

**Cross-Sensor Retrieval:** NOT WORKING (0% success)  
**Root Cause:** Sensor-biased embeddings from training protocol  
**Documentation:** Transparent and complete  
**Future Work:** Clearly defined and feasible  

### Presentation Strategy

**DO:**
✅ Be honest about 0% cross-sensor success
✅ Explain root cause (sensor-biased embeddings)
✅ Show quantitative bias analysis (0.4027 bias score)
✅ Distinguish training recall (64.9%) from real retrieval (0%)
✅ Present as functional prototype with known limitation
✅ Document cross-sensor invariance as future work

**DO NOT:**
❌ Claim cross-sensor correspondence capability
❌ Present training recall as real retrieval performance
❌ Hide the 0% cross-sensor failure
❌ Mislead judges about current capabilities
❌ Omit sensor bias analysis from presentation

---

## SUMMARY

**Bottom Line:**
- Cross-sensor retrieval currently fails (0% success)
- Root cause: Sensor-biased embeddings (proven quantitatively)
- Training protocol limitation: Same-sensor triplets only
- Required fix: Expanded dataset + cross-sensor training
- Timeline: 4-6 months for full implementation

**For SIH Presentation:**
- Present as functional prototype with known limitation
- Be transparent about 0% cross-sensor success
- Explain root cause clearly (sensor bias = 0.4027)
- Distinguish training metrics from real retrieval
- Document cross-sensor invariance as future work

---

**Documentation Status:** COMPLETE  
**Last Updated:** 2026-09-28  
**Next Review:** After SIH Grand Finale
