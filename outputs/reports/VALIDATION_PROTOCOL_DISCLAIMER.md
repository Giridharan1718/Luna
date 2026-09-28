# Validation Protocol Disclaimer for LunarAI

**Document Purpose:** Official disclaimer to be added to all validation reports and presentations.

**Date:** 2026-09-28  
**Problem Statement:** SIH 2026 PS26166  
**Project:** LunarAI - Multi-Modal Lunar Image Correspondence & Registration

---

## ⚠️ IMPORTANT VALIDATION PROTOCOL DISCLAIMER

### Current Validation Protocol

All validation results reported in LunarAI use a **controlled validation protocol** with same-image offset pairs.

**What This Means:**
- Query and reference images are from the **same sensor**
- Ground truth transformation is **synthetic** (known artificial offsets)
- Validation tests **pipeline correctness** and **technical implementation**
- Results demonstrate **sub-pixel accuracy** on controlled pairs

**What This Does NOT Mean:**
- Results do **NOT** validate real cross-sensor correspondence
- Results do **NOT** represent real-world orbital geometry
- Results do **NOT** prove multi-sensor invariance
- Results are **NOT** applicable to operational scenarios

---

## CONTROLLED PROTOCOL RESULTS

### Multi-Modal Validation (Controlled Pairs)

| Sensor Pair | Success Rate | RMSE (px) | Protocol |
|-------------|-------------|-----------|----------|
| OHRC ↔ TMC-2 | 100% | 0.36 | Controlled |
| OHRC ↔ IIRS | 100% | 0.45 | Controlled |
| OHRC ↔ LRO NAC | 100% | 0.25 | Controlled |
| TMC-2 ↔ IIRS | 100% | 0.27 | Controlled |
| TMC-2 ↔ LRO NAC | 100% | 0.24 | Controlled |

**Interpretation:** These results demonstrate that the pipeline works correctly on controlled pairs with known ground truth. They do not represent real cross-sensor correspondence.

### Sun-Angle Validation (Controlled Pairs)

| Sun Bin | Success Rate | RMSE (px) | Protocol |
|---------|-------------|-----------|----------|
| 0-10° | 100% | 0.16 | Controlled |
| 20-40° | 100% | 0.25 | Controlled |
| 40-60° | 100% | 0.18 | Controlled |

**Interpretation:** These results demonstrate illumination robustness on controlled pairs. They do not represent real sun-angle variation scenarios.

### Scale Validation (Controlled Pairs)

| Scale Factor | Success Rate | RMSE (px) | Protocol |
|--------------|-------------|-----------|----------|
| 0.5× | 100% | 0.26 | Controlled |
| 1.0× | 100% | 0.24 | Controlled |
| 2.0× | 75% | 0.31 | Controlled |
| 4.0× | 0% | N/A | Controlled |

**Interpretation:** These results demonstrate scale invariance up to 2× on controlled pairs. 4× scale failure is a known limitation.

---

## REAL CROSS-SENSOR VALIDATION (CURRENT LIMITATION)

### Cross-Sensor Retrieval Performance

| Metric | Value | Protocol |
|--------|-------|----------|
| Cross-Sensor Recall@1 | 0.0% | Real Cross-Sensor |
| Cross-Sensor Recall@5 | 0.0% | Real Cross-Sensor |
| Cross-Sensor Recall@10 | 0.0% | Real Cross-Sensor |
| Same-Sensor Recall@1 | 73.3% | Same-Sensor |
| Cross-Sensor Fraction in Top-10 | 0.7% | Real Cross-Sensor |

**Interpretation:** Real cross-sensor retrieval shows 0% success. This is a known limitation.

### Root Cause Analysis

**Embedding Sensor Bias:**
- Silhouette Score: 0.1774 (positive = sensor clustering)
- Intra-Sensor Similarity: 0.9068 (very high)
- Inter-Sensor Similarity: 0.5041 (moderate)
- Sensor Bias Score: 0.4027 (positive = sensor-biased)
- **Conclusion:** Embeddings are sensor-biased, not location-invariant

**Training Protocol Limitation:**
- Training used same-sensor triplets only
- No cross-sensor positive pairs
- No cross-sensor hard negatives
- Small dataset (712 triplets total)
- **Conclusion:** Model learned sensor-specific features, not location-invariant features

---

## REQUIRED DISCLAIMERS FOR ALL REPORTS

### For Validation Reports

> **Validation Protocol Disclaimer:** All validation results use a controlled protocol with same-image offset pairs. These results demonstrate pipeline correctness and sub-pixel accuracy on controlled pairs. Real cross-sensor correspondence validation is identified as future work.

### For Retrieval Reports

> **Cross-Sensor Retrieval Disclaimer:** Cross-sensor retrieval currently shows 0% success due to sensor-biased embeddings (silhouette score: 0.1774, bias score: 0.4027). This is a known limitation caused by training with same-sensor triplets only. Cross-sensor invariance requires expanded dataset and retraining with cross-sensor positive pairs.

### For Training Reports

> **Training Overfitting Disclaimer:** Training shows overfitting due to small dataset (712 triplets). Best validation epoch is 13 (val loss: 0.079). Data augmentation and regularization are identified as future work.

### For Coverage Reports

> **Coverage Metric Disclaimer:** Coverage metric represents patch-level feature coverage (fraction of image with detectable features in selected 256×256 patches). 22.6% is reasonable for this metric definition on large lunar images.

---

## DISTINCTION CLARIFICATION

### Controlled Protocol (Current Implementation)

**Definition:** Same-image offset pairs with synthetic ground truth

**Purpose:** Validate pipeline correctness and technical implementation

**Results:** 100% success, sub-pixel accuracy (0.2-0.4 px RMSE)

**Validity:** Demonstrates technical feasibility, not real-world performance

### Real Cross-Sensor Validation (Future Work)

**Definition:** Actual multi-sensor image pairs from different orbits

**Purpose:** Validate true cross-sensor correspondence capability

**Current Status:** Not implemented

**Current Performance:** 0% success (sensor-biased embeddings)

**Required:** Expanded dataset, cross-sensor training, real ground truth

---

## HONEST ASSESSMENT

### What LunarAI Currently Demonstrates

✅ **Technical Implementation:** Complete end-to-end pipeline working correctly  
✅ **Pipeline Robustness:** Works well on controlled pairs  
✅ **Sub-Pixel Accuracy:** Achieved on controlled protocol  
✅ **Engineering Quality:** Well-architected, modular, reproducible  
✅ **Multi-Sensor Support:** All 5 sensors included in dataset  
✅ **Same-Sensor Retrieval:** 73.3% success (validated)

### What LunarAI Does NOT Currently Demonstrate

❌ **True Cross-Sensor Correspondence:** Real multi-sensor matching not validated  
❌ **Cross-Sensor Retrieval:** 0% success on real cross-sensor queries  
❌ **Sensor-Invariant Embeddings:** Embeddings sensor-biased (proven)  
❌ **Real Orbital Geometry:** Validation uses synthetic offsets, not real scenarios  
❌ **Operational Readiness:** Not ready for real-world deployment

---

## FUTURE WORK ROADMAP

### Short-Term (Documentation - Before SIH)

1. ✅ Add validation protocol disclaimer to all reports
2. ✅ Document cross-sensor retrieval limitation
3. ✅ Clarify coverage metric definition
4. ✅ Document training overfitting

### Medium-Term (Post-SIH - Model Improvements)

1. Expand dataset (target: 5000+ triplets)
2. Implement cross-sensor training with proper triplet loss
3. Use contrastive learning (InfoNCE) for better cross-sensor invariance
4. Add data augmentation and regularization

### Long-Term (Full System Upgrade)

1. Acquire real cross-sensor ground truth correspondences
2. Implement real cross-sensor validation
3. Test on operational scenarios
4. Deploy for real-world lunar image analysis

---

## ETHICAL PRESENTATION GUIDELINES

### DO

✅ Clearly label all results as "controlled protocol"  
✅ Distinguish between pipeline implementation and scientific validation  
✅ Be transparent about cross-sensor retrieval limitation  
✅ Present current results as technical demonstration  
✅ Document limitations in reports and presentation  
✅ Present as functional prototype with clear improvement path

### DO NOT

❌ Claim cross-sensor correspondence capability without evidence  
❌ Present controlled results as real cross-sensor validation  
❌ Hide the 0% cross-sensor retrieval failure  
❌ Mislead judges about protocol limitations  
❌ Omit known limitations from presentation

---

## SUMMARY

**Bottom Line:**
- LunarAI is a **technically complete pipeline** that works correctly on controlled validation
- LunarAI does **NOT currently demonstrate** real cross-sensor correspondence capability
- The distinction between controlled and real cross-sensor validation is critical
- We are transparent about this limitation in all documentation
- True cross-sensor validation is identified as future work requiring expanded dataset

**Judges' Questions:**
- Expect questions about validation protocol
- Be prepared to distinguish controlled vs real cross-sensor
- Be honest about current limitations
- Present as a functional prototype with clear improvement path

---

**Disclaimer Status:** APPROVED FOR SIH PRESENTATION  
**Last Updated:** 2026-09-28  
**Approved By:** LunarAI Team
