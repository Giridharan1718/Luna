# Validation Protocol Clarification for LunarAI

**Document Purpose:** Clarify the validation protocol used in LunarAI experiments and distinguish between controlled and real cross-sensor validation.

**Date:** 2026-09-28  
**Problem Statement:** SIH 2026 PS26166  
**Project:** LunarAI - Multi-Modal Lunar Image Correspondence & Registration

---

## IMPORTANT DISTINCTION

### Controlled Protocol vs Real Cross-Sensor Validation

LunarAI uses **two different validation protocols** in the experiments. It is critical to distinguish between them:

---

## 1. CONTROLLED PROTOCOL (Current Implementation)

### Definition:
- Uses **same-image offset pairs** for validation
- Pairs are created by taking one image and applying synthetic offsets
- Both query and reference come from the **same sensor**
- Known ground truth transformation (synthetic offsets)

### Usage in LunarAI:
- **Multi-modal validation:** Controlled pairs for OHRC↔TMC-2, OHRC↔IIRS, etc.
- **Sun-angle validation:** Controlled pairs across sun-angle bins
- **Scale validation:** Controlled pairs at different scales
- **Results reported:** 100% success rate, sub-pixel accuracy

### Strengths:
- ✅ Known ground truth transformation
- ✅ Consistent evaluation across experiments
- ✅ Good for testing pipeline robustness
- ✅ Demonstrates technical implementation correctness

### Limitations:
- ⚠️ **Not real cross-sensor correspondence**
- ⚠️ Same-sensor pairs (not truly multi-modal)
- ⚠️ Artificial offsets (not real orbital geometry)
- ⚠️ Does not test true cross-sensor invariance

### Results:
- Success rate: 100% (controlled pairs)
- RMSE: 0.2-0.4 px (controlled pairs)
- Coverage: 60-80% (controlled pairs)

---

## 2. REAL CROSS-SENSOR VALIDATION (Current Limitation)

### Definition:
- Uses **actual multi-sensor image pairs** from different orbits
- Query and reference come from **different sensors**
- Unknown or approximate ground truth transformation
- Tests true cross-sensor invariance

### Current Status in LunarAI:
- ❌ **Not implemented** in validation experiments
- ❌ **Retrieval stage** shows 0% recall@1 on real cross-sensor queries
- ❌ **Embeddings** not truly cross-sensor invariant
- ❌ **Training** used same-sensor triplets only

### Why It Fails:
1. **Training protocol:** Triplets were same-sensor, geo-proximity based
2. **No cross-sensor negatives:** Training did not include cross-sensor hard negatives
3. **Sensor bias:** Embeddings learned sensor-specific features, not location-invariant
4. **Insufficient data:** Small dataset limited cross-sensor coverage

### Results:
- Retrieval recall@1: 0.0% (real cross-sensor)
- Retrieval recall@5: 0.0% (real cross-sensor)
- Retrieval recall@10: 0.0% (real cross-sensor)
- Cross-sensor fraction in top-10: 0.7%

---

## 3. WHAT THIS MEANS FOR SIH EVALUATION

### What LunarAI Currently Demonstrates:
✅ **Technical Implementation:** Complete end-to-end pipeline working correctly  
✅ **Pipeline Robustness:** Works well on controlled pairs  
✅ **Sub-pixel Accuracy:** Achieved on controlled protocol  
✅ **Engineering Quality:** Well-architected, modular, reproducible  
✅ **Multi-Sensor Support:** All 5 sensors included in dataset  

### What LunarAI Does NOT Currently Demonstrate:
❌ **True Cross-Sensor Correspondence:** Real multi-sensor matching not validated  
❌ **Cross-Sensor Retrieval:** 0% success on real cross-sensor queries  
❌ **Sensor-Invariant Embeddings:** Embeddings sensor-biased, not location-biased  
❌ **Real Orbital Geometry:** Validation uses synthetic offsets, not real scenarios  

---

## 4. HONEST ASSESSMENT FOR JUDGES

### During Presentation:
- **Lead with:** "LunarAI demonstrates a complete pipeline with controlled protocol validation"
- **Clarify early:** "Current validation uses same-image offset pairs, not real cross-sensor"
- **Show metrics:** Present controlled protocol results (100% success, sub-pixel accuracy)
- **Be transparent:** "Real cross-sensor retrieval is a known limitation (0% success)"
- **Future work:** "True cross-sensor validation requires expanded dataset and retraining"

### During Q&A:
- **If asked about cross-sensor:**
  - "Current validation uses controlled protocol (same-image offsets)"
  - "Real cross-sensor retrieval shows 0% success - this is a known limitation"
  - "Cross-sensor invariance requires: (1) expanded dataset, (2) cross-sensor training, (3) true ground truth"
  - "This is documented as future work in our roadmap"

- **If asked about validation rigor:**
  - "We use controlled protocol for pipeline verification"
  - "This demonstrates technical correctness but not real-world cross-sensor performance"
  - "We are transparent about this limitation in our documentation"
  - "True cross-sensor validation is identified as future work"

- **If asked about claims:**
  - "We claim pipeline completeness and engineering quality (validated)"
  - "We do NOT claim real cross-sensor correspondence capability (not validated)"
  - "Our multi-modal results are clearly labeled as controlled protocol"
  - "We distinguish between pipeline implementation and scientific validation"

---

## 5. RECOMMENDED PRESENTATION SLIDES

### Slide 1: Protocol Overview
**Title:** Validation Protocol in LunarAI  
**Content:**
- LunarAI uses two validation protocols:
  1. **Controlled Protocol:** Same-image offset pairs (current implementation)
  2. **Real Cross-Sensor:** Actual multi-sensor pairs (future work)
- Current results are from controlled protocol
- Real cross-sensor validation is identified as future work

### Slide 2: Controlled Protocol Results
**Title:** Controlled Protocol Validation Results  
**Content:**
- Multi-modal pairs: 5 sensor combinations tested
- Success rate: 100% (controlled pairs)
- RMSE: 0.2-0.4 px (controlled pairs)
- Coverage: 60-80% (controlled pairs)
- Demonstrates: Pipeline correctness, technical implementation

### Slide 3: Cross-Sensor Retrieval Limitation
**Title:** Cross-Sensor Retrieval - Current Limitation  
**Content:**
- Real cross-sensor retrieval: 0% recall@1
- Root cause: Training used same-sensor triplets
- Embeddings: Sensor-biased, not location-biased
- Status: Known limitation, documented
- Future work: Cross-sensor training with expanded dataset

### Slide 4: Limitations and Future Work
**Title:** Limitations and Future Work  
**Content:**
**Current Limitations:**
- Small dataset (17 images)
- Training overfitting
- Controlled protocol only
- Cross-sensor retrieval not validated

**Future Work:**
- Expand dataset (500+ images)
- Cross-sensor training
- Real ground truth correspondence
- True cross-sensor validation

---

## 6. ETHICAL PRESENTATION GUIDELINES

### DO:
✅ Clearly label all results as "controlled protocol"  
✅ Distinguish between pipeline implementation and scientific validation  
✅ Be transparent about cross-sensor retrieval limitation  
✅ Present current results as technical demonstration  
✅ Document limitations in reports and presentation  

### DO NOT:
❌ Claim cross-sensor correspondence capability without evidence  
❌ Present controlled results as real cross-sensor validation  
❌ Hide the 0% cross-sensor retrieval failure  
❌ Mislead judges about protocol limitations  
❌ Omit known limitations from presentation  

---

## 7. SUMMARY

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

**Verdict:**
- **Prototype Ready:** Yes (technical implementation complete)
- **Grand Finale Ready:** No (scientific validation incomplete)
- **Best Prototype Candidate:** Only if transparent about limitations

---

**Document Status:** Approved for SIH presentation  
**Last Updated:** 2026-09-28  
**Approved By:** LunarAI Team
