# Training Overfitting Documentation

**Document Purpose:** Document training overfitting issue and present best validation epoch.

**Date:** 2026-09-28  
**Problem Statement:** SIH 2026 PS26166  
**Project:** LunarAI - Multi-Modal Lunar Image Correspondence & Registration

---

## ⚠️ TRAINING OVERFITTING ISSUE

### Training Loss Curves

**Epoch-wise Loss Values (from lunadna_training_report.json):**

| Epoch | Train Loss | Val Loss | Gap |
|-------|-----------|----------|-----|
| 1 | 0.2391 | 0.1807 | 0.0584 |
| 2 | 0.0934 | 0.2102 | -0.1168 |
| 3 | 0.0172 | 0.1834 | -0.1661 |
| 4 | 0.0050 | 0.1376 | -0.1326 |
| 5 | 0.0021 | 0.1207 | -0.1186 |
| 6 | 0.0138 | 0.1124 | -0.0986 |
| 7 | 0.0059 | 0.1142 | -0.1082 |
| 8 | 0.0016 | 0.0929 | -0.0913 |
| 9 | 0.0003 | 0.1063 | -0.1060 |
| 10 | 0.0002 | 0.1053 | -0.1051 |
| 11 | 0.0077 | 0.1051 | -0.0974 |
| 12 | 0.0017 | 0.0990 | -0.0974 |
| 13 | 0.0004 | 0.0794 | -0.0790 ✅ **BEST** |
| 14 | 0.0004 | 0.1585 | -0.1581 |
| 15 | 0.0000 | 0.1420 | -0.1419 |
| 16 | 0.0005 | 0.1341 | -0.1336 |
| 17 | 0.0001 | 0.1176 | -0.1175 |
| 18 | 0.0000 | 0.1227 | -0.1227 |
| 19 | 0.0007 | 0.1198 | -0.1191 |
| 20 | 0.0124 | 0.1157 | -0.1033 |
| 21 | 0.0105 | 0.1053 | -0.0947 |

### Best Validation Epoch

**Epoch 13** is the best validation point:
- Train Loss: 0.0004
- Val Loss: 0.0794
- Gap: 0.0790
- Recall@1: 64.9%
- Recall@5: 94.8%
- Recall@10: 97.4%

**Why Epoch 13:**
- Lowest validation loss (0.0794)
- Good train-val gap (0.0790)
- High recall metrics
- Before overfitting spike at epoch 14

---

## ROOT CAUSE ANALYSIS

### 1. Insufficient Training Data

**Dataset Size:**
- Training triplets: 580
- Validation triplets: 132
- Total triplets: 712
- Images: 15 (17 after KAGUYA fix)
- Patches: 940

**Problem:**
- 580 training triplets is extremely small for deep learning
- Typical metric learning requires 10,000+ triplets
- Model cannot learn robust cross-sensor features
- High risk of memorization

### 2. No Data Augmentation

**Current Implementation:**
- No geometric augmentation (rotation, flip, scaling)
- No photometric augmentation (brightness, contrast)
- No noise injection
- No synthetic positive pairs

**Problem:**
- Model sees each patch exactly once
- Cannot learn invariance to transformations
- Increases overfitting risk

### 3. No Regularization

**Current Implementation:**
- No dropout
- No weight decay
- No batch normalization (in projection head)
- No early stopping (used but not optimal)

**Problem:**
- Model has no constraints on complexity
- Can memorize training data
- Overfits to training triplets

### 4. Training on CPU

**Current Implementation:**
- Device: CPU (not GPU)
- Slow training speed
- Limited epochs (21 total)
- Cannot use advanced techniques (mixed precision)

**Problem:**
- Cannot scale to larger datasets
- Limited experimentation
- Modern techniques unavailable

---

## OVERFITTING DIAGNOSIS

### Symptoms

1. **Train Loss → 0:**
   - Epoch 15: 0.0000 (near zero)
   - Epoch 18: 0.0000 (near zero)
   - Indicates memorization

2. **Val Loss Spikes:**
   - Epoch 14: 0.1585 (sudden spike)
   - Epoch 15: 0.1420 (still high)
   - Indicates generalization failure

3. **Train-Val Gap:**
   - Best gap: 0.0790 (epoch 13)
   - Worst gap: 0.1581 (epoch 14)
   - Gaps indicate overfitting

### Diagnosis

**Severity:** Moderate to Severe

**Evidence:**
- Train loss → 0 indicates memorization
- Val loss spikes indicate generalization failure
- Small dataset exacerbates overfitting
- No regularization compounds the problem

**Impact:**
- Poor generalization to new data
- Cross-sensor performance suffers
- Cannot handle unseen terrains
- Limited operational readiness

---

## BEST VALIDATION EPOCH

### Epoch 13 Selection

**Metrics at Epoch 13:**
- Train Loss: 0.0004
- Val Loss: 0.0794 (lowest)
- Recall@1: 64.9%
- Recall@5: 94.8%
- Recall@10: 97.4%

**Why Epoch 13 is Best:**
- Lowest validation loss (0.0794)
- Before overfitting spike at epoch 14
- Good balance of train-val gap (0.0790)
- High recall metrics

**Recommendation:**
- Use epoch 13 checkpoint for inference
- Do not use later epochs (overfitted)
- Report epoch 13 metrics as final results

---

## REQUIRED DISCLOSURE

### For Training Section

> **Training Overfitting Disclaimer:** Training shows overfitting due to small dataset (580 training triplets, 132 validation triplets). Train loss → 0.0000 by epoch 15, val loss spikes at epoch 14. Best validation epoch is 13 (val loss: 0.0794, recall@1: 64.9%). Data augmentation and regularization are identified as future work.

### For Model Section

> **Model Generalization Disclaimer:** LunaDNA model checkpoint from epoch 13 is used for inference. This epoch showed best validation performance (val loss: 0.0794) before overfitting spike. Model generalization is limited by small dataset size (712 total triplets). Expanded dataset with data augmentation is required for better generalization.

---

## FUTURE WORK FOR TRAINING IMPROVEMENT

### Short-Term (Before SIH)

1. **Document Limitation:**
   - Add overfitting disclaimer to training section
   - Present epoch 13 as best validation point
   - Explain small dataset constraint

2. **Use Best Checkpoint:**
   - Ensure epoch 13 checkpoint is used for inference
   - Report epoch 13 metrics as final results
   - Do not use later epochs

### Medium-Term (Post-SIH)

1. **Data Augmentation:**
   - Random rotations (0°, 90°, 180°, 270°)
   - Random flips (horizontal, vertical)
   - Random brightness/contrast adjustments
   - Gaussian noise injection
   - Random crops and scaling

2. **Regularization:**
   - Add dropout to projection head (0.3-0.5)
   - Add L2 weight decay (1e-4 to 1e-3)
   - Add batch normalization
   - Implement proper early stopping

3. **Expanded Dataset:**
   - Target: 5000+ training triplets
   - Include KAGUYA in training
   - Add overlapping sensor coverage
   - Generate synthetic positive pairs

### Long-Term (Full System Upgrade)

1. **Contrastive Learning:**
   - Implement InfoNCE loss
   - Self-supervised pre-training
   - Hard negative mining
   - Cross-sensor positive pairs

2. **Advanced Regularization:**
   - Mixup augmentation
   - CutMix augmentation
   - AutoAugment
   - RandAugment

3. **Architecture Improvements:**
   - Reduce model capacity if needed
   - Add attention mechanisms
   - Use pre-trained on lunar data
   - Implement ensemble methods

---

## HONEST ASSESSMENT

### Current State

**Training Overfitting:** MODERATE TO SEVERE  
**Best Epoch:** 13 (val loss: 0.0794)  
**Generalization:** Limited by small dataset  
**Documentation:** Transparent and complete  
**Future Work:** Clearly defined and feasible  

### Presentation Strategy

**DO:**
✅ Be honest about overfitting
✅ Present epoch 13 as best validation point
✅ Explain small dataset constraint (712 triplets)
✅ Show training loss curves if possible
✅ Document as known limitation
✅ Present future work roadmap

**DO NOT:**
❌ Claim model generalizes well
❌ Use later epochs (overfitted)
❌ Hide train loss → 0
❌ Mislead about dataset size
❌ Omit overfitting from presentation

---

## SUMMARY

**Bottom Line:**
- Training shows overfitting (train loss → 0)
- Best validation epoch: 13 (val loss: 0.0794)
- Root cause: Small dataset (712 triplets), no augmentation, no regularization
- Impact: Poor generalization, limited cross-sensor performance
- Required fix: Expanded dataset + augmentation + regularization

**For SIH Presentation:**
- Present epoch 13 as best validation point
- Be honest about overfitting (train loss → 0)
- Explain small dataset constraint (712 triplets)
- Document as known limitation
- Present future work roadmap

---

**Documentation Status:** COMPLETE  
**Last Updated:** 2026-09-28  
**Next Review:** After SIH Grand Finale
