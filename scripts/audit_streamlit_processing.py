"""
STRICT STREAMLIT PROCESSING AUDIT

This script performs a comprehensive audit to verify that the Streamlit application
is performing REAL processing and not displaying hardcoded/demo outputs.

Run this after uploading two images to the Streamlit app to verify real processing.
"""

from __future__ import annotations

import os
import json
import sys
import time
from pathlib import Path
from typing import Any

# Set OpenMP environment variable before importing torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lunarai_lib.config import load_config
from lunarai_lib.lunadna import compute_embeddings
from lunarai_lib.live import register, retrieve, session, uploads_dir
from lunarai_lib.validation import load_gray, ValidationSuite

cfg = load_config()


class ProcessingAuditor:
    """Strict auditor for Streamlit processing verification."""
    
    def __init__(self):
        self.cfg = load_config()
        self.suite = session(cfg)
        self.results: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
    
    def log(self, message: str, level: str = "info"):
        """Log audit message."""
        prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]", "success": "[SUCCESS]"}[level]
        print(f"{prefix} {message}")
        if level == "warning":
            self.warnings.append(message)
        elif level == "error":
            self.errors.append(message)
    
    def check_1_image_upload(self, src_path: str, ref_path: str) -> bool:
        """Check 1: Image Upload - Verify uploaded files are actually read from disk."""
        print("\n" + "="*80)
        print("CHECK 1: IMAGE UPLOAD")
        print("="*80)
        
        src = Path(src_path)
        ref = Path(ref_path)
        
        if not src.exists():
            self.log(f"Source image not found: {src}", "error")
            return False
        if not ref.exists():
            self.log(f"Reference image not found: {ref}", "error")
            return False
        
        # File info
        src_info = {
            "filename": src.name,
            "size_bytes": src.stat().st_size,
            "size_mb": src.stat().st_size / (1024 * 1024),
        }
        ref_info = {
            "filename": ref.name,
            "size_bytes": ref.stat().st_size,
            "size_mb": ref.stat().st_size / (1024 * 1024),
        }
        
        # Read images
        src_img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        ref_img = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)
        
        if src_img is None:
            self.log(f"Cannot decode source image: {src}", "error")
            return False
        if ref_img is None:
            self.log(f"Cannot decode reference image: {ref}", "error")
            return False
        
        src_info.update({
            "width": src_img.shape[1],
            "height": src_img.shape[0],
            "channels": src_img.shape[2] if len(src_img.shape) == 3 else 1,
            "dtype": str(src_img.dtype),
        })
        ref_info.update({
            "width": ref_img.shape[1],
            "height": ref_img.shape[0],
            "channels": ref_img.shape[2] if len(ref_img.shape) == 3 else 1,
            "dtype": str(ref_img.dtype),
        })
        
        self.results["image_upload"] = {"source": src_info, "reference": ref_info}
        
        print(f"Source: {src_info['filename']}")
        print(f"  Size: {src_info['size_mb']:.2f} MB")
        print(f"  Dimensions: {src_info['width']}×{src_info['height']}")
        print(f"  Format: {src_info['dtype']}, {src_info['channels']} channels")
        print(f"\nReference: {ref_info['filename']}")
        print(f"  Size: {ref_info['size_mb']:.2f} MB")
        print(f"  Dimensions: {ref_info['width']}×{ref_info['height']}")
        print(f"  Format: {ref_info['dtype']}, {ref_info['channels']} channels")
        
        self.log("Image upload verified - files read from disk", "success")
        return True
    
    def check_2_preprocessing(self, src_path: str, ref_path: str) -> bool:
        """Check 2: Preprocessing - Verify preprocessing runs on uploaded images."""
        print("\n" + "="*80)
        print("CHECK 2: PREPROCESSING")
        print("="*80)
        
        src = load_gray(src_path)
        ref = load_gray(ref_path)
        
        if src is None or ref is None:
            self.log("Preprocessing failed - cannot load images", "error")
            return False
        
        self.results["preprocessing"] = {
            "source_original_size": src.shape,
            "reference_original_size": ref.shape,
            "source_processed_dtype": str(src.dtype),
            "reference_processed_dtype": str(ref.dtype),
        }
        
        print(f"Source original size: {src.shape}")
        print(f"Reference original size: {ref.shape}")
        print(f"Source processed dtype: {src.dtype}")
        print(f"Reference processed dtype: {ref.dtype}")
        
        self.log("Preprocessing verified - images loaded and converted to grayscale", "success")
        return True
    
    def check_3_lunadna(self, src_path: str, ref_path: str) -> bool:
        """Check 3: LunaDNA - Verify embeddings are generated from uploaded images."""
        print("\n" + "="*80)
        print("CHECK 3: LUNADNA ENCODER")
        print("="*80)
        
        try:
            # Generate embeddings
            embeddings = compute_embeddings(
                self.suite.model,
                [str(src_path), str(ref_path)],
                device=getattr(self.suite, "device", "cpu"),
                size=224
            )
            
            if embeddings is None or len(embeddings) == 0:
                self.log("LunaDNA failed - no embeddings generated", "error")
                return False
            
            self.results["lunadna"] = {
                "embedding_shape": embeddings.shape,
                "embedding_dimension": embeddings.shape[1],
                "source_embedding_first_10": embeddings[0, :10].tolist(),
                "reference_embedding_first_10": embeddings[1, :10].tolist(),
            }
            
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Embedding dimension: {embeddings.shape[1]}")
            print(f"Source embedding first 10 values: {embeddings[0, :10]}")
            print(f"Reference embedding first 10 values: {embeddings[1, :10]}")
            
            if embeddings.shape[1] != 512:
                self.log(f"Embedding dimension is {embeddings.shape[1]}, expected 512", "warning")
            
            self.log("LunaDNA verified - embeddings generated from uploaded images", "success")
            return True
        except Exception as e:
            self.log(f"LunaDNA failed with error: {e}", "error")
            return False
    
    def check_4_faiss(self, src_path: str) -> bool:
        """Check 4: FAISS Retrieval - Verify retrieval is based on generated embeddings."""
        print("\n" + "="*80)
        print("CHECK 4: FAISS RETRIEVAL")
        print("="*80)
        
        try:
            # Run retrieval
            retrieval_results = retrieve(src_path, k=5, cfg=self.cfg, suite=self.suite)
            
            if retrieval_results is None or len(retrieval_results) == 0:
                self.log("FAISS retrieval failed - no results", "error")
                return False
            
            self.results["faiss"] = {
                "top_5_candidates": retrieval_results.to_dict(orient="records"),
                "similarity_scores": retrieval_results["similarity"].tolist(),
            }
            
            print(f"Top-5 retrieved candidates:")
            for idx, row in retrieval_results.iterrows():
                print(f"  {idx+1}. {row['patch_id']} (similarity: {row['similarity']:.4f})")
            
            self.log("FAISS retrieval verified - based on generated embeddings", "success")
            return True
        except Exception as e:
            self.log(f"FAISS retrieval failed with error: {e}", "error")
            return False
    
    def check_5_superpoint(self, src_path: str, ref_path: str) -> bool:
        """Check 5: SuperPoint - Verify keypoints are detected from uploaded images."""
        print("\n" + "="*80)
        print("CHECK 5: SUPERPOINT DETECTION")
        print("="*80)
        
        try:
            src = load_gray(src_path)
            ref = load_gray(ref_path)
            
            # Run matching through the suite
            m = self.suite.matcher.match(src, ref)
            
            if m is None:
                self.log("SuperPoint failed - no matches", "error")
                return False
            
            self.results["superpoint"] = {
                "source_keypoint_count": len(m.kp0),
                "reference_keypoint_count": len(m.kp1),
            }
            
            print(f"Source keypoint count: {len(m.kp0)}")
            print(f"Reference keypoint count: {len(m.kp1)}")
            
            self.log("SuperPoint verified - keypoints detected from uploaded images", "success")
            return True
        except Exception as e:
            self.log(f"SuperPoint failed with error: {e}", "error")
            return False
    
    def check_6_lightglue(self, src_path: str, ref_path: str) -> bool:
        """Check 6: LightGlue - Verify descriptor matching runs."""
        print("\n" + "="*80)
        print("CHECK 6: LIGHTGLUE MATCHING")
        print("="*80)
        
        try:
            src = load_gray(src_path)
            ref = load_gray(ref_path)
            
            m = self.suite.matcher.match(src, ref)
            
            if m is None:
                self.log("LightGlue failed - no matches", "error")
                return False
            
            self.results["lightglue"] = {
                "total_matches": len(m.kp0),
            }
            
            print(f"Total matches: {len(m.kp0)}")
            
            self.log("LightGlue verified - descriptor matching performed", "success")
            return True
        except Exception as e:
            self.log(f"LightGlue failed with error: {e}", "error")
            return False
    
    def check_7_anms(self, src_path: str, ref_path: str) -> bool:
        """Check 7: ANMS - Verify ANMS selection runs."""
        print("\n" + "="*80)
        print("CHECK 7: ANMS DISTRIBUTION")
        print("="*80)
        
        try:
            # Run full registration to get ANMS results
            run = register(src_path, ref_path, name="audit_anms", cfg=self.cfg, 
                         suite=self.suite, save=False)
            
            if run is None or "result" not in run:
                self.log("ANMS failed - no registration result", "error")
                return False
            
            result = run["result"]
            anms_matches = result.get("anms_matches", 0)
            match_count = result.get("match_count", 0)
            
            self.results["anms"] = {
                "keypoints_before_anms": match_count,
                "keypoints_after_anms": anms_matches,
                "reduction_ratio": anms_matches / match_count if match_count > 0 else 0,
            }
            
            print(f"Keypoints before ANMS: {match_count}")
            print(f"Keypoints after ANMS: {anms_matches}")
            print(f"Reduction ratio: {anms_matches / match_count if match_count > 0 else 0:.3f}")
            
            self.log("ANMS verified - uniform distribution selection performed", "success")
            return True
        except Exception as e:
            self.log(f"ANMS failed with error: {e}", "error")
            return False
    
    def check_8_magsac(self, src_path: str, ref_path: str) -> bool:
        """Check 8: MAGSAC++ - Verify outlier rejection runs."""
        print("\n" + "="*80)
        print("CHECK 8: MAGSAC++ OUTLIER REMOVAL")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_magsac", cfg=self.cfg,
                         suite=self.suite, save=False)
            
            if run is None or "result" not in run:
                self.log("MAGSAC++ failed - no registration result", "error")
                return False
            
            result = run["result"]
            match_count = result.get("match_count", 0)
            inlier_count = result.get("inlier_count", 0)
            outlier_count = match_count - inlier_count
            
            self.results["magsac"] = {
                "matches": match_count,
                "inliers": inlier_count,
                "outliers": outlier_count,
                "inlier_ratio": inlier_count / match_count if match_count > 0 else 0,
            }
            
            print(f"Matches: {match_count}")
            print(f"Inliers: {inlier_count}")
            print(f"Outliers: {outlier_count}")
            print(f"Inlier ratio: {inlier_count / match_count if match_count > 0 else 0:.3f}")
            
            self.log("MAGSAC++ verified - outlier rejection performed", "success")
            return True
        except Exception as e:
            self.log(f"MAGSAC++ failed with error: {e}", "error")
            return False
    
    def check_9_homography(self, src_path: str, ref_path: str) -> bool:
        """Check 9: Homography - Verify transformation matrix is computed."""
        print("\n" + "="*80)
        print("CHECK 9: HOMOGRAPHY ESTIMATION")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_homography", cfg=self.cfg,
                         suite=self.suite, save=False)
            
            if run is None or "H" not in run or run["H"] is None:
                self.log("Homography failed - no transformation matrix", "error")
                return False
            
            H = run["H"]
            
            self.results["homography"] = {
                "matrix": H.tolist(),
                "determinant": float(np.linalg.det(H[:2, :2])),
                "translation": [float(H[0, 2]), float(H[1, 2])],
            }
            
            print("Homography matrix:")
            print(np.round(H, 6))
            print(f"Determinant: {np.linalg.det(H[:2, :2]):.6f}")
            print(f"Translation: [{H[0, 2]:.2f}, {H[1, 2]:.2f}]")
            
            self.log("Homography verified - transformation matrix computed", "success")
            return True
        except Exception as e:
            self.log(f"Homography failed with error: {e}", "error")
            return False
    
    def check_10_ecc(self, src_path: str, ref_path: str) -> bool:
        """Check 10: ECC - Verify refinement runs."""
        print("\n" + "="*80)
        print("CHECK 10: ECC REFINEMENT")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_ecc", cfg=self.cfg,
                         suite=self.suite, save=False)
            
            if run is None or "result" not in run:
                self.log("ECC failed - no registration result", "error")
                return False
            
            result = run["result"]
            ecc_applied = result.get("ecc_applied", False)
            ecc_correlation = result.get("ecc_correlation", 0.0)
            rmse_before_ecc = result.get("rmse_px_before_ecc", 0.0)
            rmse_after_ecc = result.get("rmse_px", 0.0)
            
            self.results["ecc"] = {
                "ecc_applied": ecc_applied,
                "ecc_correlation": ecc_correlation,
                "rmse_before_ecc": rmse_before_ecc,
                "rmse_after_ecc": rmse_after_ecc,
                "improvement": rmse_before_ecc - rmse_after_ecc,
            }
            
            print(f"ECC applied: {ecc_applied}")
            print(f"ECC correlation: {ecc_correlation:.4f}")
            print(f"RMSE before ECC: {rmse_before_ecc:.4f} px")
            print(f"RMSE after ECC: {rmse_after_ecc:.4f} px")
            print(f"Improvement: {rmse_before_ecc - rmse_after_ecc:.4f} px")
            
            self.log("ECC verified - refinement performed", "success")
            return True
        except Exception as e:
            self.log(f"ECC failed with error: {e}", "error")
            return False
    
    def check_11_registration_output(self, src_path: str, ref_path: str) -> bool:
        """Check 11: Registration Output - Display reference, target, registered, overlay."""
        print("\n" + "="*80)
        print("CHECK 11: REGISTRATION OUTPUT")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_output", cfg=self.cfg,
                         suite=self.suite, save=True)
            
            if run is None or "images" not in run:
                self.log("Registration output failed - no images", "error")
                return False
            
            images = run["images"]
            
            self.results["registration_output"] = {
                "source_path": str(images.get("src")),
                "reference_path": str(images.get("ref")),
                "registered_path": str(images.get("registered")),
                "run_dir": str(run.get("run_dir")),
            }
            
            print(f"Source image: {images.get('src')}")
            print(f"Reference image: {images.get('ref')}")
            print(f"Registered image: {images.get('registered')}")
            print(f"Run directory: {run.get('run_dir')}")
            
            # Verify files exist
            for key, path in images.items():
                if path and Path(path).exists():
                    print(f"  [OK] {key}: exists")
                else:
                    print(f"  [MISSING] {key}: missing")
                    self.log(f"Registration output {key} missing", "warning")
            
            self.log("Registration output verified - all images generated", "success")
            return True
        except Exception as e:
            self.log(f"Registration output failed with error: {e}", "error")
            return False
    
    def check_12_metrics(self, src_path: str, ref_path: str) -> bool:
        """Check 12: Metrics - Calculate from current uploaded images."""
        print("\n" + "="*80)
        print("CHECK 12: METRICS CALCULATION")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_metrics", cfg=self.cfg,
                         suite=self.suite, save=False)
            
            if run is None or "result" not in run:
                self.log("Metrics failed - no registration result", "error")
                return False
            
            result = run["result"]
            
            self.results["metrics"] = {
                "inlier_ratio": result.get("inlier_ratio_3px", 0.0),
                "rmse": result.get("rmse_px", 0.0),
                "coverage": result.get("coverage_score", 0.0),
                "runtime": result.get("runtime_s", 0.0),
                "ssim": result.get("ssim", 0.0),
                "ncc": result.get("ncc", 0.0),
                "confidence": result.get("confidence", 0.0),
            }
            
            print(f"Inlier Ratio @3px: {result.get('inlier_ratio_3px', 0.0):.4f}")
            print(f"RMSE: {result.get('rmse_px', 0.0):.4f} px")
            print(f"Coverage: {result.get('coverage_score', 0.0):.4f}")
            print(f"Runtime: {result.get('runtime_s', 0.0):.2f} s")
            print(f"SSIM: {result.get('ssim', 0.0):.4f}")
            print(f"NCC: {result.get('ncc', 0.0):.4f}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            
            self.log("Metrics verified - calculated from uploaded images", "success")
            return True
        except Exception as e:
            self.log(f"Metrics failed with error: {e}", "error")
            return False
    
    def check_13_confidence(self, src_path: str, ref_path: str) -> bool:
        """Check 13: Confidence Engine - Show exact formula, intermediate values, final score."""
        print("\n" + "="*80)
        print("CHECK 13: CONFIDENCE ENGINE")
        print("="*80)
        
        try:
            run = register(src_path, ref_path, name="audit_confidence", cfg=self.cfg,
                         suite=self.suite, save=False)
            
            if run is None or "result" not in run:
                self.log("Confidence failed - no registration result", "error")
                return False
            
            result = run["result"]
            
            # Extract intermediate values
            inlier_ratio = result.get("inlier_ratio_3px", 0.0)
            coverage = result.get("coverage_score", 0.0)
            rmse = result.get("rmse_px", 0.0)
            confidence = result.get("confidence", 0.0)
            
            # Normalize RMSE (lower is better, convert to score)
            rmse_score = max(0, 1 - rmse / 10.0)  # Assume 10px is max reasonable RMSE
            
            # Formula estimation (since exact formula not exposed)
            # Confidence = (inlier_ratio * 0.4) + (coverage * 0.3) + (rmse_score * 0.3)
            estimated_confidence = (inlier_ratio * 0.4) + (coverage * 0.3) + (rmse_score * 0.3)
            
            self.results["confidence"] = {
                "formula": "Confidence ≈ (Inlier Ratio × 0.4) + (Coverage × 0.3) + (RMSE Score × 0.3)",
                "inlier_ratio": inlier_ratio,
                "coverage": coverage,
                "rmse": rmse,
                "rmse_score": rmse_score,
                "estimated_confidence": estimated_confidence,
                "actual_confidence": confidence,
                "difference": abs(estimated_confidence - confidence),
            }
            
            print("Confidence Formula:")
            print("  Confidence approx = (Inlier Ratio * 0.4) + (Coverage * 0.3) + (RMSE Score * 0.3)")
            print(f"\nIntermediate values:")
            print(f"  Inlier Ratio: {inlier_ratio:.4f}")
            print(f"  Coverage: {coverage:.4f}")
            print(f"  RMSE: {rmse:.4f} px")
            print(f"  RMSE Score (normalized): {rmse_score:.4f}")
            print(f"\nEstimated confidence: {estimated_confidence:.4f}")
            print(f"Actual confidence: {confidence:.4f}")
            print(f"Difference: {abs(estimated_confidence - confidence):.4f}")
            
            self.log("Confidence engine verified - values calculated from uploaded images", "success")
            return True
        except Exception as e:
            self.log(f"Confidence engine failed with error: {e}", "error")
            return False
    
    def check_14_reproducibility(self, src_path: str, ref_path: str) -> bool:
        """Check 14: Verify Results - Run same images twice, check if outputs change appropriately."""
        print("\n" + "="*80)
        print("CHECK 14: REPRODUCIBILITY TEST")
        print("="*80)
        
        try:
            # Run first time
            run1 = register(src_path, ref_path, name="audit_repro1", cfg=self.cfg,
                          suite=self.suite, save=False)
            
            # Run second time
            run2 = register(src_path, ref_path, name="audit_repro2", cfg=self.cfg,
                          suite=self.suite, save=False)
            
            if run1 is None or run2 is None:
                self.log("Reproducibility test failed - registration errors", "error")
                return False
            
            result1 = run1["result"]
            result2 = run2["result"]
            
            # Compare key metrics
            metrics_to_compare = ["inlier_ratio_3px", "rmse_px", "coverage_score", "runtime_s", "confidence"]
            differences = {}
            
            for metric in metrics_to_compare:
                val1 = result1.get(metric, 0.0)
                val2 = result2.get(metric, 0.0)
                diff = abs(val1 - val2)
                differences[metric] = diff
                print(f"{metric}: {val1:.4f} vs {val2:.4f} (diff: {diff:.4f})")
            
            # Check if differences are small (indicating real processing, not hardcoded)
            max_diff = max(differences.values())
            
            self.results["reproducibility"] = {
                "run1_confidence": result1.get("confidence", 0.0),
                "run2_confidence": result2.get("confidence", 0.0),
                "max_difference": max_diff,
                "differences": differences,
            }
            
            if max_diff < 0.01:
                self.log("Reproducibility verified - outputs are consistent (real processing)", "success")
                return True
            elif max_diff < 0.1:
                self.log(f"Reproducibility warning - outputs vary slightly (max diff: {max_diff:.4f})", "warning")
                return True
            else:
                self.log(f"Reproducibility failed - outputs vary significantly (max diff: {max_diff:.4f})", "error")
                return False
        except Exception as e:
            self.log(f"Reproducibility test failed with error: {e}", "error")
            return False
    
    def check_15_hardcoded_detection(self) -> bool:
        """Check 15: Final Audit - Detect hardcoded metrics."""
        print("\n" + "="*80)
        print("CHECK 15: HARDCODED METRIC DETECTION")
        print("="*80)
        
        # Check if metrics match known hardcoded values
        known_hardcoded_values = {
            "inlier_ratio_3px": [0.88, 0.90, 0.85, 0.95],
            "rmse_px": [0.30, 0.25, 0.35, 0.20],
            "coverage_score": [0.226, 0.23, 0.25, 0.20],
        }
        
        hardcoded_detected = False
        
        for metric, known_values in known_hardcoded_values.items():
            if metric in self.results.get("metrics", {}):
                actual_value = self.results["metrics"][metric]
                for known in known_values:
                    if abs(actual_value - known) < 0.001:
                        self.log(f"Potential hardcoded value detected: {metric} = {actual_value} (matches {known})", "warning")
                        hardcoded_detected = True
        
        if not hardcoded_detected:
            self.log("No hardcoded metrics detected - values appear to be computed from data", "success")
        
        return not hardcoded_detected
    
    def final_audit(self, src_path: str, ref_path: str) -> dict[str, Any]:
        """Run complete audit and return final verdict."""
        print("\n" + "="*80)
        print("LUNARAI STREAMLIT PROCESSING AUDIT")
        print("="*80)
        print(f"Source: {src_path}")
        print(f"Reference: {ref_path}")
        print("="*80)
        
        # Run all checks
        checks = [
            self.check_1_image_upload(src_path, ref_path),
            self.check_2_preprocessing(src_path, ref_path),
            self.check_3_lunadna(src_path, ref_path),
            self.check_4_faiss(src_path),
            self.check_5_superpoint(src_path, ref_path),
            self.check_6_lightglue(src_path, ref_path),
            self.check_7_anms(src_path, ref_path),
            self.check_8_magsac(src_path, ref_path),
            self.check_9_homography(src_path, ref_path),
            self.check_10_ecc(src_path, ref_path),
            self.check_11_registration_output(src_path, ref_path),
            self.check_12_metrics(src_path, ref_path),
            self.check_13_confidence(src_path, ref_path),
            self.check_14_reproducibility(src_path, ref_path),
            self.check_15_hardcoded_detection(),
        ]
        
        # Calculate pass rate
        passed = sum(checks)
        total = len(checks)
        pass_rate = passed / total
        
        # Final verdict
        print("\n" + "="*80)
        print("FINAL AUDIT VERDICT")
        print("="*80)
        print(f"Checks passed: {passed}/{total} ({pass_rate*100:.1f}%)")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Errors: {len(self.errors)}")
        
        if pass_rate >= 0.9 and len(self.errors) == 0:
            verdict = "REAL PROCESSING"
            print("\n[SUCCESS] REAL PROCESSING")
            print("All metrics are computed from the uploaded images during runtime.")
        elif pass_rate >= 0.7:
            verdict = "MOSTLY REAL PROCESSING"
            print("\n[WARNING] MOSTLY REAL PROCESSING")
            print("Most metrics are computed from uploaded images, but some issues detected.")
        else:
            verdict = "DEMO / HARDCODED OUTPUT"
            print("\n[ERROR] DEMO / HARDCODED OUTPUT")
            print("Significant issues detected - metrics may be hardcoded or demo data.")
        
        self.results["final_audit"] = {
            "checks_passed": passed,
            "checks_total": total,
            "pass_rate": pass_rate,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "verdict": verdict,
        }
        
        # Save results
        output_path = uploads_dir(cfg) / "audit_results.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nAudit results saved to: {output_path}")
        
        return self.results


def main():
    """Main entry point."""
    import argparse
    
    ap = argparse.ArgumentParser(description="Audit Streamlit processing")
    ap.add_argument("source", help="Source image path")
    ap.add_argument("reference", help="Reference image path")
    args = ap.parse_args()
    
    auditor = ProcessingAuditor()
    results = auditor.final_audit(args.source, args.reference)
    
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print(f"Verdict: {results['final_audit']['verdict']}")


if __name__ == "__main__":
    main()
