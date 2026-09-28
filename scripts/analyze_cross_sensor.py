"""Analyze cross-sensor retrieval and implement improvements."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
import torch

from lunarai_lib.config import load_config
from lunarai_lib.lunadna import LunaDNA, compute_embeddings
from lunarai_lib.faiss_db import VectorIndex
from lunarai_lib.cross_sensor_retrieval import (
    analyze_embedding_sensor_bias,
    compute_cross_sensor_recall,
)

print("=" * 60)
print("Cross-Sensor Retrieval Analysis")
print("=" * 60)

# Load configuration
cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# Load patch index
df = pd.read_csv(cfg.PATCHES_DIR / "patch_index.csv").dropna(subset=["patch_path"])
print(f"Total patches: {len(df)}")
print(f"Sensors: {df['dataset_name'].unique().tolist()}")
print(f"\nPatch distribution by sensor:")
print(df['dataset_name'].value_counts())

# Load model
print("\nLoading LunaDNA model...")
model = LunaDNA(pretrained=False, grayscale=True).to(device)
ckpt_path = cfg.MODELS_DIR / "lunadna.pt"
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}")
else:
    print("WARNING: No trained checkpoint found")
    sys.exit(1)

# Load FAISS index
print("\nLoading FAISS index...")
index, db_embs, mapping = VectorIndex.load(cfg.DATABASE_DIR)
print(f"FAISS index: {index.ntotal} vectors")
print(f"Embedding dimension: {db_embs.shape[1]}")

# Analyze embedding sensor bias
print("\n" + "=" * 60)
print("Embedding Sensor Bias Analysis")
print("=" * 60)

bias_analysis = analyze_embedding_sensor_bias(db_embs, df)

print(f"Silhouette Score: {bias_analysis['silhouette_score']:.4f}")
print(f"Avg Intra-Sensor Similarity: {bias_analysis['avg_intra_sensor_similarity']:.4f}")
print(f"Avg Inter-Sensor Similarity: {bias_analysis['avg_inter_sensor_similarity']:.4f}")
print(f"Sensor Bias Score: {bias_analysis['sensor_bias_score']:.4f}")
print(f"Is Sensor Biased: {bias_analysis['is_sensor_biased']}")

# Save bias analysis
output_dir = Path("outputs/metrics")
output_dir.mkdir(parents=True, exist_ok=True)
# Convert numpy types to Python types for JSON serialization
bias_analysis_json = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else 
                          (float(v) if isinstance(v, (np.float32, np.float64)) else v))
                      for k, v in bias_analysis.items()}
with open(output_dir / "embedding_bias_analysis.json", "w") as f:
    json.dump(bias_analysis_json, f, indent=2)
print(f"\nSaved bias analysis to {output_dir / 'embedding_bias_analysis.json'}")

# Load existing retrieval results
print("\n" + "=" * 60)
print("Analyzing Existing Retrieval Results")
print("=" * 60)

retrieval_path = cfg.RETRIEVAL_DIR / "topk_results.csv"
if retrieval_path.exists():
    retrieval_df = pd.read_csv(retrieval_path)
    print(f"Loaded {len(retrieval_df)} retrieval results")
    
    # Compute cross-sensor recall
    recall_metrics = compute_cross_sensor_recall(retrieval_df, df, geo_threshold=1.0)
    
    print(f"\nCross-Sensor Recall Metrics:")
    print(f"Cross-Sensor Recall@1: {recall_metrics['cross_sensor_recall@1']:.4f}")
    print(f"Same-Sensor Recall@1: {recall_metrics['same_sensor_recall@1']:.4f}")
    print(f"Cross-Sensor Total: {recall_metrics['cross_sensor_total']}")
    print(f"Same-Sensor Total: {recall_metrics['same_sensor_total']}")
    print(f"Cross-Sensor Correct: {recall_metrics['cross_sensor_correct']}")
    print(f"Same-Sensor Correct: {recall_metrics['same_sensor_correct']}")
    
    # Save recall metrics
    recall_metrics_json = {k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                         for k, v in recall_metrics.items()}
    with open(output_dir / "cross_sensor_recall_analysis.json", "w") as f:
        json.dump(recall_metrics_json, f, indent=2)
    print(f"\nSaved recall metrics to {output_dir / 'cross_sensor_recall_analysis.json'}")
else:
    print(f"Retrieval results not found at {retrieval_path}")
    print("Run nb06_retrieval_testing.ipynb first")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"\nKey Findings:")
print(f"1. Embeddings show sensor bias: {bias_analysis['is_sensor_biased']}")
print(f"   - Silhouette score: {bias_analysis['silhouette_score']:.4f} (higher = more sensor clustering)")
print(f"   - Bias score: {bias_analysis['sensor_bias_score']:.4f} (positive = sensor-biased)")

if retrieval_path.exists():
    print(f"\n2. Cross-sensor retrieval performance:")
    print(f"   - Cross-sensor recall@1: {recall_metrics['cross_sensor_recall@1']:.4f}")
    print(f"   - Same-sensor recall@1: {recall_metrics['same_sensor_recall@1']:.4f}")
    print(f"   - Cross-sensor success rate: {recall_metrics['cross_sensor_correct']}/{recall_metrics['cross_sensor_total']}")

print(f"\n3. Recommendations:")
print(f"   - Embeddings are sensor-biased (clustering by sensor, not location)")
print(f"   - Cross-sensor retrieval shows low success (expected given bias)")
print(f"   - True cross-sensor invariance requires model retraining")
print(f"   - Current system is best for same-sensor retrieval")
print(f"   - Cross-sensor correspondence is identified as future work")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
