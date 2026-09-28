"""Cross-sensor retrieval improvements for LunarAI.

This module implements sensor-aware retrieval strategies to improve cross-sensor
correspondence without requiring full model retraining.

Key improvements:
1. Sensor-aware retrieval: Boost cross-sensor matches
2. Geographic filtering: Prioritize geographically close candidates
3. Ensemble retrieval: Combine multiple retrieval strategies
4. Adaptive thresholds: Sensor-specific similarity thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SensorAwareRetrieval:
    """Implements sensor-aware retrieval to improve cross-sensor correspondence."""
    
    def __init__(self, patch_index: pd.DataFrame, embedding_dim: int = 512):
        """
        Initialize sensor-aware retrieval.
        
        Args:
            patch_index: DataFrame with patch metadata (patch_id, sensor, lat, lon, etc.)
            embedding_dim: Dimension of embeddings
        """
        self.patch_index = patch_index
        self.embedding_dim = embedding_dim
        
        # Build sensor lookup
        self.sensor_to_patches: Dict[str, List[str]] = {}
        for sensor in patch_index['dataset_name'].unique():
            self.sensor_to_patches[sensor] = patch_index[
                patch_index['dataset_name'] == sensor
            ]['patch_id'].tolist()
        
        # Build geographic index (simple grid-based)
        self.geo_grid = self._build_geo_grid()
        
        # Sensor-specific similarity thresholds (empirically determined)
        self.sensor_thresholds = {
            'OHRC': 0.85,
            'TMC-2': 0.80,
            'IIRS': 0.75,
            'LRO NAC': 0.82,
            'KAGUYA': 0.78,
        }
    
    def _build_geo_grid(self, grid_size: float = 1.0) -> Dict[Tuple[int, int], List[str]]:
        """Build a geographic grid for fast proximity queries."""
        grid: Dict[Tuple[int, int], List[str]] = {}
        
        for _, row in self.patch_index.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
            
            grid_x = int(row['latitude'] / grid_size)
            grid_y = int(row['longitude'] / grid_size)
            
            key = (grid_x, grid_y)
            if key not in grid:
                grid[key] = []
            grid[key].append(row['patch_id'])
        
        return grid
    
    def find_nearby_patches(self, query_lat: float, query_lon: float, 
                          radius: float = 2.0) -> List[str]:
        """Find patches within geographic radius."""
        nearby = []
        
        for (grid_x, grid_y), patch_ids in self.geo_grid.items():
            grid_lat = grid_x * 1.0  # grid_size
            grid_lon = grid_y * 1.0
            
            dist = np.sqrt((grid_lat - query_lat)**2 + (grid_lon - query_lon)**2)
            if dist <= radius:
                nearby.extend(patch_ids)
        
        return nearby
    
    def boost_cross_sensor_scores(self, scores: np.ndarray, 
                                   query_sensor: str,
                                   candidate_sensors: List[str],
                                   boost_factor: float = 0.1) -> np.ndarray:
        """
        Boost scores for cross-sensor candidates.
        
        Args:
            scores: Original similarity scores
            query_sensor: Sensor of query patch
            candidate_sensors: List of candidate sensors
            boost_factor: Factor to boost cross-sensor scores
        
        Returns:
            Boosted scores
        """
        boosted = scores.copy()
        
        for i, cand_sensor in enumerate(candidate_sensors):
            if cand_sensor != query_sensor:
                # Boost cross-sensor matches
                boosted[i] = min(1.0, boosted[i] + boost_factor)
        
        return boosted
    
    def filter_by_geography(self, candidates: List[str], 
                          query_lat: float, query_lon: float,
                          max_distance: float = 3.0) -> List[str]:
        """Filter candidates by geographic proximity."""
        filtered = []
        
        for patch_id in candidates:
            patch_row = self.patch_index[self.patch_index['patch_id'] == patch_id]
            if len(patch_row) == 0:
                continue
            
            patch_row = patch_row.iloc[0]
            if pd.isna(patch_row['latitude']) or pd.isna(patch_row['longitude']):
                continue
            
            dist = np.sqrt(
                (patch_row['latitude'] - query_lat)**2 + 
                (patch_row['longitude'] - query_lon)**2
            )
            
            if dist <= max_distance:
                filtered.append(patch_id)
        
        return filtered
    
    def adaptive_threshold_retrieval(self, query_sensor: str,
                                     scores: np.ndarray) -> np.ndarray:
        """Apply sensor-specific similarity thresholds."""
        threshold = self.sensor_thresholds.get(query_sensor, 0.80)
        
        # Filter candidates below threshold
        mask = scores >= threshold
        return mask
    
    def ensemble_retrieval(self, query_embedding: np.ndarray,
                          index, mapping: List[str],
                          query_patch_id: str,
                          top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble retrieval combining multiple strategies.
        
        Args:
            query_embedding: Query patch embedding
            index: FAISS index
            mapping: Patch ID mapping
            query_patch_id: Query patch ID
            top_k: Number of results to return
        
        Returns:
            (scores, indices) of retrieved candidates
        """
        # Get query metadata
        query_row = self.patch_index[self.patch_index['patch_id'] == query_patch_id]
        if len(query_row) == 0:
            # Fallback to standard retrieval
            return index.search(query_embedding, top_k)
        
        query_row = query_row.iloc[0]
        query_sensor = query_row['dataset_name']
        query_lat = query_row.get('latitude', np.nan)
        query_lon = query_row.get('longitude', np.nan)
        
        # Standard FAISS retrieval
        scores, indices = index.search(query_embedding, top_k * 2)  # Get more candidates
        
        # Get candidate sensors
        candidate_sensors = []
        candidate_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(mapping):
                patch_id = Path(mapping[idx]).name.replace('.png', '')
                candidate_ids.append(patch_id)
                cand_row = self.patch_index[self.patch_index['patch_id'] == patch_id]
                if len(cand_row) > 0:
                    candidate_sensors.append(cand_row.iloc[0]['dataset_name'])
                else:
                    candidate_sensors.append('UNKNOWN')
        
        # Boost cross-sensor scores
        boosted_scores = self.boost_cross_sensor_scores(
            scores[0], query_sensor, candidate_sensors, boost_factor=0.05
        )
        
        # Geographic filtering if coordinates available
        if not pd.isna(query_lat) and not pd.isna(query_lon):
            geo_filtered = self.filter_by_geography(
                candidate_ids, query_lat, query_lon, max_distance=3.0
            )
            # Prioritize geo-filtered candidates
            geo_boost = np.array([1.1 if cid in geo_filtered else 1.0 
                                 for cid in candidate_ids])
            boosted_scores = boosted_scores * geo_boost
        
        # Re-rank by boosted scores
        sorted_indices = np.argsort(-boosted_scores)
        top_indices = sorted_indices[:top_k]
        
        final_scores = boosted_scores[top_indices]
        final_indices = indices[0][top_indices]
        
        return final_scores.reshape(1, -1), final_indices.reshape(1, -1)


def compute_cross_sensor_recall(retrieval_results: pd.DataFrame,
                               patch_index: pd.DataFrame,
                               geo_threshold: float = 1.0) -> Dict[str, float]:
    """
    Compute cross-sensor recall metrics.
    
    Args:
        retrieval_results: DataFrame with retrieval results
        patch_index: DataFrame with patch metadata
        geo_threshold: Geographic distance threshold for correct match
    
    Returns:
        Dictionary with recall metrics
    """
    # Build lookup
    meta = patch_index.set_index('patch_id')
    
    # Count cross-sensor correct retrievals
    cross_sensor_correct = 0
    cross_sensor_total = 0
    same_sensor_correct = 0
    same_sensor_total = 0
    
    for query_id, group in retrieval_results.groupby('query_patch'):
        if query_id not in meta.index:
            continue
        
        query_row = meta.loc[query_id]
        query_sensor = query_row['dataset_name']
        query_lat = query_row.get('latitude', np.nan)
        query_lon = query_row.get('longitude', np.nan)
        
        # Check top-1 retrieval
        top1 = group[group['rank'] == 1]
        if len(top1) == 0:
            continue
        
        top1_row = top1.iloc[0]
        retrieved_id = top1_row['retrieved_patch'].replace('.png', '')
        
        if retrieved_id not in meta.index:
            continue
        
        retrieved_row = meta.loc[retrieved_id]
        retrieved_sensor = retrieved_row['dataset_name']
        
        # Check if geographically close
        if not pd.isna(query_lat) and not pd.isna(query_lon):
            if not pd.isna(retrieved_row['latitude']) and not pd.isna(retrieved_row['longitude']):
                geo_dist = np.sqrt(
                    (query_lat - retrieved_row['latitude'])**2 +
                    (query_lon - retrieved_row['longitude'])**2
                )
                is_geo_match = geo_dist < geo_threshold
            else:
                is_geo_match = False
        else:
            is_geo_match = False
        
        # Count cross-sensor vs same-sensor
        if retrieved_sensor != query_sensor:
            cross_sensor_total += 1
            if is_geo_match:
                cross_sensor_correct += 1
        else:
            same_sensor_total += 1
            if is_geo_match:
                same_sensor_correct += 1
    
    # Compute metrics
    cross_sensor_recall = (
        cross_sensor_correct / cross_sensor_total if cross_sensor_total > 0 else 0.0
    )
    same_sensor_recall = (
        same_sensor_correct / same_sensor_total if same_sensor_total > 0 else 0.0
    )
    
    return {
        'cross_sensor_recall@1': float(cross_sensor_recall),
        'same_sensor_recall@1': float(same_sensor_recall),
        'cross_sensor_total': int(cross_sensor_total),
        'same_sensor_total': int(same_sensor_total),
        'cross_sensor_correct': int(cross_sensor_correct),
        'same_sensor_correct': int(same_sensor_correct),
    }


def analyze_embedding_sensor_bias(embeddings: np.ndarray,
                                  patch_index: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze if embeddings are sensor-biased.
    
    Args:
        embeddings: Embedding matrix (n_patches, dim)
        patch_index: DataFrame with patch metadata
    
    Returns:
        Dictionary with bias analysis metrics
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    # Get sensor labels
    patch_id_to_sensor = dict(zip(
        patch_index['patch_id'],
        patch_index['dataset_name']
    ))
    
    # Build sensor list in same order as embeddings
    sensors = []
    for patch_id in patch_index['patch_id']:
        sensors.append(patch_id_to_sensor.get(patch_id, 'UNKNOWN'))
    
    # PCA visualization
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    
    # Compute silhouette score (measures clustering by sensor)
    if len(set(sensors)) > 1:
        silhouette = silhouette_score(pca_embeddings, sensors)
    else:
        silhouette = 0.0
    
    # Compute intra-sensor vs inter-sensor similarity
    sensor_to_indices = {}
    for i, sensor in enumerate(sensors):
        if sensor not in sensor_to_indices:
            sensor_to_indices[sensor] = []
        sensor_to_indices[sensor].append(i)
    
    intra_similarities = []
    inter_similarities = []
    
    for sensor, indices in sensor_to_indices.items():
        if len(indices) < 2:
            continue
        
        # Intra-sensor similarity
        sensor_embeddings = embeddings[indices]
        intra_sim = np.mean(
            np.dot(sensor_embeddings, sensor_embeddings.T)
        )
        intra_similarities.append(intra_sim)
        
        # Inter-sensor similarity
        for other_sensor, other_indices in sensor_to_indices.items():
            if other_sensor == sensor:
                continue
            
            other_embeddings = embeddings[other_indices]
            inter_sim = np.mean(
                np.dot(sensor_embeddings, other_embeddings.T)
            )
            inter_similarities.append(inter_sim)
    
    avg_intra = np.mean(intra_similarities) if intra_similarities else 0.0
    avg_inter = np.mean(inter_similarities) if inter_similarities else 0.0
    
    # Bias score: higher = more sensor-biased
    bias_score = avg_intra - avg_inter
    
    return {
        'silhouette_score': float(silhouette),
        'avg_intra_sensor_similarity': float(avg_intra),
        'avg_inter_sensor_similarity': float(avg_inter),
        'sensor_bias_score': float(bias_score),
        'is_sensor_biased': bias_score > 0.1,  # Threshold for bias
    }
