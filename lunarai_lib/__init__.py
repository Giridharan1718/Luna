"""LunarAI shared library — SIH 26166.

Modules:
- config:       YAML configuration and resolved paths
- metadata:     PDS4/ENVI dataset discovery and metadata parsing
- io_utils:     binary/image IO, JSON/CSV persistence
- db:           SQLite schema per 05_Backend_Schema
- preprocessing: CLAHE / hist-eq / normalization / denoise / resize
- patching:     256x256 stride-128 patch extraction + patch_index.csv
- lunadna:      ResNet18 512-D embedding model + triplet training
- faiss_db:     FAISS IndexFlatIP (numpy-cosine fallback) vector database
- matching:     SuperPoint+LightGlue, ANMS, coverage/uniformity
- geometry:     MAGSAC++, homography, ECC, RMSE/SSIM/NCC, visualizations
- pipeline:     end-to-end orchestrator
- reporting:    CSV/JSON/PDF exports
"""
__version__ = "1.0.0"
