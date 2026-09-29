"""
Flask Backend for LunarAI Dashboard

This provides the backend API for the HTML dashboard to make buttons functional.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lunarai_lib.config import load_config
from lunarai_lib.live import register, retrieve, session, uploads_dir
from lunarai_lib.validation import load_gray

app = Flask(__name__)
CORS(app)

cfg = load_config()
suite = session(cfg)


@app.route('/')
def index():
    """Serve the HTML dashboard."""
    return send_from_directory(str(Path(__file__).parent), 'index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    info = {
        "status": "operational",
        "device": getattr(suite, "device", "cpu"),
        "matcher": getattr(suite, "matcher_backend", None),
        "checkpoint_epoch": getattr(suite, "ckpt_epoch", None),
        "ntotal": int(getattr(suite.index, "ntotal", 0) or 0),
    }
    return jsonify(info)


@app.route('/api/kpi')
def kpi():
    """KPIs computed live from artifacts - no hard-coded numbers."""
    try:
        import pandas as pd

        from lunarai_lib import appdata
        h = appdata.headline(cfg)
        sub = appdata.table("subpixel_accuracy_report.csv")
        modal = appdata.table("multi_modal_validation.csv")
        return jsonify({
            "images_processed": h.get("images"),
            "patches_indexed": h.get("patches"),
            "checkpoint_epoch": h.get("epoch"),
            "matcher": h.get("matcher"),
            "registered_pairs": int(len(sub)) if sub is not None else 0,
            "median_rmse_px": (float(pd.to_numeric(sub.rmse_after_ecc_px,
                                                   errors="coerce").median())
                               if sub is not None else None),
            "share_err_lt_1px": h.get("share_err_lt_1px"),
            "inlier_ratio_3px_mean": (float(pd.to_numeric(modal.inlier_ratio_3px,
                                                          errors="coerce").mean())
                                      if modal is not None else None),
            "coverage_mean": (float(pd.to_numeric(modal.coverage_score,
                                                  errors="coerce").mean())
                              if modal is not None else None),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/datasets')
def datasets():
    """Get dataset information."""
    try:
        from lunarai_lib import appdata
        idx = appdata.patch_index(cfg)
        if idx is not None:
            datasets = idx.groupby("dataset_name").agg({
                "patch_path": "count",
            }).rename(columns={"patch_path": "image_count"}).to_dict("index")
            # Add static sensor information
            sensor_info = {
                "OHRC": {"resolution": "0.25 m/pixel", "sensor_type": "Optical"},
                "TMC-2": {"resolution": "2.0 m/pixel", "sensor_type": "Optical"},
                "IIRS": {"resolution": "N/A", "sensor_type": "Spectroscopic"},
                "LRO NAC": {"resolution": "0.5 m/pixel", "sensor_type": "Optical"},
                "KAGUYA": {"resolution": "10 m/pixel", "sensor_type": "Optical"},
            }
            for sensor in datasets:
                if sensor in sensor_info:
                    datasets[sensor].update(sensor_info[sensor])
                else:
                    datasets[sensor]["resolution"] = "N/A"
                    datasets[sensor]["sensor_type"] = "Unknown"
            return jsonify(datasets)
        else:
            return jsonify({"error": "No dataset index found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/experiments')
def experiments():
    """Get experiment table data."""
    try:
        report_path = cfg.OUTPUTS_ROOT / "reports" / "multi_modal_validation.csv"
        if report_path.exists():
            import pandas as pd
            df = pd.read_csv(report_path)
            return jsonify(df.to_dict(orient="records"))
        else:
            return jsonify({"error": "Experiment table not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register_pair():
    """Run registration on a pair of images."""
    try:
        data = request.json
        src_path = data.get('source')
        ref_path = data.get('reference')
        
        if not src_path or not ref_path:
            return jsonify({"error": "Source and reference paths required"}), 400
        
        # Run registration
        run = register(src_path, ref_path, name="api_pair", cfg=cfg, suite=suite, save=True)
        
        if run is None or "result" not in run:
            return jsonify({"error": "Registration failed"}), 500
        
        result = run["result"]
        
        # Convert numpy arrays to lists for JSON serialization
        homography = run.get("H")
        if homography is not None:
            homography = homography.tolist() if hasattr(homography, 'tolist') else homography
        
        return jsonify({
            "status": result.get("status"),
            "inlier_ratio": result.get("inlier_ratio_3px"),
            "rmse": result.get("rmse_px"),
            "coverage": result.get("coverage_score"),
            "runtime": result.get("runtime_s"),
            "ssim": result.get("ssim"),
            "ncc": result.get("ncc"),
            "confidence": result.get("confidence"),
            "homography": homography,
            "images": {k: str(v) for k, v in (run.get("images") or {}).items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/retrieve', methods=['POST'])
def retrieve_patches():
    """Run retrieval on a query image."""
    try:
        data = request.json
        query_path = data.get('query')
        k = data.get('k', 5)
        
        if not query_path:
            return jsonify({"error": "Query path required"}), 400
        
        # Run retrieval
        results = retrieve(query_path, k=k, cfg=cfg, suite=suite)
        
        if results is None:
            return jsonify({"error": "Retrieval failed"}), 500
        
        return jsonify(results.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        uploads = uploads_dir(cfg)
        filename = file.filename
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        file_path = uploads / safe_filename
        file.save(str(file_path))
        
        return jsonify({
            "status": "success",
            "filename": safe_filename,
            "path": str(file_path),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/<format>')
def export_report(format):
    """Export report in specified format."""
    try:
        if format == 'csv':
            report_path = cfg.OUTPUTS_ROOT / "reports" / "multi_modal_validation.csv"
            if report_path.exists():
                return send_from_directory(str(report_path.parent), report_path.name, mimetype='text/csv')
            else:
                return jsonify({"error": "CSV not found"}), 404
        elif format == 'json':
            report_path = cfg.OUTPUTS_ROOT / "metrics" / "evaluation_report.json"
            if report_path.exists():
                return send_from_directory(str(report_path.parent), report_path.name, mimetype='application/json')
            else:
                return jsonify({"error": "JSON not found"}), 404
        else:
            return jsonify({"error": "Unsupported format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting LunarAI Backend Server...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Uploads directory: {uploads_dir(cfg)}")
    print("Server running at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
