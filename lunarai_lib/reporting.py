"""Report generation: PDF via reportlab, plus CSV/JSON export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io_utils import save_json, save_rows_csv


def export_metrics_csv(metrics: dict[str, Any], path: Path) -> Path:
    rows = [{"metric": k, "value": v} for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool))]
    return save_rows_csv(rows, path)


def export_results_json(obj: dict[str, Any], path: Path) -> Path:
    clean = {}
    for k, v in obj.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            clean[k] = v
    return save_json(clean, path)


def build_pdf_report(path: Path, *, title: str, subtitle: str,
                     metrics: dict[str, Any], image_paths: list[Path] | None = None,
                     summary_lines: list[str] | None = None) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    c.setFillColorRGB(0.043, 0.063, 0.125)
    c.rect(0, height - 3.2 * cm, width, 3.2 * cm, fill=1, stroke=0)
    c.setFillColorRGB(0.9, 0.95, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(1.2 * cm, height - 2.1 * cm, title)
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.55, 0.75, 0.95)
    c.drawString(1.2 * cm, height - 2.8 * cm, subtitle)

    y = height - 4.2 * cm
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.2 * cm, y, "Evaluation Metrics")
    y -= 0.7 * cm
    c.setFont("Courier", 10)
    for k, v in metrics.items():
        if isinstance(v, float):
            text = f"{k:<28} {v:.4f}"
        else:
            text = f"{k:<28} {v}"
        c.drawString(1.4 * cm, y, text)
        y -= 0.55 * cm
        if y < 6 * cm:
            c.showPage()
            y = height - 2 * cm
            c.setFont("Courier", 10)

    if summary_lines:
        y -= 0.4 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1.2 * cm, y, "Summary")
        y -= 0.7 * cm
        c.setFont("Helvetica", 9.5)
        for line in summary_lines:
            c.drawString(1.4 * cm, y, line[:110])
            y -= 0.5 * cm

    if image_paths:
        for img_path in image_paths:
            if not Path(img_path).exists():
                continue
            if y < 8 * cm:
                c.showPage()
                y = height - 2 * cm
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            max_w, max_h = width - 3 * cm, 7.5 * cm
            scale = min(max_w / iw, max_h / ih)
            c.drawImage(str(img_path), 1.5 * cm, y - ih * scale,
                        iw * scale, ih * scale)
            y -= ih * scale + 0.6 * cm

    c.save()
    return path
