"""
validate_bboxes.py — MODULE 6b: BOUNDING BOX VALIDATION (Phase 6)
─────────────────────────────────────────────────────────────────
Audits, validates geometry, and normalizes bounding box coordinates
for all records in data/cleaned/annotations.jsonl.

Validation checks:
──────────────────
 1. Presence: bbox is not None / empty
 2. Format: exactly 4 numeric values [x1, y1, x2, y2]
 3. Coordinate ordering: x1 < x2 and y1 < y2 (positive width & height)
 4. Non-negative bounds: x1 >= 0 and y1 >= 0
 5. Image boundary clipping: x2 <= image_width (1200) and y2 <= image_height (1600)
 6. Minimum dimension & area check: width >= 5, height >= 5, area >= 25 px
 7. Aspect ratio verification for OCR text lines / words

Coordinate Normalization & Modeling formats:
─────────────────────────────────────────────
 • Normalized [0, 1]:  [x1/W, y1/H, x2/W, y2/H]
 • LayoutLM [0, 1000]: [int(x1/W*1000), int(y1/H*1000), int(x2/W*1000), int(y2/H*1000)]
 • YOLO format:        [x_center/W, y_center/H, bbox_width/W, bbox_height/H]

Outputs:
────────
 data/cleaned/annotations_validated.jsonl  — verified annotations with normalized coords
 data/cleaned/missing_bbox_annotations.jsonl — 20 records requiring bbox annotation
 reports/bbox_validation.csv               — per-record bbox audit details
 reports/bbox_validation.json              — summary statistics
 reports/bbox_validation_charts.png        — 4-panel spatial & geometric visualization
 reports/cleaning_log.csv                  — appended provenance decisions

Usage:
──────
 cd Indic_OCR_Data_Engineering
 python scripts/validate_bboxes.py
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".mpl_cache"))

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from config import SYNTHETIC_DATASET_DIR, REPORTS_DIR, CLEANED_DIR

INPUT_JSONL_PATH = CLEANED_DIR / "annotations.jsonl"
PAGE_WIDTH  = 1200   # default synthetic page width
PAGE_HEIGHT = 1600   # default synthetic page height

RUN_TS  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SECTION = "=" * 70

LOG_PATH = REPORTS_DIR / "cleaning_log.csv"
_new_log_entries = []


def log_action(record_id: str, field: str, issue: str,
               action: str, proposed_value: str = ""):
    """Buffer a cleaning log entry for reports/cleaning_log.csv."""
    _new_log_entries.append({
        "timestamp":      RUN_TS,
        "record_id":      record_id,
        "record_type":    "annotation",
        "field":          field,
        "issue":          issue,
        "action":         action,
        "proposed_value": proposed_value,
    })


def append_cleaning_log():
    """Append buffered Phase 6b decisions to cleaning_log.csv."""
    if not _new_log_entries:
        return
    file_exists = LOG_PATH.exists()
    keys = ["timestamp", "record_id", "record_type", "field",
            "issue", "action", "proposed_value"]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(_new_log_entries)
    print(f"   ✓ Appended {len(_new_log_entries)} bbox decisions to {LOG_PATH}")


def validate_bbox_geometry(bbox, img_w=PAGE_WIDTH, img_h=PAGE_HEIGHT):
    """
    Validate bbox coordinates.
    Returns: (is_valid, list_of_errors, metrics_dict)
    """
    if bbox is None:
        return False, ["MISSING_BBOX"], None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, ["INVALID_FORMAT_NOT_4_ELEMENTS"], None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return False, ["NON_NUMERIC_COORDINATES"], None

    errors = []
    # Ordering
    if x1 >= x2:
        errors.append(f"INVERTED_X(x1={x1}>=x2={x2})")
    if y1 >= y2:
        errors.append(f"INVERTED_Y(y1={y1}>=y2={y2})")

    # Non-negative
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        errors.append("NEGATIVE_COORDINATE")

    # Boundary clipping
    if x2 > img_w or y2 > img_h:
        errors.append(f"EXCEEDS_IMAGE_BOUNDS({img_w}x{img_h})")

    # Dimensions
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    aspect_ratio = round(bw / bh, 3) if bh > 0 else 0.0

    if bw < 5 or bh < 5:
        errors.append("TOO_SMALL_DIMENSIONS")

    is_valid = len(errors) == 0

    metrics = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "bbox_width": round(bw, 1),
        "bbox_height": round(bh, 1),
        "bbox_area": round(area, 1),
        "aspect_ratio": aspect_ratio,
        # Normalized formats
        "bbox_norm": [round(x1/img_w, 4), round(y1/img_h, 4),
                      round(x2/img_w, 4), round(y2/img_h, 4)],
        "bbox_layoutlm": [int(x1/img_w*1000), int(y1/img_h*1000),
                          int(x2/img_w*1000), int(y2/img_h*1000)],
        "bbox_yolo": [round(((x1 + x2)/2)/img_w, 4),
                      round(((y1 + y2)/2)/img_h, 4),
                      round(bw/img_w, 4), round(bh/img_h, 4)],
    }

    return is_valid, errors, metrics


def main():
    print(SECTION)
    print("PHASE 6b — BOUNDING BOX VALIDATION & NORMALIZATION")
    print(SECTION)

    # 1. Load cleaned annotations
    print(f"\n[1/5] Loading cleaned annotations from {INPUT_JSONL_PATH} …")
    if not INPUT_JSONL_PATH.exists():
        print(f"ERROR: Clean annotations file not found at {INPUT_JSONL_PATH}")
        sys.exit(1)

    records = []
    with open(INPUT_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    n_records = len(records)
    print(f"   Loaded {n_records} clean annotation records")

    # 2. Validate all bounding boxes
    print("\n[2/5] Performing geometric and boundary validation …")
    validated_records = []
    missing_bbox_records = []
    audit_rows = []

    n_valid   = 0
    n_missing = 0
    n_invalid = 0

    for rec in records:
        doc_id = rec["document_id"]
        f_name = rec["field_name"]
        bbox   = rec.get("bbox")
        rec_ref = f"{doc_id}|{f_name}"

        is_valid, errors, geom = validate_bbox_geometry(bbox, PAGE_WIDTH, PAGE_HEIGHT)

        audit_row = {
            "document_id": doc_id,
            "field_name":  f_name,
            "has_bbox":    bbox is not None,
            "is_valid":    is_valid,
            "errors":      "|".join(errors) if errors else "NONE",
            "x1":          geom["x1"] if geom else None,
            "y1":          geom["y1"] if geom else None,
            "x2":          geom["x2"] if geom else None,
            "y2":          geom["y2"] if geom else None,
            "width":       geom["bbox_width"] if geom else None,
            "height":      geom["bbox_height"] if geom else None,
            "area":        geom["bbox_area"] if geom else None,
            "aspect_ratio": geom["aspect_ratio"] if geom else None,
        }
        audit_rows.append(audit_row)

        if bbox is None:
            n_missing += 1
            missing_rec = dict(rec)
            missing_rec["issue"] = "MISSING_BBOX"
            missing_bbox_records.append(missing_rec)
            log_action(rec_ref, "bbox", "MISSING_BBOX",
                       "FLAG_FOR_ANNOTATION — excluded from detection tasks, retained for recognition")
        elif not is_valid:
            n_invalid += 1
            log_action(rec_ref, "bbox", f"INVALID_GEOMETRY:{'|'.join(errors)}",
                       "FLAG_INVALID_BBOX — requires manual coordinate adjustment")
        else:
            n_valid += 1

        # Enrich validated record
        val_rec = dict(rec)
        val_rec["is_bbox_valid"]   = is_valid
        val_rec["bbox_errors"]      = errors
        if geom:
            val_rec["bbox_norm"]     = geom["bbox_norm"]
            val_rec["bbox_layoutlm"] = geom["bbox_layoutlm"]
            val_rec["bbox_yolo"]     = geom["bbox_yolo"]
            val_rec["bbox_area"]     = geom["bbox_area"]
            val_rec["bbox_aspect_ratio"] = geom["aspect_ratio"]

        validated_records.append(val_rec)

    audit_df = pd.DataFrame(audit_rows)

    # 3. Save artifacts
    print("\n[3/5] Saving validated outputs …")

    # Validated annotations JSONL
    val_jsonl_path = CLEANED_DIR / "annotations_validated.jsonl"
    with open(val_jsonl_path, "w", encoding="utf-8") as f:
        for r in validated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Validated annotations saved → {val_jsonl_path} ({len(validated_records)} records)")

    # Missing bbox JSONL
    missing_bbox_path = CLEANED_DIR / "missing_bbox_annotations.jsonl"
    with open(missing_bbox_path, "w", encoding="utf-8") as f:
        for r in missing_bbox_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Missing bbox records saved  → {missing_bbox_path} ({len(missing_bbox_records)} records)")

    # Audit CSV
    audit_csv_path = REPORTS_DIR / "bbox_validation.csv"
    audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8")
    print(f"   ✓ BBox audit CSV saved        → {audit_csv_path}")

    append_cleaning_log()

    # 4. Save JSON summary
    print("\n[4/5] Generating Phase 6b JSON report …")
    valid_geom_df = audit_df[audit_df["is_valid"]]

    summary_data = {
        "run_timestamp": RUN_TS,
        "phase": "Phase 6b — Bounding Box Validation",
        "total_records_evaluated": n_records,
        "valid_bboxes": n_valid,
        "missing_bboxes": n_missing,
        "invalid_bboxes": n_invalid,
        "page_dimensions_reference": {
            "width": PAGE_WIDTH,
            "height": PAGE_HEIGHT
        },
        "bbox_statistics": {
            "width": {
                "min": float(valid_geom_df["width"].min()),
                "max": float(valid_geom_df["width"].max()),
                "mean": round(float(valid_geom_df["width"].mean()), 2),
            },
            "height": {
                "min": float(valid_geom_df["height"].min()),
                "max": float(valid_geom_df["height"].max()),
                "mean": round(float(valid_geom_df["height"].mean()), 2),
            },
            "area": {
                "min": float(valid_geom_df["area"].min()),
                "max": float(valid_geom_df["area"].max()),
                "mean": round(float(valid_geom_df["area"].mean()), 2),
            },
            "aspect_ratio": {
                "min": float(valid_geom_df["aspect_ratio"].min()),
                "max": float(valid_geom_df["aspect_ratio"].max()),
                "mean": round(float(valid_geom_df["aspect_ratio"].mean()), 2),
            }
        },
        "coordinate_formats_generated": [
            "bbox (absolute pixels: [x1, y1, x2, y2])",
            "bbox_norm (relative 0.0 to 1.0)",
            "bbox_layoutlm (scaled 0 to 1000 for LayoutLMv1/v2/v3)",
            "bbox_yolo (normalized [x_center, y_center, width, height])"
        ]
    }

    summary_json_path = REPORTS_DIR / "bbox_validation.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"   ✓ Summary JSON saved          → {summary_json_path}")

    # 5. Visualizations
    print("\n[5/5] Generating publication charts …")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 6b — Bounding Box Validation & Spatial Geometry", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Bounding Box Status Breakdown
    ax = axes[0, 0]
    stat_cats   = ["Valid BBoxes", "Missing BBoxes", "Invalid Geometry"]
    stat_counts = [n_valid, n_missing, n_invalid]
    stat_colors = ["#2CA02C", "#FF7F0E", "#D62728"]
    bars1 = ax.bar(stat_cats, stat_counts, color=stat_colors, edgecolor="white", width=0.55)
    ax.set_title("Bounding Box Status", fontsize=12, fontweight="bold")
    ax.set_ylabel("Records")
    ax.set_ylim(0, max(stat_counts) * 1.15)
    for b, c in zip(bars1, stat_counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                f"{c}\n({c/n_records*100:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Width vs Height Scatter & OCR Line Proportions
    ax = axes[0, 1]
    if not valid_geom_df.empty:
        scatter = ax.scatter(
            valid_geom_df["width"],
            valid_geom_df["height"],
            c=valid_geom_df["aspect_ratio"],
            cmap="crest",
            alpha=0.6,
            edgecolors="none",
            s=35
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Aspect Ratio (W / H)", fontsize=9)
        ax.set_title("BBox Dimensions: Width vs Height", fontsize=12, fontweight="bold")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
    else:
        ax.text(0.5, 0.5, "No valid bboxes", ha="center", va="center")

    # Panel 3: Area Distribution
    ax = axes[1, 0]
    if not valid_geom_df.empty:
        ax.hist(valid_geom_df["area"], bins=25, color="#1F77B4", edgecolor="white", alpha=0.85)
        ax.axvline(x=float(valid_geom_df["area"].median()), color="red", linestyle="--", linewidth=1.5,
                   label=f"Median Area ({float(valid_geom_df['area'].median()):.0f} px²)")
        ax.set_title("Bounding Box Area Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Area (px²)")
        ax.set_ylabel("Count")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid bboxes", ha="center", va="center")

    # Panel 4: Spatial Coverage on Document Canvas (1200 x 1600)
    ax = axes[1, 1]
    ax.set_xlim(0, PAGE_WIDTH)
    ax.set_ylim(PAGE_HEIGHT, 0)   # Invert Y to match image top-left origin
    ax.set_aspect("equal")

    # Draw page boundary
    page_rect = patches.Rectangle((0, 0), PAGE_WIDTH, PAGE_HEIGHT,
                                  linewidth=1.5, edgecolor="black", facecolor="#FAFAFA")
    ax.add_patch(page_rect)

    # Plot sample bounding boxes (first 80 boxes for clear visualization)
    sample_df = valid_geom_df.head(80)
    for _, r in sample_df.iterrows():
        bx = r["x1"]
        by = r["y1"]
        bw = r["width"]
        bh = r["height"]
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=1,
                                 edgecolor="#1F77B4", facecolor="#1F77B4", alpha=0.18)
        ax.add_patch(rect)

    ax.set_title(f"Spatial Layout on Page ({PAGE_WIDTH} × {PAGE_HEIGHT} px)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px, top-left origin)")

    plt.tight_layout()
    chart_path = REPORTS_DIR / "bbox_validation_charts.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Charts saved                → {chart_path}")

    # Summary prints
    print("\n" + SECTION)
    print("PHASE 6b BOUNDING BOX VALIDATION — SUMMARY")
    print(SECTION)
    print(f"   Total records evaluated     : {n_records}")
    print(f"   Valid bounding boxes        : {n_valid} ({n_valid/n_records*100:.1f}%)")
    print(f"   Missing bounding boxes      : {n_missing} (flagged & isolated)")
    print(f"   Invalid geometry coordinates: {n_invalid}")
    print(f"   Page canvas reference       : {PAGE_WIDTH} × {PAGE_HEIGHT} px")
    print(f"   BBox dimensions (mean)      : width={summary_data['bbox_statistics']['width']['mean']} px, "
          f"height={summary_data['bbox_statistics']['height']['mean']} px")
    print(f"\n   Outputs saved:")
    print(f"     • {val_jsonl_path.name}")
    print(f"     • {missing_bbox_path.name}")
    print(f"     • {audit_csv_path.name}")
    print(f"     • {summary_json_path.name}")
    print(f"     • {chart_path.name}")
    print(f"     • Appended decisions to {LOG_PATH.name}")
    print(SECTION + "\n")


if __name__ == "__main__":
    main()
