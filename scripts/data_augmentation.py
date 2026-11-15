"""
data_augmentation.py — MODULE 10: DATA AUGMENTATION (Phase 10)
─────────────────────────────────────────────────────────────
Generates realistic, verifiable OCR data augmentations strictly for valid
training split documents while preserving text ground truth, transforming
bounding box coordinates, and preventing data leakage.

Key Principles & Safeguards:
───────────────────────────
 1. Train-Only Augmentation (Zero Leakage):
    - Strictly filters candidate documents using train.txt splits.
    - Never augments validation or test documents.
 2. Blank Placeholder Defense:
    - Audits every image for blank / uniform white placeholder status (contrast_std < 1.0).
    - Refuses to augment blank canvases; logs and reports limitation if detected.
 3. Realistic OCR Transforms:
    - Slight rotation (±1.0° to ±2.0°) with white background fill.
    - Photometric brightness and contrast jitter.
    - Additive Gaussian sensor noise.
    - Mild optical blur (scanner lens / focus softness).
    - Centered scaling / perspective tilt.
 4. Geometric Bounding Box Transformation:
    - Mathematically transforms 4-corner polygons under rotation/scale affine matrices.
    - Computes tight axis-aligned bounding boxes clipped to [0, W] and [0, H].
    - Re-generates all coordinate representations: [x1, y1, x2, y2], bbox_norm,
      bbox_layoutlm, and bbox_yolo.
 5. Complete Provenance & Lineage Tracking:
    - Original documents are never overwritten or modified.
    - Augmented images saved in data/augmented/images/.
    - Maintains parent_document_id and transformation recipe metadata.
    - Appends all decisions to reports/cleaning_log.csv.

Outputs:
────────
  data/augmented/images/               — augmented OCR document PNG images
  data/augmented/annotations_augmented.jsonl — transformed annotation records
  data/augmented/annotations_augmented.csv   — CSV tabular representation
  data/augmented/augmented_catalog.csv       — document-level augmented catalog
  reports/augmentation_summary.csv           — per-recipe / per-source statistics
  reports/augmentation_summary.json          — execution summary and parameters
  reports/augmentation_charts.png            — 4-panel visual audit dashboard
  reports/cleaning_log.csv                   — appended provenance decisions

Usage:
──────
  cd Indic_OCR_Data_Engineering
  python scripts/data_augmentation.py
"""

import sys
import os
import json
import csv
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# ── Project root and paths ───────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR         = PROJECT_ROOT / "data"
RAW_DIR          = DATA_DIR / "raw"
CLEANED_DIR      = DATA_DIR / "cleaned"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
AUGMENTED_DIR    = DATA_DIR / "augmented"
FINAL_DIR        = DATA_DIR / "final"
REPORTS_DIR      = PROJECT_ROOT / "reports"

AUG_IMAGES_DIR   = AUGMENTED_DIR / "images"
AUG_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".mpl_cache"))

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from config import (
    SYNTHETIC_DATASET_DIR, RANDOM_SEED,
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT,
    BLUR_THRESHOLD, CONTRAST_LOW_THRESH
)

# Set global deterministic random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RUN_TS  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SECTION = "=" * 70

LOG_PATH = REPORTS_DIR / "cleaning_log.csv"
_new_log_entries = []


def log_action(record_id: str, field: str, issue: str,
               action: str, proposed_value: str = ""):
    """Buffer a cleaning/augmentation log entry for reports/cleaning_log.csv."""
    _new_log_entries.append({
        "timestamp":      RUN_TS,
        "record_id":      record_id,
        "record_type":    "augmentation",
        "field":          field,
        "issue":          issue,
        "action":         action,
        "proposed_value": proposed_value,
    })


def append_cleaning_log():
    """Append buffered Phase 10 decisions to reports/cleaning_log.csv."""
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
    print(f"   ✓ Appended {len(_new_log_entries)} augmentation actions to {LOG_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC & COORDINATE TRANSFORMATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def transform_bounding_box(bbox: list, M: np.ndarray, img_w: int, img_h: int) -> tuple:
    """
    Transform axis-aligned bounding box [x1, y1, x2, y2] under affine matrix M.
    Returns:
        (new_bbox, new_norm, new_layoutlm, new_yolo, new_area, new_aspect_ratio, is_valid)
    """
    if bbox is None or len(bbox) != 4:
        return None, None, None, None, 0.0, 0.0, False

    x1, y1, x2, y2 = bbox
    # 4 corner points in homogeneous coordinates
    corners = np.array([
        [x1, y1, 1.0],
        [x2, y1, 1.0],
        [x2, y2, 1.0],
        [x1, y2, 1.0]
    ], dtype=np.float32)

    transformed = M.dot(corners.T).T  # Shape: (4, 2)

    new_x1 = int(np.floor(np.min(transformed[:, 0])))
    new_y1 = int(np.floor(np.min(transformed[:, 1])))
    new_x2 = int(np.ceil(np.max(transformed[:, 0])))
    new_y2 = int(np.ceil(np.max(transformed[:, 1])))

    # Clip to image boundary
    new_x1 = max(0, min(img_w - 1, new_x1))
    new_y1 = max(0, min(img_h - 1, new_y1))
    new_x2 = max(new_x1 + 1, min(img_w, new_x2))
    new_y2 = max(new_y1 + 1, min(img_h, new_y2))

    bw = new_x2 - new_x1
    bh = new_y2 - new_y1
    area = round(float(bw * bh), 1)
    aspect_ratio = round(float(bw / bh), 3) if bh > 0 else 0.0
    is_valid = (bw >= 5 and bh >= 5 and area >= 25.0)

    new_bbox = [new_x1, new_y1, new_x2, new_y2]
    new_norm = [
        round(new_x1 / img_w, 4),
        round(new_y1 / img_h, 4),
        round(new_x2 / img_w, 4),
        round(new_y2 / img_h, 4)
    ]
    new_layoutlm = [
        int(round(new_x1 / img_w * 1000)),
        int(round(new_y1 / img_h * 1000)),
        int(round(new_x2 / img_w * 1000)),
        int(round(new_y2 / img_h * 1000))
    ]
    new_yolo = [
        round(((new_x1 + new_x2) / 2.0) / img_w, 4),
        round(((new_y1 + new_y2) / 2.0) / img_h, 4),
        round(bw / img_w, 4),
        round(bh / img_h, 4)
    ]

    return new_bbox, new_norm, new_layoutlm, new_yolo, area, aspect_ratio, is_valid


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION OPERATORS
# ─────────────────────────────────────────────────────────────────────────────
def apply_rotation(img: np.ndarray, angle_deg: float) -> tuple:
    """Rotate image around center by angle_deg with white border fill."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # White background for documents (255)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated, M


def apply_scaling(img: np.ndarray, scale_factor: float) -> tuple:
    """Scale image slightly around center with white border fill."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, scale_factor)
    scaled = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return scaled, M


def apply_brightness_contrast(img: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """
    Adjust contrast (alpha) and brightness (beta).
    new_pixel = clip(alpha * pixel + beta, 0, 255)
    """
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def apply_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise to simulate sensor/paper grain."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_gaussian_blur(img: np.ndarray, kernel_size: int = 3, sigma: float = 0.8) -> np.ndarray:
    """Apply mild Gaussian blur to simulate scanner lens softness."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    return blurred


def apply_illumination_gradient(img: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    """Simulate non-uniform ambient lighting / phone shadow across document page."""
    h, w = img.shape[:2]
    # Linear gradient from top-left to bottom-right
    x = np.linspace(1.0 - intensity, 1.0 + intensity, w, dtype=np.float32)
    y = np.linspace(1.0 - intensity, 1.0 + intensity, h, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    gradient = (xv * yv) / (1.0)
    gradient = np.clip(gradient, 0.75, 1.25)
    shaded = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
    return shaded


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION RECIPES
# ─────────────────────────────────────────────────────────────────────────────
def augment_mild(img: np.ndarray, seed_val: int) -> tuple:
    """
    Recipe 1: Mild Scan / Document Capture
    - Subtle rotation: ±0.8°
    - Mild contrast adjustment: 0.95 to 1.05
    - Mild brightness jitter: ±8
    - Very mild Gaussian blur: sigma=0.6
    """
    rng = random.Random(seed_val)
    angle = rng.choice([-0.8, -0.5, 0.5, 0.8])
    alpha = rng.uniform(0.95, 1.05)
    beta = rng.randint(-8, 8)

    # 1. Geometry (Rotation)
    aug_img, M = apply_rotation(img, angle)
    # 2. Photometric (Brightness & Contrast)
    aug_img = apply_brightness_contrast(aug_img, alpha=alpha, beta=beta)
    # 3. Softness (Mild Blur)
    aug_img = apply_gaussian_blur(aug_img, kernel_size=3, sigma=0.6)

    transforms = [
        f"rotation_{angle:+.1f}deg",
        f"contrast_{alpha:.2f}",
        f"brightness_{beta:+d}",
        "gaussian_blur_s0.6"
    ]
    return aug_img, M, transforms


def augment_moderate(img: np.ndarray, seed_val: int) -> tuple:
    """
    Recipe 2: Moderate Mobile / Field Capture
    - Rotation: ±1.5°
    - Scaling: 0.98 or 1.02
    - Additive Gaussian noise: sigma=12
    - Contrast attenuation: 0.88 (simulating aged/faded ink)
    - Mild Gaussian blur: sigma=0.9
    """
    rng = random.Random(seed_val)
    angle = rng.choice([-1.5, -1.2, 1.2, 1.5])
    scale = rng.choice([0.98, 1.02])
    alpha = rng.uniform(0.85, 0.95)
    beta = rng.randint(-12, 5)
    noise_sigma = 12.0

    # 1. Geometry: Rotation + Scaling combined
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 2. Photometric: Contrast & Brightness
    aug_img = apply_brightness_contrast(aug_img, alpha=alpha, beta=beta)
    # 3. Sensor Noise
    aug_img = apply_gaussian_noise(aug_img, sigma=noise_sigma)
    # 4. Blur
    aug_img = apply_gaussian_blur(aug_img, kernel_size=3, sigma=0.9)

    transforms = [
        f"rotation_{angle:+.1f}deg",
        f"scale_{scale:.2f}",
        f"contrast_{alpha:.2f}",
        f"noise_sigma{noise_sigma:.0f}",
        "gaussian_blur_s0.9"
    ]
    return aug_img, M, transforms


# ─────────────────────────────────────────────────────────────────────────────
# BLANK PLACEHOLDER AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def audit_image_validity(img: np.ndarray) -> tuple:
    """
    Determine if image is valid, corrupt, or a blank placeholder.
    Returns:
        (status, contrast_std, dynamic_range, mean_val, reason)
    """
    if img is None or img.size == 0:
        return "CORRUPT", 0.0, 0, 0.0, "Unreadable or empty image array"

    std_val = float(np.std(img))
    min_val = int(np.min(img))
    max_val = int(np.max(img))
    mean_val = float(np.mean(img))
    dyn_range = max_val - min_val

    # Blank / uniform placeholder test:
    # A blank white page has std < 1.0 and mean > 250 with near-zero dynamic range
    if std_val < 1.0 or dyn_range < 5 or (mean_val > 250.0 and dyn_range <= 2):
        return "BLANK_PLACEHOLDER", std_val, dyn_range, mean_val, (
            f"Uniform blank canvas detected (std={std_val:.2f}, range={dyn_range}, mean={mean_val:.1f})"
        )

    return "VALID_DOCUMENT", std_val, dyn_range, mean_val, "Valid document image with text strokes"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 10: Verifiable OCR Data Augmentation Engine")
    parser.add_argument("--source", choices=["preprocessed", "final", "raw", "auto"],
                        default="auto", help="Source directory for candidate document images")
    parser.add_argument("--recipes", nargs="+", default=["mild", "moderate"],
                        choices=["mild", "moderate"], help="Augmentation recipes to apply")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(SECTION)
    print("PHASE 10 — VERIFIABLE INDIC OCR DATA AUGMENTATION")
    print(SECTION)

    # 1. Determine Source Image Directory
    # Priority in auto mode: preprocessed/images -> final/images -> raw
    source_choice = args.source
    if source_choice == "auto":
        if (PREPROCESSED_DIR / "images").exists() and any((PREPROCESSED_DIR / "images").glob("*.png")):
            source_dir = PREPROCESSED_DIR / "images"
            source_label = "preprocessed"
        elif (FINAL_DIR / "images").exists() and any((FINAL_DIR / "images").glob("*.png")):
            source_dir = FINAL_DIR / "images"
            source_label = "final"
        else:
            source_dir = SYNTHETIC_DATASET_DIR / "images"
            source_label = "raw"
    elif source_choice == "preprocessed":
        source_dir = PREPROCESSED_DIR / "images"
        source_label = "preprocessed"
    elif source_choice == "final":
        source_dir = FINAL_DIR / "images"
        source_label = "final"
    else:
        source_dir = SYNTHETIC_DATASET_DIR / "images"
        source_label = "raw"

    print(f"\n[1/6] Initializing Data Augmentation Engine …")
    print(f"   Selected Image Source : {source_dir} [{source_label}]")
    print(f"   Augmentation Recipes  : {args.recipes}")
    print(f"   Deterministic Seed    : {args.seed}")

    # 2. Ingest Splits to Enforce Zero-Leakage Guardrail
    print("\n[2/6] Loading splits to enforce Train-Only restriction …")
    splits_dir = SYNTHETIC_DATASET_DIR / "splits"
    train_file = splits_dir / "train.txt"
    val_file   = splits_dir / "validation.txt"
    test_file  = splits_dir / "test.txt"

    if not train_file.exists():
        print(f"ERROR: Splits file not found: {train_file}")
        sys.exit(1)

    train_ids = set(train_file.read_text(encoding="utf-8").splitlines())
    val_ids   = set(val_file.read_text(encoding="utf-8").splitlines()) if val_file.exists() else set()
    test_ids  = set(test_file.read_text(encoding="utf-8").splitlines()) if test_file.exists() else set()

    print(f"   Registered splits:")
    print(f"     • Train doc IDs       : {len(train_ids)} (ELIGIBLE FOR AUGMENTATION)")
    print(f"     • Validation doc IDs  : {len(val_ids)} (STRICTLY EXCLUDED)")
    print(f"     • Test doc IDs        : {len(test_ids)} (STRICTLY EXCLUDED)")

    # 3. Load Validated Annotations
    print("\n[3/6] Loading annotations to maintain ground-truth binding …")
    annotations_path = CLEANED_DIR / "annotations_validated.jsonl"
    if not annotations_path.exists():
        annotations_path = FINAL_DIR / "annotations_master.jsonl"

    if not annotations_path.exists():
        print(f"ERROR: No annotations file found in {CLEANED_DIR} or {FINAL_DIR}")
        sys.exit(1)

    doc_to_annotations = defaultdict(list)
    total_raw_anns = 0
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                rec = json.loads(line_str)
                doc_to_annotations[rec["document_id"]].append(rec)
                total_raw_anns += 1

    print(f"   Loaded {total_raw_anns} annotations across {len(doc_to_annotations)} documents from {annotations_path.name}")

    # 4. Audit Candidate Training Images for Blank Placeholders and Corruption
    print(f"\n[4/6] Auditing candidate training images in {source_dir.name} …")
    valid_train_images = []
    blank_placeholder_images = []
    corrupt_images = []
    missing_images = []

    for doc_id in sorted(train_ids):
        img_path = source_dir / f"{doc_id}.png"
        if not img_path.exists():
            missing_images.append(doc_id)
            log_action(doc_id, "image_file", "MISSING_ON_DISK", "SKIP — file not found", str(img_path))
            continue

        if img_path.stat().st_size == 0:
            corrupt_images.append(doc_id)
            log_action(doc_id, "image_file", "CORRUPT_ZERO_BYTE", "SKIP — quarantined corrupt image", str(img_path))
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        status, std_val, dyn_range, mean_val, reason = audit_image_validity(img)

        if status == "BLANK_PLACEHOLDER":
            blank_placeholder_images.append((doc_id, std_val, reason))
            log_action(doc_id, "image_content", "BLANK_PLACEHOLDER",
                       "SKIP — refused to augment blank canvas to prevent false data generation",
                       f"std={std_val:.2f}")
        elif status == "CORRUPT":
            corrupt_images.append(doc_id)
            log_action(doc_id, "image_content", "UNREADABLE", "SKIP — corrupted pixels", reason)
        else:
            valid_train_images.append((doc_id, img_path, img))

    print(f"   Candidate Audit Results:")
    print(f"     • Valid OCR-Ready Documents : {len(valid_train_images)}")
    print(f"     • Blank Placeholder Images  : {len(blank_placeholder_images)} (Skipped)")
    print(f"     • Corrupt / Zero-byte Files : {len(corrupt_images)} (Skipped)")
    print(f"     • Missing Files on Disk     : {len(missing_images)} (Skipped)")

    # ── DEFENSE: If all candidate images are blank placeholders, halt meaningfully
    if len(valid_train_images) == 0 and len(blank_placeholder_images) > 0:
        limitation_msg = (
            f"DATASET LIMITATION DETECTED: All {len(blank_placeholder_images)} training images in "
            f"'{source_dir}' are identical blank white placeholders. As mandated by project safety "
            f"standards, meaningful OCR augmentation was stopped to prevent false data generation. "
            f"Please ensure preprocessed document images with rendered text (Phase 7) are available."
        )
        print(f"\n[!] {limitation_msg}")

        # Generate summary reflecting the limitation audit
        summary_data = {
            "run_timestamp": RUN_TS,
            "phase": "Phase 10 — Data Augmentation",
            "source_directory": str(source_dir),
            "status": "STOPPED_BLANK_PLACEHOLDER_LIMITATION",
            "total_candidate_train_images": len(train_ids),
            "valid_augmented_images_generated": 0,
            "blank_placeholders_detected": len(blank_placeholder_images),
            "corrupt_images_skipped": len(corrupt_images),
            "limitation_report": limitation_msg
        }

        with open(REPORTS_DIR / "augmentation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        summary_rows = [{
            "metric": "Status", "value": "STOPPED_BLANK_PLACEHOLDER_LIMITATION",
            "count": len(blank_placeholder_images), "details": limitation_msg
        }]
        pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "augmentation_summary.csv", index=False)
        append_cleaning_log()
        print("\nExiting gracefully per blank placeholder defense policy.")
        return

    # 5. Execute Verifiable Augmentation Pipeline on Valid Training Images
    print(f"\n[5/6] Executing realistic OCR augmentation on {len(valid_train_images)} valid training images …")
    augmented_records = []
    catalog_rows = []
    per_recipe_counts = defaultdict(int)
    doc_index = 0

    for doc_id, img_path, orig_gray in valid_train_images:
        h, w = orig_gray.shape[:2]
        parent_anns = doc_to_annotations.get(doc_id, [])

        for recipe in args.recipes:
            sub_seed = args.seed + (doc_index * 13) + (101 if recipe == "moderate" else 37)
            aug_doc_id = f"{doc_id}_aug_{recipe}"

            if recipe == "mild":
                aug_img, M, transforms = augment_mild(orig_gray, sub_seed)
            else:
                aug_img, M, transforms = augment_moderate(orig_gray, sub_seed)

            # Save augmented image separately in data/augmented/images/
            out_img_path = AUG_IMAGES_DIR / f"{aug_doc_id}.png"
            cv2.imwrite(str(out_img_path), aug_img)

            # Transform bounding boxes for all annotations linked to this document
            num_valid_boxes = 0
            for orig_rec in parent_anns:
                aug_rec = dict(orig_rec)
                aug_rec["document_id"] = aug_doc_id
                aug_rec["parent_document_id"] = doc_id
                aug_rec["is_augmented"] = True
                aug_rec["augmentation_recipe"] = recipe
                aug_rec["transforms_applied"] = transforms

                # Transform coordinates
                orig_bbox = orig_rec.get("bbox")
                if orig_bbox and len(orig_bbox) == 4:
                    new_bbox, new_norm, new_llm, new_yolo, area, ar, is_valid = transform_bounding_box(
                        orig_bbox, M, w, h
                    )
                    aug_rec["bbox"] = new_bbox
                    aug_rec["bbox_norm"] = new_norm
                    aug_rec["bbox_layoutlm"] = new_llm
                    aug_rec["bbox_yolo"] = new_yolo
                    aug_rec["bbox_area"] = area
                    aug_rec["bbox_aspect_ratio"] = ar
                    aug_rec["is_bbox_valid"] = is_valid
                    if is_valid:
                        num_valid_boxes += 1
                else:
                    aug_rec["bbox"] = None
                    aug_rec["is_bbox_valid"] = False

                augmented_records.append(aug_rec)

            per_recipe_counts[recipe] += 1
            catalog_rows.append({
                "augmented_doc_id":  aug_doc_id,
                "parent_doc_id":     doc_id,
                "recipe":            recipe,
                "script":            parent_anns[0].get("script", "unknown") if parent_anns else "unknown",
                "doc_type":          parent_anns[0].get("doc_type", "document") if parent_anns else "document",
                "transforms":        " | ".join(transforms),
                "num_fields":        len(parent_anns),
                "valid_bboxes":      num_valid_boxes,
                "image_width":       w,
                "image_height":      h,
                "file_path":         str(out_img_path.relative_to(PROJECT_ROOT)),
                "split":             "train"  # Strictly train split
            })

            log_action(aug_doc_id, "image_augmentation", f"AUG_{recipe.upper()}",
                       f"APPLY_RECIPE — parent={doc_id}, transforms={','.join(transforms)}",
                       str(out_img_path.relative_to(PROJECT_ROOT)))

        doc_index += 1

    total_augmented_images = len(catalog_rows)
    total_augmented_anns   = len(augmented_records)
    print(f"   ✓ Generated {total_augmented_images} augmented document images")
    print(f"   ✓ Generated {total_augmented_anns} augmented annotation records")
    for r, count in per_recipe_counts.items():
        print(f"       • Recipe '{r}': {count} documents")

    # 6. Save Artifacts, Catalogs, Summaries & Visualizations
    print(f"\n[6/6] Saving Phase 10 artifacts, summaries and visualizations …")

    # Annotations JSONL
    aug_jsonl_path = AUGMENTED_DIR / "annotations_augmented.jsonl"
    with open(aug_jsonl_path, "w", encoding="utf-8") as f:
        for rec in augmented_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"   ✓ Augmented annotations JSONL saved → {aug_jsonl_path} ({total_augmented_anns} records)")

    # Annotations CSV
    aug_csv_path = AUGMENTED_DIR / "annotations_augmented.csv"
    pd.DataFrame(augmented_records).to_csv(aug_csv_path, index=False, encoding="utf-8")
    print(f"   ✓ Augmented annotations CSV saved   → {aug_csv_path}")

    # Master Augmented Catalog CSV
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = AUGMENTED_DIR / "augmented_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False, encoding="utf-8")
    print(f"   ✓ Augmented catalog CSV saved       → {catalog_path}")

    # Reports: augmentation_summary.csv and augmentation_summary.json
    summary_data = {
        "run_timestamp": RUN_TS,
        "phase": "Phase 10 — Data Augmentation",
        "source_directory": str(source_dir.relative_to(PROJECT_ROOT)),
        "source_type": source_label,
        "total_candidate_train_ids": len(train_ids),
        "valid_parent_images_augmented": len(valid_train_images),
        "blank_placeholders_skipped": len(blank_placeholder_images),
        "corrupt_images_skipped": len(corrupt_images),
        "validation_test_images_excluded": len(val_ids) + len(test_ids),
        "total_augmented_images_generated": total_augmented_images,
        "total_augmented_annotations_generated": total_augmented_anns,
        "recipe_breakdown": dict(per_recipe_counts),
        "script_distribution": catalog_df["script"].value_counts().to_dict(),
        "random_seed": args.seed,
        "zero_leakage_guarantee": True,
        "bounding_box_transformation": "Affine 4-corner polygon projection with clipping"
    }

    summary_json_path = REPORTS_DIR / "augmentation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"   ✓ Summary JSON saved → {summary_json_path}")

    summary_rows = [
        {"metric": "Source Image Type", "value": source_label, "details": str(source_dir)},
        {"metric": "Train Split IDs", "value": str(len(train_ids)), "details": "From data/raw/indic_ocr_dataset/splits/train.txt"},
        {"metric": "Valid Training Documents", "value": str(len(valid_train_images)), "details": "Passed non-blank and non-corrupt audits"},
        {"metric": "Blank Placeholders Skipped", "value": str(len(blank_placeholder_images)), "details": "Safeguard against augmenting blank white canvases"},
        {"metric": "Corrupt Images Skipped", "value": str(len(corrupt_images)), "details": "Zero-byte / unreadable files quarantined in Phase 5"},
        {"metric": "Val & Test Images Protected", "value": str(len(val_ids) + len(test_ids)), "details": "Zero leakage guarantee enforced"},
        {"metric": "Augmented Images Created", "value": str(total_augmented_images), "details": f"{per_recipe_counts.get('mild', 0)} mild + {per_recipe_counts.get('moderate', 0)} moderate"},
        {"metric": "Augmented Annotations Created", "value": str(total_augmented_anns), "details": "Text preserved verbatim, bboxes transformed"},
        {"metric": "Random Seed", "value": str(args.seed), "details": "Deterministic reproducibility"}
    ]
    summary_csv_path = REPORTS_DIR / "augmentation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False, encoding="utf-8")
    print(f"   ✓ Summary CSV saved  → {summary_csv_path}")

    # Append provenance records to cleaning_log.csv
    append_cleaning_log()

    # Generate 4-panel visual audit dashboard
    print(f"   Generating Phase 10 visual audit charts …")
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 10 — OCR Data Augmentation & Lineage Audit", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Candidate Image Eligibility & Defense Audit
    ax1 = axes[0, 0]
    audit_labels = ["Valid Train\nAugmented", "Corrupt\nSkipped", "Blank Canvas\nProtected", "Val & Test\nExcluded"]
    audit_counts = [len(valid_train_images), len(corrupt_images), len(blank_placeholder_images), len(val_ids) + len(test_ids)]
    colors1 = ["#2CA02C", "#D62728", "#FF7F0E", "#1F77B4"]
    bars1 = ax1.bar(audit_labels, audit_counts, color=colors1, width=0.55, edgecolor="white")
    ax1.set_title("Training Split Eligibility & Safeguard Breakdown", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Document Image Count")
    for b, c in zip(bars1, audit_counts):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, str(c), ha="center", va="bottom", fontweight="bold")

    # Panel 2: Augmented Corpus Volume by Recipe
    ax2 = axes[0, 1]
    recipe_names = list(per_recipe_counts.keys())
    recipe_vals = [per_recipe_counts[k] for k in recipe_names]
    bars2 = ax2.bar([r.title() + " Recipe" for r in recipe_names], recipe_vals, color=["#4472C4", "#ED7D31"], width=0.5, edgecolor="white")
    ax2.set_title(f"Augmented Document Generation ({total_augmented_images} Total)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Generated Images")
    for b, c in zip(bars2, recipe_vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{c} images", ha="center", va="bottom", fontweight="bold")

    # Panel 3: Script Balance in Augmented Data
    ax3 = axes[1, 0]
    script_counts = catalog_df["script"].value_counts()
    script_map = {"en": "English (en)", "hi": "Hindi (hi)", "bn": "Bengali (bn)"}
    ax3.bar([script_map.get(k, k) for k in script_counts.index], script_counts.values,
            color=["#17BECF", "#BCBD22", "#9467BD"], width=0.5, edgecolor="white")
    ax3.set_title("Script Distribution in Augmented Dataset", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Augmented Documents")
    for idx, (k, val) in enumerate(script_counts.items()):
        ax3.text(idx, val + 3, f"{val} ({val/total_augmented_images*100:.1f}%)", ha="center", va="bottom", fontweight="bold")

    # Panel 4: Sample Original vs Augmented Bounding Box Alignment
    ax4 = axes[1, 1]
    if valid_train_images:
        sample_doc_id, _, sample_img = valid_train_images[0]
        # Show moderate augmented sample with bboxes overlaid
        sample_aug_path = AUG_IMAGES_DIR / f"{sample_doc_id}_aug_moderate.png"
        if sample_aug_path.exists():
            disp_img = cv2.imread(str(sample_aug_path), cv2.IMREAD_GRAYSCALE)
            ax4.imshow(disp_img, cmap="gray")
            # Draw sample bboxes
            sample_anns = [r for r in augmented_records if r["document_id"] == f"{sample_doc_id}_aug_moderate"]
            for ann in sample_anns:
                bbox = ann.get("bbox")
                if bbox:
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                             linewidth=1.5, edgecolor="red", facecolor="none")
                    ax4.add_patch(rect)
            ax4.set_title(f"Transformed BBoxes on {sample_doc_id}_aug_moderate", fontsize=12, fontweight="bold")
            ax4.axis("off")
        else:
            ax4.text(0.5, 0.5, "Sample preview available on disk", ha="center", va="center")
            ax4.axis("off")
    else:
        ax4.text(0.5, 0.5, "No valid training images to display", ha="center", va="center")
        ax4.axis("off")

    plt.tight_layout()
    chart_path = REPORTS_DIR / "augmentation_charts.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Charts saved       → {chart_path}")

    # Summary prints
    print("\n" + SECTION)
    print("PHASE 10 DATA AUGMENTATION — COMPLETE SUMMARY")
    print(SECTION)
    print(f"   Total Train IDs Evaluated          : {len(train_ids)}")
    print(f"   Valid Non-Blank Parents Augmented : {len(valid_train_images)}")
    print(f"   Blank Placeholders Skipped         : {len(blank_placeholder_images)}")
    print(f"   Corrupt Images Quarantined         : {len(corrupt_images)}")
    print(f"   Val/Test Images Excluded           : {len(val_ids) + len(test_ids)} (Zero Data Leakage)")
    print(f"   Total Augmented Images Created     : {total_augmented_images}")
    print(f"   Total Augmented Annotations        : {total_augmented_anns}")
    print(f"   Storage Directories:")
    print(f"     • Images       : {AUG_IMAGES_DIR}")
    print(f"     • Annotations  : {aug_jsonl_path.name}, {aug_csv_path.name}")
    print(f"     • Catalogs     : {catalog_path.name}")
    print(f"     • Reports      : {summary_json_path.name}, {summary_csv_path.name}, {chart_path.name}")
    print(SECTION + "\n")


if __name__ == "__main__":
    main()

