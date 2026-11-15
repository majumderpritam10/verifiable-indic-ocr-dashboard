"""
image_quality.py — MODULE 5: IMAGE QUALITY ANALYSIS (Phase 5)
─────────────────────────────────────────────────────────────
Performs a comprehensive quality assessment of all document images in
data/raw/indic_ocr_dataset/images/ to detect visual degradations before
preprocessing and model training.

Checks & Metrics evaluated
──────────────────────────
 1. Integrity & readability:
    - Zero-byte / truncated / unreadable image detection (corrupt files)
 2. Spatial resolution & geometry:
    - Width, height, aspect ratio, megapixels
    - Resolution check against MIN_IMAGE_WIDTH & MIN_IMAGE_HEIGHT (100 px)
 3. Blur & Sharpness:
    - Variance of the Laplacian (cv2.Laplacian)
    - Flagged blurry if Laplacian variance < BLUR_THRESHOLD (100.0)
    - Sobel edge intensity (mean gradient magnitude)
 4. Contrast:
    - Pixel intensity standard deviation
    - Flagged low contrast if std-dev < CONTRAST_LOW_THRESH (30.0)
    - Dynamic intensity range (max - min)
 5. Exposure & Brightness:
    - Mean grayscale pixel intensity
    - Flagged underexposed (< 40) or overexposed (> 240 / blank white)
 6. Noise level:
    - Robust high-frequency noise estimate via MAD of Laplacian
    - Flagged noisy if noise estimate > NOISE_HIGH_THRESH (50.0)
 7. Blank / Uniform page detection:
    - Detected if contrast std < 1.0 and brightness > 250 (e.g. synthetic placeholders)
 8. Quality classification & action recommendations:
    - Quality labels: 'good', 'fair', 'poor', 'corrupt'
    - Recommended pipeline action: 'use_as_is', 'preprocess', 'quarantine'

Outputs
───────
  reports/image_quality.csv         — per-image detailed quality metrics
  reports/image_quality.json        — aggregated quality profile & defect stats
  reports/image_quality_charts.png  — 6-panel visualization figure
  reports/cleaning_log.csv          — appended with Phase 5 quality assignments

Usage
─────
  cd Indic_OCR_Data_Engineering
  python scripts/image_quality.py
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter

# Set MPLCONFIGDIR early to avoid cache / permission issues
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".mpl_cache"))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    SYNTHETIC_DATASET_DIR, REPORTS_DIR,
    BLUR_THRESHOLD, CONTRAST_LOW_THRESH,
    NOISE_HIGH_THRESH, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT,
    SUPPORTED_IMAGE_EXTS
)

IMAGES_DIR = SYNTHETIC_DATASET_DIR / "images"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
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
        "record_type":    "image",
        "field":          field,
        "issue":          issue,
        "action":         action,
        "proposed_value": proposed_value,
    })


def append_cleaning_log():
    """Append buffered Phase 5 decisions to cleaning_log.csv."""
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
    print(f"   ✓ Appended {len(_new_log_entries)} quality decisions to {LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def estimate_noise_mad(laplacian: np.ndarray) -> float:
    """Robust noise standard deviation using Median Absolute Deviation on Laplacian."""
    abs_lap = np.abs(laplacian)
    med = np.median(abs_lap)
    mad = np.median(np.abs(abs_lap - med))
    # Normal consistency factor 0.6745
    return float(mad / 0.6745) if mad > 0 else 0.0


def compute_sobel_gradient_magnitude(gray: np.ndarray) -> float:
    """Mean Sobel gradient magnitude for edge sharpness."""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(mag))


def analyze_single_image(img_path: Path) -> dict:
    """
    Perform deep image quality metrics computation on a single image file.
    Returns dictionary with all metrics, defect flags, and quality label.
    """
    doc_id = img_path.stem
    file_size = img_path.stat().st_size

    # Check for corrupt or zero-byte file
    if file_size == 0:
        return {
            "document_id":        doc_id,
            "file_name":          img_path.name,
            "file_size_bytes":    0,
            "is_corrupt":         True,
            "width":              None,
            "height":             None,
            "aspect_ratio":       None,
            "megapixels":         None,
            "mean_brightness":    None,
            "contrast_std":       None,
            "blur_laplacian_var": None,
            "edge_density":       None,
            "noise_estimate":     None,
            "dynamic_range":      None,
            "is_blurry":          False,
            "is_low_contrast":    False,
            "is_underexposed":    False,
            "is_overexposed":     False,
            "is_noisy":           False,
            "is_low_resolution":  False,
            "is_blank":           False,
            "quality_score":      0.0,
            "quality_label":      "corrupt",
            "recommended_action": "quarantine",
        }

    # Attempt image read via OpenCV
    # Using cv2.imdecode to handle unicode paths reliably on Windows
    try:
        raw_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception:
        # Fallback to PIL
        try:
            pil_img = Image.open(img_path)
            pil_img.verify()
            pil_img = Image.open(img_path)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return {
                "document_id":        doc_id,
                "file_name":          img_path.name,
                "file_size_bytes":    file_size,
                "is_corrupt":         True,
                "width":              None,
                "height":             None,
                "aspect_ratio":       None,
                "megapixels":         None,
                "mean_brightness":    None,
                "contrast_std":       None,
                "blur_laplacian_var": None,
                "edge_density":       None,
                "noise_estimate":     None,
                "dynamic_range":      None,
                "is_blurry":          False,
                "is_low_contrast":    False,
                "is_underexposed":    False,
                "is_overexposed":     False,
                "is_noisy":           False,
                "is_low_resolution":  False,
                "is_blank":           False,
                "quality_score":      0.0,
                "quality_label":      "corrupt",
                "recommended_action": "quarantine",
            }

    h, w = img_bgr.shape[:2]
    aspect_ratio = round(w / h, 4) if h > 0 else 0.0
    megapixels = round((w * h) / 1_000_000, 4)

    # Convert to grayscale for standard quality metrics
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Blur via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Edge density via Sobel
    edge_mag = compute_sobel_gradient_magnitude(gray)

    # 3. Contrast via standard deviation of pixel intensities
    contrast_std = float(np.std(gray))
    dyn_range = int(np.max(gray) - np.min(gray))

    # 4. Brightness / exposure
    brightness = float(np.mean(gray))

    # 5. Noise estimation
    noise_est = estimate_noise_mad(laplacian)

    # Flag conditions
    is_low_res = (w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT)
    is_blurry  = (blur_score < BLUR_THRESHOLD)
    is_low_con = (contrast_std < CONTRAST_LOW_THRESH)
    is_underexp = (brightness < 40.0)
    is_overexp  = (brightness > 240.0)
    is_noisy    = (noise_est > NOISE_HIGH_THRESH)
    is_blank    = (contrast_std < 1.0 and brightness > 250.0)

    # Compute a continuous quality score (0 to 100)
    # Penalize blur, low contrast, underexposure/extreme overexposure, high noise
    score = 100.0
    if is_blank:
        score = 20.0   # blank placeholder
    else:
        if is_low_res:
            score -= 30.0
        if is_blurry:
            # Scale penalty based on how far below threshold
            ratio = max(0.0, blur_score / BLUR_THRESHOLD)
            score -= (1.0 - ratio) * 35.0
        if is_low_con:
            ratio = max(0.0, contrast_std / CONTRAST_LOW_THRESH)
            score -= (1.0 - ratio) * 25.0
        if is_underexp:
            score -= 20.0
        elif is_overexp and not is_blank:
            score -= 15.0
        if is_noisy:
            score -= min(25.0, (noise_est - NOISE_HIGH_THRESH) * 0.5)

    score = max(0.0, min(100.0, round(score, 1)))

    # Determine discrete quality label & recommended action
    if is_blank:
        # Document placeholder
        quality_label = "poor"
        action = "preprocess"   # requires contrast/binarization or rendering
    elif score >= 75.0:
        quality_label = "good"
        action = "use_as_is"
    elif score >= 45.0:
        quality_label = "fair"
        action = "preprocess"   # will benefit from CLAHE / deskew / denoise
    else:
        quality_label = "poor"
        action = "preprocess"

    return {
        "document_id":        doc_id,
        "file_name":          img_path.name,
        "file_size_bytes":    file_size,
        "is_corrupt":         False,
        "width":              w,
        "height":             h,
        "aspect_ratio":       aspect_ratio,
        "megapixels":         megapixels,
        "mean_brightness":    round(brightness, 2),
        "contrast_std":       round(contrast_std, 2),
        "blur_laplacian_var": round(blur_score, 2),
        "edge_density":       round(edge_mag, 2),
        "noise_estimate":     round(noise_est, 2),
        "dynamic_range":      dyn_range,
        "is_blurry":          is_blurry,
        "is_low_contrast":    is_low_con,
        "is_underexposed":    is_underexp,
        "is_overexposed":     is_overexp,
        "is_noisy":           is_noisy,
        "is_low_resolution":  is_low_res,
        "is_blank":           is_blank,
        "quality_score":      score,
        "quality_label":      quality_label,
        "recommended_action": action,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(SECTION)
    print("PHASE 5 — IMAGE QUALITY ANALYSIS")
    print(SECTION)

    # 1. Enumerate images
    print("\n[1/5] Scanning dataset images …")
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images directory not found at {IMAGES_DIR}")
        sys.exit(1)

    image_paths = sorted([
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])
    print(f"   Found {len(image_paths)} images in {IMAGES_DIR}")

    # 2. Quality analysis
    print("\n[2/5] Computing image quality metrics (blur, contrast, brightness, noise, resolution) …")
    results = []
    for i, p in enumerate(image_paths, 1):
        res = analyze_single_image(p)
        results.append(res)
        if i % 60 == 0 or i == len(image_paths):
            print(f"   Analyzed {i}/{len(image_paths)} images …")

    df = pd.DataFrame(results)

    # 3. Quality summary & logging
    print("\n[3/5] Evaluating defects & quality distribution …")
    corrupt_mask = df["is_corrupt"]
    valid_df = df[~corrupt_mask]

    n_total       = len(df)
    n_corrupt     = int(corrupt_mask.sum())
    n_valid       = len(valid_df)
    n_blurry      = int(valid_df["is_blurry"].sum())
    n_low_con     = int(valid_df["is_low_contrast"].sum())
    n_overexp     = int(valid_df["is_overexposed"].sum())
    n_underexp    = int(valid_df["is_underexposed"].sum())
    n_noisy       = int(valid_df["is_noisy"].sum())
    n_low_res     = int(valid_df["is_low_resolution"].sum())
    n_blank       = int(valid_df["is_blank"].sum())

    label_counts = df["quality_label"].value_counts().to_dict()
    action_counts = df["recommended_action"].value_counts().to_dict()

    print(f"   Total images analyzed   : {n_total}")
    print(f"   Usable images           : {n_valid}")
    print(f"   Corrupt (zero-byte)     : {n_corrupt}")
    print(f"   Blank page placeholders : {n_blank}")
    print(f"   Low contrast (< {CONTRAST_LOW_THRESH})  : {n_low_con}")
    print(f"   Blurry (< {BLUR_THRESHOLD})     : {n_blurry}")
    print(f"   Overexposed (> 240)     : {n_overexp}")
    print(f"   Underexposed (< 40)     : {n_underexp}")
    print(f"   High noise (> {NOISE_HIGH_THRESH})    : {n_noisy}")
    print(f"   Quality label breakdown : {label_counts}")

    # Log handling decisions for each image to cleaning_log.csv
    print("\n[4/5] Logging Phase 5 quality assignments to cleaning_log.csv …")
    for _, row in df.iterrows():
        did = row["document_id"]
        q_label = row["quality_label"]
        rec_act = row["recommended_action"]

        if row["is_corrupt"]:
            log_action(did, "quality_label", "CORRUPT_IMAGE",
                       "QUARANTINE — zero-byte/unreadable image excluded", "corrupt")
        else:
            issues = []
            if row["is_blank"]:
                issues.append("BLANK_PLACEHOLDER")
            if row["is_blurry"]:
                issues.append("BLURRY")
            if row["is_low_contrast"]:
                issues.append("LOW_CONTRAST")
            if row["is_overexposed"] and not row["is_blank"]:
                issues.append("OVEREXPOSED")
            if row["is_underexposed"]:
                issues.append("UNDEREXPOSED")
            if row["is_noisy"]:
                issues.append("HIGH_NOISE")
            if row["is_low_resolution"]:
                issues.append("LOW_RESOLUTION")

            issue_str = "|".join(issues) if issues else "CLEAN"
            action_desc = f"ASSIGN_QUALITY — labeled '{q_label}' (score={row['quality_score']}); action={rec_act}"
            log_action(did, "quality_label", issue_str, action_desc, q_label)

    append_cleaning_log()

    # 4. Save reports
    print("\n[5/5] Saving reports and generating publication charts …")

    # CSV report
    csv_path = REPORTS_DIR / "image_quality.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✓ CSV report saved   → {csv_path}")

    # JSON report
    corrupt_ids = df[df["is_corrupt"]]["document_id"].tolist()
    json_path = REPORTS_DIR / "image_quality.json"
    summary_data = {
        "run_timestamp": RUN_TS,
        "phase": "Phase 5 — Image Quality Analysis",
        "total_images": n_total,
        "usable_images": n_valid,
        "corrupt_images": n_corrupt,
        "corrupt_doc_ids": corrupt_ids,
        "thresholds_used": {
            "blur_threshold_laplacian_var": BLUR_THRESHOLD,
            "contrast_low_threshold_std": CONTRAST_LOW_THRESH,
            "noise_high_threshold_mad": NOISE_HIGH_THRESH,
            "min_image_width": MIN_IMAGE_WIDTH,
            "min_image_height": MIN_IMAGE_HEIGHT,
        },
        "quality_label_distribution": label_counts,
        "recommended_action_distribution": action_counts,
        "defect_counts": {
            "is_corrupt": n_corrupt,
            "is_blank_placeholder": n_blank,
            "is_low_contrast": n_low_con,
            "is_blurry": n_blurry,
            "is_overexposed": n_overexp,
            "is_underexposed": n_underexp,
            "is_noisy": n_noisy,
            "is_low_resolution": n_low_res,
        },
        "metric_statistics_valid_images": {
            "width": {
                "min": int(valid_df["width"].min()),
                "max": int(valid_df["width"].max()),
                "mean": round(float(valid_df["width"].mean()), 2),
            },
            "height": {
                "min": int(valid_df["height"].min()),
                "max": int(valid_df["height"].max()),
                "mean": round(float(valid_df["height"].mean()), 2),
            },
            "brightness_mean": {
                "min": round(float(valid_df["mean_brightness"].min()), 2),
                "max": round(float(valid_df["mean_brightness"].max()), 2),
                "mean": round(float(valid_df["mean_brightness"].mean()), 2),
            },
            "contrast_std": {
                "min": round(float(valid_df["contrast_std"].min()), 2),
                "max": round(float(valid_df["contrast_std"].max()), 2),
                "mean": round(float(valid_df["contrast_std"].mean()), 2),
            },
            "blur_laplacian_var": {
                "min": round(float(valid_df["blur_laplacian_var"].min()), 2),
                "max": round(float(valid_df["blur_laplacian_var"].max()), 2),
                "mean": round(float(valid_df["blur_laplacian_var"].mean()), 2),
            },
            "noise_estimate_mad": {
                "min": round(float(valid_df["noise_estimate"].min()), 2),
                "max": round(float(valid_df["noise_estimate"].max()), 2),
                "mean": round(float(valid_df["noise_estimate"].mean()), 2),
            },
            "quality_score": {
                "min": round(float(valid_df["quality_score"].min()), 2),
                "max": round(float(valid_df["quality_score"].max()), 2),
                "mean": round(float(valid_df["quality_score"].mean()), 2),
            }
        },
        "downstream_recommendations": {
            "phase_6_clean_annotations": "Filter out annotations associated with the 4 corrupt images.",
            "phase_7_image_preprocessing": (
                "Blank placeholder images require synthetic character rendering / CLAHE contrast stretching "
                "before feeding into OCR recognition models."
            ),
            "phase_10_augmentation": "Apply mild/moderate geometric and photometric transforms only to the training set."
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"   ✓ JSON report saved  → {json_path}")

    # Charts: 6-panel visualization figure
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 5 — Image Quality Analysis Report", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Quality Label Breakdown
    ax = axes[0, 0]
    ql_order = ["good", "fair", "poor", "corrupt"]
    ql_counts = [label_counts.get(k, 0) for k in ql_order]
    ql_colors = ["#2CA02C", "#FFA500", "#FF6B6B", "#7F7F7F"]
    bars1 = ax.bar(ql_order, ql_counts, color=ql_colors, edgecolor="white", width=0.55)
    ax.set_title("Image Quality Classification", fontsize=12, fontweight="bold")
    ax.set_ylabel("Document Count")
    ax.set_ylim(0, max(ql_counts) * 1.15 if ql_counts else 10)
    for b, cnt in zip(bars1, ql_counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                f"{cnt}\n({cnt/n_total*100:.1f}%)", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Panel 2: Detected Defect Frequencies
    ax = axes[0, 1]
    defect_names = ["Corrupt\n(0-byte)", "Blank\nPage", "Low\nContrast",
                    "Blurry\n(< 100)", "Over\nexposed", "High\nNoise"]
    defect_vals  = [n_corrupt, n_blank, n_low_con, n_blurry, n_overexp, n_noisy]
    d_colors     = ["#D62728", "#E377C2", "#FF7F0E", "#1F77B4", "#BCBD22", "#9467BD"]
    bars2 = ax.bar(defect_names, defect_vals, color=d_colors, edgecolor="white", width=0.6)
    ax.set_title("Detected Quality Defects", fontsize=12, fontweight="bold")
    ax.set_ylabel("Occurrences")
    ax.set_ylim(0, max(defect_vals) * 1.15 if defect_vals else 10)
    for b, val in zip(bars2, defect_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 3: Recommended Action Distribution
    ax = axes[0, 2]
    act_order = ["use_as_is", "preprocess", "quarantine"]
    act_counts = [action_counts.get(k, 0) for k in act_order]
    act_colors = ["#2CA02C", "#1F77B4", "#D62728"]
    wedges, texts, autotexts = ax.pie(
        act_counts,
        labels=[k.replace("_", " ").title() for k in act_order],
        autopct="%1.1f%%",
        colors=act_colors,
        startangle=140,
        textprops={"fontsize": 10, "fontweight": "bold"}
    )
    ax.set_title("Recommended Pipeline Actions", fontsize=12, fontweight="bold")

    # Panel 4: Contrast vs Brightness Distribution
    ax = axes[1, 0]
    if not valid_df.empty:
        scatter = ax.scatter(
            valid_df["mean_brightness"],
            valid_df["contrast_std"],
            c=valid_df["quality_score"],
            cmap="viridis",
            alpha=0.7,
            s=40,
            edgecolor="none"
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Quality Score (0-100)", fontsize=9)
        ax.axhline(y=CONTRAST_LOW_THRESH, color="red", linestyle="--", linewidth=1.5,
                   label=f"Low Contrast ({CONTRAST_LOW_THRESH})")
        ax.axvline(x=40, color="orange", linestyle=":", linewidth=1.2, label="Underexposed (40)")
        ax.axvline(x=240, color="purple", linestyle=":", linewidth=1.2, label="Overexposed (240)")
        ax.set_title("Contrast vs Brightness Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Brightness (Pixel Intensity 0-255)")
        ax.set_ylabel("Contrast (Standard Deviation)")
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No valid images", ha="center", va="center")

    # Panel 5: Blur / Laplacian Variance Distribution
    ax = axes[1, 1]
    if not valid_df.empty:
        # Most synthetic images have laplacian var = 0, so show a bar or hist
        lap_vals = valid_df["blur_laplacian_var"]
        ax.hist(lap_vals, bins=20, color="#1F77B4", edgecolor="white", alpha=0.8)
        ax.axvline(x=BLUR_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                   label=f"Blur Threshold ({BLUR_THRESHOLD})")
        ax.set_title("Sharpness / Laplacian Variance", fontsize=12, fontweight="bold")
        ax.set_xlabel("Laplacian Variance (Var > 100 = Sharp)")
        ax.set_ylabel("Document Count")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No valid images", ha="center", va="center")

    # Panel 6: Image Dimensions & Aspect Ratio
    ax = axes[1, 2]
    if not valid_df.empty:
        dims = valid_df[["width", "height"]].drop_duplicates()
        dim_str = [f"{int(r.width)} × {int(r.height)}" for _, r in dims.iterrows()]
        dim_counts = [len(valid_df[(valid_df['width'] == r.width) & (valid_df['height'] == r.height)])
                      for _, r in dims.iterrows()]
        bars6 = ax.bar(dim_str, dim_counts, color="#4472C4", edgecolor="white", width=0.4)
        ax.set_title("Image Dimensions (Width × Height)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Document Count")
        for b, cnt in zip(bars6, dim_counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                    f"{cnt} images", ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No valid images", ha="center", va="center")

    plt.tight_layout()
    chart_path = REPORTS_DIR / "image_quality_charts.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Charts saved       → {chart_path}")

    # Summary prints
    print("\n" + SECTION)
    print("PHASE 5 IMAGE QUALITY ANALYSIS — SUMMARY")
    print(SECTION)
    print(f"   Total images audited       : {n_total}")
    print(f"   Corrupt images quarantined : {n_corrupt} {corrupt_ids}")
    print(f"   Usable images evaluated    : {n_valid}")
    print(f"   Identified as blank page   : {n_blank} (synthetic placeholder canvas)")
    print(f"   Quality classification     : {label_counts}")
    print(f"   Recommended action count   : {action_counts}")
    print(f"\n   Reports generated:")
    print(f"     • {csv_path.name}")
    print(f"     • {json_path.name}")
    print(f"     • {chart_path.name}")
    print(f"     • Appended decisions to {LOG_PATH.name}")
    print(SECTION + "\n")


if __name__ == "__main__":
    main()
