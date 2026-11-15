"""
clean_annotations.py — MODULE 6a: ANNOTATION CLEANING (Phase 6)
─────────────────────────────────────────────────────────────
Cleans, normalizes, and audits annotation records from
data/raw/indic_ocr_dataset/annotations/annotations.jsonl.

Principles applied:
───────────────────
  • Never silently delete records: separate into clean vs quarantined files.
  • Keep both original (raw) text and normalized text.
  • Unicode NFC normalization for Indic scripts (Bengali, Hindi) and English.
  • Quarantine annotations linked to corrupt/unusable images (Phase 5).
  • Flag records with missing ground-truth text for manual completion.
  • Deduplicate redundant (document_id, field_name) records.
  • Log every action and provenance trail to reports/cleaning_log.csv.

Outputs:
────────
  data/cleaned/annotations.jsonl           — verified, cleaned dataset
  data/cleaned/annotations.csv             — tabular view of cleaned annotations
  data/cleaned/quarantined_annotations.jsonl — annotations for corrupt images
  data/cleaned/missing_gt_annotations.jsonl — flagged missing ground-truth
  data/cleaned/duplicate_annotations.jsonl  — redundant duplicates
  reports/clean_annotations_summary.json   — summary statistics
  reports/clean_annotations_charts.png     — visualization figure
  reports/cleaning_log.csv                 — appended provenance decisions

Usage:
──────
  cd Indic_OCR_Data_Engineering
  python scripts/clean_annotations.py
"""

import sys
import os
import json
import csv
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

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
import seaborn as sns

from config import (
    SYNTHETIC_DATASET_DIR, REPORTS_DIR, CLEANED_DIR,
    VALID_SCRIPTS, UNICODE_RANGES, REQUIRED_ANNOTATION_KEYS
)

RAW_JSONL_PATH = SYNTHETIC_DATASET_DIR / "annotations" / "annotations.jsonl"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
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
        "record_type":    "annotation",
        "field":          field,
        "issue":          issue,
        "action":         action,
        "proposed_value": proposed_value,
    })


def append_cleaning_log():
    """Append buffered Phase 6a decisions to cleaning_log.csv."""
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
    print(f"   ✓ Appended {len(_new_log_entries)} cleaning actions to {LOG_PATH}")


def normalize_indic_text(text: str) -> str:
    """
    Standardize Indic text:
      - Unicode NFC normalization
      - Remove zero-width & non-printable control characters
      - Normalize spaces / whitespace
    """
    if text is None:
        return ""
    # 1. Unicode NFC
    norm = unicodedata.normalize("NFC", str(text))
    # 2. Remove invisible zero-width characters (except virama/hasanta which are within blocks)
    # \u200b: zero-width space, \ufeff: BOM, \u200e/\u200f: direction marks
    norm = re.sub(r"[\u200b\u200e\u200f\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", norm)
    # 3. Collapse whitespace
    norm = " ".join(norm.split())
    return norm


def detect_script(text: str) -> str:
    """Detect dominant script from character counts."""
    if not text:
        return "unknown"
    counts = Counter()
    for char in text:
        code = ord(char)
        if 0x0980 <= code <= 0x09FF:
            counts["bn"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["hi"] += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            counts["en"] += 1

    if not counts:
        return "en" if any(c.isascii() for c in text) else "unknown"
    top_script, top_cnt = counts.most_common(1)[0]
    total_letters = sum(counts.values())
    if top_cnt / total_letters < 0.65 and len(counts) > 1:
        return "mixed"
    return top_script


def main():
    print(SECTION)
    print("PHASE 6a — ANNOTATION CLEANING & NORMALIZATION")
    print(SECTION)

    # 1. Load corrupt image list from Phase 5 report
    print("\n[1/5] Loading corrupt images list (Phase 5 report) …")
    quality_json_path = REPORTS_DIR / "image_quality.json"
    corrupt_image_ids = set()
    if quality_json_path.exists():
        with open(quality_json_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
            corrupt_image_ids = set(q_data.get("corrupt_doc_ids", []))
    else:
        corrupt_image_ids = {"INDIC_0050", "INDIC_0100", "INDIC_0199", "INDIC_0299"}
    print(f"   Corrupt image doc IDs quarantined: {sorted(corrupt_image_ids)}")

    # 2. Read raw annotations
    print("\n[2/5] Reading raw annotations from annotations.jsonl …")
    if not RAW_JSONL_PATH.exists():
        print(f"ERROR: File not found: {RAW_JSONL_PATH}")
        sys.exit(1)

    raw_records = []
    with open(RAW_JSONL_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            rec = json.loads(line_str)
            rec["_line_no"] = line_no
            raw_records.append(rec)

    total_records = len(raw_records)
    print(f"   Total raw annotation records loaded: {total_records}")

    # 3. Audit, Clean, Normalize, and Partition
    print("\n[3/5] Cleaning, normalizing text, and resolving issues …")
    clean_records       = []
    quarantined_records = []  # From corrupt images
    missing_gt_records  = []  # Missing ground-truth text
    duplicate_records   = []  # Duplicate (doc_id, field_name) pairs

    seen_pairs = set()

    for rec in raw_records:
        line_no = rec["_line_no"]
        doc_id  = rec.get("document_id", "")
        f_name  = rec.get("field_name", "")
        raw_txt = rec.get("text")
        bbox    = rec.get("bbox")
        script  = rec.get("script", "unknown")

        pair_key = (doc_id, f_name)
        record_ref = f"{doc_id}|{f_name}|L{line_no}"

        # Check 1: Quarantine if associated with corrupt image
        if doc_id in corrupt_image_ids:
            rec_quarantine = dict(rec)
            rec_quarantine["quarantine_reason"] = "CORRUPT_SOURCE_IMAGE"
            quarantined_records.append(rec_quarantine)
            log_action(record_ref, "document_id", "CORRUPT_IMAGE_ORPHAN",
                       "QUARANTINE — image file is corrupt zero-byte; record excluded from training")
            continue

        # Check 2: Duplicate annotation check (same doc_id + field_name)
        if pair_key in seen_pairs:
            rec_dup = dict(rec)
            rec_dup["duplicate_reason"] = "REDUNDANT_FIELD_OCCURRENCE"
            duplicate_records.append(rec_dup)
            log_action(record_ref, "document_id+field_name", "DUPLICATE_ANNOTATION",
                       "QUARANTINE — duplicate field instance; primary occurrence kept")
            continue
        seen_pairs.add(pair_key)

        # Check 3: Missing ground truth text
        is_missing_gt = (raw_txt is None or str(raw_txt).strip() == "")
        if is_missing_gt:
            rec_missing = dict(rec)
            rec_missing["issue"] = "EMPTY_GROUND_TRUTH"
            missing_gt_records.append(rec_missing)
            log_action(record_ref, "text", "EMPTY_GROUND_TRUTH",
                       "FLAG_FOR_ANNOTATION — retained with needs_manual_annotation=True")

        # 4. Normalize text
        norm_txt = normalize_indic_text(raw_txt) if not is_missing_gt else ""

        # 5. Script validation & alignment
        detected = detect_script(norm_txt) if norm_txt else script
        if script not in VALID_SCRIPTS:
            script_final = detected
            log_action(record_ref, "script", "INVALID_SCRIPT_LABEL",
                       f"CORRECT — replaced '{script}' with '{detected}'", detected)
        else:
            script_final = script

        clean_rec = {
            "document_id":              doc_id,
            "field_name":               f_name,
            "text_raw":                 raw_txt if raw_txt is not None else "",
            "text":                     norm_txt,
            "bbox":                     bbox,
            "script":                   script_final,
            "detected_script":          detected,
            "has_bbox":                 bbox is not None,
            "needs_manual_annotation":  is_missing_gt,
            "source_line_no":           line_no,
        }
        clean_records.append(clean_rec)

    # 4. Save cleaned and partitioned artifacts
    print("\n[4/5] Writing cleaned outputs to data/cleaned/ …")

    # Cleaned JSONL
    clean_jsonl_path = CLEANED_DIR / "annotations.jsonl"
    with open(clean_jsonl_path, "w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Clean annotations JSONL saved → {clean_jsonl_path} ({len(clean_records)} records)")

    # Cleaned CSV
    clean_csv_path = CLEANED_DIR / "annotations.csv"
    clean_df = pd.DataFrame(clean_records)
    clean_df.to_csv(clean_csv_path, index=False, encoding="utf-8")
    print(f"   ✓ Clean annotations CSV saved   → {clean_csv_path}")

    # Quarantined JSONL (corrupt image orphans)
    quarantine_path = CLEANED_DIR / "quarantined_annotations.jsonl"
    with open(quarantine_path, "w", encoding="utf-8") as f:
        for r in quarantined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Quarantined annotations saved  → {quarantine_path} ({len(quarantined_records)} records)")

    # Missing GT JSONL
    missing_gt_path = CLEANED_DIR / "missing_gt_annotations.jsonl"
    with open(missing_gt_path, "w", encoding="utf-8") as f:
        for r in missing_gt_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Missing GT records saved       → {missing_gt_path} ({len(missing_gt_records)} records)")

    # Duplicate records JSONL
    dup_path = CLEANED_DIR / "duplicate_annotations.jsonl"
    with open(dup_path, "w", encoding="utf-8") as f:
        for r in duplicate_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✓ Duplicate records saved        → {dup_path} ({len(duplicate_records)} records)")

    append_cleaning_log()

    # 5. Generate Summary and Charts
    print("\n[5/5] Generating Phase 6a summary report and visualizations …")

    # Script distribution
    script_counts = clean_df["script"].value_counts().to_dict()
    field_counts  = clean_df["field_name"].value_counts().to_dict()

    summary_data = {
        "run_timestamp": RUN_TS,
        "phase": "Phase 6a — Annotation Cleaning",
        "input_records_total": total_records,
        "clean_records_retained": len(clean_records),
        "quarantined_corrupt_records": len(quarantined_records),
        "quarantined_duplicate_records": len(duplicate_records),
        "missing_gt_flagged_records": len(missing_gt_records),
        "missing_bbox_records": int((~clean_df["has_bbox"]).sum()),
        "script_distribution": script_counts,
        "field_distribution": field_counts,
        "normalization_rules": [
            "Unicode NFC normalization",
            "Removal of non-printable / zero-width characters",
            "Whitespace collapse and trimming",
            "Preservation of raw text (text_raw) and normalized text (text)"
        ]
    }

    summary_json_path = REPORTS_DIR / "clean_annotations_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"   ✓ Summary JSON saved → {summary_json_path}")

    # Charts
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Phase 6a — Annotation Cleaning & Normalization Summary", fontsize=14, fontweight="bold")

    # Panel 1: Record disposition
    ax = axes[0]
    cats = ["Clean\nRetained", "Quarantined\n(Corrupt Img)", "Quarantined\n(Duplicates)", "Flagged\n(Missing GT)"]
    vals = [len(clean_records), len(quarantined_records), len(duplicate_records), len(missing_gt_records)]
    colors = ["#2CA02C", "#D62728", "#FF7F0E", "#FFA500"]
    bars1 = ax.bar(cats, vals, color=colors, edgecolor="white", width=0.55)
    ax.set_title("Annotation Records Disposition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Records")
    ax.set_ylim(0, max(vals) * 1.15)
    for b, v in zip(bars1, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 25,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Clean script distribution
    ax = axes[1]
    s_names = list(script_counts.keys())
    s_vals  = [script_counts[k] for k in s_names]
    s_colors = ["#1F77B4", "#2CA02C", "#FF7F0E"]
    bars2 = ax.bar([k.upper() for k in s_names], s_vals, color=s_colors[:len(s_names)], edgecolor="white", width=0.5)
    ax.set_title("Clean Script Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Records")
    for b, v in zip(bars2, s_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                f"{v}\n({v/len(clean_records)*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 3: Field name representation in clean dataset
    ax = axes[2]
    f_names = list(field_counts.keys())
    f_vals  = [field_counts[k] for k in f_names]
    ax.barh(f_names, f_vals, color="#4472C4", edgecolor="white", height=0.6)
    ax.set_title("Clean Field Frequency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    ax.invert_yaxis()

    plt.tight_layout()
    chart_path = REPORTS_DIR / "clean_annotations_charts.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Charts saved       → {chart_path}")

    # Summary prints
    print("\n" + SECTION)
    print("PHASE 6a ANNOTATION CLEANING — SUMMARY")
    print(SECTION)
    print(f"   Total raw records evaluated     : {total_records}")
    print(f"   Clean records retained          : {len(clean_records)}")
    print(f"   Quarantined corrupt img records : {len(quarantined_records)} (from 4 corrupt images)")
    print(f"   Quarantined duplicate records   : {len(duplicate_records)}")
    print(f"   Flagged missing ground truth    : {len(missing_gt_records)}")
    print(f"   Missing bounding boxes noted    : {int((~clean_df['has_bbox']).sum())}")
    print(f"   Script distribution in clean    : {script_counts}")
    print(f"\n   Outputs created in {CLEANED_DIR}:")
    print(f"     • {clean_jsonl_path.name}")
    print(f"     • {clean_csv_path.name}")
    print(f"     • {quarantine_path.name}")
    print(f"     • {missing_gt_path.name}")
    print(f"     • {dup_path.name}")
    print(f"     • Appended decisions to {LOG_PATH.name}")
    print(SECTION + "\n")


if __name__ == "__main__":
    main()
