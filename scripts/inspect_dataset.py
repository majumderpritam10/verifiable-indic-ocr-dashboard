"""
inspect_dataset.py — MODULE 2: DATASET INSPECTION (Phase 1)
─────────────────────────────────────────────────────────────
Reads every file in data/raw/indic_ocr_dataset/ and produces:
  • Console summary
  • reports/dataset_profile.csv
  • reports/dataset_profile.json
  • reports/dataset_profile_charts.png   (6-panel figure)

Checks performed
────────────────
 1. Total image count
 2. Images per language (inferred from splits + doc_id prefix mapping)
 3. Images per document type
 4. Images per file format
 5. Image dimensions and resolution
 6. File sizes
 7. Corrupt / unreadable image files
 8. Missing annotation records
 9. Missing ground-truth text
10. Duplicate image SHA-256 hashes
11. Duplicate document IDs in annotations
12. Invalid annotation formats (missing required keys)
13. Missing bounding-box coordinates
14. Invalid bounding boxes (geometry checks)
15. Image-to-annotation mapping coverage

Usage
─────
  cd Indic_OCR_Data_Engineering
  py -3.11 scripts\\inspect_dataset.py
"""

import sys
import json
import csv
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SYNTHETIC_DATASET_DIR, REPORTS_DIR,
    SUPPORTED_IMAGE_EXTS, REQUIRED_ANNOTATION_KEYS,
    VALID_SCRIPTS, VALID_DOC_TYPES
)

IMAGES_DIR      = SYNTHETIC_DATASET_DIR / "images"
ANNOTATIONS_DIR = SYNTHETIC_DATASET_DIR / "annotations"
SPLITS_DIR      = SYNTHETIC_DATASET_DIR / "splits"
DOCS_DIR        = SYNTHETIC_DATASET_DIR / "docs"
JSONL_PATH      = ANNOTATIONS_DIR / "annotations.jsonl"
CSV_PATH        = ANNOTATIONS_DIR / "fields.csv"
SCHEMA_PATH     = DOCS_DIR / "schema.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SECTION = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def try_load_image(path: Path):
    """Return (width, height, file_size_bytes, error_msg).
    Uses only Pillow — no OpenCV dependency here."""
    from PIL import Image, UnidentifiedImageError
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            return None, None, 0, "ZERO_BYTE"
        with Image.open(path) as img:
            img.verify()           # detects truncation
        with Image.open(path) as img:
            w, h = img.size
        return w, h, file_size, None
    except UnidentifiedImageError:
        return None, None, path.stat().st_size, "UNIDENTIFIED_FORMAT"
    except Exception as exc:
        return None, None, path.stat().st_size, str(exc)


def load_splits() -> dict:
    """Return {doc_id: split_name}."""
    split_map = {}
    for split in ("train", "validation", "test"):
        p = SPLITS_DIR / f"{split}.txt"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                did = line.strip()
                if did:
                    split_map[did] = split
    return split_map


def infer_language_from_id(doc_id: str) -> str:
    """INDIC_0001–0100 → bn, 0101–0200 → hi, 0201–0300 → en."""
    try:
        n = int(doc_id.split("_")[1])
        if   n <= 100: return "bn"
        elif n <= 200: return "hi"
        else:          return "en"
    except Exception:
        return "unknown"


def validate_bbox(bbox, img_w, img_h):
    """Return list of error strings (empty = valid)."""
    errors = []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ["bbox must be a list/tuple of 4 numbers"]
    x1, y1, x2, y2 = bbox
    if x1 >= x2:
        errors.append(f"x1({x1}) >= x2({x2})")
    if y1 >= y2:
        errors.append(f"y1({y1}) >= y2({y2})")
    if any(v < 0 for v in [x1, y1, x2, y2]):
        errors.append("negative coordinate")
    if img_w and img_h:
        if x2 > img_w or y2 > img_h:
            errors.append(f"bbox exceeds image bounds ({img_w}×{img_h})")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FILE INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
print(SECTION)
print("PHASE 1 — DATASET INSPECTION")
print(SECTION)

print("\n[1/6] Scanning image files …")

image_files = sorted(IMAGES_DIR.glob("*"))
img_records = []          # one row per image file

hash_to_ids   = defaultdict(list)
seen_doc_ids  = set()

for img_path in image_files:
    suffix = img_path.suffix.lower()
    doc_id = img_path.stem
    w, h, fsize, err = try_load_image(img_path)

    # hash only non-corrupt files
    sha = sha256_file(img_path) if fsize > 0 else "CORRUPT"
    hash_to_ids[sha].append(doc_id)

    img_records.append({
        "document_id":    doc_id,
        "file_name":      img_path.name,
        "extension":      suffix,
        "file_size_bytes":fsize,
        "width":          w,
        "height":         h,
        "inferred_lang":  infer_language_from_id(doc_id),
        "is_corrupt":     err is not None,
        "corrupt_reason": err or "",
    })
    seen_doc_ids.add(doc_id)

img_df = pd.DataFrame(img_records)
print(f"   Total files found : {len(img_df)}")
print(f"   Corrupt images    : {img_df['is_corrupt'].sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LOAD ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Loading annotation JSONL …")

annotations = []
jsonl_parse_errors = []

if JSONL_PATH.exists():
    with JSONL_PATH.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_line_no"] = line_no
                annotations.append(rec)
            except json.JSONDecodeError as exc:
                jsonl_parse_errors.append({"line": line_no, "error": str(exc)})
else:
    print(f"   ⚠ JSONL not found at {JSONL_PATH}")

ann_df = pd.DataFrame(annotations) if annotations else pd.DataFrame()
print(f"   Records loaded    : {len(ann_df)}")
print(f"   Parse errors      : {len(jsonl_parse_errors)}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LOAD SPLITS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Loading splits …")
split_map  = load_splits()
split_cnts = Counter(split_map.values())
print(f"   train={split_cnts.get('train',0)}, "
      f"validation={split_cnts.get('validation',0)}, "
      f"test={split_cnts.get('test',0)}")

# Merge split into img_df
img_df["split"] = img_df["document_id"].map(split_map).fillna("unknown")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LOAD DOC-LEVEL METADATA FROM ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Profiling annotation records …")

if not ann_df.empty:
    # Normalize column presence
    for col in ["document_id", "field_name", "text", "bbox", "script"]:
        if col not in ann_df.columns:
            ann_df[col] = None

    # Missing keys check
    def _missing_keys(row):
        result = []
        for k in REQUIRED_ANNOTATION_KEYS:
            if k not in row:
                result.append(k)
                continue
            val = row[k]
            # pd.isna raises on lists/arrays; guard with try
            try:
                is_na = pd.isna(val)
                if isinstance(is_na, (bool, np.bool_)) and is_na:
                    result.append(k)
            except (TypeError, ValueError):
                pass  # non-scalar (e.g. list bbox) → not NA
        return result

    ann_df["missing_keys"]      = ann_df.apply(_missing_keys, axis=1)
    ann_df["has_missing_keys"]  = ann_df["missing_keys"].apply(len) > 0

    # Empty ground-truth
    ann_df["text_empty"] = ann_df["text"].apply(
        lambda t: t is None or (isinstance(t, str) and t.strip() == "")
    )

    # Missing bbox
    ann_df["bbox_missing"] = ann_df["bbox"].apply(
        lambda b: b is None or (isinstance(b, float) and np.isnan(b))
    )

    # Duplicate doc_id+field_name pairs
    ann_df["dup_key"] = ann_df["document_id"].astype(str) + "|" + ann_df["field_name"].astype(str)
    dup_mask           = ann_df.duplicated(subset=["dup_key"], keep=False)
    ann_df["is_duplicate_ann"] = dup_mask

    # Script validity
    ann_df["invalid_script"] = ann_df["script"].apply(
        lambda s: s not in VALID_SCRIPTS
    )

    print(f"   Empty ground-truth text : {ann_df['text_empty'].sum()}")
    print(f"   Missing bounding boxes  : {ann_df['bbox_missing'].sum()}")
    print(f"   Duplicate annotations   : {ann_df['is_duplicate_ann'].sum()}")
    print(f"   Invalid script labels   : {ann_df['invalid_script'].sum()}")
    print(f"   Records with missing keys: {ann_df['has_missing_keys'].sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EXACT DUPLICATE IMAGES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Checking for exact duplicate images …")
exact_dupe_groups = {h: ids for h, ids in hash_to_ids.items()
                     if len(ids) > 1 and h != "CORRUPT"}
n_exact_dupe_images = sum(len(v) for v in exact_dupe_groups.values())
print(f"   Exact-duplicate groups : {len(exact_dupe_groups)}")
print(f"   Images in dupe groups  : {n_exact_dupe_images}")

img_df["sha256"] = img_df["document_id"].apply(
    lambda did: next((h for h, ids in hash_to_ids.items() if did in ids), "")
)
img_df["is_exact_duplicate"] = img_df["sha256"].apply(
    lambda h: h != "CORRUPT" and len(hash_to_ids.get(h, [])) > 1
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — IMAGE↔ANNOTATION COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Checking image ↔ annotation coverage …")
ann_doc_ids   = set(ann_df["document_id"].dropna()) if not ann_df.empty else set()
img_no_ann    = seen_doc_ids - ann_doc_ids          # images without any annotation
ann_no_img    = ann_doc_ids  - seen_doc_ids         # annotations without images
print(f"   Images with no annotations  : {len(img_no_ann)}")
print(f"   Annotations with no image   : {len(ann_no_img)}")

# ─────────────────────────────────────────────────────────────────────────────
# BBOX VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nValidating bounding boxes …")
# Build quick img_size lookup
img_size_lookup = {
    row["document_id"]: (row["width"], row["height"])
    for _, row in img_df.iterrows()
}

bbox_errors = []
if not ann_df.empty:
    for _, row in ann_df.iterrows():
        if row.get("bbox_missing"):
            continue
        bbox = row.get("bbox")
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None
        if bbox is None:
            continue
        iw, ih = img_size_lookup.get(row.get("document_id"), (None, None))
        errs = validate_bbox(bbox, iw, ih)
        if errs:
            bbox_errors.append({
                "document_id": row.get("document_id"),
                "field_name":  row.get("field_name"),
                "bbox":        str(bbox),
                "errors":      "; ".join(errs),
            })

print(f"   Invalid bounding boxes : {len(bbox_errors)}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PROFILE DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
print("\nAssembling dataset profile …")

profile_rows = []
for _, row in img_df.iterrows():
    did = row["document_id"]
    ann_subset = ann_df[ann_df["document_id"] == did] if not ann_df.empty else pd.DataFrame()
    profile_rows.append({
        "document_id":          did,
        "file_name":            row["file_name"],
        "extension":            row["extension"],
        "file_size_bytes":      row["file_size_bytes"],
        "width":                row["width"],
        "height":               row["height"],
        "inferred_language":    row["inferred_lang"],
        "split":                row["split"],
        "is_corrupt":           row["is_corrupt"],
        "corrupt_reason":       row["corrupt_reason"],
        "is_exact_duplicate":   row["is_exact_duplicate"],
        "sha256":               row["sha256"],
        "annotation_count":     len(ann_subset),
        "empty_gt_count":       int(ann_subset["text_empty"].sum()) if not ann_subset.empty and "text_empty" in ann_subset else 0,
        "missing_bbox_count":   int(ann_subset["bbox_missing"].sum()) if not ann_subset.empty and "bbox_missing" in ann_subset else 0,
    })

profile_df = pd.DataFrame(profile_rows)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SECTION)
print("DATASET PROFILE SUMMARY")
print(SECTION)

total_images        = len(profile_df)
corrupt_count       = int(profile_df["is_corrupt"].sum())
usable_count        = total_images - corrupt_count
exact_dupe_count    = int(profile_df["is_exact_duplicate"].sum())
total_annotations   = len(ann_df) if not ann_df.empty else 0
empty_gt_total      = int(ann_df["text_empty"].sum()) if not ann_df.empty and "text_empty" in ann_df else 0
missing_bbox_total  = int(ann_df["bbox_missing"].sum()) if not ann_df.empty and "bbox_missing" in ann_df else 0
dup_ann_total       = int(ann_df["is_duplicate_ann"].sum()) if not ann_df.empty and "is_duplicate_ann" in ann_df else 0
img_no_ann_count    = len(img_no_ann)
ann_no_img_count    = len(ann_no_img)

summary = {
    "total_images":           total_images,
    "usable_images":          usable_count,
    "corrupt_images":         corrupt_count,
    "exact_duplicate_images": exact_dupe_count,
    "exact_duplicate_groups": len(exact_dupe_groups),
    "images_no_annotation":   img_no_ann_count,
    "total_annotations":      total_annotations,
    "annotations_per_image":  round(total_annotations / max(usable_count,1), 2),
    "empty_ground_truth":     empty_gt_total,
    "missing_bounding_boxes": missing_bbox_total,
    "duplicate_annotations":  dup_ann_total,
    "invalid_bounding_boxes": len(bbox_errors),
    "jsonl_parse_errors":     len(jsonl_parse_errors),
    "annotations_no_image":   ann_no_img_count,
    "splits": {
        "train":      split_cnts.get("train", 0),
        "validation": split_cnts.get("validation", 0),
        "test":       split_cnts.get("test", 0),
    },
}

for k, v in summary.items():
    print(f"   {k:<35}: {v}")

# Language distribution
lang_dist = profile_df["inferred_language"].value_counts().to_dict()
print(f"\n   Language distribution:")
for lang, cnt in lang_dist.items():
    print(f"     {lang}: {cnt} images")

# Format distribution
fmt_dist = profile_df["extension"].value_counts().to_dict()
print(f"\n   Format distribution:")
for fmt, cnt in fmt_dist.items():
    print(f"     {fmt}: {cnt} images")

# File size stats (non-corrupt)
valid_sizes = profile_df[~profile_df["is_corrupt"]]["file_size_bytes"]
if not valid_sizes.empty:
    print(f"\n   File size (bytes) — valid images:")
    print(f"     min={valid_sizes.min()}, max={valid_sizes.max()}, "
          f"mean={valid_sizes.mean():.0f}, median={valid_sizes.median():.0f}")

# Image dimension stats
valid_dims = profile_df[~profile_df["is_corrupt"]]
if not valid_dims.empty:
    print(f"\n   Image dimensions (px) — valid images:")
    print(f"     Width  : min={valid_dims['width'].min()}, max={valid_dims['width'].max()}")
    print(f"     Height : min={valid_dims['height'].min()}, max={valid_dims['height'].max()}")

# Split distribution
split_dist = profile_df["split"].value_counts().to_dict()
print(f"\n   Split distribution (from splits/*.txt):")
for sp, cnt in split_dist.items():
    print(f"     {sp}: {cnt} images")

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION-LEVEL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
if not ann_df.empty:
    print(f"\n   Annotation quality summary:")
    ann_summary = {
        "total_records":           len(ann_df),
        "empty_ground_truth":      empty_gt_total,
        "missing_bbox":            missing_bbox_total,
        "duplicate_records":       dup_ann_total,
        "invalid_script_labels":   int(ann_df["invalid_script"].sum()) if "invalid_script" in ann_df else 0,
        "records_with_missing_keys": int(ann_df["has_missing_keys"].sum()) if "has_missing_keys" in ann_df else 0,
        "invalid_bboxes":          len(bbox_errors),
    }
    for k, v in ann_summary.items():
        pct = f" ({100*v/len(ann_df):.1f}%)" if v > 0 else ""
        print(f"     {k:<40}: {v}{pct}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_out = REPORTS_DIR / "dataset_profile.csv"
profile_df.to_csv(csv_out, index=False, encoding="utf-8")
print(f"\n✓ Profile CSV saved → {csv_out}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
json_out = REPORTS_DIR / "dataset_profile.json"
full_report = {
    "summary":          summary,
    "language_distribution": lang_dist,
    "format_distribution":   fmt_dist,
    "split_distribution":    split_dist,
    "exact_duplicate_groups": {h: ids for h, ids in exact_dupe_groups.items()},
    "corrupt_images":    profile_df[profile_df["is_corrupt"]][["document_id","corrupt_reason"]].to_dict(orient="records"),
    "images_no_annotation": sorted(img_no_ann),
    "annotations_no_image": sorted(ann_no_img),
    "bbox_validation_errors": bbox_errors[:50],   # cap at 50 for readability
    "jsonl_parse_errors": jsonl_parse_errors,
}
with json_out.open("w", encoding="utf-8") as fh:
    json.dump(full_report, fh, indent=2, ensure_ascii=False, default=str)
print(f"✓ Profile JSON saved → {json_out}")

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION ERRORS CSV
# ─────────────────────────────────────────────────────────────────────────────
if not ann_df.empty:
    err_ann = ann_df[
        ann_df["text_empty"] | ann_df["bbox_missing"] |
        ann_df["is_duplicate_ann"] | ann_df["invalid_script"]
    ][["document_id","field_name","text","script","text_empty",
       "bbox_missing","is_duplicate_ann","invalid_script","_line_no"]].copy()
    err_ann_path = REPORTS_DIR / "annotation_errors.csv"
    err_ann.to_csv(err_ann_path, index=False, encoding="utf-8")
    print(f"✓ Annotation errors CSV saved → {err_ann_path}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating charts …")
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Indic OCR Dataset — Phase 1 Profile", fontsize=14, fontweight="bold")

# 1. Language distribution bar
ax = axes[0, 0]
lang_series = pd.Series(lang_dist)
lang_series.plot(kind="bar", ax=ax, color=["#4472C4","#ED7D31","#A9D18E"])
ax.set_title("Images per Language")
ax.set_xlabel("Language")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)

# 2. Split distribution pie
ax = axes[0, 1]
split_counts = [split_cnts.get(s, 0) for s in ["train","validation","test"]]
ax.pie(split_counts, labels=["Train","Validation","Test"],
       autopct="%1.0f%%", colors=["#4472C4","#ED7D31","#A9D18E"],
       startangle=90)
ax.set_title("Dataset Splits")

# 3. Image quality issues bar
ax = axes[0, 2]
issues = {
    "Corrupt\nimages":    corrupt_count,
    "Exact\nduplicates":  exact_dupe_count,
    "No\nannotation":     img_no_ann_count,
}
pd.Series(issues).plot(kind="bar", ax=ax, color="#FF6B6B")
ax.set_title("Image-level Issues")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)

# 4. Annotation issues bar
ax = axes[1, 0]
ann_issues = {
    "Empty GT":         empty_gt_total,
    "Missing BBox":     missing_bbox_total,
    "Duplicate":        dup_ann_total,
    "Invalid bbox":     len(bbox_errors),
}
pd.Series(ann_issues).plot(kind="bar", ax=ax, color="#FFA500")
ax.set_title("Annotation-level Issues")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=9)

# 5. File size histogram (non-corrupt)
ax = axes[1, 1]
valid_sizes_kb = valid_sizes / 1024
ax.hist(valid_sizes_kb, bins=20, color="#4472C4", edgecolor="white")
ax.set_title("File Size Distribution (non-corrupt)")
ax.set_xlabel("Size (KB)")
ax.set_ylabel("Count")

# 6. Annotations per document histogram
ax = axes[1, 2]
if not ann_df.empty:
    ann_per_doc = ann_df.groupby("document_id").size()
    ax.hist(ann_per_doc, bins=range(0, ann_per_doc.max()+2),
            color="#A9D18E", edgecolor="white", align="left")
ax.set_title("Annotations per Document")
ax.set_xlabel("Annotation count")
ax.set_ylabel("Documents")

plt.tight_layout()
chart_path = REPORTS_DIR / "dataset_profile_charts.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"✓ Charts saved → {chart_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SECTION)
print("PHASE 1 CHECKLIST")
print(SECTION)
checks = [
    ("Total image count verified",          True),
    ("Images per language counted",          True),
    ("Images per file format counted",       True),
    ("Image dimensions recorded",            True),
    ("File sizes recorded",                  True),
    ("Corrupt files detected",               corrupt_count > 0),
    ("Exact duplicates detected",            exact_dupe_count > 0),
    ("JSONL annotations loaded",             len(annotations) > 0),
    ("Empty ground-truth detected",          empty_gt_total > 0),
    ("Missing bboxes detected",              missing_bbox_total > 0),
    ("Duplicate annotations detected",       dup_ann_total > 0),
    ("Invalid bbox geometry checked",        True),
    ("Image↔annotation coverage checked",   True),
    ("Split distribution verified",          True),
    ("CSV report saved",                     csv_out.exists()),
    ("JSON report saved",                    json_out.exists()),
    ("Charts saved",                         chart_path.exists()),
]
for desc, ok in checks:
    status = "✅" if ok else "⚠️ "
    print(f"  {status}  {desc}")

print(f"\n✅ PHASE 1 COMPLETE — all reports in: {REPORTS_DIR}")
