"""
duplicate_detection.py — MODULE 4: DUPLICATE DETECTION (Phase 4)
─────────────────────────────────────────────────────────────────
Detects five classes of duplicates:

  1. Exact duplicate image files   — SHA-256 hash match
  2. Near-duplicate images         — Perceptual hash (pHash) Hamming distance
  3. Duplicate document IDs        — Same doc_id appears > once in image folder
  4. Duplicate annotation records  — Same (document_id, field_name) pair
  5. Duplicate text samples        — Identical normalized ground-truth strings

Design rules
────────────
  • Near-duplicate images are FLAGGED, not removed.
    Two documents may share the same printed template but contain
    different field values — only a human can confirm they are truly redundant.
  • Exact image duplicates that are confirmed identical (same SHA-256 AND
    same doc_id assignment) are recorded but not auto-deleted.
  • All decisions are appended to reports/cleaning_log.csv.
  • The raw dataset is never modified.

Outputs
───────
  reports/duplicates.csv           — per-record duplicate flags
  reports/duplicates.json          — grouped duplicate clusters
  reports/duplicate_charts.png     — visual summary
  reports/cleaning_log.csv         — appended with Phase 4 decisions

Usage
─────
  cd Indic_OCR_Data_Engineering
  py -3.11 scripts\\duplicate_detection.py
"""

import sys
import json
import hashlib
import unicodedata
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── project root ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SYNTHETIC_DATASET_DIR, REPORTS_DIR,
    PHASH_THRESHOLD, SUPPORTED_IMAGE_EXTS
)

IMAGES_DIR      = SYNTHETIC_DATASET_DIR / "images"
ANNOTATIONS_DIR = SYNTHETIC_DATASET_DIR / "annotations"
JSONL_PATH      = ANNOTATIONS_DIR / "annotations.jsonl"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SECTION = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING LOG (append to existing log from Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH = REPORTS_DIR / "cleaning_log.csv"
_new_log_entries = []

def log_action(record_id, record_type, field, issue, action, proposed_value=""):
    _new_log_entries.append({
        "timestamp":      RUN_TS,
        "record_id":      str(record_id),
        "record_type":    record_type,
        "field":          field,
        "issue":          issue,
        "action":         action,
        "proposed_value": str(proposed_value),
    })

def flush_log():
    """Append new entries to the existing cleaning_log.csv."""
    if not _new_log_entries:
        return
    df = pd.DataFrame(_new_log_entries)
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(LOG_PATH, index=False, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of file, or None if file is unreadable/corrupt."""
    try:
        if path.stat().st_size == 0:
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def compute_phash(path: Path):
    """
    Compute a 64-bit perceptual hash using only Pillow (no imagehash lib).
    Algorithm: resize to 32×32 greyscale, compute DCT-like mean hash
    over 8×8 blocks → returns a 64-bit integer.

    Falls back to a pure-Python dHash if Pillow is unavailable.
    Returns None for corrupt files.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("L").resize((32, 32), Image.LANCZOS)
            pixels = list(img.getdata())

        # Build 8×8 block averages (each block = 4×4 pixels in the 32×32 grid)
        blocks = []
        for block_r in range(8):
            for block_c in range(8):
                vals = []
                for pr in range(block_r * 4, (block_r + 1) * 4):
                    for pc in range(block_c * 4, (block_c + 1) * 4):
                        vals.append(pixels[pr * 32 + pc])
                blocks.append(sum(vals) / len(vals))

        mean_val = sum(blocks) / len(blocks)
        bits = [1 if b >= mean_val else 0 for b in blocks]
        hash_int = int("".join(str(b) for b in bits), 2)
        return hash_int
    except Exception:
        return None


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def normalize_text(text: str) -> str:
    """NFC-normalize and strip for duplicate text comparison."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text.strip().lower())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print(SECTION)
print("PHASE 4 — DUPLICATE DETECTION")
print(SECTION)

print("\n[1/6] Loading image files …")
image_paths = sorted(IMAGES_DIR.glob("*.png"))
print(f"   Total image files : {len(image_paths)}")

print("\n[2/6] Loading annotation JSONL …")
annotations = []
with JSONL_PATH.open("r", encoding="utf-8") as fh:
    for ln, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rec["_line_no"] = ln
            annotations.append(rec)
        except json.JSONDecodeError:
            pass
ann_df = pd.DataFrame(annotations)
print(f"   Annotation records: {len(ann_df)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXACT DUPLICATE IMAGES (SHA-256)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Computing SHA-256 hashes (exact duplicates) …")

hash_records = []
hash_to_paths = defaultdict(list)

for path in image_paths:
    sha = sha256_file(path)
    hash_records.append({
        "document_id": path.stem,
        "file_name":   path.name,
        "sha256":      sha if sha else "CORRUPT",
        "is_corrupt":  sha is None,
    })
    if sha:
        hash_to_paths[sha].append(path.stem)

hash_df = pd.DataFrame(hash_records)

# Groups with > 1 image
exact_dupe_groups = {h: ids for h, ids in hash_to_paths.items() if len(ids) > 1}
n_exact_dupe_images = sum(len(v) for v in exact_dupe_groups.values())

hash_df["exact_duplicate"]       = hash_df["sha256"].apply(
    lambda h: h != "CORRUPT" and h in exact_dupe_groups
)
hash_df["exact_duplicate_group"] = hash_df["sha256"].apply(
    lambda h: list(exact_dupe_groups.get(h, []))
)

print(f"   Unique SHA-256 hashes          : {hash_df[~hash_df['is_corrupt']]['sha256'].nunique()}")
print(f"   Exact duplicate groups         : {len(exact_dupe_groups)}")
print(f"   Images in exact dupe groups    : {n_exact_dupe_images}")

# Log
for h, ids in exact_dupe_groups.items():
    for did in ids:
        log_action(did, "image", "sha256", "EXACT_DUPLICATE",
                   "FLAGGED — keep all, mark group; human review before removal",
                   proposed_value=f"group_hash={h[:16]}…")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — NEAR-DUPLICATE IMAGES (perceptual hash)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Computing perceptual hashes (near-duplicates) …")
print(f"   Hamming distance threshold : {PHASH_THRESHOLD}")

phash_records = []
valid_paths = [p for p in image_paths if sha256_file(p) is not None]
print(f"   Processing {len(valid_paths)} non-corrupt images …")

for path in valid_paths:
    ph = compute_phash(path)
    phash_records.append({
        "document_id": path.stem,
        "phash":       ph,
    })

phash_df = pd.DataFrame(phash_records).dropna(subset=["phash"])
phash_df["phash"] = phash_df["phash"].astype(int)

# O(n²) pairwise comparison — tractable for ≤300 images on i3 / 8GB RAM
near_dupe_pairs   = []
near_dupe_set     = set()

rows = phash_df.to_dict(orient="records")
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        dist = hamming_distance(rows[i]["phash"], rows[j]["phash"])
        if dist <= PHASH_THRESHOLD:
            did_a = rows[i]["document_id"]
            did_b = rows[j]["document_id"]
            near_dupe_pairs.append({
                "doc_id_a":        did_a,
                "doc_id_b":        did_b,
                "hamming_distance": dist,
                "is_exact":        dist == 0,
            })
            near_dupe_set.add(did_a)
            near_dupe_set.add(did_b)

near_dupe_df = pd.DataFrame(near_dupe_pairs)

print(f"   Near-duplicate pairs found     : {len(near_dupe_pairs)}")
print(f"   Unique images in near-dup pairs: {len(near_dupe_set)}")

# Add near-dup flag to hash_df
hash_df["near_duplicate"] = hash_df["document_id"].isin(near_dupe_set)

# Log
for _, row in near_dupe_df.iterrows():
    log_action(
        f"{row['doc_id_a']}↔{row['doc_id_b']}",
        "image", "phash", "NEAR_DUPLICATE",
        "FLAGGED — same split assignment required; human must confirm if truly redundant",
        proposed_value=f"hamming={row['hamming_distance']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DUPLICATE DOCUMENT IDs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Checking duplicate document IDs on disk …")

# In a well-formed dataset doc_ids are unique by filename; check anyway
doc_id_counts = hash_df["document_id"].value_counts()
dup_doc_ids   = doc_id_counts[doc_id_counts > 1]
print(f"   Duplicate document IDs on disk : {len(dup_doc_ids)}")

for did in dup_doc_ids.index:
    log_action(did, "image", "document_id", "DUPLICATE_DOC_ID",
               "FLAG — resolve naming conflict before integration")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DUPLICATE ANNOTATION RECORDS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Checking duplicate annotation records …")

# 5a — Exact (document_id + field_name) duplicates
ann_df["dup_key"]         = ann_df["document_id"].astype(str) + "|" + ann_df["field_name"].astype(str)
ann_df["is_dup_ann"]      = ann_df.duplicated(subset=["dup_key"], keep=False)
n_dup_ann                 = int(ann_df["is_dup_ann"].sum())

# Group them
dup_ann_groups = (
    ann_df[ann_df["is_dup_ann"]]
    .groupby("dup_key")["_line_no"]
    .apply(list)
    .reset_index()
    .rename(columns={"_line_no": "line_numbers"})
)

print(f"   Duplicate (doc_id+field_name) records : {n_dup_ann}")
print(f"   Unique duplicate keys                 : {len(dup_ann_groups)}")

for _, row in ann_df[ann_df["is_dup_ann"]].iterrows():
    log_action(
        f"{row['document_id']}|{row['field_name']}|L{row['_line_no']}",
        "annotation", "document_id+field_name", "DUPLICATE_ANNOTATION",
        "FLAGGED — keep first occurrence; mark others for review",
        proposed_value=f"dup_key={row['dup_key']}"
    )

# 5b — Duplicate normalized text strings (across all annotations)
ann_df["normalized_text"] = ann_df["text"].apply(
    lambda t: normalize_text(str(t)) if pd.notna(t) else ""
)
# Only flag non-empty text
nonempty_text             = ann_df[ann_df["normalized_text"] != ""]
text_counts               = nonempty_text["normalized_text"].value_counts()
dup_texts                 = text_counts[text_counts > 1]
n_dup_text_records        = int(nonempty_text["normalized_text"].isin(dup_texts.index).sum())

ann_df["is_dup_text"] = ann_df["normalized_text"].apply(
    lambda t: t != "" and t in dup_texts.index
)

print(f"   Duplicate normalized text strings     : {len(dup_texts)}")
print(f"   Annotation records with duplicate text: {n_dup_text_records}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD COMBINED DUPLICATE REPORT
# ─────────────────────────────────────────────────────────────────────────────
# Per-image duplicate summary
dup_img_df = hash_df[["document_id","sha256","exact_duplicate","near_duplicate","is_corrupt"]].copy()
dup_img_df["exact_dup_group_size"] = dup_img_df["sha256"].apply(
    lambda h: len(exact_dupe_groups.get(h, [])) if h != "CORRUPT" else 0
)

# Per-annotation duplicate summary
dup_ann_out = ann_df[["document_id","field_name","text","normalized_text",
                       "is_dup_ann","is_dup_text","_line_no"]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
# Image-level duplicates
dup_img_path = REPORTS_DIR / "duplicates.csv"
dup_img_df.to_csv(dup_img_path, index=False, encoding="utf-8")
print(f"\n✓ Duplicates CSV (images)       → {dup_img_path}")

# Near-duplicate pairs
if not near_dupe_df.empty:
    nd_path = REPORTS_DIR / "near_duplicate_pairs.csv"
    near_dupe_df.to_csv(nd_path, index=False, encoding="utf-8")
    print(f"✓ Near-duplicate pairs CSV      → {nd_path}")

# Annotation duplicates
dup_ann_path = REPORTS_DIR / "duplicate_annotations.csv"
dup_ann_out[dup_ann_out["is_dup_ann"] | dup_ann_out["is_dup_text"]].to_csv(
    dup_ann_path, index=False, encoding="utf-8"
)
print(f"✓ Duplicate annotations CSV     → {dup_ann_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
dup_json = {
    "run_timestamp": RUN_TS,
    "image_duplicates": {
        "exact_duplicate_groups":       {h[:16]+"…": ids for h, ids in list(exact_dupe_groups.items())[:10]},
        "total_exact_dupe_groups":      len(exact_dupe_groups),
        "total_exact_dupe_images":      n_exact_dupe_images,
        "total_near_dupe_pairs":        len(near_dupe_pairs),
        "total_near_dupe_images":       len(near_dupe_set),
        "near_dupe_threshold_hamming":  PHASH_THRESHOLD,
        "near_duplicate_pairs_sample":  near_dupe_pairs[:20],
    },
    "annotation_duplicates": {
        "duplicate_doc_id_field_pairs": n_dup_ann,
        "duplicate_text_strings":       len(dup_texts),
        "duplicate_text_records":       n_dup_text_records,
        "duplicate_groups": dup_ann_groups.head(20).to_dict(orient="records"),
    },
    "handling_rules": {
        "exact_duplicate_images":    "FLAGGED — all copies kept; mark group; require human confirmation before deletion",
        "near_duplicate_images":     "FLAGGED — keep all copies in SAME split to prevent leakage; human confirms if redundant",
        "duplicate_doc_ids":         "FLAG — resolve naming conflict before integration",
        "duplicate_annotations":     "FLAGGED — keep first occurrence; mark remaining for human review",
        "duplicate_text":            "INFORMATIONAL — same text may appear in different documents legitimately",
    },
}

dup_json_path = REPORTS_DIR / "duplicates.json"
with dup_json_path.open("w", encoding="utf-8") as fh:
    json.dump(dup_json, fh, indent=2, ensure_ascii=False, default=str)
print(f"✓ Duplicates JSON               → {dup_json_path}")

# Flush cleaning log
flush_log()
print(f"✓ Cleaning log appended         → {LOG_PATH}  (+{len(_new_log_entries)} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating charts …")
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Phase 4 — Duplicate Detection", fontsize=14, fontweight="bold")

# Panel 1: Duplicate image breakdown (stacked bar)
ax = axes[0]
categories  = ["Clean", "Corrupt", "Exact Dupe", "Near Dupe"]
corrupt_cnt = int(hash_df["is_corrupt"].sum())
exact_cnt   = int(hash_df["exact_duplicate"].sum())
near_cnt    = int(hash_df["near_duplicate"].sum())
clean_cnt   = len(hash_df) - corrupt_cnt - max(0, exact_cnt - near_cnt)

counts  = [clean_cnt, corrupt_cnt, exact_cnt, near_cnt]
colors  = ["#4472C4","#FF6B6B","#FFA500","#FFD700"]
bars    = ax.bar(categories, counts, color=colors, edgecolor="white", width=0.6)
ax.set_title("Image Duplicate Categories")
ax.set_ylabel("Count")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")

# Panel 2: Near-duplicate Hamming distance distribution
ax = axes[1]
if not near_dupe_df.empty:
    dist_counts = near_dupe_df["hamming_distance"].value_counts().sort_index()
    ax.bar(dist_counts.index, dist_counts.values, color="#4472C4", edgecolor="white")
    ax.axvline(x=PHASH_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold={PHASH_THRESHOLD}")
    ax.set_title("Near-Duplicate Hamming Distance Distribution")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Pair Count")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No near-duplicate\npairs detected",
            ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_title("Near-Duplicate Hamming Distance")

# Panel 3: Annotation duplicate summary
ax = axes[2]
ann_cats   = ["Total\nAnnotations", "Duplicate\n(doc+field)", "Duplicate\nText"]
ann_counts = [len(ann_df), n_dup_ann, n_dup_text_records]
ann_colors = ["#4472C4","#FF6B6B","#FFA500"]
bars2 = ax.bar(ann_cats, ann_counts, color=ann_colors, edgecolor="white", width=0.6)
ax.set_title("Annotation Duplicate Summary")
ax.set_ylabel("Count")
for bar, cnt in zip(bars2, ann_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
chart_path = REPORTS_DIR / "duplicate_charts.png"
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"✓ Charts saved → {chart_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY + CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SECTION)
print("PHASE 4 DUPLICATE DETECTION — SUMMARY")
print(SECTION)
print(f"\n   Total images scanned              : {len(image_paths)}")
print(f"   Corrupt (skipped for hashing)     : {corrupt_cnt}")
print(f"   Exact duplicate groups            : {len(exact_dupe_groups)}")
print(f"   Images in exact dupe groups       : {n_exact_dupe_images}")
print(f"   Near-duplicate pairs (phash)      : {len(near_dupe_pairs)}")
print(f"   Images in near-dup pairs          : {len(near_dupe_set)}")
print(f"   Duplicate doc IDs on disk         : {len(dup_doc_ids)}")
print(f"   Duplicate annotation records      : {n_dup_ann}")
print(f"   Duplicate normalized text strings : {len(dup_texts)}")

print("\n── PHASE 4 CHECKLIST ───────────────────────────────────────")
checks = [
    ("SHA-256 exact duplicate detection",         True),
    ("Perceptual hash near-duplicate detection",  True),
    ("Duplicate document ID check",               True),
    ("Duplicate annotation record check",         True),
    ("Duplicate text sample check",               True),
    ("Exact dupe groups detected",                len(exact_dupe_groups) > 0),
    ("Near-dup pairs detected",                   len(near_dupe_pairs) > 0),
    ("Duplicate annotations detected",            n_dup_ann > 0),
    ("All duplicates flagged (not auto-deleted)", True),
    ("Cleaning log appended",                     LOG_PATH.exists()),
    ("duplicates.csv saved",                      dup_img_path.exists()),
    ("duplicates.json saved",                     dup_json_path.exists()),
    ("duplicate_annotations.csv saved",           dup_ann_path.exists()),
    ("Charts saved",                              chart_path.exists()),
]
for desc, ok in checks:
    print(f"  {'✅' if ok else '⚠️ '}  {desc}")

print(f"\n✅ PHASE 4 COMPLETE — reports in: {REPORTS_DIR}")
