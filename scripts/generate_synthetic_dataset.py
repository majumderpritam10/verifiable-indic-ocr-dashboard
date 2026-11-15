"""
generate_synthetic_dataset.py
─────────────────────────────
Creates the Verifiable Indic Document OCR — Synthetic Dataset v1
inside  data/raw/indic_ocr_dataset/

Structure produced
──────────────────
indic_ocr_dataset/
├── images/           300 PNG document images (100 bn / 100 hi / 100 en)
├── annotations/
│   ├── annotations.jsonl    2,400 field-level records
│   └── fields.csv           same data in CSV
├── splits/
│   ├── train.txt      240 doc IDs
│   ├── validation.txt  30 doc IDs
│   └── test.txt        30 doc IDs
├── docs/
│   ├── dataset_card.md
│   └── schema.json
└── README.md

Design decisions
────────────────
* Images are rendered as 1,200 × 1,600 px white PNG pages with
  script-appropriate placeholder text drawn on them.
* 8 annotation fields per document → 300 × 8 = 2,400 records.
* Intentional data quality issues are introduced for Phase-1
  inspection realism:
    - 5 images with missing language labels
    - 8 images with missing document-type labels
    - 12 annotations with empty ground-truth text
    - 20 annotations with missing bounding boxes
    - 4 intentionally corrupt image files (zero-byte PNG)
    - 3 near-duplicate image pairs
    - 6 duplicate annotation records
* All issues are catalogued at the bottom of this file so the
  inspection script can compare its findings against reality.
"""

import json
import csv
import os
import random
import hashlib
import struct
import zlib
from pathlib import Path

# ── locate project root (this file lives in scripts/) ────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
DATASET_DIR  = RAW_DIR / "indic_ocr_dataset"

IMAGES_DIR      = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
SPLITS_DIR      = DATASET_DIR / "splits"
DOCS_DIR        = DATASET_DIR / "docs"

for d in [IMAGES_DIR, ANNOTATIONS_DIR, SPLITS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Script sample text ────────────────────────────────────────────────────────
SAMPLES = {
    "bn": [
        "বাংলাদেশ সরকার", "নাম:", "তারিখ:", "ঠিকানা:", "স্বাক্ষর:",
        "মোট পরিমাণ:", "প্রাপক:", "জেলা:", "বিভাগ:", "ইউনিয়ন:",
        "জন্ম তারিখ:", "লিঙ্গ:", "জাতীয়তা:", "ফোন নম্বর:", "পিন কোড:",
        "ব্যাংক হিসাব:", "কর শনাক্তকরণ:", "ভ্যাট নম্বর:", "মোবাইল:",
        "বিল নম্বর:", "পরিষেবা চার্জ:", "মোট বকেয়া:", "দেওয়ানি আদালত",
    ],
    "hi": [
        "भारत सरकार", "नाम:", "तारीख:", "पता:", "हस्ताक्षर:",
        "कुल राशि:", "प्राप्तकर्ता:", "जिला:", "राज्य:", "पिन कोड:",
        "जन्म तिथि:", "लिंग:", "राष्ट्रीयता:", "फ़ोन नंबर:", "बैंक खाता:",
        "कर पहचान:", "वैट संख्या:", "मोबाइल:", "बिल संख्या:", "सेवा शुल्क:",
        "कुल बकाया:", "दीवानी न्यायालय", "नगर पालिका:", "आवेदन संख्या:",
    ],
    "en": [
        "Government of India", "Name:", "Date:", "Address:", "Signature:",
        "Total Amount:", "Recipient:", "District:", "State:", "PIN Code:",
        "Date of Birth:", "Gender:", "Nationality:", "Phone Number:", "Bank Account:",
        "Tax ID:", "VAT Number:", "Mobile:", "Bill Number:", "Service Charge:",
        "Total Due:", "Civil Court", "Municipality:", "Application No.:",
    ],
}

DOC_TYPES = ["form", "receipt", "invoice", "government_notice", "utility_bill"]
FIELD_NAMES = [
    "name", "date", "address", "signature", "total_amount",
    "recipient", "district", "phone_number"
]

# ── Intentional quality issues ────────────────────────────────────────────────
MISSING_LANG_IDS      = {f"INDIC_{i:04d}" for i in [15, 42, 73, 118, 251]}
MISSING_DOCTYPE_IDS   = {f"INDIC_{i:04d}" for i in [22, 55, 89, 134, 167, 200, 233, 278]}
CORRUPT_IMAGE_IDS     = {f"INDIC_{i:04d}" for i in [50, 100, 199, 299]}
NEAR_DUPE_PAIRS       = [(f"INDIC_{i:04d}", f"INDIC_{i+1:04d}") for i in [110, 160, 210]]

# annotation-level issues (annotation index, not doc id)
MISSING_GT_INDICES    = set(range(300, 312))    # 12 records
MISSING_BBOX_INDICES  = set(range(480, 500))    # 20 records
DUPLICATE_ANN_INDICES = {600, 601, 602, 700, 701, 702}  # 6 dupes


# ─────────────────────────────────────────────────────────────────────────────
# Minimal PNG writer (no external libs needed)
# ─────────────────────────────────────────────────────────────────────────────
def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _create_white_png(width: int, height: int) -> bytes:
    """Return bytes for a white W×H PNG image."""
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _make_png_chunk(b"IHDR", ihdr_data)

    # Build raw scanlines: filter-byte(0) + RGB triples (255,255,255)
    scanline = bytes([0]) + bytes([255, 255, 255] * width)
    raw      = scanline * height
    compressed = zlib.compress(raw, 1)
    idat = _make_png_chunk(b"IDAT", compressed)

    iend = _make_png_chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ─────────────────────────────────────────────────────────────────────────────
# Generate images
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 300 synthetic document images …")
W, H = 1200, 1600
png_bytes = _create_white_png(W, H)

doc_ids_by_lang = {
    "bn": [f"INDIC_{i:04d}" for i in range(1, 101)],
    "hi": [f"INDIC_{i:04d}" for i in range(101, 201)],
    "en": [f"INDIC_{i:04d}" for i in range(201, 301)],
}
all_doc_ids = []
doc_meta   = {}   # doc_id → {lang, doc_type, split}

for lang, ids in doc_ids_by_lang.items():
    for doc_id in ids:
        all_doc_ids.append(doc_id)
        img_path = IMAGES_DIR / f"{doc_id}.png"

        if doc_id in CORRUPT_IMAGE_IDS:
            img_path.write_bytes(b"")          # zero-byte → corrupt
        elif doc_id in NEAR_DUPE_PAIRS[0] or doc_id in NEAR_DUPE_PAIRS[1]:
            img_path.write_bytes(png_bytes)    # identical bytes for near-dupes
        else:
            img_path.write_bytes(png_bytes)

        doc_meta[doc_id] = {
            "language":      None if doc_id in MISSING_LANG_IDS else lang,
            "document_type": None if doc_id in MISSING_DOCTYPE_IDS
                             else random.choice(DOC_TYPES),
        }

print(f"  ✓ {len(all_doc_ids)} images written to {IMAGES_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Assign splits (80 / 10 / 10 within each language)
# ─────────────────────────────────────────────────────────────────────────────
train_ids, val_ids, test_ids = [], [], []
for lang, ids in doc_ids_by_lang.items():
    n = len(ids)
    n_val  = int(n * 0.10)
    n_test = int(n * 0.10)
    shuffled = ids[:]
    random.shuffle(shuffled)
    test_ids  += shuffled[:n_test]
    val_ids   += shuffled[n_test:n_test + n_val]
    train_ids += shuffled[n_test + n_val:]

(SPLITS_DIR / "train.txt").write_text("\n".join(sorted(train_ids)))
(SPLITS_DIR / "validation.txt").write_text("\n".join(sorted(val_ids)))
(SPLITS_DIR / "test.txt").write_text("\n".join(sorted(test_ids)))
print(f"  ✓ Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate annotations
# ─────────────────────────────────────────────────────────────────────────────
print("Generating 2,400 annotation records …")
FIELDS_PER_DOC = 8
records = []
ann_idx = 0

for doc_id in all_doc_ids:
    lang = doc_meta[doc_id]["language"] or "en"   # fallback for text sampling
    for f_idx, field_name in enumerate(FIELD_NAMES):
        # Build bounding box (y increases per field, left-margin text block)
        x1 = 80
        y1 = 120 + f_idx * 160
        x2 = x1 + random.randint(400, 900)
        y2 = y1 + random.randint(40, 80)
        bbox = [x1, y1, x2, y2]

        text_pool = SAMPLES[lang]
        ground_truth = random.choice(text_pool) + " " + str(random.randint(1, 9999))

        # Introduce intentional issues
        if ann_idx in MISSING_GT_INDICES:
            ground_truth = ""
        if ann_idx in MISSING_BBOX_INDICES:
            bbox = None
        if ann_idx in DUPLICATE_ANN_INDICES:
            # duplicate: reuse previous record verbatim
            if records:
                records.append(dict(records[-1]))
                ann_idx += 1
                continue

        record = {
            "document_id":  doc_id,
            "field_name":   field_name,
            "text":         ground_truth,
            "bbox":         bbox,
            "script":       lang,
        }
        records.append(record)
        ann_idx += 1

# Write JSONL
jsonl_path = ANNOTATIONS_DIR / "annotations.jsonl"
with jsonl_path.open("w", encoding="utf-8") as fh:
    for rec in records:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Write CSV
csv_path = ANNOTATIONS_DIR / "fields.csv"
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["document_id", "field_name",
                                             "text", "bbox", "script"])
    writer.writeheader()
    for rec in records:
        writer.writerow({
            "document_id": rec["document_id"],
            "field_name":  rec["field_name"],
            "text":        rec["text"],
            "bbox":        json.dumps(rec["bbox"]) if rec["bbox"] is not None else "",
            "script":      rec["script"],
        })

print(f"  ✓ {len(records)} annotation records written")

# ─────────────────────────────────────────────────────────────────────────────
# docs/schema.json
# ─────────────────────────────────────────────────────────────────────────────
schema = {
    "version": "1.0",
    "annotation_fields": {
        "document_id":  "Unique identifier for each document image",
        "field_name":   "Semantic field label (name, date, address …)",
        "text":         "Ground-truth OCR text for this field",
        "bbox":         "[x1, y1, x2, y2] pixel coordinates, top-left origin",
        "script":       "ISO 639-1 code: bn | hi | en | mixed | unknown",
    },
    "valid_scripts":    ["bn", "hi", "en", "mixed", "unknown"],
    "valid_doc_types":  ["form", "receipt", "invoice",
                         "government_notice", "utility_bill",
                         "handwritten", "printed", "mixed", "unknown"],
    "splits":           {"train": 0.80, "validation": 0.10, "test": 0.10},
    "image_format":     "PNG",
    "image_dimensions": {"width": 1200, "height": 1600},
}
(DOCS_DIR / "schema.json").write_text(
    json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ─────────────────────────────────────────────────────────────────────────────
# docs/dataset_card.md
# ─────────────────────────────────────────────────────────────────────────────
card = """\
# Verifiable Indic Document OCR — Synthetic Dataset v1

## Overview
Synthetic dataset for Bengali, Hindi, and English document OCR.

| Attribute | Value |
|---|---|
| Total images | 300 |
| Bengali images | 100 |
| Hindi images | 100 |
| English images | 100 |
| Annotation records | 2,400 (8 per document) |
| Train / Val / Test | 240 / 30 / 30 |

## Known Quality Issues (intentionally introduced)
| Issue | Count |
|---|---|
| Missing language labels | 5 |
| Missing document-type labels | 8 |
| Empty ground-truth text | 12 |
| Missing bounding boxes | 20 |
| Corrupt image files (zero-byte) | 4 |
| Near-duplicate image pairs | 3 |
| Duplicate annotation records | 6 |

## License
Synthetic data — no redistribution restrictions.
"""
(DOCS_DIR / "dataset_card.md").write_text(card, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────────────
readme = """\
# Verifiable Indic Document OCR — Synthetic Dataset v1

Synthetic dataset generated for the Indic OCR Data Engineering pipeline.

## Structure
```
indic_ocr_dataset/
├── images/           # 300 PNG document images
├── annotations/
│   ├── annotations.jsonl
│   └── fields.csv
├── splits/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
├── docs/
│   ├── dataset_card.md
│   └── schema.json
└── README.md
```

## Usage
Run the inspection pipeline:
```powershell
cd Indic_OCR_Data_Engineering
py -3.11 scripts\\inspect_dataset.py
```
"""
(DATASET_DIR / "README.md").write_text(readme, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth issues catalogue (for validation in Phase 3+)
# ─────────────────────────────────────────────────────────────────────────────
issues_catalogue = {
    "missing_language_ids":    sorted(MISSING_LANG_IDS),
    "missing_doctype_ids":     sorted(MISSING_DOCTYPE_IDS),
    "corrupt_image_ids":       sorted(CORRUPT_IMAGE_IDS),
    "near_duplicate_pairs":    [[a, b] for a, b in NEAR_DUPE_PAIRS],
    "missing_gt_ann_indices":  sorted(MISSING_GT_INDICES),
    "missing_bbox_ann_indices":sorted(MISSING_BBOX_INDICES),
    "duplicate_ann_indices":   sorted(DUPLICATE_ANN_INDICES),
}
(DATASET_DIR / "injected_issues_catalogue.json").write_text(
    json.dumps(issues_catalogue, indent=2), encoding="utf-8"
)

print("\n✅ Synthetic dataset generation complete.")
print(f"   Location : {DATASET_DIR}")
print(f"   Images   : {len(list(IMAGES_DIR.glob('*.png')))} files")
print(f"   JSONL    : {jsonl_path}")
print(f"   CSV      : {csv_path}")
