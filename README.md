# Indic OCR Data Engineering Pipeline

**Project:** Verifiable Indic Document OCR — Dataset Preparation  
**Purpose:** Collect, inspect, clean, preprocess, and package Bengali, Hindi, and English document image datasets for downstream OCR model development.

---

## Project Structure

```
Indic_OCR_Data_Engineering/
├── data/
│   ├── raw/                   ← Immutable source data (never overwrite)
│   │   └── indic_ocr_dataset/ ← Synthetic dataset v1
│   ├── cleaned/               ← After annotation cleaning
│   ├── preprocessed/          ← After image preprocessing
│   ├── augmented/             ← Augmented training images
│   └── final/                 ← Integrated + split final dataset
├── scripts/
│   ├── generate_synthetic_dataset.py   ← Creates the synthetic dataset
│   ├── inspect_dataset.py              ← PHASE 1 — Dataset profiling
│   ├── missing_values.py               ← PHASE 3 — Missing value analysis
│   ├── duplicate_detection.py          ← PHASE 4 — Duplicate detection
│   ├── image_quality.py                ← PHASE 5 — Quality analysis
│   ├── clean_annotations.py            ← PHASE 6a — Annotation cleaning
│   ├── validate_bboxes.py              ← PHASE 6b — BBox validation
│   ├── preprocess_images.py            ← PHASE 7 — Image preprocessing
│   ├── script_analysis.py              ← Script/language ID
│   ├── augmentation.py                 ← PHASE 10 — Augmentation
│   ├── integrate_datasets.py           ← PHASE 9 — Dataset integration
│   ├── create_splits.py                ← PHASE 11 — Train/val/test splits
│   └── final_audit.py                  ← PHASE 12 — Final audit
├── reports/                   ← All generated CSV/JSON/PNG reports
├── notebooks/                 ← Jupyter exploration notebooks
├── tests/                     ← Unit tests
├── config.py                  ← Central configuration
├── requirements.txt
└── README.md                  ← This file
```

---

## Setup

### Requirements
- Python 3.11 or a compatible newer Python version
- Windows, macOS, or Linux

### Install dependencies (Windows PowerShell)
Run these commands from the project root folder:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use this command in the same terminal and retry activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

For macOS/Linux, create and activate the environment with:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Execution Order (Phase by Phase)

> **Important:** Always run from the `Indic_OCR_Data_Engineering\` directory.
> Never skip a phase. Never delete raw data.

### Step 0 — Generate synthetic dataset (one-time)
```powershell
python scripts\generate_synthetic_dataset.py
```

### Phase 1 — Dataset Inspection
```powershell
python scripts\inspect_dataset.py
```
**Outputs:** `reports/dataset_profile.csv`, `reports/dataset_profile.json`, `reports/dataset_profile_charts.png`, `reports/annotation_errors.csv`

### Phase 3 — Missing Values (coming next)
```powershell
python scripts\missing_values.py
```

### Phase 4 — Duplicate Detection
```powershell
python scripts\duplicate_detection.py
```

### Phase 5 — Image Quality Analysis
```powershell
python scripts\image_quality.py
```

### Phase 6 — Clean Annotations + Validate BBoxes
```powershell
python scripts\clean_annotations.py
python scripts\validate_bboxes.py
```

### Phase 7 — Image Preprocessing
```powershell
python scripts\preprocess_images.py
```

### Phase 10 — Augmentation
```powershell
python scripts\data_augmentation.py
# or
python scripts\augmentation.py
```
**Outputs:** `data/augmented/images/` (472 PNGs), `data/augmented/annotations_augmented.jsonl`, `data/augmented/annotations_augmented.csv`, `data/augmented/augmented_catalog.csv`, `reports/augmentation_summary.csv`, `reports/augmentation_summary.json`, `reports/augmentation_charts.png`

### Phase 11 — Create Splits
```powershell
python scripts\create_splits.py
```
**Outputs:**
- `data/splits/train/` (`manifest.csv`, `annotations.jsonl`, `annotations.csv`, `documents.txt`)
- `data/splits/validation/` (`manifest.csv`, `annotations.jsonl`, `annotations.csv`, `documents.txt`)
- `data/splits/test/` (`manifest.csv`, `annotations.jsonl`, `annotations.csv`, `documents.txt`)
- `data/splits/manifest_master.csv`
- `data/splits/train.txt`, `data/splits/validation.txt`, `data/splits/test.txt`
- `reports/split_summary.csv`, `reports/split_summary.json`
- `reports/split_distribution_charts.png`
- Updated `data/final/integrated_catalog.csv` with `split` column
- Updated `data/final/annotations_master.jsonl` with `split` field

### Phase 12 — Final Audit
```powershell
python scripts\final_audit.py
```
**Outputs:**
- `reports/final_audit_report.csv` (12-check compliance scorecard)
- `reports/final_audit_summary.json` (Structured audit diagnostics & limitation analysis)
- `reports/final_audit_charts.png` (4-panel visual audit dashboard)
- Appended provenance records to `reports/cleaning_log.csv`

---

## Dataset Correction & Handover (Post-Phase 12)

A comprehensive, audited dataset correction was executed to resolve all confirmed Phase 12 findings without modifying raw data or previous phase outputs.

### Correction Scripts
```powershell
# 1. Execute Dataset Correction Engine
python scripts\dataset_correction.py

# 2. Run Comprehensive Quality Audit on Corrected Dataset
python scripts\audit_corrected_dataset.py
```

### Corrected Dataset Directory Structure (`data/corrected/`)
Downstream team members developing OCR models should use `data/corrected/`:
```
data/corrected/
├── train/
│   ├── manifest.csv               # 705 training images (237 orig + 468 aug)
│   ├── annotations.jsonl          # 5,607 clean annotations
│   ├── annotations.csv
│   └── documents.txt
├── validation/
│   ├── manifest.csv               # 29 validation images (29 orig + 0 aug)
│   ├── annotations.jsonl          # 229 clean annotations
│   ├── annotations.csv
│   └── documents.txt
├── test/
│   ├── manifest.csv               # 30 test images (30 orig + 0 aug)
│   ├── annotations.jsonl          # 240 clean annotations
│   ├── annotations.csv
│   └── documents.txt
├── negative_samples/
│   ├── manifest.csv               # 3 negative sample training images (INDIC_0038 + augs)
│   ├── annotations.jsonl          # 12 empty ground-truth text records
│   ├── annotations.csv
│   └── documents.txt
├── quarantine/
│   ├── quarantined_manifest.csv   # 64 segregated images (57 redundant + 7 defective)
│   ├── quarantined_annotations.jsonl # 524 excluded annotations
│   └── documents.txt
├── manifest_master.csv            # 764 total usable images
├── annotations_master.jsonl       # 6,076 total usable clean annotations
└── disposition_log.csv            # Complete itemized provenance of all 828 images
```

### Key Correction Deliverables & Reports
- `reports/duplicate_resolution_report.json` & `.csv`: Exact SHA-256 duplicate template resolution & near-duplicate pair analysis.
- `reports/dataset_correction_log.csv`: Full audit trail of every segregated and quarantined record.
- `reports/corrected_split_summary.json` & `.csv`: Final split metrics, percentages, and language distribution.
- `reports/corrected_final_audit_report.csv`: 12-check compliance scorecard (10 PASS, 2 WARNING, 0 FAIL).
- `reports/corrected_final_audit_summary.json`: Complete audit metrics for corrected dataset.
- `reports/corrected_final_audit_charts.png`: 4-panel visual audit dashboard.

---

## Key Principles

| Rule | Reason |
|------|--------|
| Raw data is immutable | Preserve original for reproducibility |
| Never silently delete records | Always quarantine or flag |
| Keep original + normalized text | Different models may need either form |
| Augment only training split | Prevent data leakage |
| Record every cleaning decision | Full provenance trail |

---

## Dataset (Synthetic v1)

| Attribute | Value |
|-----------|-------|
| Total images | 300 |
| Bengali | 100 |
| Hindi | 100 |
| English | 100 |
| Annotation records | 2,400 |
| Train / Val / Test | 240 / 30 / 30 |
| Image size | 1200 × 1600 px PNG |

### Intentionally injected quality issues (for pipeline testing)
| Issue | Count |
|-------|-------|
| Corrupt image files | 4 |
| Missing language labels | 5 |
| Missing document-type labels | 8 |
| Empty ground-truth text | 12 |
| Missing bounding boxes | 20 |
| Near-duplicate image pairs | 3 |
| Duplicate annotation records | 6 |

---

## Technologies
Python 3.11 · Pandas · NumPy · Pillow · OpenCV (headless) · Matplotlib · Seaborn · scikit-learn · imagehash
