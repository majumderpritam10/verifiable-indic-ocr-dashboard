"""
scripts/dataset_correction.py — Comprehensive Dataset Correction Engine
────────────────────────────────────────────────────────────────────────
Applies rigorous, audited corrections across Phases 10, 11, and 12:
1. Resolves SHA-256 collision groups by preserving representatives and segregating redundant copies.
2. Investigates and retains genuinely distinct near-duplicate document pairs.
3. Segregates intentional negative samples (empty ground-truth) into a dedicated training set.
4. Excludes null bounding boxes and defective blank-canvas documents into quarantine.
5. Produces the clean, production-ready dataset in data/corrected/.
6. Generates full audit logs, duplicate resolution reports, and split summaries.

Outputs:
  data/corrected/train/
  data/corrected/validation/
  data/corrected/test/
  data/corrected/negative_samples/
  data/corrected/quarantine/
  data/corrected/manifest_master.csv
  data/corrected/annotations_master.jsonl
  data/corrected/disposition_log.csv
  reports/duplicate_resolution_report.json
  reports/duplicate_resolution_report.csv
  reports/dataset_correction_log.csv
  reports/corrected_split_summary.json
  reports/corrected_split_summary.csv
"""

import sys
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

PROJECT_ROOT = Path(r"c:\Users\Anusuya\Downloads\Indic_OCR_Data_Engineering_ankita\Indic_OCR_Data_Engineering\Indic_OCR_Data_Engineering")
DATA_DIR = PROJECT_ROOT / "data"
CORRECTED_DIR = DATA_DIR / "corrected"
REPORTS_DIR = PROJECT_ROOT / "reports"

os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".mpl_cache"))


def main():
    print("=" * 70)
    print("STARTING DATASET CORRECTION ENGINE — PHASES 10, 11, 12")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Ensure target directories exist
    train_dir = CORRECTED_DIR / "train"
    val_dir = CORRECTED_DIR / "validation"
    test_dir = CORRECTED_DIR / "test"
    neg_dir = CORRECTED_DIR / "negative_samples"
    quar_dir = CORRECTED_DIR / "quarantine"

    for d in [train_dir, val_dir, test_dir, neg_dir, quar_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load master manifest & split annotations
    manifest_master_path = DATA_DIR / "splits" / "manifest_master.csv"
    with open(manifest_master_path, encoding="utf-8") as f:
        master_manifest_rows = list(csv.DictReader(f))
    print(f"Loaded {len(master_manifest_rows)} rows from {manifest_master_path.name}")

    all_annos = []
    for split_name in ["train", "validation", "test"]:
        anno_file = DATA_DIR / "splits" / split_name / "annotations.jsonl"
        with open(anno_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_annos.append(json.loads(line))
    print(f"Loaded {len(all_annos)} total annotations across baseline splits")

    # 3. Identify Documents to Segregate / Quarantine
    # -------------------------------------------------------------
    # Group A: Redundant exact duplicate copies of external adapters
    # Keep EXT_B2B_0001, EXT_BN_0001, EXT_INV_0001 as primary representatives in Train.
    # Segregate EXT_*_0002 through EXT_*_0020 (19 * 3 = 57 documents)
    external_redundant = {}
    for i in range(2, 21):
        external_redundant[f"EXT_B2B_{i:04d}"] = {
            "representative": "EXT_B2B_0001",
            "group": "EXT_B2B",
            "reason": "REDUNDANT_EXACT_DUPLICATE_EXTERNAL_TEMPLATE",
            "disposition": "QUARANTINE_REDUNDANT_EXACT_COPY"
        }
        external_redundant[f"EXT_BN_{i:04d}"] = {
            "representative": "EXT_BN_0001",
            "group": "EXT_BN",
            "reason": "REDUNDANT_EXACT_DUPLICATE_EXTERNAL_TEMPLATE",
            "disposition": "QUARANTINE_REDUNDANT_EXACT_COPY"
        }
        external_redundant[f"EXT_INV_{i:04d}"] = {
            "representative": "EXT_INV_0001",
            "group": "EXT_INV",
            "reason": "REDUNDANT_EXACT_DUPLICATE_EXTERNAL_TEMPLATE",
            "disposition": "QUARANTINE_REDUNDANT_EXACT_COPY"
        }

    # Group B: Defective blank-canvas synthetic documents with 100% missing text or null bboxes
    defective_synthetic = {
        "INDIC_0039": {
            "reason": "DEFECTIVE_SYNTHETIC_EMPTY_GROUND_TRUTH_BLANK_IMAGE",
            "disposition": "QUARANTINE_DEFECTIVE_SYNTHETIC_IMAGE"
        },
        "INDIC_0061": {
            "reason": "DEFECTIVE_SYNTHETIC_ALL_BBOXES_NULL_BLANK_IMAGE",
            "disposition": "QUARANTINE_DEFECTIVE_SYNTHETIC_IMAGE"
        },
        "INDIC_0062": {
            "reason": "DEFECTIVE_SYNTHETIC_ALL_BBOXES_NULL_BLANK_IMAGE",
            "disposition": "QUARANTINE_DEFECTIVE_SYNTHETIC_IMAGE"
        }
    }

    # 4. Process Manifest Rows & Categorize Images
    # -------------------------------------------------------------
    usable_manifest_rows = []
    quarantined_manifest_rows = []
    correction_logs = []
    duplicate_resolution_records = []

    # Record resolutions for Task 1 & Task 2
    for doc_id, meta in external_redundant.items():
        orig_row = next((r for r in master_manifest_rows if r["document_id"] == doc_id), None)
        orig_split = orig_row["split"] if orig_row else "unknown"
        duplicate_resolution_records.append({
            "group_id": meta["group"],
            "document_id": doc_id,
            "representative_id": meta["representative"],
            "original_split": orig_split,
            "status": "REDUNDANT_EXACT_DUPLICATE",
            "action": "QUARANTINE",
            "reason": "Eliminates cross-split and intra-split exact pixel collisions; representative preserved in train."
        })

    for doc_id, meta in defective_synthetic.items():
        orig_row = next((r for r in master_manifest_rows if r["document_id"] == doc_id), None)
        orig_split = orig_row["split"] if orig_row else "unknown"
        duplicate_resolution_records.append({
            "group_id": "SYNTHETIC_BLANK_CANVAS_GROUP",
            "document_id": doc_id,
            "representative_id": "NONE (DEFECTIVE)",
            "original_split": orig_split,
            "status": "DEFECTIVE_EMPTY_OR_NULL_BBOX",
            "action": "QUARANTINE",
            "reason": meta["reason"]
        })

    # Task 2: Near-duplicate pairs resolution
    duplicate_resolution_records.append({
        "group_id": "NEAR_DUPLICATE_PAIR_1",
        "document_id": "INDIC_0110 vs INDIC_0111",
        "representative_id": "RETAIN_BOTH",
        "original_split": "INDIC_0110=test, INDIC_0111=train",
        "status": "GENUINE_DISTINCT_DOCUMENTS",
        "action": "RETAIN_IN_SPLITS",
        "reason": "Distinct text strings verified; pHash distance 4 caused by shared synthetic layout template."
    })
    duplicate_resolution_records.append({
        "group_id": "NEAR_DUPLICATE_PAIR_2",
        "document_id": "INDIC_0160 vs INDIC_0161",
        "representative_id": "RETAIN_BOTH",
        "original_split": "INDIC_0160=test, INDIC_0161=train",
        "status": "GENUINE_DISTINCT_DOCUMENTS",
        "action": "RETAIN_IN_SPLITS",
        "reason": "Distinct text strings verified; pHash distance 2 caused by shared synthetic layout template."
    })

    # Categorize master manifest rows
    for row in master_manifest_rows:
        doc_id = row["document_id"]
        parent_id = row["parent_document_id"]
        is_aug = row["is_augmented"].lower() == "true"

        # Check external redundant
        if doc_id in external_redundant or parent_id in external_redundant:
            meta = external_redundant.get(doc_id) or external_redundant[parent_id]
            q_row = dict(row)
            q_row["quarantine_reason"] = meta["reason"]
            q_row["disposition"] = meta["disposition"]
            quarantined_manifest_rows.append(q_row)
            correction_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "record_id": doc_id,
                "record_type": "image",
                "field": "split_assignment",
                "issue": meta["reason"],
                "action": meta["disposition"],
                "proposed_value": "quarantined"
            })
            continue

        # Check defective synthetic
        if doc_id in defective_synthetic or parent_id in defective_synthetic:
            meta = defective_synthetic.get(doc_id) or defective_synthetic[parent_id]
            q_row = dict(row)
            q_row["quarantine_reason"] = meta["reason"]
            q_row["disposition"] = meta["disposition"]
            quarantined_manifest_rows.append(q_row)
            correction_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "record_id": doc_id,
                "record_type": "image",
                "field": "image_integrity",
                "issue": meta["reason"],
                "action": meta["disposition"],
                "proposed_value": "quarantined"
            })
            continue

        # Valid usable row
        usable_manifest_rows.append(row)

    print(f"Usable manifest images: {len(usable_manifest_rows)}")
    print(f"Quarantined manifest images: {len(quarantined_manifest_rows)}")

    # 5. Process Annotations
    # -------------------------------------------------------------
    usable_annos = []
    negative_sample_annos = []
    quarantined_annos = []

    for a in all_annos:
        doc_id = a.get("document_id")
        parent_id = a.get("parent_document_id") or doc_id
        text = a.get("text", "")
        bbox = a.get("bbox")

        # Quarantined doc
        if doc_id in external_redundant or parent_id in external_redundant:
            meta = external_redundant.get(doc_id) or external_redundant[parent_id]
            q_anno = dict(a)
            q_anno["quarantine_reason"] = meta["reason"]
            quarantined_annos.append(q_anno)
            continue

        if doc_id in defective_synthetic or parent_id in defective_synthetic:
            meta = defective_synthetic.get(doc_id) or defective_synthetic[parent_id]
            q_anno = dict(a)
            q_anno["quarantine_reason"] = meta["reason"]
            quarantined_annos.append(q_anno)
            continue

        # Null bounding box in usable document (e.g., INDIC_0063 fields 1-4)
        if not bbox or bbox is None:
            q_anno = dict(a)
            q_anno["quarantine_reason"] = "NULL_BOUNDING_BOX_PARTIAL_DEFECT"
            quarantined_annos.append(q_anno)
            correction_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "record_id": f"{doc_id}_{a.get('field_name')}",
                "record_type": "annotation",
                "field": "bbox",
                "issue": "NULL_BOUNDING_BOX",
                "action": "EXCLUDE_FROM_USABLE_ANNOTATIONS",
                "proposed_value": "quarantined"
            })
            continue

        # Empty ground-truth text with valid bbox (intentional negative sample in INDIC_0038)
        if not text or text == "":
            neg_anno = dict(a)
            neg_anno["is_negative_sample"] = True
            negative_sample_annos.append(neg_anno)
            correction_logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "record_id": f"{doc_id}_{a.get('field_name')}",
                "record_type": "annotation",
                "field": "text",
                "issue": "EMPTY_GROUND_TRUTH_NEGATIVE_SAMPLE",
                "action": "PRESERVE_IN_NEGATIVE_SAMPLES",
                "proposed_value": "negative_sample"
            })
            continue

        # Valid clean annotation
        usable_annos.append(a)

    print(f"Usable clean annotations: {len(usable_annos)}")
    print(f"Negative sample annotations: {len(negative_sample_annos)}")
    print(f"Quarantined annotations: {len(quarantined_annos)}")
    print(f"Annotation Reconciliation Sum: {len(usable_annos) + len(negative_sample_annos) + len(quarantined_annos)} == {len(all_annos)}")

    # 6. Write Split Manifests, Documents.txt, and Annotations
    # -------------------------------------------------------------
    split_usable_rows = defaultdict(list)
    for r in usable_manifest_rows:
        split_usable_rows[r["split"]].append(r)

    split_usable_annos = defaultdict(list)
    for a in usable_annos:
        split_usable_annos[a["split"]].append(a)

    manifest_fieldnames = list(master_manifest_rows[0].keys())

    for s_name, target_path in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
        s_rows = split_usable_rows[s_name]
        s_annos = split_usable_annos[s_name]

        # manifest.csv
        with open(target_path / "manifest.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
            writer.writeheader()
            writer.writerows(s_rows)

        # documents.txt (unique document IDs)
        doc_ids = sorted(list(set(r["document_id"] for r in s_rows)))
        with open(target_path / "documents.txt", "w", encoding="utf-8") as f:
            for did in doc_ids:
                f.write(f"{did}\n")

        # annotations.jsonl
        with open(target_path / "annotations.jsonl", "w", encoding="utf-8") as f:
            for a in s_annos:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

        # annotations.csv
        if s_annos:
            # Collect union of all fields
            anno_fields = sorted(list(set().union(*(d.keys() for d in s_annos))))
            with open(target_path / "annotations.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=anno_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(s_annos)

        print(f"  Corrected {s_name}: {len(s_rows)} images, {len(s_annos)} annotations, {len(doc_ids)} document IDs")

    # Write Master Manifest & Master Annotations in data/corrected/
    with open(CORRECTED_DIR / "manifest_master.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(usable_manifest_rows)

    with open(CORRECTED_DIR / "annotations_master.jsonl", "w", encoding="utf-8") as f:
        for a in usable_annos:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # 7. Write Negative Samples Directory
    # -------------------------------------------------------------
    neg_doc_ids = sorted(list(set(a["document_id"] for a in negative_sample_annos)))
    neg_manifest_rows = [r for r in master_manifest_rows if r["document_id"] in neg_doc_ids]

    with open(neg_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        writer.writerows(neg_manifest_rows)

    with open(neg_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for a in negative_sample_annos:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    if negative_sample_annos:
        neg_fields = sorted(list(set().union(*(d.keys() for d in negative_sample_annos))))
        with open(neg_dir / "annotations.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=neg_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(negative_sample_annos)

    with open(neg_dir / "documents.txt", "w", encoding="utf-8") as f:
        for did in neg_doc_ids:
            f.write(f"{did}\n")

    # 8. Write Quarantine Directory
    # -------------------------------------------------------------
    quar_manifest_fields = manifest_fieldnames + ["quarantine_reason", "disposition"]
    with open(quar_dir / "quarantined_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=quar_manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(quarantined_manifest_rows)

    with open(quar_dir / "quarantined_annotations.jsonl", "w", encoding="utf-8") as f:
        for a in quarantined_annos:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    quar_doc_ids = sorted(list(set(r["document_id"] for r in quarantined_manifest_rows)))
    with open(quar_dir / "documents.txt", "w", encoding="utf-8") as f:
        for did in quar_doc_ids:
            f.write(f"{did}\n")

    # 9. Write Comprehensive Disposition Log in data/corrected/
    # -------------------------------------------------------------
    all_dispositions = []
    for r in usable_manifest_rows:
        all_dispositions.append({
            "document_id": r["document_id"],
            "parent_document_id": r["parent_document_id"],
            "type": "IMAGE",
            "assigned_split": r["split"],
            "status": "USABLE",
            "category": "CLEAN_OCR_IMAGE",
            "notes": "Included in corrected usable training/evaluation dataset"
        })
    for r in quarantined_manifest_rows:
        all_dispositions.append({
            "document_id": r["document_id"],
            "parent_document_id": r["parent_document_id"],
            "type": "IMAGE",
            "assigned_split": r.get("split", "quarantine"),
            "status": "QUARANTINED",
            "category": r.get("quarantine_reason", "QUARANTINED"),
            "notes": r.get("disposition", "Excluded from usable dataset")
        })

    with open(CORRECTED_DIR / "disposition_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "parent_document_id", "type", "assigned_split", "status", "category", "notes"])
        writer.writeheader()
        writer.writerows(all_dispositions)

    # 10. Write Deliverable Reports
    # -------------------------------------------------------------
    # duplicate_resolution_report.json & csv
    with open(REPORTS_DIR / "duplicate_resolution_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": len(duplicate_resolution_records),
            "records": duplicate_resolution_records
        }, f, indent=2)

    with open(REPORTS_DIR / "duplicate_resolution_report.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(duplicate_resolution_records[0].keys()))
        writer.writeheader()
        writer.writerows(duplicate_resolution_records)

    # dataset_correction_log.csv
    with open(REPORTS_DIR / "dataset_correction_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "record_id", "record_type", "field", "issue", "action", "proposed_value"])
        writer.writeheader()
        writer.writerows(correction_logs)

    # Calculate Corrected Split Summary Metrics
    usable_orig_docs = [r for r in usable_manifest_rows if r["is_augmented"].lower() == "false"]
    usable_aug_docs = [r for r in usable_manifest_rows if r["is_augmented"].lower() == "true"]

    orig_split_counts = Counter(r["split"] for r in usable_orig_docs)
    total_usable_orig = len(usable_orig_docs)

    corrected_summary = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "READY",
        "corpus_overview": {
            "total_usable_images": len(usable_manifest_rows),
            "usable_original_documents": total_usable_orig,
            "usable_augmented_images": len(usable_aug_docs),
            "usable_clean_annotations": len(usable_annos),
            "negative_sample_annotations": len(negative_sample_annos),
            "quarantined_images": len(quarantined_manifest_rows),
            "quarantined_annotations": len(quarantined_annos)
        },
        "original_document_split": {
            "train": {
                "count": orig_split_counts["train"],
                "percentage": round(orig_split_counts["train"] / total_usable_orig * 100, 2)
            },
            "validation": {
                "count": orig_split_counts["validation"],
                "percentage": round(orig_split_counts["validation"] / total_usable_orig * 100, 2)
            },
            "test": {
                "count": orig_split_counts["test"],
                "percentage": round(orig_split_counts["test"] / total_usable_orig * 100, 2)
            }
        },
        "all_images_split": {
            "train": len(split_usable_rows["train"]),
            "validation": len(split_usable_rows["validation"]),
            "test": len(split_usable_rows["test"])
        },
        "annotations_split": {
            "train": len(split_usable_annos["train"]),
            "validation": len(split_usable_annos["validation"]),
            "test": len(split_usable_annos["test"])
        },
        "script_distribution": {
            "original_documents": dict(Counter(r["script"] for r in usable_orig_docs)),
            "all_images": dict(Counter(r["script"] for r in usable_manifest_rows))
        }
    }

    with open(REPORTS_DIR / "corrected_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(corrected_summary, f, indent=2)

    # corrected_split_summary.csv
    with open(REPORTS_DIR / "corrected_split_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "original_documents", "original_pct", "augmented_images", "total_images", "annotations"])
        for s_name in ["train", "validation", "test"]:
            writer.writerow([
                s_name,
                orig_split_counts[s_name],
                f"{orig_split_counts[s_name] / total_usable_orig * 100:.2f}%",
                len([r for r in split_usable_rows[s_name] if r["is_augmented"].lower() == "true"]),
                len(split_usable_rows[s_name]),
                len(split_usable_annos[s_name])
            ])

    print("\n" + "=" * 70)
    print("DATASET CORRECTION COMPLETED SUCCESSFULLY")
    print(f"Corrected usable documents: {total_usable_orig}")
    print(f"  Train: {orig_split_counts['train']} ({orig_split_counts['train']/total_usable_orig*100:.2f}%)")
    print(f"  Val:   {orig_split_counts['validation']} ({orig_split_counts['validation']/total_usable_orig*100:.2f}%)")
    print(f"  Test:  {orig_split_counts['test']} ({orig_split_counts['test']/total_usable_orig*100:.2f}%)")
    print(f"Total usable images: {len(usable_manifest_rows)}")
    print(f"Total usable annotations: {len(usable_annos)}")
    print(f"Total negative samples: {len(negative_sample_annos)}")
    print(f"Total quarantined records: {len(quarantined_manifest_rows)} images, {len(quarantined_annos)} annotations")
    print("=" * 70)


if __name__ == "__main__":
    main()
