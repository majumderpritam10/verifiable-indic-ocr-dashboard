"""Read-only checks for the corrected dataset before website integration.

Run from the project root:
    python scripts/verify_predeployment.py

This script does not modify dataset files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / "data" / "corrected"
REPORTS = ROOT / "reports"


def resolve_manifest_image(value: object) -> Path:
    """Resolve Windows- or POSIX-style relative paths portably."""
    normalized = str(value).replace("\\", "/")
    return ROOT / Path(normalized)


def main() -> None:
    manifest_path = CORRECTED / "manifest_master.csv"
    manifest = pd.read_csv(manifest_path)
    required = {"document_id", "split", "is_augmented", "num_annotations", "image_path"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise SystemExit(f"Manifest is missing required columns: {missing_columns}")

    missing_images: list[str] = []
    unreadable_images: list[str] = []
    wrong_dimensions: list[dict] = []
    for raw_path in manifest["image_path"].dropna():
        image_path = resolve_manifest_image(raw_path)
        if not image_path.is_file():
            missing_images.append(str(raw_path))
            continue
        try:
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(image_path) as im:
                if im.size != (1200, 1600):
                    wrong_dimensions.append({"path": str(raw_path), "dimensions": list(im.size)})
        except Exception as exc:  # report exact item; keep checking remaining files
            unreadable_images.append(f"{raw_path}: {exc}")

    split_counts = manifest["split"].value_counts(dropna=False).to_dict()
    augmented_counts = (
        manifest.groupby("split")["is_augmented"].apply(lambda s: int(s.astype(str).str.lower().isin(["true", "1"]).sum())).to_dict()
    )

    annotation_file = CORRECTED / "annotations_master.jsonl"
    clean_annotation_count = sum(1 for line in annotation_file.open(encoding="utf-8") if line.strip())
    negative_file = CORRECTED / "negative_samples" / "annotations.jsonl"
    negative_annotation_count = sum(1 for line in negative_file.open(encoding="utf-8") if line.strip()) if negative_file.exists() else 0
    quarantine_file = CORRECTED / "quarantine" / "quarantined_annotations.jsonl"
    quarantined_annotation_count = sum(1 for line in quarantine_file.open(encoding="utf-8") if line.strip()) if quarantine_file.exists() else 0

    audit_path = REPORTS / "corrected_final_audit_summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scorecard = audit.get("scorecard_summary", {})

    result = {
        "project_root": ".",
        "corrected_manifest_rows": int(len(manifest)),
        "split_image_counts": {str(k): int(v) for k, v in split_counts.items()},
        "augmented_image_counts_by_split": {str(k): int(v) for k, v in augmented_counts.items()},
        "manifest_image_files_missing": len(missing_images),
        "manifest_images_unreadable": len(unreadable_images),
        "manifest_images_wrong_dimensions": len(wrong_dimensions),
        "clean_annotation_records": clean_annotation_count,
        "negative_sample_annotation_records": negative_annotation_count,
        "quarantined_annotation_records": quarantined_annotation_count,
        "final_audit_status": audit.get("final_dataset_status"),
        "final_audit_scorecard": scorecard,
        "missing_image_examples": missing_images[:10],
        "unreadable_image_examples": unreadable_images[:10],
        "wrong_dimension_examples": wrong_dimensions[:10],
        "read_only": True,
    }
    out = REPORTS / "predeployment_verification.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {out.relative_to(ROOT)}")

    if missing_images or unreadable_images or wrong_dimensions:
        raise SystemExit("Predeployment verification found image integrity issues; inspect the report.")


if __name__ == "__main__":
    main()
