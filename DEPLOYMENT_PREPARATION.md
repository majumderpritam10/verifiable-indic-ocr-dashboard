# Deployment Preparation Notes

This folder is a portability-prepared copy of the supplied project. The original uploaded ZIP was not modified.

## Changes made

1. Removed machine-specific Python executable selection from `config.py`; it now uses `sys.executable` from the active environment.
2. Replaced the README's machine-specific Python setup commands with portable virtual-environment instructions.
3. Added Streamlit and Plotly to `requirements.txt` for the upcoming website implementation.
4. Added `.gitignore` entries for Python caches, virtual environments, editor files, local secrets, and temporary logs.
5. Excluded local Matplotlib caches and Python bytecode from this prepared copy.

## Dataset source of truth for the website

Use `data/corrected/` and the `corrected_*` reports as the authoritative corrected dataset and audit outputs. Do not combine older pre-correction reports with corrected reports without clearly labelling their stage.

Important corrected metrics recorded in the supplied audit:
- Usable images: 764
- Original usable documents: 296
- Usable augmented images: 468
- Usable clean annotation records: 6,076
- Negative-sample annotation records: 12
- Quarantined images: 64
- Quarantined annotation records: 524
- Final audit: 10 PASS, 2 WARNING, 0 FAIL (READY WITH WARNINGS)

## Annotation-count note

The `num_annotations` field in the corrected manifests is a source-record/disposition count for affected documents, not always the number of clean rows in `annotations_master.jsonl` alone. For example, some empty-text records are stored separately under `negative_samples/`, and some excluded records are stored under `quarantine/`. Therefore, do not overwrite the manifest values using only a count of `annotations_master.jsonl`; doing so would erase the distinction between clean, negative, and quarantined records.

## Known items to resolve or disclose before final public deployment

1. The corrected audit is `READY WITH WARNINGS`, not a clean PASS. Preserve and display both warnings: perceptual near-duplicate/layout similarity and one cross-split text/template string overlap.
2. The external dataset catalog uses `external_adapter` provenance labels. Before claiming that all listed items were downloaded from Kaggle or another public source, the team should attach verifiable source URLs, licenses, and acquisition evidence.
3. The corrected manifests store image paths with Windows backslashes. Application code should normalize these paths with `str(path).replace("\\\\", "/")` (or an equivalent portable path resolver) before opening files on Linux.
4. Streamlit application code has not yet been added. This is a prepared project base, not a deployed website.
5. Before publishing a public GitHub repository, confirm that the group permits public source/data publication and that no restricted or licensed dataset content is included.

## Recommended deployment sequence

1. Implement and test the Streamlit app locally.
2. Add a lightweight app-specific dependency set if needed; keep all required packages in the root `requirements.txt` for Streamlit Community Cloud.
3. Check the repository size and individual file sizes before pushing to GitHub.
4. Confirm dataset licensing/provenance and repository visibility with the group.
5. Push the tested project to the personal GitHub repository.
6. Connect the repository to Streamlit Community Cloud and deploy the root app entry point.
