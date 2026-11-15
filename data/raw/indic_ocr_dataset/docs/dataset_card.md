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
