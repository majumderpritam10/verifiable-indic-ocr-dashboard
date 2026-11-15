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
py -3.11 scripts\inspect_dataset.py
```
