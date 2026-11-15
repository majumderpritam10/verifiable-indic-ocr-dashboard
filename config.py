"""
config.py — Central configuration for Indic OCR Data Engineering pipeline.

All paths, thresholds, and schema constants are defined here.
Import this module in every script.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# ROOT PATHS
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
CLEANED_DIR  = DATA_DIR / "cleaned"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
AUGMENTED_DIR    = DATA_DIR / "augmented"
FINAL_DIR        = DATA_DIR / "final"

REPORTS_DIR   = PROJECT_ROOT / "reports"
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR     = PROJECT_ROOT / "tests"

# Source synthetic dataset location inside raw/
SYNTHETIC_DATASET_DIR = RAW_DIR / "indic_ocr_dataset"

# ─────────────────────────────────────────────────────────────
# SUPPORTED FILE FORMATS
# ─────────────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_DOC_EXTS   = {".pdf"}

# ─────────────────────────────────────────────────────────────
# LANGUAGE / SCRIPT CONSTANTS
# ─────────────────────────────────────────────────────────────
VALID_SCRIPTS = {"bn", "hi", "en", "mixed", "unknown"}

UNICODE_RANGES = {
    "bn": (0x0980, 0x09FF),   # Bengali block
    "hi": (0x0900, 0x097F),   # Devanagari block
    "en": (0x0000, 0x007F),   # Basic Latin
}

# ─────────────────────────────────────────────────────────────
# DOCUMENT TYPES
# ─────────────────────────────────────────────────────────────
VALID_DOC_TYPES = {
    "form", "receipt", "invoice", "government_notice",
    "utility_bill", "handwritten", "printed", "mixed", "unknown"
}

# ─────────────────────────────────────────────────────────────
# QUALITY THRESHOLDS
# ─────────────────────────────────────────────────────────────
BLUR_THRESHOLD        = 100.0   # Laplacian variance below this → blurry
MIN_RESOLUTION        = 100     # DPI below this → low resolution
MIN_IMAGE_WIDTH       = 100     # pixels
MIN_IMAGE_HEIGHT      = 100     # pixels
CONTRAST_LOW_THRESH   = 30.0    # std-dev of pixel intensities
NOISE_HIGH_THRESH     = 50.0    # estimated noise std-dev

# ─────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────
PHASH_THRESHOLD = 10   # Hamming distance for near-duplicate detection

# ─────────────────────────────────────────────────────────────
# DATASET SPLITS
# ─────────────────────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
RANDOM_SEED  = 42

# ─────────────────────────────────────────────────────────────
# AUGMENTATION LEVELS
# ─────────────────────────────────────────────────────────────
AUGMENTATION_LEVELS = {
    0: "clean",
    1: "mild",
    2: "moderate",
    3: "severe",
}

# ─────────────────────────────────────────────────────────────
# ANNOTATION SCHEMA (expected JSON keys)
# ─────────────────────────────────────────────────────────────
REQUIRED_ANNOTATION_KEYS = {
    "document_id", "field_name", "text", "bbox", "script"
}

# ─────────────────────────────────────────────────────────────
# PYTHON EXECUTABLE
# ─────────────────────────────────────────────────────────────
# Use the Python interpreter running the current environment.
# This keeps the project portable across Windows, macOS, Linux,
# and Streamlit Community Cloud.
import sys
PYTHON_EXE = sys.executable

# ─────────────────────────────────────────────────────────────
# ENSURE DIRECTORIES EXIST AT IMPORT TIME
# ─────────────────────────────────────────────────────────────
for _d in [RAW_DIR, CLEANED_DIR, PREPROCESSED_DIR,
           AUGMENTED_DIR, FINAL_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
