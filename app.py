
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Verifiable Indic Document OCR",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MANIFEST_PATH = (
    BASE_DIR / "data" / "corrected" / "manifest_master.csv"
)


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_dataset():

    if not MANIFEST_PATH.exists():
        return None

    return pd.read_csv(MANIFEST_PATH)


df = load_dataset()


# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetric"] {
    background-color: #172554;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #bfdbfe;
}

[data-testid="stMetricValue"] {
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📄 INDIC OCR")

st.sidebar.caption(
    "Verifiable Indic Document OCR Platform"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Dataset Dashboard",
        "Dataset Explorer",
        "Data Analysis",
        "Image Processing",
        "Data Quality",
        "Dataset Splitting",
        "Final Audit",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Academic Project\n\n"
    "UEM Kolkata | CSE"
)


# ==========================================
# CHECK DATASET
# ==========================================

if df is None:

    st.error(
        "Master dataset manifest not found. "
        "Please verify the project directory."
    )

    st.stop()


# ==========================================
# COMMON DATA
# ==========================================

total_documents = len(df)

split_counts = (
    df["split"].value_counts()
    if "split" in df.columns
    else pd.Series(dtype=int)
)

language_counts = (
    df["script"].value_counts()
    if "script" in df.columns
    else pd.Series(dtype=int)
)

augmented_count = (
    df["is_augmented"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
    .sum()
    if "is_augmented" in df.columns
    else 0
)


# ==========================================
# HOME
# ==========================================

if page == "Home":

    st.title("📄 Verifiable Indic Document OCR")

    st.subheader(
        "Intelligent Dataset Engineering Platform"
    )

    st.markdown("---")

    st.write(
        """
        Welcome to our Verifiable Indic Document OCR platform.

        This project focuses on preparing, analysing and
        validating document datasets for Optical Character
        Recognition (OCR) applications involving Indic scripts.
        """
    )

    st.markdown("### Supported Languages")

    c1, c2, c3 = st.columns(3)

    c1.info("🇬🇧 English")
    c2.info("🇮🇳 Hindi")
    c3.info("🇧🇩 Bengali")

    st.markdown("---")

    st.success(
        "Dataset successfully connected to the application."
    )

    st.metric(
        "Total Documents",
        f"{total_documents:,}"
    )


# ==========================================
# DATASET DASHBOARD
# ==========================================

elif page == "Dataset Dashboard":

    st.title("📊 Dataset Dashboard")

    st.caption(
        "Live statistics from the corrected master manifest"
    )

    st.markdown("---")

    train_count = int(split_counts.get("train", 0))
    validation_count = int(split_counts.get("validation", 0))
    test_count = int(split_counts.get("test", 0))

    # METRIC CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Documents",
        f"{total_documents:,}"
    )

    c2.metric(
        "Training",
        f"{train_count:,}"
    )

    c3.metric(
        "Validation",
        f"{validation_count:,}"
    )

    c4.metric(
        "Testing",
        f"{test_count:,}"
    )

    st.markdown("---")

    c5, c6 = st.columns(2)

    c5.metric(
        "Augmented Documents",
        f"{augmented_count:,}"
    )

    c6.metric(
        "Languages",
        len(language_counts)
    )

    st.markdown("---")

    # SPLIT CHART

    st.subheader("Dataset Split Distribution")

    split_df = (
        split_counts
        .rename_axis("Dataset Split")
        .reset_index(name="Documents")
    )

    fig_split = px.pie(
        split_df,
        names="Dataset Split",
        values="Documents",
        hole=0.45,
        title="Training / Validation / Testing"
    )

    st.plotly_chart(
        fig_split,
        use_container_width=True
    )

    st.markdown("---")

    # LANGUAGE CHART

    st.subheader("Language Distribution")

    language_names = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali"
    }

    language_df = (
        language_counts
        .rename_axis("Language")
        .reset_index(name="Documents")
    )

    language_df["Language"] = (
        language_df["Language"]
        .map(language_names)
        .fillna(language_df["Language"])
    )

    fig_language = px.bar(
        language_df,
        x="Language",
        y="Documents",
        color="Language",
        text="Documents",
        title="Documents by Language"
    )

    st.plotly_chart(
        fig_language,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Dataset Manifest Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==========================================
# DATASET EXPLORER
# ==========================================

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")

    st.write(
        "Explore actual document images from the dataset."
    )

    st.markdown("---")

    filtered_df = df.copy()

    # LANGUAGE FILTER

    if "script" in filtered_df.columns:

        scripts = sorted(
            filtered_df["script"].dropna().unique()
        )

        selected_script = st.selectbox(
            "Select Language",
            ["All"] + scripts
        )

        if selected_script != "All":

            filtered_df = filtered_df[
                filtered_df["script"] == selected_script
            ]

    # SPLIT FILTER

    if "split" in filtered_df.columns:

        splits = sorted(
            filtered_df["split"].dropna().unique()
        )

        selected_split = st.selectbox(
            "Select Dataset Split",
            ["All"] + splits
        )

        if selected_split != "All":

            filtered_df = filtered_df[
                filtered_df["split"] == selected_split
            ]

    st.write(
        f"Matching documents: {len(filtered_df)}"
    )

    if len(filtered_df) > 0:

        selected_index = st.selectbox(
            "Choose Document",
            filtered_df.index,
            format_func=lambda i: str(
                filtered_df.loc[i, "document_id"]
            )
        )

        record = filtered_df.loc[selected_index]

        relative_path = str(
            record["image_path"]
        ).replace("\\", "/")

        image_path = BASE_DIR / relative_path

        if image_path.is_file():

            st.image(
                str(image_path),
                caption=str(record["document_id"]),
                use_container_width=True
            )

            st.subheader("Document Information")

            st.json({
                "Document ID": str(record["document_id"]),
                "Language": str(record["script"]),
                "Split": str(record["split"]),
                "Document Type": str(record["doc_type"]),
                "Annotations": int(record["num_annotations"]),
                "Augmented": str(record["is_augmented"])
            })

        else:

            st.error(
                f"Image not found: {image_path}"
            )

    else:

        st.warning("No matching documents found.")


# ==========================================
# DATA ANALYSIS
# ==========================================

elif page == "Data Analysis":

    st.title("📈 Data Analysis")

    st.subheader("Language Statistics")

    st.dataframe(
        language_counts.rename("Documents"),
        use_container_width=True
    )

    st.subheader("Document Type Distribution")

    if "doc_type" in df.columns:

        type_counts = (
            df["doc_type"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Document Type")
            .reset_index(name="Documents")
        )

        fig = px.bar(
            type_counts,
            x="Document Type",
            y="Documents",
            text="Documents"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# IMAGE PROCESSING
# ==========================================

elif page == "Image Processing":

    st.title("🖼️ Image Processing")
    st.write(
        "Compare original document images with their "
        "preprocessed versions."
    )

    catalog_path = (
        BASE_DIR / "data" / "preprocessed" / "preprocessed_catalog.csv"
    )

    original_dir = BASE_DIR / "data" / "final" / "images"

    processed_dir = (
        BASE_DIR / "data" / "preprocessed" / "images"
    )

    if catalog_path.exists():

        try:
            catalog = pd.read_csv(catalog_path)

            st.success("Preprocessing catalog loaded successfully!")

            # Dataset summary
            st.subheader("📊 Preprocessing Summary")

            total_images = len(catalog)

            clean_images = (
                catalog["quality_status"]
                .astype(str)
                .str.lower()
                .eq("clean_preprocessed")
                .sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Processed Images", total_images)
            col2.metric("Clean Preprocessed", int(clean_images))
            col3.metric(
                "Available Scripts",
                catalog["script"].nunique()
            )

            st.divider()

            # Language filter
            st.subheader("🔎 Select a Document")

            scripts = ["All"] + sorted(
                catalog["script"].dropna().astype(str).unique().tolist()
            )

            selected_script = st.selectbox(
                "Filter by Script",
                scripts
            )

            filtered_catalog = catalog.copy()

            if selected_script != "All":
                filtered_catalog = filtered_catalog[
                    filtered_catalog["script"].astype(str)
                    == selected_script
                ]

            if not filtered_catalog.empty:

                document_ids = filtered_catalog[
                    "document_id"
                ].astype(str).tolist()

                selected_id = st.selectbox(
                    "Choose Document ID",
                    document_ids
                )

                row = filtered_catalog[
                    filtered_catalog["document_id"].astype(str)
                    == selected_id
                ].iloc[0]

                raw_path = original_dir / str(row["raw_file_name"])

                processed_path = (
                    processed_dir / str(row["preprocessed_file_name"])
                )

                st.divider()

                # Image comparison
                st.subheader("🖼️ Original vs Preprocessed")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Original Image")

                    if raw_path.exists():
                        st.image(
                            str(raw_path),
                            caption=selected_id + " - Original",
                            use_container_width=True
                        )
                    else:
                        st.error("Original image not found.")

                with col2:
                    st.markdown("### Preprocessed Image")

                    if processed_path.exists():
                        st.image(
                            str(processed_path),
                            caption=selected_id + " - Preprocessed",
                            use_container_width=True
                        )
                    else:
                        st.error("Preprocessed image not found.")

                st.divider()

                # Document information
                st.subheader("📋 Document Information")

                col1, col2, col3 = st.columns(3)

                col1.metric("Document ID", selected_id)
                col2.metric("Script", str(row["script"]))
                col3.metric(
                    "Quality Status",
                    str(row["quality_status"])
                )

                st.divider()

                # Processing measurements
                st.subheader("📈 Before and After Measurements")

                def show_metric(column_name, label):
                    if (
                        column_name in row.index
                        and pd.notna(row[column_name])
                    ):
                        return f"{float(row[column_name]):.2f}"
                    return "N/A"

                measurements = {
                    "Measurement": [
                        "Contrast (Standard Deviation)",
                        "Blur (Laplacian Variance)",
                        "Brightness (Mean)",
                    ],
                    "Original": [
                        show_metric("raw_contrast_std", "Contrast"),
                        show_metric(
                            "raw_blur_laplacian_var", "Blur"
                        ),
                        show_metric(
                            "raw_brightness_mean", "Brightness"
                        ),
                    ],
                    "Preprocessed": [
                        show_metric("prep_contrast_std", "Contrast"),
                        show_metric(
                            "prep_blur_laplacian_var", "Blur"
                        ),
                        show_metric(
                            "prep_brightness_mean", "Brightness"
                        ),
                    ],
                }

                st.dataframe(
                    pd.DataFrame(measurements),
                    use_container_width=True,
                    hide_index=True
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Skew Angle",
                    show_metric("skew_angle_deg", "Skew") + "°"
                )

                col2.metric(
                    "Foreground Pixel Ratio",
                    show_metric("foreground_pixel_ratio", "Foreground")
                )

                col3.metric(
                    "Dynamic Range",
                    show_metric("prep_dynamic_range", "Range")
                )

                st.info(
                    "These measurements are taken directly from "
                    "the team's preprocessing catalog."
                )

            else:
                st.warning("No documents found for this script.")

        except Exception as e:
            st.error(f"Unable to load preprocessing data: {e}")

    else:
        st.error("Preprocessing catalog not found.")
        st.code(
            "data/preprocessed/preprocessed_catalog.csv"
        )

# ==========================================
# DATA QUALITY
# ==========================================

elif page == "Data Quality":

    st.title("🧹 Data Quality")

    st.subheader("Dataset Records")

    st.write(
        "Total manifest records:",
        total_documents
    )

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .rename_axis("Column")
        .reset_index(name="Missing Values")
    )

    st.dataframe(
        missing,
        use_container_width=True
    )


# ==========================================
# DATASET SPLITTING
# ==========================================

elif page == "Dataset Splitting":

    st.title("📂 Dataset Splitting")

    st.write(
        "Actual training, validation and testing distribution."
    )

    st.bar_chart(split_counts)


# ==========================================
# FINAL AUDIT
# ==========================================

elif page == "Final Audit":
    st.title("✅ Final Quality Audit")
    st.write("Final validation results of the corrected Indic OCR dataset.")

    audit_path = BASE_DIR / "reports" / "corrected_final_audit_summary.json"

    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as file:
                audit = json.load(file)

            st.success(
                f"Dataset Status: {audit.get('final_dataset_status', 'Not available')}"
            )

            scorecard = audit.get("scorecard_summary", {})

            st.subheader("📊 Audit Scorecard")

            col1, col2, col3 = st.columns(3)
            col1.metric("PASS", scorecard.get("PASS", 0))
            col2.metric("WARNING", scorecard.get("WARNING", 0))
            col3.metric("FAIL", scorecard.get("FAIL", 0))

            metrics = audit.get("corpus_metrics", {})

            st.subheader("📁 Dataset Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("Usable Images", metrics.get("total_usable_images", 0))
            col2.metric("Original Documents", metrics.get("usable_original_documents", 0))
            col3.metric("Augmented Images", metrics.get("usable_augmented_images", 0))

            col1, col2, col3 = st.columns(3)
            col1.metric("Clean Annotations", metrics.get("usable_clean_annotations", 0))
            col2.metric("Negative Samples", metrics.get("negative_samples_annotations", 0))
            col3.metric("Quarantined Images", metrics.get("quarantined_images", 0))

            st.subheader("🔍 Individual Audit Checks")

            checks = audit.get("scorecard", [])

            if checks:
                checks_df = pd.DataFrame(checks)
                st.dataframe(checks_df, use_container_width=True)
            else:
                st.info("Individual audit checks were not found in the report.")

            st.subheader("⚠️ Warnings")

            warnings = [
                check for check in checks
                if str(check.get("status", "")).upper() == "WARNING"
            ]

            if warnings:
                for warning in warnings:
                    st.warning(
                        f"{warning.get('check_id', '')} — "
                        f"{warning.get('check_name', '')}\n\n"
                        f"{warning.get('actual', '')}"
                    )
            else:
                st.success("No warnings reported.")

            st.subheader("📝 Audit Findings")

            findings = audit.get("documented_findings", [])

            if findings:
                for finding in findings:
                    st.write("•", finding.get("description", finding))
            else:
                st.info("No additional findings available.")

        except Exception as e:
            st.error(f"Unable to read the audit report: {e}")

    else:
        st.error("Audit report file not found.")
        st.write(
            "Please make sure the reports folder is inside the main project folder."
        )
        st.code("reports/corrected_final_audit_summary.json")

    


# ==========================================
# ABOUT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About Project")

    st.write(
        """
        Project: Verifiable Indic Document OCR

        Institution: University of Engineering and Management,
        Kolkata.

        Department: Computer Science and Engineering.

        Technology Stack:
        - Python
        - Streamlit
        - Pandas
        - Plotly
        - OpenCV
        - Pillow
        """
    )
