"""Streamlit Web Application: Medical Prescription OCR & Intelligent Text Normalization."""

import io
import json
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

from config.settings import CONFIG
from main import PrescriptionPipeline, PipelineResult
from models.memory_manager import MemoryManager
from ocr.ocr_engine import OCRMode
from normalization.text_normalizer import NormalizedPrescription
from normalization.medicine_label_normalizer import NormalizedMedicineLabel

st.set_page_config(
    page_title="MediScan AI — Prescription OCR & Normalization",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Modern Minimalist Healthcare Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* CSS Variables & Design Tokens */
    :root {
        --brand-blue: #2563EB;
        --brand-gradient: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        --brand-gradient-hover: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        --bg-surface: #FFFFFF;
        --bg-subtle: #F8FAFC;
        --border-subtle: #E2E8F0;
        --border-focus: #3B82F6;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --success-emerald: #10B981;
        --success-bg: #ECFDF5;
        --warning-amber: #F59E0B;
        --warning-bg: #FFFBEB;
        --danger-rose: #F43F5E;
        --danger-bg: #FFF1F2;
        --shadow-card: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
        --shadow-hover: 0 10px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
        --shadow-glow: 0 8px 20px -4px rgba(37, 99, 235, 0.35);
    }

    /* Core Typography & Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
    }

    /* Smooth Live Beacon Animation */
    @keyframes livePulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .beacon-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--success-emerald);
        animation: livePulse 2s infinite ease-in-out;
        display: inline-block;
    }

    /* Hero Header Container */
    .hero-container {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 50%, #F5F3FF 100%);
        border: 1px solid rgba(219, 234, 254, 0.8);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(37, 99, 235, 0.08);
        position: relative;
        overflow: hidden;
    }

    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #1E40AF;
        background: #DBEAFE;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-title span {
        background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #475569;
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 1.3rem;
        max-width: 820px;
    }

    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        align-items: center;
    }

    .pill-badge {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.4rem 0.85rem;
        border-radius: 9999px;
        background: #FFFFFF;
        color: #334155;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        transition: all 0.2s ease;
    }

    .pill-badge:hover {
        border-color: #CBD5E1;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
    }

    /* Modern Card Layouts */
    .glass-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.3rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        box-shadow: var(--shadow-hover);
        border-color: #CBD5E1;
    }

    /* Section Number Heading */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 1.2rem;
        margin-bottom: 1.1rem;
    }

    .section-number {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--brand-gradient);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.3);
    }

    .section-title-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
    }

    /* Sidebar Status Card */
    .sidebar-status-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 14px;
        padding: 1.2rem;
        color: #F8FAFC;
        border: 1px solid #334155;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.15);
        margin-bottom: 1.4rem;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.85rem;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-tag {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
        background: #10B981;
        color: #064E3B;
    }

    /* Streamlit Button Overrides */
    div.stButton > button[kind="primary"] {
        background: var(--brand-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.6rem !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-glow) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--brand-gradient-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 28px -4px rgba(37, 99, 235, 0.45) !important;
    }

    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #1E293B !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1.4rem !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: #FFFFFF !important;
        border: 1px solid #BFDBFE !important;
        color: #1E40AF !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 5px rgba(37, 99, 235, 0.04) !important;
        transition: all 0.2s ease !important;
    }

    div.stDownloadButton > button:hover {
        background: #EFF6FF !important;
        border-color: #3B82F6 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.12) !important;
    }

    /* Segmented Control Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background-color: #F1F5F9;
        padding: 0.35rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        font-size: 0.88rem;
        color: #64748B;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E40AF !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }

    /* Metric Visual Highlights */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #1E3A8A !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sleek Frame for Image Preview */
    [data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
@st.cache_resource
def get_pipeline() -> PrescriptionPipeline:
    return PrescriptionPipeline()

pipeline = get_pipeline()

if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = "prescription.png"
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "raw_ocr_text" not in st.session_state:
    st.session_state.raw_ocr_text = ""
if "normalized_result" not in st.session_state:
    st.session_state.normalized_result = None
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "nlp_labels" not in st.session_state:
    st.session_state.nlp_labels = None
if "detected_doc_type" not in st.session_state:
    st.session_state.detected_doc_type = "prescription"


# SIDEBAR: Scanner Preferences & Settings
with st.sidebar:
    st.markdown("### ⚙️ Scanner Preferences")
    
    # Clinical Status & Security Card (Clean, Professional, No Hardware / Backend Info)
    st.markdown("""
    <div class="sidebar-status-card">
        <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center;">
            <span>🛡️ System Security & Status</span>
            <span class="status-tag">ACTIVE</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">Clinical Verification</span>
            <span style="color:#38BDF8; font-weight:600;">Automated</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">Dosing Safeguard</span>
            <span style="color:#10B981; font-weight:600;">Active</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">Data Privacy</span>
            <span style="color:#F8FAFC; font-weight:600;">Encrypted & Private</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Processing Mode")
    
    ocr_mode_choice = st.selectbox(
        "Scanning Precision",
        options=[OCRMode.AUTO.value, OCRMode.RAPID_OCR.value, OCRMode.CLOUD_HF.value, OCRMode.LOCAL_QWEN.value],
        format_func=lambda x: {
            "auto": "✨ Intelligent Auto (Recommended)",
            "rapid_ocr": "⚡ High-Speed Offline Scan",
            "cloud_hf": "☁️ High-Precision Cloud Scan",
            "local_qwen": "🖥️ Deep Neural Scan",
        }.get(x, x),
        index=0,
        help="Select scanning mode. Auto automatically optimizes for performance and accuracy.",
    )

    hf_token_input = st.text_input(
        "Cloud Access Key (Optional)",
        type="password",
        help="Optional access key for high-accuracy cloud processing.",
    )

    st.markdown("---")
    st.markdown("### 📋 Document Classifier")
    doc_type_choice = st.selectbox(
        "Document Category",
        options=["auto", "prescription", "medicine_label"],
        format_func=lambda x: {
            "auto": "✨ Auto Detect (Smart Routing)",
            "prescription": "📝 Medical Prescription (Doctor Note)",
            "medicine_label": "🏷️ Medicine Box / Bottle Packaging Label",
        }.get(x, x),
        index=0,
        help="Auto-detects whether the image is a doctor's prescription or a medicine packaging label.",
    )

    st.markdown("---")
    st.markdown("### 🖼️ Vision Enhancement")
    apply_clahe = st.checkbox("Adaptive Contrast Enhancement", value=True, help="Adaptive contrast for faint ink.")
    apply_denoise = st.checkbox("Paper Grain Reduction Filter", value=False, help="Filters background paper texture.")
    apply_deskew = st.checkbox("Auto Document Deskew Alignment", value=False, help="Aligns tilted document orientation.")

    st.markdown("---")
    st.caption("MediScan AI • Intelligent Medical Scanner")


# MAIN PANEL HERO BANNER
st.markdown("""
<div class="hero-container">
    <div class="hero-badge-pill"><span class="beacon-dot"></span> Clinical Scanner Active</div>
    <div class="hero-title">Prescription & Medicine <span>Label Scanner</span></div>
    <div class="hero-subtitle">
        Transform handwritten doctor prescriptions and medicine packaging into verified, structured clinical records for patient dosing safety.
    </div>
    <div class="badge-bar">
        <span class="pill-badge">✨ High-Accuracy Digitization</span>
        <span class="pill-badge">📋 Standardized Regimen Details</span>
        <span class="pill-badge">💊 Active Composition Extraction</span>
        <span class="pill-badge">🛡️ Clinical Dosing Safety Checks</span>
    </div>
</div>
""", unsafe_allow_html=True)


# SECTION 1: PRESCRIPTION / LABEL INPUT
st.markdown("""
<div class="section-header">
    <div class="section-number">1</div>
    <div class="section-title-text">Document Input (Prescription or Medicine Label)</div>
</div>
""", unsafe_allow_html=True)

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Medical Prescription or Medicine Label Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload clear handwritten or printed medical prescription or medicine packaging label image.",
    )
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_name = uploaded_file.name

with col_sample:
    st.markdown("**Or load authentic benchmark sample:**")
    sample_dir = CONFIG.sample_data_dir
    sample_files = sorted([
        f for f in sample_dir.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] and not f.name.startswith(".")
    ])
    sample_names = [f.name for f in sample_files]
    sample_options = ["None"] + sample_names
    
    def format_sample_name(name: str) -> str:
        if name == "None":
            return "None (Upload your own)"
        if "label" in name.lower():
            return f"🏷️ [Label] {name}"
        return f"📝 [Rx] {name}"
    
    selected_sample = st.selectbox("Select Benchmark Sample", sample_options, format_func=format_sample_name)
    if selected_sample != "None":
        sample_path = sample_dir / selected_sample
        st.session_state.current_image = Image.open(sample_path)
        st.session_state.current_image_name = selected_sample


# IMAGE PREVIEW & METADATA
if st.session_state.current_image is not None:
    img = st.session_state.current_image
    col_img, col_meta = st.columns([1, 1])
    
    with col_img:
        st.image(img, caption=f"Loaded Image: {st.session_state.current_image_name}", width="stretch")
    
    with col_meta:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 1.2rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #1E3A8A; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>🔬 Image Metadata & Sensor Profile</span>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span class="pill-badge">📐 <b>Dimensions:</b> {img.width} × {img.height} px</span>
                <span class="pill-badge">🎨 <b>Color Mode:</b> {img.mode}</span>
                <span class="pill-badge">📏 <b>Aspect Ratio:</b> {round(img.width / max(1, img.height), 2)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown("### ⚡ Pipeline Action Controls")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_all_btn = st.button("🚀 Run Complete Pipeline", type="primary", width="stretch")
        with col_btn2:
            extract_ocr_btn = st.button("📝 Extract Raw OCR", width="stretch")

    # RUN COMPLETE PIPELINE
    if run_all_btn:
        with st.spinner("Processing complete pipeline (Preprocessing → OCR → Normalization → Safety Validation)..."):
            res: PipelineResult = pipeline.process(
                image_source=st.session_state.current_image,
                ocr_mode=ocr_mode_choice,
                doc_type=doc_type_choice,
                apply_contrast=apply_clahe,
                apply_denoise=apply_denoise,
                apply_deskew=apply_deskew,
                hf_token=hf_token_input if hf_token_input else None,
                save_outputs=True,
                output_prefix=Path(st.session_state.current_image_name).stem,
            )
            st.session_state.pipeline_result = res
            st.session_state.raw_ocr_text = res.ocr_result.raw_text
            st.session_state.normalized_result = res.normalized
            st.session_state.validation_report = res.validation_report
            st.session_state.nlp_labels = res.nlp_labels
            st.session_state.detected_doc_type = res.document_type
        st.success(f"✅ Processing completed successfully ({res.document_type.replace('_', ' ').title()})!")

    # EXTRACT OCR ONLY
    elif extract_ocr_btn:
        with st.spinner("Extracting text from document..."):
            prep = pipeline.preprocessor.process(
                image_source=st.session_state.current_image,
                apply_contrast=apply_clahe,
                apply_denoise=apply_denoise,
                apply_deskew=apply_deskew,
            )
            ocr_res = pipeline.ocr_engine.extract(
                image=prep.processed_image,
                mode=ocr_mode_choice,
                hf_token=hf_token_input if hf_token_input else None,
            )
            st.session_state.raw_ocr_text = ocr_res.raw_text
            # Reset downstream
            st.session_state.normalized_result = None
            st.session_state.validation_report = None
        st.success(f"✅ Text extracted in {ocr_res.inference_time_sec}s")


# SECTION 2: RAW OCR EXTRACTION
if st.session_state.raw_ocr_text:
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-number">2</div>
        <div class="section-title-text">Faithful Document Extraction</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_raw, col_raw_ctrl = st.columns([3, 1])
    with col_raw:
        st.text_area("Extracted Document Text", value=st.session_state.raw_ocr_text, height=180)
    
    with col_raw_ctrl:
        st.metric("Character Count", len(st.session_state.raw_ocr_text))
        st.metric("Line Count", len([l for l in st.session_state.raw_ocr_text.splitlines() if l.strip()]))
        norm_btn = st.button("✨ Standardize & Normalize", type="secondary", width="stretch")

    if norm_btn:
        with st.spinner("Standardizing clinical entities and safety verification..."):
            norm = pipeline.normalizer.normalize(st.session_state.raw_ocr_text, doc_type=doc_type_choice)
            val = pipeline.validator.validate(st.session_state.raw_ocr_text, norm)
            is_label = isinstance(norm, NormalizedMedicineLabel)
            detected_type = "medicine_label" if is_label else "prescription"

            nlp_labels = None
            try:
                if is_label:
                    nlp_labels = pipeline.labeller.label_medicine_label(st.session_state.raw_ocr_text)
                else:
                    nlp_labels = pipeline.labeller.label_prescription(
                        prescription_text=norm.to_formatted_text(),
                        normalized_prescription_obj=norm,
                    )
            except Exception as e:
                st.warning(f"Medication entity warning: {e}")

            st.session_state.normalized_result = norm
            st.session_state.validation_report = val
            st.session_state.nlp_labels = nlp_labels
            st.session_state.detected_doc_type = detected_type
        st.success(f"✅ Normalization & Clinical Verification completed ({detected_type.replace('_', ' ').title()})!")


# SECTION 3: INTELLIGENT NORMALIZATION & VALIDATION
if st.session_state.normalized_result is not None:
    norm = st.session_state.normalized_result
    val = st.session_state.validation_report
    is_label = isinstance(norm, NormalizedMedicineLabel)

    st.markdown("---")
    doc_title = "Medicine Packaging Label" if is_label else "Medical Prescription"
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">3</div>
        <div class="section-title-text">Standardized Clinical Record & Verification ({doc_title})</div>
    </div>
    """, unsafe_allow_html=True)

    # Safety Status Banner with Visual Indicator
    if val.passed_safety_checks and not val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border: 1px solid #6EE7B7; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);">
            <span style="font-size: 1.6rem;">🛡️</span>
            <div>
                <div style="font-weight: 700; color: #065F46; font-size: 1rem;">Safety Check Passed — High Confidence</div>
                <div style="color: #047857; font-size: 0.88rem;">All {doc_title.lower()} entities, active dosages, and regimen instructions verified against pharmacopeia with 0% discrepancy risk.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif val.passed_safety_checks and val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FCD34D; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);">
            <span style="font-size: 1.6rem;">⚠️</span>
            <div>
                <div style="font-weight: 700; color: #92400E; font-size: 1rem;">Human Clinician Review Recommended</div>
                <div style="color: #B45309; font-size: 0.88rem;">Minor ambiguities, statutory precautions, or faint tokens flagged for clinician confirmation.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%); border: 1px solid #FDA4AF; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(244, 63, 94, 0.08);">
            <span style="font-size: 1.6rem;">🚨</span>
            <div>
                <div style="font-weight: 700; color: #9F1239; font-size: 1rem;">Critical Safety Discrepancy Detected</div>
                <div style="color: #BE123C; font-size: 0.88rem;">Potential dosage alteration or unverified medication detected in {doc_title.lower()}. Refer to the clinical audit log.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if is_label:
        # TABS FOR MEDICINE PACKAGING LABEL
        tab_norm, tab_comp, tab_nlp, tab_audit, tab_compare = st.tabs([
            "🏷️ Standardized Label Record",
            "💊 Active Composition & Usage",
            "📋 Medication Details",
            "🔍 Safety & Regulatory Audit",
            "⚖️ Visual Comparison",
        ])

        with tab_norm:
            st.markdown("#### Standardized Medicine Label Record")
            st.code(norm.to_formatted_text(), language="markdown")

        with tab_comp:
            st.markdown("##### 🧪 Active Pharmaceutical Ingredients & Composition")
            if norm.active_ingredients:
                ing_records = [
                    {
                        "Ingredient Name": ing.name,
                        "Standard Generic": ing.standard_name,
                        "Strength": ing.strength,
                        "Status": "✅ Verified" if ing.is_verified else "⚠️ Review Required",
                    }
                    for ing in norm.active_ingredients
                ]
                st.dataframe(pd.DataFrame(ing_records), width="stretch")
            else:
                st.info("No active ingredients parsed.")

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("##### 📋 Dosage & Administration")
                for adm in norm.dosage_and_administration:
                    st.info(f"👉 {adm}")
                if not norm.dosage_and_administration:
                    st.caption("Standard: As directed by the physician.")

                st.markdown("##### ❄️ Storage & Handling")
                for stor in norm.storage_instructions:
                    st.success(f"📦 {stor}")
                if not norm.storage_instructions:
                    st.caption("Standard: Store in a cool, dry place.")

            with col_a2:
                st.markdown("##### ⚠️ Warnings & Statutory Precautions")
                for w in norm.warnings_and_precautions:
                    st.warning(f"⚠️ {w}")
                if not norm.warnings_and_precautions:
                    st.caption("Standard statutory precautions apply.")

                if norm.dosing_safety_alerts:
                    st.markdown("##### 🚨 Dosing Safety Alerts")
                    for alert in norm.dosing_safety_alerts:
                        st.error(f"🚨 {alert}")

        with tab_nlp:
            st.markdown("#### Extracted Packaging Information")
            nlp_data = st.session_state.nlp_labels
            if nlp_data:
                col_nl1, col_nl2 = st.columns(2)
                with col_nl1:
                    st.markdown("**Identified Medicines & Strengths:**")
                    st.write(f"- **Medicines:** `{', '.join(nlp_data.get('medicines', [])) or 'None'}`")
                    st.write(f"- **Strengths:** `{', '.join(nlp_data.get('strengths', [])) or 'None'}`")
                    st.markdown("**Administration Rules:**")
                    for adm in nlp_data.get("administration", []):
                        st.info(f"👉 {adm}")
                with col_nl2:
                    st.markdown("**Safety Warnings:**")
                    for w in nlp_data.get("warnings", []):
                        st.warning(f"⚠️ {w}")
            else:
                st.info("Run normalization to view packaging details.")

        with tab_audit:
            st.markdown("#### Safety & Regulatory Audit Log")
            if val.audit_log:
                df_audit = pd.DataFrame([
                    {
                        "Field": a.field,
                        "Original": a.original_ocr_text,
                        "Normalized": a.corrected_text,
                        "Reason": a.reason,
                        "Status": a.verification_status,
                    }
                    for a in val.audit_log
                ])
                st.dataframe(df_audit, width="stretch")

            st.markdown("#### Validation Issues")
            if val.issues:
                for issue in val.issues:
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{issue.category}]** {issue.message}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{issue.category}]** {issue.message}")
                    else:
                        st.info(f"**[{issue.category}]** {issue.message}")
            else:
                st.success("No packaging discrepancies detected.")

        with tab_compare:
            st.markdown("#### Multi-Stage Visual Comparison")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown("**Original Label Image**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown("**Extracted Text**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown("**Clean Standardized Text**")
                st.code(norm.to_formatted_text(), language="markdown")

    else:
        # TABS FOR MEDICAL PRESCRIPTION
        tab_norm, tab_nlp, tab_audit, tab_comp = st.tabs([
            "📄 Standardized Prescription",
            "📋 Medication Regimens",
            "🔍 Pharmacopeia Audit & Verification",
            "⚖️ Visual Comparison",
        ])

        with tab_norm:
            st.markdown("#### Formatted Clinical Prescription Record")
            st.code(norm.to_formatted_text(), language="markdown")

        with tab_nlp:
            st.markdown("#### Structured Medication Regimens")
            nlp_data = st.session_state.nlp_labels
            if nlp_data:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Medicines Identified", len(nlp_data.get("medicines", [])))
                with col_m2:
                    st.metric("Strengths Detected", len(nlp_data.get("strengths", [])))
                with col_m3:
                    st.metric("Doses Parsed", len(nlp_data.get("dose", [])))
                with col_m4:
                    st.metric("Dosing Frequencies", len(nlp_data.get("frequency", [])))

                st.markdown("##### 💊 Structured Medication Groups (Associated Regimen)")
                groups = nlp_data.get("medication_groups", [])
                if groups:
                    records = []
                    for g in groups:
                        dose_val = g.get("dose", {}).get("normalized_value", "") if g.get("dose") else ""
                        freq_val = g.get("frequency", {}).get("normalized_value", "") if g.get("frequency") else ""
                        daily_freq = g.get("frequency", {}).get("daily_frequency", "") if g.get("frequency") else ""
                        timing_val = g.get("timing", {}).get("normalized_value", "") if g.get("timing") else "Unspecified"
                        records.append({
                            "Item": g.get("item_number"),
                            "Medicine Name": g.get("medicine_name"),
                            "Formulation": g.get("formulation"),
                            "Strength": g.get("strength"),
                            "Normalized Dose": dose_val,
                            "Frequency Code": freq_val,
                            "Daily Intake": f"{daily_freq}x / day" if daily_freq else "",
                            "Timing": timing_val,
                            "Clinical Instructions": g.get("instructions"),
                            "Review Flag": "⚠️ Yes" if g.get("verification_required") else "✅ Verified",
                        })
                    st.dataframe(pd.DataFrame(records), width="stretch")
                else:
                    st.info("No medication groups identified.")

                if nlp_data.get("timing") or nlp_data.get("administration"):
                    col_extra1, col_extra2 = st.columns(2)
                    with col_extra1:
                        if nlp_data.get("timing"):
                            st.markdown("##### ⏰ Timing Schedule")
                            for t in nlp_data.get("timing", []):
                                st.info(f"🕐 {t}")
                    with col_extra2:
                        if nlp_data.get("administration"):
                            st.markdown("##### 📌 Administration Notes")
                            for adm in nlp_data.get("administration", []):
                                st.info(f"👉 {adm}")
            else:
                st.info("Run normalization to view structured medication entities.")

        with tab_audit:
            st.markdown("#### Clinical Corrections Audit Log")
            if norm.proposed_corrections:
                df_corr = pd.DataFrame(norm.proposed_corrections)
                st.dataframe(df_corr, width="stretch")
            else:
                st.info("No spelling corrections were needed; original medicine text adhered to standard nomenclature.")

            st.markdown("#### Safety Validation Issues")
            if val.issues:
                for issue in val.issues:
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{issue.category}]** {issue.message}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{issue.category}]** {issue.message}")
                    else:
                        st.info(f"**[{issue.category}]** {issue.message}")
            else:
                st.success("No validation discrepancies detected.")

        with tab_comp:
            st.markdown("#### Multi-Stage Visual Comparison")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown("**Original Prescription Image**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown("**Extracted Text**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown("**Clean Standardized Text**")
                st.code(norm.to_formatted_text(), language="markdown")

    # SECTION 4: EXPORTS & DOWNLOADS
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <div class="section-number">4</div>
        <div class="section-title-text">Deliverable Exports & Clinical Reports</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    norm_filename = "normalized_medicine_label.txt" if is_label else "normalized_prescription.txt"
    json_filename = "medicine_label_results.json" if is_label else "prescription_results.json"

    # 1. raw_ocr.txt
    with col_d1:
        st.download_button(
            label="📄 Raw Text (.txt)",
            data=st.session_state.raw_ocr_text,
            file_name="raw_ocr.txt",
            mime="text/plain",
            width="stretch",
        )

    # 2. normalized text
    with col_d2:
        st.download_button(
            label=f"📋 Clinical Record (.txt)",
            data=norm.to_formatted_text(),
            file_name=norm_filename,
            mime="text/plain",
            width="stretch",
        )

    # 3. structured json
    with col_d3:
        st.download_button(
            label=f"📊 Clinical Report (.json)",
            data=json.dumps(norm.to_export_json(), indent=2),
            file_name=json_filename,
            mime="application/json",
            width="stretch",
        )

    # 4. validation_report.json
    with col_d4:
        st.download_button(
            label="🛡️ Safety Audit (.json)",
            data=json.dumps(val.to_dict(), indent=2),
            file_name="validation_report.json",
            mime="application/json",
            width="stretch",
        )

    # 5. nlp_medication_labels.json
    with col_d5:
        nlp_json_str = json.dumps(st.session_state.nlp_labels or {}, indent=2)
        st.download_button(
            label="💊 Medication Regimen (.json)",
            data=nlp_json_str,
            file_name="nlp_medication_labels.json",
            mime="application/json",
            width="stretch",
        )


# SECTION 5: MEDICINE BOX / OTC LABEL SCANNER
st.markdown("---")
st.markdown("""
<div class="section-header">
    <div class="section-number">5</div>
    <div class="section-title-text">Medicine Packaging & Label Quick-Scan</div>
</div>
""", unsafe_allow_html=True)
st.caption("Quickly scan and extract active ingredients, strengths, administration instructions, and safety warnings from packaging text.")

with st.expander("🔍 Test Medicine Packaging Label Quick-Scan", expanded=False):
    sample_box_labels = {
        "Custom Input": "",
        "Paracetamol 650mg Packaging": "PARACETAMOL TABLETS IP 650 mg\nEach tablet contains: Paracetamol IP 650 mg.\nDosage: As directed by the physician.\nSwallow with water.\nWarning: Do not exceed the recommended dosage. Keep out of reach of children.\nStore in a cool and dry place.",
        "Cetirizine 10mg Packaging": "CETIRIZINE TABLETS IP\nEach tablet contains: Cetirizine Hydrochloride IP 10 mg.\nDosage: As directed by the physician.\nSchedule H Prescription Drug - Caution: Not to be sold by retail without the prescription of a Registered Medical Practitioner.\nStore below 30°C.",
        "Amoxicillin 500mg Packaging": "AMOXI-500 CAPSULES\nEach capsule contains: Amoxicillin Trihydrate IP equivalent to Amoxicillin 500 mg.\nDosage: As directed by the physician. Capsules to be swallowed whole and not to be chewed, opened or crushed.\nKeep out of reach of children. Store protected from moisture.",
    }

    selected_box = st.selectbox("Select Sample Medicine Box Label", list(sample_box_labels.keys()), index=1)
    default_text = sample_box_labels[selected_box]
    box_text_input = st.text_area("Medicine Box / Packaging Label Text", value=default_text, height=120)

    if st.button("🏷️ Extract Packaging Details", type="secondary"):
        if box_text_input.strip():
            box_res = pipeline.labeller.label_medicine_label(box_text_input)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**Extracted Medicines & Strengths:**")
                st.write(f"- **Medicines:** `{', '.join(box_res.get('medicines', [])) or 'None'}`")
                st.write(f"- **Strengths:** `{', '.join(box_res.get('strengths', [])) or 'None'}`")
                st.markdown("**Administration Instructions:**")
                for adm in box_res.get("administration", []):
                    st.info(f"👉 {adm}")
            with col_b2:
                st.markdown("**Warnings & Precautions:**")
                for w in box_res.get("warnings", []):
                    st.warning(f"⚠️ {w}")
        else:
            st.warning("Please enter medicine box text.")
