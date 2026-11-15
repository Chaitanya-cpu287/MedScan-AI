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
from translations import t, LANGUAGES
import streamlit.components.v1 as components
from tts_reader import (
    build_regimens_speech_text,
    build_audit_speech_text,
    build_label_composition_speech_text,
    generate_speech_audio,
    get_browser_speech_html,
)
from clinical_translator import (
    translate_formulation,
    translate_timing,
    translate_daily_intake,
    translate_dose,
    translate_clinical_instruction,
    translate_audit_reason,
    translate_safety_message,
    format_audit_dataframe,
    build_localized_prescription_card,
    build_localized_prescription_text,
)

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
if "user_lang" not in st.session_state:
    st.session_state.user_lang = "en"

def tr(key: str) -> str:
    """Translation helper for current user language."""
    return t(key, st.session_state.user_lang)


# TOP NAVIGATION & MULTILINGUAL LANGUAGE SELECTOR BAR
col_nav_brand, col_nav_lang = st.columns([2.5, 1.5])
with col_nav_brand:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; padding: 0.2rem 0 0.8rem 0;">
        <span style="font-size: 1.5rem;">💊</span>
        <span style="font-weight: 800; font-size: 1.3rem; color: #1E3A8A; letter-spacing: -0.02em;">MediScan AI</span>
        <span style="background: #EFF6FF; color: #2563EB; font-size: 0.72rem; font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 9999px; border: 1px solid #BFDBFE;">HEALTHCARE</span>
    </div>
    """, unsafe_allow_html=True)

with col_nav_lang:
    selected_lang = st.selectbox(
        "🌐 " + tr("language_selector_label"),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: f"{LANGUAGES[code]['flag']} {LANGUAGES[code]['native']} ({LANGUAGES[code]['name']})",
        index=list(LANGUAGES.keys()).index(st.session_state.user_lang),
        key="top_language_selector",
        help="Select language: English, हिन्दी, తెలుగు, தமிழ், বাংলা, मराठी, ಕನ್ನಡ, ગુજરાતી, മലയാളം, ਪੰਜਾਬੀ",
    )
    if selected_lang != st.session_state.user_lang:
        st.session_state.user_lang = selected_lang
        st.rerun()


# SIDEBAR: Scanner Preferences & Settings
with st.sidebar:
    st.markdown(f"### {tr('sidebar_preferences')}")
    
    # Clinical Status & Security Card
    st.markdown(f"""
    <div class="sidebar-status-card">
        <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center;">
            <span>{tr('sidebar_status_title')}</span>
            <span class="status-tag">{tr('sidebar_status_active')}</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{tr('sidebar_verification')}</span>
            <span style="color:#38BDF8; font-weight:600;">{tr('sidebar_verification_val')}</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{tr('sidebar_safeguard')}</span>
            <span style="color:#10B981; font-weight:600;">{tr('sidebar_safeguard_val')}</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{tr('sidebar_privacy')}</span>
            <span style="color:#F8FAFC; font-weight:600;">{tr('sidebar_privacy_val')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {tr('sidebar_mode_title')}")
    
    ocr_mode_choice = st.selectbox(
        tr("sidebar_precision_label"),
        options=[OCRMode.AUTO.value, OCRMode.RAPID_OCR.value, OCRMode.CLOUD_HF.value, OCRMode.LOCAL_QWEN.value],
        format_func=lambda x: {
            "auto": tr("mode_auto"),
            "rapid_ocr": tr("mode_rapid"),
            "cloud_hf": tr("mode_cloud"),
            "local_qwen": tr("mode_neural"),
        }.get(x, x),
        index=0,
        help="Auto automatically optimizes for performance and accuracy.",
    )

    hf_token_input = st.text_input(
        tr("sidebar_key_label"),
        type="password",
        help=tr("sidebar_key_help"),
    )

    st.markdown("---")
    st.markdown(f"### {tr('sidebar_classifier_title')}")
    doc_type_choice = st.selectbox(
        tr("sidebar_doc_category"),
        options=["auto", "prescription", "medicine_label"],
        format_func=lambda x: {
            "auto": tr("doc_auto"),
            "prescription": tr("doc_rx"),
            "medicine_label": tr("doc_label"),
        }.get(x, x),
        index=0,
        help="Auto-detects whether the image is a doctor's prescription or a medicine packaging label.",
    )

    st.markdown("---")
    st.markdown(f"### {tr('sidebar_vision_title')}")
    apply_clahe = st.checkbox(tr("enhance_clahe"), value=True, help="Adaptive contrast for faint ink.")
    apply_denoise = st.checkbox(tr("enhance_denoise"), value=False, help="Filters background paper texture.")
    apply_deskew = st.checkbox(tr("enhance_deskew"), value=True, help="Aligns tilted document orientation.")

    st.markdown("---")
    st.caption(tr("sidebar_footer"))


# MAIN PANEL HERO BANNER
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge-pill"><span class="beacon-dot"></span> {tr('scanner_active')}</div>
    <div class="hero-title">{tr('hero_title_prefix')} <span>{tr('hero_title_suffix')}</span></div>
    <div class="hero-subtitle">
        {tr('hero_subtitle')}
    </div>
    <div class="badge-bar">
        <span class="pill-badge">{tr('badge_digitization')}</span>
        <span class="pill-badge">{tr('badge_regimen')}</span>
        <span class="pill-badge">{tr('badge_composition')}</span>
        <span class="pill-badge">{tr('badge_safety')}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# SECTION 1: PRESCRIPTION / LABEL INPUT
st.markdown(f"""
<div class="section-header">
    <div class="section-number">1</div>
    <div class="section-title-text">{tr('sec1_title')}</div>
</div>
""", unsafe_allow_html=True)

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        tr("upload_label"),
        type=["jpg", "jpeg", "png", "webp"],
        help=tr("upload_help"),
    )
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_name = uploaded_file.name

with col_sample:
    st.markdown(f"**{tr('sample_heading')}**")
    sample_dir = CONFIG.sample_data_dir
    sample_files = sorted([
        f for f in sample_dir.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] and not f.name.startswith(".")
    ])
    sample_names = [f.name for f in sample_files]
    sample_options = ["None"] + sample_names
    
    def format_sample_name(name: str) -> str:
        if name == "None":
            return tr("sample_none")
        if "label" in name.lower():
            return f"🏷️ [Label] {name}"
        return f"📝 [Rx] {name}"
    
    selected_sample = st.selectbox(tr("sample_select_label"), sample_options, format_func=format_sample_name)
    if selected_sample != "None":
        sample_path = sample_dir / selected_sample
        st.session_state.current_image = Image.open(sample_path)
        st.session_state.current_image_name = selected_sample


# IMAGE PREVIEW & METADATA
if st.session_state.current_image is not None:
    img = st.session_state.current_image
    col_img, col_meta = st.columns([1, 1])
    
    with col_img:
        st.image(img, caption=f"{tr('loaded_image_caption')}: {st.session_state.current_image_name}", width="stretch")
    
    with col_meta:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 1.2rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #1E3A8A; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>{tr('meta_title')}</span>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span class="pill-badge">📐 <b>{tr('meta_dimensions')}:</b> {img.width} × {img.height} px</span>
                <span class="pill-badge">🎨 <b>{tr('meta_color')}:</b> {img.mode}</span>
                <span class="pill-badge">📏 <b>{tr('meta_aspect')}:</b> {round(img.width / max(1, img.height), 2)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown(f"### {tr('pipeline_action_controls')}")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_all_btn = st.button(tr("btn_run_all"), type="primary", width="stretch")
        with col_btn2:
            extract_ocr_btn = st.button(tr("btn_extract_ocr"), width="stretch")

    # RUN COMPLETE PIPELINE
    if run_all_btn:
        with st.spinner(tr("spinner_pipeline")):
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
        doc_disp = tr("doc_label") if res.document_type == "medicine_label" else tr("doc_rx")
        st.success(f"✅ {tr('success_pipeline')} ({doc_disp})!")

    # EXTRACT OCR ONLY
    elif extract_ocr_btn:
        with st.spinner(tr("spinner_ocr")):
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
        st.success(f"✅ {tr('success_ocr')} {ocr_res.inference_time_sec}s")


# SECTION 2: RAW OCR EXTRACTION
if st.session_state.raw_ocr_text:
    st.markdown("---")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">2</div>
        <div class="section-title-text">{tr('sec2_title')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_raw, col_raw_ctrl = st.columns([3, 1])
    with col_raw:
        st.text_area(tr("extracted_text_label"), value=st.session_state.raw_ocr_text, height=180)
    
    with col_raw_ctrl:
        st.metric(tr("char_count"), len(st.session_state.raw_ocr_text))
        st.metric(tr("line_count"), len([l for l in st.session_state.raw_ocr_text.splitlines() if l.strip()]))
        norm_btn = st.button(tr("btn_normalize"), type="secondary", width="stretch")

    if norm_btn:
        with st.spinner(tr("spinner_normalize")):
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
        doc_disp = tr("doc_label") if detected_type == "medicine_label" else tr("doc_rx")
        st.success(f"✅ {tr('success_normalize')} ({doc_disp})!")


# SECTION 3: INTELLIGENT NORMALIZATION & VALIDATION
if st.session_state.normalized_result is not None:
    norm = st.session_state.normalized_result
    val = st.session_state.validation_report
    is_label = isinstance(norm, NormalizedMedicineLabel)

    st.markdown("---")
    doc_title = tr("doc_label") if is_label else tr("doc_rx")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">3</div>
        <div class="section-title-text">{tr('sec3_title')} ({doc_title})</div>
    </div>
    """, unsafe_allow_html=True)

    # Safety Status Banner with Visual Indicator
    if val.passed_safety_checks and not val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border: 1px solid #6EE7B7; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);">
            <span style="font-size: 1.6rem;">🛡️</span>
            <div>
                <div style="font-weight: 700; color: #065F46; font-size: 1rem;">{tr('banner_safe_title')}</div>
                <div style="color: #047857; font-size: 0.88rem;">{tr('banner_safe_desc')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif val.passed_safety_checks and val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FCD34D; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);">
            <span style="font-size: 1.6rem;">⚠️</span>
            <div>
                <div style="font-weight: 700; color: #92400E; font-size: 1rem;">{tr('banner_review_title')}</div>
                <div style="color: #B45309; font-size: 0.88rem;">{tr('banner_review_desc')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%); border: 1px solid #FDA4AF; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(244, 63, 94, 0.08);">
            <span style="font-size: 1.6rem;">🚨</span>
            <div>
                <div style="font-weight: 700; color: #9F1239; font-size: 1rem;">{tr('banner_critical_title')}</div>
                <div style="color: #BE123C; font-size: 0.88rem;">{tr('banner_critical_desc')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if is_label:
        # TABS FOR MEDICINE PACKAGING LABEL
        tab_norm, tab_comp, tab_nlp, tab_audit, tab_compare = st.tabs([
            tr("tab_lbl_standard"),
            tr("tab_lbl_comp"),
            tr("tab_lbl_details"),
            tr("tab_lbl_audit"),
            tr("tab_lbl_compare"),
        ])

        with tab_norm:
            st.markdown(f"#### {tr('lbl_record_heading')}")
            st.code(norm.to_formatted_text(), language="markdown")

        with tab_comp:
            col_comp_h, col_comp_btn = st.columns([3, 1])
            with col_comp_h:
                st.markdown(f"##### {tr('lbl_comp_heading')}")
            with col_comp_btn:
                speak_comp_btn = st.button(tr("btn_read_out"), key="speak_comp_btn", width="stretch")

            if speak_comp_btn and norm.active_ingredients:
                with st.spinner("Generating speech in selected language..."):
                    speech_text = build_label_composition_speech_text(norm.active_ingredients, lang=st.session_state.user_lang)
                    audio_bytes = generate_speech_audio(speech_text, lang=st.session_state.user_lang)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    else:
                        html_code = get_browser_speech_html(
                            speech_text,
                            lang=st.session_state.user_lang,
                            button_id="browser_speak_comp",
                            button_label=tr("btn_read_out")
                        )
                        components.html(html_code, height=45)

            if norm.active_ingredients:
                ing_records = [
                    {
                        tr("lbl_col_ingredient"): ing.name,
                        tr("lbl_col_generic"): ing.standard_name,
                        tr("lbl_col_strength"): ing.strength,
                        tr("lbl_col_status"): tr("flag_verified") if ing.is_verified else tr("flag_review"),
                    }
                    for ing in norm.active_ingredients
                ]
                st.dataframe(pd.DataFrame(ing_records), width="stretch")
            else:
                st.info(tr("lbl_no_ing"))

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"##### {tr('lbl_admin_heading')}")
                for adm in norm.dosage_and_administration:
                    st.info(f"👉 {adm}")
                if not norm.dosage_and_administration:
                    st.caption(tr("lbl_std_admin"))

                st.markdown(f"##### {tr('lbl_storage_heading')}")
                for stor in norm.storage_instructions:
                    st.success(f"📦 {stor}")
                if not norm.storage_instructions:
                    st.caption(tr("lbl_std_storage"))

            with col_a2:
                st.markdown(f"##### {tr('lbl_warnings_heading')}")
                for w in norm.warnings_and_precautions:
                    st.warning(f"⚠️ {w}")
                if not norm.warnings_and_precautions:
                    st.caption(tr("lbl_std_warnings"))

                if norm.dosing_safety_alerts:
                    st.markdown(f"##### {tr('lbl_alerts_heading')}")
                    for alert in norm.dosing_safety_alerts:
                        st.error(f"🚨 {alert}")

        with tab_nlp:
            st.markdown(f"#### {tr('lbl_extracted_info')}")
            nlp_data = st.session_state.nlp_labels
            if nlp_data:
                col_nl1, col_nl2 = st.columns(2)
                with col_nl1:
                    st.markdown(f"**{tr('lbl_ident_meds')}**")
                    st.write(f"- **{tr('col_medicine')}:** `{', '.join(nlp_data.get('medicines', [])) or '—'}`")
                    st.write(f"- **{tr('col_strength')}:** `{', '.join(nlp_data.get('strengths', [])) or '—'}`")
                    st.markdown(f"**{tr('lbl_admin_rules')}**")
                    for adm in nlp_data.get("administration", []):
                        st.info(f"👉 {adm}")
                with col_nl2:
                    st.markdown(f"**{tr('lbl_safety_warn')}**")
                    for w in nlp_data.get("warnings", []):
                        st.warning(f"⚠️ {w}")
            else:
                st.info(tr("lbl_no_details"))

        with tab_audit:
            col_laud_h, col_laud_btn = st.columns([3, 1])
            with col_laud_h:
                st.markdown(f"#### {tr('lbl_audit_log_heading')}")
            with col_laud_btn:
                speak_laud_btn = st.button(tr("btn_read_out"), key="speak_laud_btn", width="stretch")

            if speak_laud_btn:
                with st.spinner("Generating speech in selected language..."):
                    corr_list = [
                        {"original": a.original_ocr_text, "normalized": a.corrected_text, "status": a.verification_status, "reason": a.reason}
                        for a in (val.audit_log or [])
                    ]
                    speech_text = build_audit_speech_text(corr_list, issues=val.issues, lang=st.session_state.user_lang)
                    audio_bytes = generate_speech_audio(speech_text, lang=st.session_state.user_lang)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    else:
                        html_code = get_browser_speech_html(
                            speech_text,
                            lang=st.session_state.user_lang,
                            button_id="browser_speak_laud",
                            button_label=tr("btn_read_out")
                        )
                        components.html(html_code, height=45)

            if val.audit_log:
                raw_log = [
                    {
                        "Field": a.field,
                        "Original": a.original_ocr_text,
                        "Normalized": a.corrected_text,
                        "Reason": a.reason,
                        "Status": a.verification_status,
                    }
                    for a in val.audit_log
                ]
                df_audit = pd.DataFrame(format_audit_dataframe(raw_log, st.session_state.user_lang))
                st.dataframe(df_audit, width="stretch")

            st.markdown(f"#### {tr('validation_issues_heading')}")
            if val.issues:
                for issue in val.issues:
                    cat_txt, msg_txt = translate_safety_message(issue.message, issue.category, st.session_state.user_lang)
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{cat_txt}]** {msg_txt}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{cat_txt}]** {msg_txt}")
                    else:
                        st.info(f"**[{cat_txt}]** {msg_txt}")
            else:
                st.success(tr("lbl_audit_clean"))

        with tab_compare:
            st.markdown(f"#### {tr('compare_heading')}")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown(f"**{tr('compare_orig_lbl')}**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown(f"**{tr('compare_extracted')}**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown(f"**{tr('compare_clean')}**")
                st.code(norm.to_formatted_text(), language="markdown")

    else:
        # TABS FOR MEDICAL PRESCRIPTION
        tab_norm, tab_nlp, tab_audit, tab_comp = st.tabs([
            tr("tab_rx_standard"),
            tr("tab_rx_regimens"),
            tr("tab_rx_audit"),
            tr("tab_rx_compare"),
        ])

        with tab_norm:
            st.markdown(f"#### {tr('rx_record_heading')}")
            if st.session_state.user_lang != "en":
                st.markdown(build_localized_prescription_card(norm, st.session_state.user_lang), unsafe_allow_html=True)
                st.code(build_localized_prescription_text(norm, st.session_state.user_lang), language="markdown")
            else:
                st.code(norm.to_formatted_text(), language="markdown")

        with tab_nlp:
            st.markdown(f"#### {tr('regimens_heading')}")
            nlp_data = st.session_state.nlp_labels or {}
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(tr("metric_medicines"), len(nlp_data.get("medicines", [])) or (len(norm.medications) if hasattr(norm, "medications") else 0))
            with col_m2:
                st.metric(tr("metric_strengths"), len(nlp_data.get("strengths", [])))
            with col_m3:
                st.metric(tr("metric_doses"), len(nlp_data.get("dose", [])))
            with col_m4:
                st.metric(tr("metric_frequencies"), len(nlp_data.get("frequency", [])))

            groups = nlp_data.get("medication_groups", [])
            if not groups and hasattr(norm, "medications") and norm.medications:
                groups = [
                    {
                        "item_number": idx,
                        "medicine_name": m.medicine_name,
                        "formulation": m.formulation or "",
                        "strength": m.strength or "",
                        "dose": m.dosage_instruction or "",
                        "frequency": m.frequency or "",
                        "timing": m.food_timing or "",
                        "instructions": m.dosage_instruction or "",
                        "verification_required": not m.is_verified,
                    }
                    for idx, m in enumerate(norm.medications, 1)
                ]

            col_tbl_h, col_tbl_btn = st.columns([3, 1])
            with col_tbl_h:
                st.markdown(f"##### {tr('groups_heading')}")
            with col_tbl_btn:
                speak_regimens_btn = st.button(tr("btn_read_out"), key="speak_regimens_btn", width="stretch")

            if speak_regimens_btn:
                if groups:
                    with st.spinner("Generating speech in selected language..."):
                        speech_text = build_regimens_speech_text(groups, lang=st.session_state.user_lang)
                        audio_bytes = generate_speech_audio(speech_text, lang=st.session_state.user_lang)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        else:
                            html_code = get_browser_speech_html(
                                speech_text,
                                lang=st.session_state.user_lang,
                                button_id="browser_speak_regimens",
                                button_label=tr("btn_read_out")
                            )
                            components.html(html_code, height=45)
                else:
                    st.info(tr("no_groups"))

            if groups:
                records = []
                for g in groups:
                    dose_val = g.get("dose", {}).get("normalized_value", "") if isinstance(g.get("dose"), dict) else (g.get("dose") or "")
                    freq_val = g.get("frequency", {}).get("normalized_value", "") if isinstance(g.get("frequency"), dict) else (g.get("frequency") or "")
                    daily_freq = g.get("frequency", {}).get("daily_frequency", "") if isinstance(g.get("frequency"), dict) else ""
                    timing_obj = g.get("timing")
                    if isinstance(timing_obj, dict):
                        timing_val = timing_obj.get("timing_type") or timing_obj.get("normalized_value") or "Unspecified"
                    else:
                        timing_val = timing_obj or "Unspecified"

                    form_val = translate_formulation(g.get("formulation", ""), st.session_state.user_lang)
                    dose_trans = translate_dose(dose_val, st.session_state.user_lang)
                    daily_trans = translate_daily_intake(f"{daily_freq}x / day" if daily_freq else "", st.session_state.user_lang)
                    timing_trans = translate_timing(timing_val, st.session_state.user_lang)
                    instructions_trans = translate_clinical_instruction(g.get("instructions", ""), st.session_state.user_lang)

                    records.append({
                        tr("col_item"): g.get("item_number"),
                        tr("col_medicine"): g.get("medicine_name"),
                        tr("col_formulation"): form_val,
                        tr("col_strength"): g.get("strength"),
                        tr("col_dose"): dose_trans,
                        tr("col_freq"): freq_val,
                        tr("col_daily"): daily_trans,
                        tr("col_timing"): timing_trans,
                        tr("col_instructions"): instructions_trans,
                        tr("col_review"): tr("flag_review") if g.get("verification_required") else tr("flag_verified"),
                    })
                st.dataframe(pd.DataFrame(records), width="stretch")
            else:
                st.info(tr("no_groups"))

            if nlp_data.get("timing") or nlp_data.get("administration"):
                col_extra1, col_extra2 = st.columns(2)
                with col_extra1:
                    if nlp_data.get("timing"):
                        st.markdown(f"##### {tr('timing_schedule')}")
                        for timing_item in nlp_data.get("timing", []):
                            if isinstance(timing_item, dict):
                                t_raw = timing_item.get("timing_type") or timing_item.get("normalized_value") or timing_item.get("original_text", "")
                            else:
                                t_raw = str(timing_item)
                            t_disp = translate_timing(t_raw, st.session_state.user_lang)
                            if t_disp:
                                st.info(f"🕐 {t_disp}")
                with col_extra2:
                    if nlp_data.get("administration"):
                        st.markdown(f"##### {tr('admin_notes')}")
                        for adm in nlp_data.get("administration", []):
                            st.info(f"👉 {adm}")

        with tab_audit:
            col_aud_h, col_aud_btn = st.columns([3, 1])
            with col_aud_h:
                st.markdown(f"#### {tr('audit_heading')}")
            with col_aud_btn:
                speak_audit_btn = st.button(tr("btn_read_out"), key="speak_audit_btn", width="stretch")

            if speak_audit_btn:
                with st.spinner("Generating speech in selected language..."):
                    corrections_list = norm.proposed_corrections if norm.proposed_corrections else []
                    speech_text = build_audit_speech_text(corrections_list, issues=val.issues, lang=st.session_state.user_lang)
                    audio_bytes = generate_speech_audio(speech_text, lang=st.session_state.user_lang)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    else:
                        html_code = get_browser_speech_html(
                            speech_text,
                            lang=st.session_state.user_lang,
                            button_id="browser_speak_audit",
                            button_label=tr("btn_read_out")
                        )
                        components.html(html_code, height=45)

            if norm.proposed_corrections:
                formatted_corr = format_audit_dataframe(norm.proposed_corrections, st.session_state.user_lang)
                df_corr = pd.DataFrame(formatted_corr)
                st.dataframe(df_corr, width="stretch")
            else:
                st.info(tr("audit_clean"))

            st.markdown(f"#### {tr('validation_issues_heading')}")
            if val.issues:
                for issue in val.issues:
                    cat_txt, msg_txt = translate_safety_message(issue.message, issue.category, st.session_state.user_lang)
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{cat_txt}]** {msg_txt}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{cat_txt}]** {msg_txt}")
                    else:
                        st.info(f"**[{cat_txt}]** {msg_txt}")
            else:
                st.success(tr("validation_clean"))

        with tab_comp:
            st.markdown(f"#### {tr('compare_heading')}")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown(f"**{tr('compare_orig_rx')}**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown(f"**{tr('compare_extracted')}**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown(f"**{tr('compare_clean')}**")
                compare_text = (
                    build_localized_prescription_text(norm, st.session_state.user_lang)
                    if st.session_state.user_lang != "en"
                    else norm.to_formatted_text()
                )
                st.code(compare_text, language="markdown")

    # SECTION 4: EXPORTS & DOWNLOADS
    st.markdown("---")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">4</div>
        <div class="section-title-text">{tr('sec4_title')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    norm_filename = "normalized_medicine_label.txt" if is_label else "normalized_prescription.txt"
    json_filename = "medicine_label_results.json" if is_label else "prescription_results.json"

    # 1. raw_ocr.txt
    with col_d1:
        st.download_button(
            label=tr("dl_raw"),
            data=st.session_state.raw_ocr_text,
            file_name="raw_ocr.txt",
            mime="text/plain",
            width="stretch",
        )

    # 2. normalized text
    with col_d2:
        st.download_button(
            label=tr("dl_record"),
            data=norm.to_formatted_text(),
            file_name=norm_filename,
            mime="text/plain",
            width="stretch",
        )

    # 3. structured json
    with col_d3:
        st.download_button(
            label=tr("dl_report"),
            data=json.dumps(norm.to_export_json(), indent=2),
            file_name=json_filename,
            mime="application/json",
            width="stretch",
        )

    # 4. validation_report.json
    with col_d4:
        st.download_button(
            label=tr("dl_audit"),
            data=json.dumps(val.to_dict(), indent=2),
            file_name="validation_report.json",
            mime="application/json",
            width="stretch",
        )

    # 5. nlp_medication_labels.json
    with col_d5:
        nlp_json_str = json.dumps(st.session_state.nlp_labels or {}, indent=2)
        st.download_button(
            label=tr("dl_regimen"),
            data=nlp_json_str,
            file_name="nlp_medication_labels.json",
            mime="application/json",
            width="stretch",
        )


# SECTION 5: MEDICINE BOX / OTC LABEL SCANNER
st.markdown("---")
st.markdown(f"""
<div class="section-header">
    <div class="section-number">5</div>
    <div class="section-title-text">{tr('sec5_title')}</div>
</div>
""", unsafe_allow_html=True)
st.caption(tr("sec5_caption"))

with st.expander(tr("sec5_expander"), expanded=False):
    sample_box_labels = {
        tr("sec5_custom"): "",
        "Paracetamol 650mg Packaging": "PARACETAMOL TABLETS IP 650 mg\nEach tablet contains: Paracetamol IP 650 mg.\nDosage: As directed by the physician.\nSwallow with water.\nWarning: Do not exceed the recommended dosage. Keep out of reach of children.\nStore in a cool and dry place.",
        "Cetirizine 10mg Packaging": "CETIRIZINE TABLETS IP\nEach tablet contains: Cetirizine Hydrochloride IP 10 mg.\nDosage: As directed by the physician.\nSchedule H Prescription Drug - Caution: Not to be sold by retail without the prescription of a Registered Medical Practitioner.\nStore below 30°C.",
        "Amoxicillin 500mg Packaging": "AMOXI-500 CAPSULES\nEach capsule contains: Amoxicillin Trihydrate IP equivalent to Amoxicillin 500 mg.\nDosage: As directed by the physician. Capsules to be swallowed whole and not to be chewed, opened or crushed.\nKeep out of reach of children. Store protected from moisture.",
    }

    selected_box = st.selectbox(tr("sec5_select_sample"), list(sample_box_labels.keys()), index=1)
    default_text = sample_box_labels[selected_box]
    box_text_input = st.text_area(tr("sec5_input_label"), value=default_text, height=120)

    if st.button(tr("sec5_btn"), type="secondary"):
        if box_text_input.strip():
            box_res = pipeline.labeller.label_medicine_label(box_text_input)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f"**{tr('lbl_ident_meds')}**")
                st.write(f"- **{tr('col_medicine')}:** `{', '.join(box_res.get('medicines', [])) or '—'}`")
                st.write(f"- **{tr('col_strength')}:** `{', '.join(box_res.get('strengths', [])) or '—'}`")
                st.markdown(f"**{tr('lbl_admin_rules')}**")
                for adm in box_res.get("administration", []):
                    st.info(f"👉 {adm}")
            with col_b2:
                st.markdown(f"**{tr('lbl_safety_warn')}**")
                for w in box_res.get("warnings", []):
                    st.warning(f"⚠️ {w}")
        else:
            st.warning(tr("sec5_enter_text"))
