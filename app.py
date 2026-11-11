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
from localization import (
    t,
    SUPPORTED_LANGUAGES,
    get_language_options,
    get_language_info,
)
from voice_assistant import (
    VoiceManager,
    SpeechTextFormatter,
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


# Initialize Session State & Singletons
@st.cache_resource
def get_pipeline() -> PrescriptionPipeline:
    return PrescriptionPipeline()

@st.cache_resource
def get_voice_manager() -> VoiceManager:
    return VoiceManager()

pipeline = get_pipeline()
voice_mgr = get_voice_manager()

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en"
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = "auto"
if "active_voice_resp" not in st.session_state:
    st.session_state.active_voice_resp = None

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

lang = st.session_state.selected_language


# SIDEBAR: Scanner Preferences & Settings
with st.sidebar:
    # 1. Interface Language Selector (Top Priority)
    st.markdown(f"### {t('sidebar_lang_selector', lang)}")
    lang_opts = get_language_options()
    lang_idx = lang_opts.index(lang) if lang in lang_opts else 0

    chosen_lang = st.selectbox(
        t("sidebar_lang_selector", lang),
        options=lang_opts,
        index=lang_idx,
        format_func=lambda c: get_language_info(c).display_label,
        key="global_lang_dropdown",
        label_visibility="collapsed",
    )
    if chosen_lang != st.session_state.selected_language:
        st.session_state.selected_language = chosen_lang
        st.session_state.active_voice_resp = None
        st.rerun()

    st.markdown("---")
    st.markdown(f"### ⚙️ {t('system_brand', lang)}")
    
    # Clinical Status & Security Card
    st.markdown(f"""
    <div class="sidebar-status-card">
        <div style="font-weight:700; font-size:0.95rem; margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center;">
            <span>{t('sidebar_security_title', lang)}</span>
            <span class="status-tag">ACTIVE</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{t('sidebar_status_active', lang)}</span>
            <span style="color:#38BDF8; font-weight:600;">{t('status_verified', lang)}</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{t('badge_safety', lang)}</span>
            <span style="color:#10B981; font-weight:600;">{t('sidebar_audit_passed', lang)}</span>
        </div>
        <div class="status-row">
            <span style="color:#94A3B8;">{t('sidebar_privacy_verified', lang)}</span>
            <span style="color:#F8FAFC; font-weight:600;">HIPAA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {t('sidebar_processing_mode', lang)}")
    
    ocr_mode_choice = st.selectbox(
        t("sidebar_scan_precision", lang),
        options=[OCRMode.AUTO.value, OCRMode.RAPID_OCR.value, OCRMode.CLOUD_HF.value, OCRMode.LOCAL_QWEN.value],
        format_func=lambda x: {
            "auto": t("mode_auto", lang),
            "rapid_ocr": t("mode_rapid", lang),
            "cloud_hf": t("mode_cloud", lang),
            "local_qwen": t("mode_local", lang),
        }.get(x, x),
        index=0,
        help=t("sidebar_scan_precision_help", lang),
    )

    hf_token_input = st.text_input(
        t("sidebar_cloud_key", lang),
        type="password",
        help=t("sidebar_cloud_key_help", lang),
    )

    st.markdown("---")
    st.markdown(f"### {t('sidebar_doc_classifier', lang)}")
    doc_type_choice = st.selectbox(
        t("sidebar_doc_category", lang),
        options=["auto", "prescription", "medicine_label"],
        format_func=lambda x: {
            "auto": t("doc_auto", lang),
            "prescription": t("doc_rx", lang),
            "medicine_label": t("doc_label", lang),
        }.get(x, x),
        index=0,
        help=t("sidebar_doc_category_help", lang),
    )

    st.markdown("---")
    st.markdown(f"### {t('sidebar_img_preproc', lang)}")
    apply_clahe = st.checkbox(t("sidebar_clahe", lang), value=True, help=t("sidebar_clahe_help", lang))
    apply_denoise = st.checkbox(t("sidebar_denoise", lang), value=False, help=t("sidebar_denoise_help", lang))
    apply_deskew = st.checkbox(t("sidebar_deskew", lang), value=False, help=t("sidebar_deskew_help", lang))

    st.markdown("---")
    st.caption(f"{t('system_brand', lang)}")


# MAIN PANEL HERO BANNER
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge-pill"><span class="beacon-dot"></span> {t('app_badge_active', lang)}</div>
    <div class="hero-title">{t('app_title', lang)}</div>
    <div class="hero-subtitle">
        {t('app_subtitle', lang)}
    </div>
    <div class="badge-bar">
        <span class="pill-badge">{t('badge_digitization', lang)}</span>
        <span class="pill-badge">{t('badge_regimen', lang)}</span>
        <span class="pill-badge">{t('badge_composition', lang)}</span>
        <span class="pill-badge">{t('badge_safety', lang)}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# SECTION 1: PRESCRIPTION / LABEL INPUT
st.markdown(f"""
<div class="section-header">
    <div class="section-number">1</div>
    <div class="section-title-text">{t('sec_1_title', lang)}</div>
</div>
""", unsafe_allow_html=True)

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        t("upload_label", lang),
        type=["jpg", "jpeg", "png", "webp"],
        help=t("upload_help", lang),
    )
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_name = uploaded_file.name

with col_sample:
    st.markdown(f"**{t('sample_or_title', lang)}**")
    sample_dir = CONFIG.sample_data_dir
    sample_files = sorted([
        f for f in sample_dir.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] and not f.name.startswith(".")
    ])
    sample_names = [f.name for f in sample_files]
    sample_options = ["None"] + sample_names
    
    def format_sample_name(name: str) -> str:
        if name == "None":
            return t("sample_none", lang)
        if "label" in name.lower():
            return f"🏷️ [Label] {name}"
        return f"📝 [Rx] {name}"
    
    selected_sample = st.selectbox(t("sample_select_label", lang), sample_options, format_func=format_sample_name)
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
                <span>{t('img_meta_title', lang)}</span>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span class="pill-badge">📐 <b>{t('meta_dimensions', lang)}:</b> {img.width} × {img.height} px</span>
                <span class="pill-badge">🎨 <b>{t('meta_color_mode', lang)}:</b> {img.mode}</span>
                <span class="pill-badge">📏 <b>{t('meta_aspect_ratio', lang)}:</b> {round(img.width / max(1, img.height), 2)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown(f"### ⚡ {t('pipeline_actions_title', lang)}")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_all_btn = st.button(t("btn_run_pipeline", lang), type="primary", width="stretch")
        with col_btn2:
            extract_ocr_btn = st.button(t("btn_extract_ocr", lang), width="stretch")

    # RUN COMPLETE PIPELINE
    if run_all_btn:
        with st.spinner(t("spinner_pipeline", lang)):
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
        st.success(f"{t('msg_pipeline_success', lang)} ({res.document_type.replace('_', ' ').title()})!")

    # EXTRACT OCR ONLY
    elif extract_ocr_btn:
        with st.spinner(t("spinner_ocr", lang)):
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
        st.success(f"{t('msg_ocr_success', lang)} {ocr_res.inference_time_sec}s")


# SECTION 2: RAW OCR EXTRACTION
if st.session_state.raw_ocr_text:
    st.markdown("---")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">2</div>
        <div class="section-title-text">{t('sec_2_title', lang)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_raw, col_raw_ctrl = st.columns([3, 1])
    with col_raw:
        st.text_area(t("extracted_text_label", lang), value=st.session_state.raw_ocr_text, height=180)
    
    with col_raw_ctrl:
        st.metric(t("metric_char_count", lang), len(st.session_state.raw_ocr_text))
        st.metric(t("metric_line_count", lang), len([l for l in st.session_state.raw_ocr_text.splitlines() if l.strip()]))
        norm_btn = st.button(t("btn_normalize", lang), type="secondary", width="stretch")

    if norm_btn:
        with st.spinner(t("spinner_normalize", lang)):
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
    sec_3_heading = t("sec_3_title_label", lang) if is_label else t("sec_3_title_rx", lang)
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">3</div>
        <div class="section-title-text">{sec_3_heading}</div>
    </div>
    """, unsafe_allow_html=True)

    # Safety Status Banner with Visual Indicator
    if val.passed_safety_checks and not val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%); border: 1px solid #6EE7B7; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);">
            <span style="font-size: 1.6rem;">🛡️</span>
            <div>
                <div style="font-weight: 700; color: #065F46; font-size: 1rem;">{t('safety_passed_title', lang)}</div>
                <div style="color: #047857; font-size: 0.88rem;">{t('safety_passed_desc', lang)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif val.passed_safety_checks and val.human_review_required:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FCD34D; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);">
            <span style="font-size: 1.6rem;">⚠️</span>
            <div>
                <div style="font-weight: 700; color: #92400E; font-size: 1rem;">{t('safety_review_title', lang)}</div>
                <div style="color: #B45309; font-size: 0.88rem;">{t('safety_review_desc', lang)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%); border: 1px solid #FDA4AF; border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; box-shadow: 0 4px 12px rgba(244, 63, 94, 0.08);">
            <span style="font-size: 1.6rem;">🚨</span>
            <div>
                <div style="font-weight: 700; color: #9F1239; font-size: 1rem;">{t('safety_critical_title', lang)}</div>
                <div style="color: #BE123C; font-size: 0.88rem;">{t('safety_critical_desc', lang)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if is_label:
        # TABS FOR MEDICINE PACKAGING LABEL
        tab_norm, tab_comp, tab_nlp, tab_audit, tab_compare = st.tabs([
            t("tab_label_standard", lang),
            t("tab_label_composition", lang),
            t("tab_label_details", lang),
            t("tab_label_audit", lang),
            t("tab_label_comparison", lang),
        ])

        with tab_norm:
            st.markdown(f"#### {t('label_header_standard', lang)}")
            st.code(norm.to_formatted_text(), language="markdown")

        with tab_comp:
            st.markdown(f"##### {t('label_header_composition', lang)}")
            if norm.active_ingredients:
                ing_records = [
                    {
                        t("col_ing_name", lang): ing.name,
                        t("col_std_generic", lang): ing.standard_name,
                        t("col_strength", lang): ing.strength,
                        t("col_status", lang): t("status_verified", lang) if ing.is_verified else t("status_review_yes", lang),
                    }
                    for ing in norm.active_ingredients
                ]
                st.dataframe(pd.DataFrame(ing_records), width="stretch")
            else:
                st.info(t("no_active_ingredients", lang))

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"##### {t('dosage_admin_header', lang)}")
                for adm in norm.dosage_and_administration:
                    st.info(f"👉 {adm}")
                if not norm.dosage_and_administration:
                    st.caption(t("std_dosage_directed", lang))

                st.markdown(f"##### {t('storage_header', lang)}")
                for stor in norm.storage_instructions:
                    st.success(f"📦 {stor}")
                if not norm.storage_instructions:
                    st.caption(t("std_storage_cool", lang))

            with col_a2:
                st.markdown(f"##### {t('warnings_precautions_header', lang)}")
                for w in norm.warnings_and_precautions:
                    st.warning(f"⚠️ {w}")
                if not norm.warnings_and_precautions:
                    st.caption(t("std_precautions_apply", lang))

                if norm.dosing_safety_alerts:
                    st.markdown(f"##### {t('dosing_safety_header', lang)}")
                    for alert in norm.dosing_safety_alerts:
                        st.error(f"🚨 {alert}")

        with tab_nlp:
            st.markdown(f"#### {t('pkg_info_header', lang)}")
            nlp_data = st.session_state.nlp_labels
            if nlp_data:
                col_nl1, col_nl2 = st.columns(2)
                with col_nl1:
                    st.markdown(f"**{t('identified_meds_strengths', lang)}:**")
                    st.write(f"- `{', '.join(nlp_data.get('medicines', [])) or 'None'}`")
                    st.write(f"- `{', '.join(nlp_data.get('strengths', [])) or 'None'}`")
                    st.markdown(f"**{t('administration_rules', lang)}:**")
                    for adm in nlp_data.get("administration", []):
                        st.info(f"👉 {adm}")
                with col_nl2:
                    st.markdown(f"**{t('warnings_precautions_header', lang)}:**")
                    for w in nlp_data.get("warnings", []):
                        st.warning(f"⚠️ {w}")
            else:
                st.info(t("no_pkg_details", lang))

        with tab_audit:
            st.markdown(f"#### {t('audit_log_title', lang)}")
            if val.audit_log:
                df_audit = pd.DataFrame([
                    {
                        t("col_audit_field", lang): a.field,
                        t("col_audit_orig", lang): a.original_ocr_text,
                        t("col_audit_norm", lang): a.corrected_text,
                        t("col_audit_reason", lang): a.reason,
                        t("col_audit_status", lang): a.verification_status,
                    }
                    for a in val.audit_log
                ])
                st.dataframe(df_audit, width="stretch")

            st.markdown(f"#### {t('validation_issues_title', lang)}")
            if val.issues:
                for issue in val.issues:
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{issue.category}]** {issue.message}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{issue.category}]** {issue.message}")
                    else:
                        st.info(f"**[{issue.category}]** {issue.message}")
            else:
                st.success(t("no_discrepancies_detected", lang))

        with tab_compare:
            st.markdown(f"#### {t('comp_header', lang)}")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown(f"**{t('comp_orig_label', lang)}**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown(f"**{t('comp_extracted_text', lang)}**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown(f"**{t('comp_clean_text', lang)}**")
                st.code(norm.to_formatted_text(), language="markdown")

    else:
        # TABS FOR MEDICAL PRESCRIPTION
        tab_norm, tab_nlp, tab_audit, tab_comp = st.tabs([
            t("tab_rx_standard", lang),
            t("tab_rx_regimens", lang),
            t("tab_rx_audit", lang),
            t("tab_rx_comparison", lang),
        ])

        with tab_norm:
            st.markdown(f"#### {t('rx_header_formatted', lang)}")
            st.code(norm.to_formatted_text(), language="markdown")

        with tab_nlp:
            st.markdown(f"#### {t('rx_header_regimens', lang)}")
            nlp_data = st.session_state.nlp_labels
            if nlp_data:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric(t("metric_meds_identified", lang), len(nlp_data.get("medicines", [])))
                with col_m2:
                    st.metric(t("metric_strengths_detected", lang), len(nlp_data.get("strengths", [])))
                with col_m3:
                    st.metric(t("metric_doses_parsed", lang), len(nlp_data.get("dose", [])))
                with col_m4:
                    st.metric(t("metric_freq_normalized", lang), len(nlp_data.get("frequency", [])))

                st.markdown(f"##### {t('tbl_med_groups_title', lang)}")
                groups = nlp_data.get("medication_groups", [])
                if groups:
                    records = []
                    for g in groups:
                        dose_val = g.get("dose", {}).get("normalized_value", "") if g.get("dose") else ""
                        freq_val = g.get("frequency", {}).get("normalized_value", "") if g.get("frequency") else ""
                        daily_freq = g.get("frequency", {}).get("daily_frequency", "") if g.get("frequency") else ""
                        timing_val = g.get("timing", {}).get("normalized_value", "") if g.get("timing") else "Unspecified"
                        records.append({
                            t("tbl_col_item", lang): g.get("item_number"),
                            t("tbl_col_medicine", lang): g.get("medicine_name"),
                            t("tbl_col_formulation", lang): g.get("formulation"),
                            t("tbl_col_strength", lang): g.get("strength"),
                            t("tbl_col_dose", lang): dose_val,
                            t("tbl_col_freq_code", lang): freq_val,
                            t("tbl_col_daily_intake", lang): f"{daily_freq}x / day" if daily_freq else "",
                            t("tbl_col_timing", lang): timing_val,
                            t("tbl_col_instructions", lang): g.get("instructions"),
                            t("tbl_col_review_flag", lang): t("status_review_yes", lang) if g.get("verification_required") else t("status_verified", lang),
                        })
                    st.dataframe(pd.DataFrame(records), width="stretch")
                else:
                    st.info(t("no_med_groups", lang))

                if nlp_data.get("timing") or nlp_data.get("administration"):
                    col_extra1, col_extra2 = st.columns(2)
                    with col_extra1:
                        if nlp_data.get("timing"):
                            st.markdown(f"##### {t('timing_schedule_title', lang)}")
                            for tm in nlp_data.get("timing", []):
                                st.info(f"🕐 {tm}")
                    with col_extra2:
                        if nlp_data.get("administration"):
                            st.markdown(f"##### {t('admin_notes_title', lang)}")
                            for adm in nlp_data.get("administration", []):
                                st.info(f"👉 {adm}")
            else:
                st.info(t("run_norm_to_view", lang))

        with tab_audit:
            st.markdown(f"#### {t('clinical_corr_title', lang)}")
            if norm.proposed_corrections:
                df_corr = pd.DataFrame(norm.proposed_corrections)
                st.dataframe(df_corr, width="stretch")
            else:
                st.info(t("no_corrections_needed", lang))

            st.markdown(f"#### {t('validation_issues_title', lang)}")
            if val.issues:
                for issue in val.issues:
                    if issue.severity == "CRITICAL":
                        st.error(f"**[{issue.category}]** {issue.message}")
                    elif issue.severity == "WARNING":
                        st.warning(f"**[{issue.category}]** {issue.message}")
                    else:
                        st.info(f"**[{issue.category}]** {issue.message}")
            else:
                st.success(t("no_discrepancies_detected", lang))

        with tab_comp:
            st.markdown(f"#### {t('comp_header', lang)}")
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown(f"**{t('comp_orig_rx', lang)}**")
                st.image(st.session_state.current_image, width="stretch")
            with col_c2:
                st.markdown(f"**{t('comp_extracted_text', lang)}**")
                st.code(st.session_state.raw_ocr_text, language="text")
            with col_c3:
                st.markdown(f"**{t('comp_clean_text', lang)}**")
                st.code(norm.to_formatted_text(), language="markdown")


    # =========================================================================
    # MULTILINGUAL CLINICAL VOICE ASSISTANT COMPONENT
    # =========================================================================
    st.markdown("---")
    st.markdown(f"""
    <div class="glass-card" style="margin-top: 1rem; margin-bottom: 1.5rem; border-left: 4px solid #2563EB;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem;">
            <div style="font-weight:700; font-size:1.15rem; color:#1E3A8A; display:flex; align-items:center; gap:0.5rem;">
                <span>{t('voice_title', lang)}</span>
            </div>
            <span class="pill-badge">🌐 {get_language_info(lang).display_label}</span>
        </div>
        <div style="color:#475569; font-size:0.9rem; margin-bottom:0.8rem;">
            {t('voice_subtitle', lang)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    voice_col1, voice_col2 = st.columns([2, 1])
    with voice_col1:
        voice_mode_choice = st.selectbox(
            t("voice_mode_title", lang),
            options=["auto", "online", "offline"],
            format_func=lambda m: {
                "auto": t("voice_mode_auto", lang),
                "online": t("voice_mode_online", lang),
                "offline": t("voice_mode_offline", lang),
            }.get(m, m),
            index=["auto", "online", "offline"].index(st.session_state.voice_mode),
            key="voice_mode_selector_dropdown",
        )
        st.session_state.voice_mode = voice_mode_choice

    with voice_col2:
        st.write("")
        st.write("")
        btn_summary_audio = st.button(t("voice_btn_play_summary", lang), type="primary", width="stretch")

    col_guide, _ = st.columns([1, 1])
    with col_guide:
        btn_guide_audio = st.button(t("voice_btn_guide", lang), width="stretch")

    # Audio synthesis handling
    if btn_summary_audio:
        with st.spinner(f"{t('voice_spinner', lang)} {get_language_info(lang).native_name}..."):
            if is_label:
                st.session_state.active_voice_resp = voice_mgr.get_medicine_label_summary_audio(
                    norm, target_lang=lang, mode=voice_mode_choice
                )
            else:
                st.session_state.active_voice_resp = voice_mgr.get_prescription_summary_audio(
                    norm, target_lang=lang, mode=voice_mode_choice
                )

    elif btn_guide_audio:
        with st.spinner(f"{t('voice_spinner', lang)} {get_language_info(lang).native_name}..."):
            st.session_state.active_voice_resp = voice_mgr.get_navigation_guide_audio(
                target_lang=lang, mode=voice_mode_choice
            )

    # Field-Wise Speech Accordion
    with st.expander(t("voice_field_wise_title", lang), expanded=False):
        if is_label:
            field_scripts = SpeechTextFormatter.format_medicine_label_field_scripts(norm)
        else:
            field_scripts = SpeechTextFormatter.format_prescription_field_scripts(norm)

        for f_label, f_script in field_scripts.items():
            f_col_text, f_col_btn = st.columns([4, 1])
            with f_col_text:
                st.markdown(f"**{f_label}:** {f_script}")
            with f_col_btn:
                if st.button(f"🔊 {t('voice_btn_play_field', lang)}", key=f"btn_voice_field_{f_label}", width="stretch"):
                    with st.spinner(f"{t('voice_spinner', lang)} {get_language_info(lang).native_name}..."):
                        st.session_state.active_voice_resp = voice_mgr.process_speech(
                            english_text=f_script,
                            target_lang=lang,
                            mode=voice_mode_choice,
                        )

    # Active Audio Player & Engine Badge
    current_audio_resp = st.session_state.get("active_voice_resp")
    if current_audio_resp and current_audio_resp.audio_bytes:
        st.audio(current_audio_resp.audio_bytes, format=current_audio_resp.mime_type)

        badge_text = t("voice_online_badge", lang) if "online" in current_audio_resp.tts_engine_used else t("voice_offline_badge", lang)
        if current_audio_resp.is_tts_fallback:
            badge_text = t("voice_fallback_badge", lang)

        st.caption(f"🎧 {badge_text} • {get_language_info(current_audio_resp.target_language).display_label}")

        with st.expander(t("voice_script_preview", lang), expanded=False):
            st.markdown(f"**Spoken Script ({get_language_info(current_audio_resp.target_language).native_name}):**")
            st.write(current_audio_resp.spoken_text)
            st.markdown("**Original English Clinical Text:**")
            st.caption(current_audio_resp.original_english_text)

    st.info(t("voice_disclaimer", lang))


# SECTION 4: EXPORTS & DOWNLOADS
if st.session_state.normalized_result is not None:
    norm = st.session_state.normalized_result
    val = st.session_state.validation_report
    is_label = isinstance(norm, NormalizedMedicineLabel)

    st.markdown("---")
    st.markdown(f"""
    <div class="section-header">
        <div class="section-number">4</div>
        <div class="section-title-text">{t('sec_4_title', lang)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    norm_filename = "normalized_medicine_label.txt" if is_label else "normalized_prescription.txt"
    json_filename = "medicine_label_results.json" if is_label else "prescription_results.json"

    # 1. raw_ocr.txt
    with col_d1:
        st.download_button(
            label=t("btn_dl_raw_text", lang),
            data=st.session_state.raw_ocr_text,
            file_name="raw_ocr.txt",
            mime="text/plain",
            width="stretch",
        )

    # 2. normalized text
    with col_d2:
        st.download_button(
            label=t("btn_dl_clinical_record", lang),
            data=norm.to_formatted_text(),
            file_name=norm_filename,
            mime="text/plain",
            width="stretch",
        )

    # 3. structured json
    with col_d3:
        st.download_button(
            label=t("btn_dl_clinical_report", lang),
            data=json.dumps(norm.to_export_json(), indent=2),
            file_name=json_filename,
            mime="application/json",
            width="stretch",
        )

    # 4. validation_report.json
    with col_d4:
        st.download_button(
            label=t("btn_dl_safety_audit", lang),
            data=json.dumps(val.to_dict(), indent=2),
            file_name="validation_report.json",
            mime="application/json",
            width="stretch",
        )

    # 5. nlp_medication_labels.json
    with col_d5:
        nlp_json_str = json.dumps(st.session_state.nlp_labels or {}, indent=2)
        st.download_button(
            label=t("btn_dl_regimen_dataset", lang),
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
    <div class="section-title-text">{t('sec_5_title', lang)}</div>
</div>
""", unsafe_allow_html=True)
st.caption(t("sec_5_caption", lang))

with st.expander(t("quickscan_expander", lang), expanded=False):
    sample_box_labels = {
        t("quickscan_custom_input", lang): "",
        "Paracetamol 650mg Packaging": "PARACETAMOL TABLETS IP 650 mg\nEach tablet contains: Paracetamol IP 650 mg.\nDosage: As directed by the physician.\nSwallow with water.\nWarning: Do not exceed the recommended dosage. Keep out of reach of children.\nStore in a cool and dry place.",
        "Cetirizine 10mg Packaging": "CETIRIZINE TABLETS IP\nEach tablet contains: Cetirizine Hydrochloride IP 10 mg.\nDosage: As directed by the physician.\nSchedule H Prescription Drug - Caution: Not to be sold by retail without the prescription of a Registered Medical Practitioner.\nStore below 30°C.",
        "Amoxicillin 500mg Packaging": "AMOXI-500 CAPSULES\nEach capsule contains: Amoxicillin Trihydrate IP equivalent to Amoxicillin 500 mg.\nDosage: As directed by the physician. Capsules to be swallowed whole and not to be chewed, opened or crushed.\nKeep out of reach of children. Store protected from moisture.",
    }

    selected_box = st.selectbox(t("quickscan_select_label", lang), list(sample_box_labels.keys()), index=1)
    default_text = sample_box_labels[selected_box]
    box_text_input = st.text_area(t("quickscan_input_label", lang), value=default_text, height=120)

    if st.button(t("btn_extract_packaging", lang), type="secondary"):
        if box_text_input.strip():
            box_res = pipeline.labeller.label_medicine_label(box_text_input)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown(f"**{t('identified_meds_strengths', lang)}:**")
                st.write(f"- `{', '.join(box_res.get('medicines', [])) or 'None'}`")
                st.write(f"- `{', '.join(box_res.get('strengths', [])) or 'None'}`")
                st.markdown(f"**{t('administration_rules', lang)}:**")
                for adm in box_res.get("administration", []):
                    st.info(f"👉 {adm}")
            with col_b2:
                st.markdown(f"**{t('warnings_precautions_header', lang)}:**")
                for w in box_res.get("warnings", []):
                    st.warning(f"⚠️ {w}")
        else:
            st.warning(t("warn_enter_box_text", lang))
