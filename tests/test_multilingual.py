"""
Tests for localization (i18n) package.
Verifies all 8 languages: en, te, hi, ta, kn, ml, bn, mr.
"""

import pytest
from localization import (
    t,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    get_language_info,
    get_language_options,
    set_current_language,
    get_current_language,
)
from localization.translations import TRANSLATIONS


def test_supported_languages_count():
    assert len(SUPPORTED_LANGUAGES) == 8
    expected = {"en", "te", "hi", "ta", "kn", "ml", "bn", "mr"}
    assert set(SUPPORTED_LANGUAGES.keys()) == expected


def test_language_info_attributes():
    for code, info in SUPPORTED_LANGUAGES.items():
        assert info.code == code
        assert len(info.english_name) > 0
        assert len(info.native_name) > 0
        assert len(info.display_label) > 0
        assert info.voice_code == code


def test_default_fallback_to_english():
    # Unknown key should return key itself if not found
    assert t("non_existent_key_xyz_123") == "non_existent_key_xyz_123"

    # English key should return properly
    assert t("app_title", lang="en") == "Prescription & Medicine Label Scanner"


def test_all_languages_have_core_keys():
    core_keys = [
        "app_title",
        "app_badge_active",
        "app_subtitle",
        "sidebar_lang_selector",
        "sidebar_security_title",
        "btn_run_pipeline",
        "btn_extract_ocr",
        "sec_1_title",
        "sec_2_title",
        "sec_3_title_rx",
        "sec_3_title_label",
        "tab_rx_standard",
        "tab_rx_regimens",
        "voice_title",
    ]
    for lang_code in SUPPORTED_LANGUAGES.keys():
        for key in core_keys:
            val = t(key, lang=lang_code)
            assert val is not None and len(val) > 0, f"Missing key '{key}' for language '{lang_code}'"


def test_native_script_rendering():
    # Verify non-empty and non-Latin native script for Indian languages
    assert "ప్రిస్క్రిప్షన్" in t("app_title", lang="te")
    assert "प्रिस्क्रिप्शन" in t("app_title", lang="hi")
    assert "மருத்துவ" in t("app_title", lang="ta")
    assert "ಪ್ರಿಸ್ಕ್ರಿಪ್ಷನ್" in t("app_title", lang="kn")
    assert "പ്രിസ്ക്രിപ്ഷൻ" in t("app_title", lang="ml")
    assert "প্রেসক্রিপশন" in t("app_title", lang="bn")
    assert "प्रिस्क्रिप्शन" in t("app_title", lang="mr")
