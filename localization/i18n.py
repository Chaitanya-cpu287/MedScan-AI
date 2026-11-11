"""
Translation lookup utility and language state management for MediScan AI.
"""

from typing import Optional
from localization.language_config import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES
from localization.translations import TRANSLATIONS


def get_current_language() -> str:
    """
    Get the currently active language code from Streamlit session state if available,
    otherwise return default ('en').
    """
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "selected_language" in st.session_state:
            return st.session_state.selected_language
    except Exception:
        pass
    return DEFAULT_LANGUAGE


def set_current_language(lang_code: str) -> None:
    """
    Set the active language code in Streamlit session state.
    """
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = DEFAULT_LANGUAGE
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            st.session_state.selected_language = lang_code
    except Exception:
        pass


def t(key: str, lang: Optional[str] = None, default: Optional[str] = None, **kwargs) -> str:
    """
    Translate a UI key into the target language with automatic English fallback.
    Supports str.format kwargs.

    Example:
        t("app_title")
        t("meta_dimensions", lang="te")
        t("welcome_user", name="Dr. Sharma")
    """
    if not lang:
        lang = get_current_language()

    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE

    # 1. Try target language dictionary
    lang_dict = TRANSLATIONS.get(lang, {})
    val = lang_dict.get(key)

    # 2. Fallback to English dictionary
    if val is None:
        en_dict = TRANSLATIONS.get(DEFAULT_LANGUAGE, {})
        val = en_dict.get(key)

    # 3. Fallback to default or raw key
    if val is None:
        val = default if default is not None else key

    # Apply formatting if kwargs provided
    if kwargs and isinstance(val, str):
        try:
            return val.format(**kwargs)
        except Exception:
            return val

    return val
