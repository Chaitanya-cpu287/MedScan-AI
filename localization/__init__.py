"""
Localization (i18n) package for MediScan AI.
Supports 8 Indian & International languages:
English, Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi.
"""

from localization.language_config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    LanguageInfo,
    get_language_info,
    get_language_options,
)
from localization.i18n import t, get_current_language, set_current_language

__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "LanguageInfo",
    "get_language_info",
    "get_language_options",
    "t",
    "get_current_language",
    "set_current_language",
]
