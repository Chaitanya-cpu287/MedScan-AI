"""
Language configuration for MediScan AI.
Supports 8 languages: English, Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    english_name: str
    native_name: str
    script: str
    voice_code: str  # code for TTS (gTTS / system voice)
    greeting: str
    flag_emoji: str = "🌐"

    @property
    def display_label(self) -> str:
        """Formatted display label for UI selector: Native (English)"""
        if self.code == "en":
            return f"{self.flag_emoji} English"
        return f"{self.flag_emoji} {self.native_name} ({self.english_name})"


# The 8 explicitly required languages
SUPPORTED_LANGUAGES: Dict[str, LanguageInfo] = {
    "en": LanguageInfo(
        code="en",
        english_name="English",
        native_name="English",
        script="Latin",
        voice_code="en",
        greeting="Welcome to MediScan AI",
        flag_emoji="🇬🇧",
    ),
    "te": LanguageInfo(
        code="te",
        english_name="Telugu",
        native_name="తెలుగు",
        script="Telugu",
        voice_code="te",
        greeting="మెడిస్కాన్ AI కి స్వాగతం",
        flag_emoji="🇮🇳",
    ),
    "hi": LanguageInfo(
        code="hi",
        english_name="Hindi",
        native_name="हिन्दी",
        script="Devanagari",
        voice_code="hi",
        greeting="मेडिस्कैन AI में आपका स्वागत है",
        flag_emoji="🇮🇳",
    ),
    "ta": LanguageInfo(
        code="ta",
        english_name="Tamil",
        native_name="தமிழ்",
        script="Tamil",
        voice_code="ta",
        greeting="மெடிஸ்கேன் AI-க்கு வரவேற்கிறோம்",
        flag_emoji="🇮🇳",
    ),
    "kn": LanguageInfo(
        code="kn",
        english_name="Kannada",
        native_name="ಕನ್ನಡ",
        script="Kannada",
        voice_code="kn",
        greeting="ಮೆಡಿಸ್ಕ್ಯಾನ್ AI ಗೆ ಸುಸ್ವಾಗತ",
        flag_emoji="🇮🇳",
    ),
    "ml": LanguageInfo(
        code="ml",
        english_name="Malayalam",
        native_name="മലയാളം",
        script="Malayalam",
        voice_code="ml",
        greeting="മെഡിസ്‌കാൻ AI-ലേക്ക് സ്വാഗതം",
        flag_emoji="🇮🇳",
    ),
    "bn": LanguageInfo(
        code="bn",
        english_name="Bengali",
        native_name="বাংলা",
        script="Bengali",
        voice_code="bn",
        greeting="মেডিস্ক্যান AI-তে স্বাগতম",
        flag_emoji="🇮🇳",
    ),
    "mr": LanguageInfo(
        code="mr",
        english_name="Marathi",
        native_name="मराठी",
        script="Devanagari",
        voice_code="mr",
        greeting="मेडिसकॅन AI मध्ये आपले स्वागत आहे",
        flag_emoji="🇮🇳",
    ),
}

DEFAULT_LANGUAGE = "en"


def get_language_info(code: str) -> LanguageInfo:
    """Get LanguageInfo for a language code, falling back to English."""
    return SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE])


def get_language_options() -> List[str]:
    """Return ordered list of language codes."""
    return list(SUPPORTED_LANGUAGES.keys())
