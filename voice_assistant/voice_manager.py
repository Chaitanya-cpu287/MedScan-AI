"""
Voice Manager for MediScan AI.
High-level coordinator for translation, speech text generation, and audio caching.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from normalization.text_normalizer import NormalizedPrescription
from normalization.medicine_label_normalizer import NormalizedMedicineLabel
from voice_assistant.speech_text_formatter import SpeechTextFormatter
from voice_assistant.translation_service import TranslationService, TranslationResult
from voice_assistant.tts_service import TTSService, TTSResult

logger = logging.getLogger("VoiceManager")


@dataclass
class VoiceResponse:
    audio_bytes: bytes
    mime_type: str
    spoken_text: str  # In target language
    original_english_text: str
    target_language: str
    tts_engine_used: str
    translation_engine_used: str
    is_tts_fallback: bool
    is_translation_fallback: bool
    integrity_verified: bool
    error_message: Optional[str] = None


class VoiceManager:
    """
    Central controller for multilingual voice assistant features.
    Provides session-level audio caching to avoid redundant synthesis.
    """

    def __init__(self):
        self.translation_service = TranslationService()
        self.tts_service = TTSService()
        self._cache: Dict[str, VoiceResponse] = {}

    def _get_cache_key(self, text: str, lang_code: str, mode: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{lang_code}_{mode}_{text_hash}"

    def process_speech(
        self,
        english_text: str,
        target_lang: str = "en",
        mode: str = "auto",
    ) -> VoiceResponse:
        """
        Translates text and synthesizes speech with caching.
        """
        cache_key = self._get_cache_key(english_text, target_lang, mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Translate
        trans_res = self.translation_service.translate(
            text=english_text,
            target_lang=target_lang,
            prefer_offline=(mode == "offline"),
        )

        # 2. Synthesize audio
        tts_res = self.tts_service.synthesize(
            text=trans_res.translated_text,
            lang_code=target_lang,
            mode=mode,
        )

        response = VoiceResponse(
            audio_bytes=tts_res.audio_bytes,
            mime_type=tts_res.mime_type,
            spoken_text=trans_res.translated_text,
            original_english_text=english_text,
            target_language=target_lang,
            tts_engine_used=tts_res.engine_used,
            translation_engine_used=trans_res.engine_used,
            is_tts_fallback=tts_res.is_fallback,
            is_translation_fallback=trans_res.is_fallback,
            integrity_verified=trans_res.integrity_verified,
            error_message=tts_res.error_message,
        )

        self._cache[cache_key] = response
        return response

    def get_prescription_summary_audio(
        self,
        prescription: NormalizedPrescription,
        target_lang: str = "en",
        mode: str = "auto",
    ) -> VoiceResponse:
        """Generate audio summary for a medical prescription."""
        script = SpeechTextFormatter.format_prescription_summary(prescription)
        return self.process_speech(script, target_lang, mode)

    def get_medicine_label_summary_audio(
        self,
        label: NormalizedMedicineLabel,
        target_lang: str = "en",
        mode: str = "auto",
    ) -> VoiceResponse:
        """Generate audio summary for a medicine packaging label."""
        script = SpeechTextFormatter.format_medicine_label_summary(label)
        return self.process_speech(script, target_lang, mode)

    def get_navigation_guide_audio(
        self,
        target_lang: str = "en",
        mode: str = "auto",
    ) -> VoiceResponse:
        """Generate audio for the interactive system guide."""
        script = SpeechTextFormatter.format_navigation_guide(target_lang)
        # Guide is already localized in translation dict
        tts_res = self.tts_service.synthesize(text=script, lang_code=target_lang, mode=mode)
        return VoiceResponse(
            audio_bytes=tts_res.audio_bytes,
            mime_type=tts_res.mime_type,
            spoken_text=script,
            original_english_text=SpeechTextFormatter.format_navigation_guide("en"),
            target_language=target_lang,
            tts_engine_used=tts_res.engine_used,
            translation_engine_used="localized_dictionary",
            is_tts_fallback=tts_res.is_fallback,
            is_translation_fallback=False,
            integrity_verified=True,
        )
