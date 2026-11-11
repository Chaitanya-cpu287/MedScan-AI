"""
Voice Assistant package for MediScan AI.
Provides hybrid multilingual voice synthesis and clinical translation for 8 languages.
"""

from voice_assistant.voice_manager import VoiceManager, VoiceResponse
from voice_assistant.tts_service import TTSService, TTSResult
from voice_assistant.translation_service import TranslationService, TranslationResult
from voice_assistant.speech_text_formatter import SpeechTextFormatter
from voice_assistant.medical_safety import MedicalSafetyGuard

__all__ = [
    "VoiceManager",
    "VoiceResponse",
    "TTSService",
    "TTSResult",
    "TranslationService",
    "TranslationResult",
    "SpeechTextFormatter",
    "MedicalSafetyGuard",
]
