"""
Tests for voice assistant and medical translation safety.
"""

import pytest
from voice_assistant.medical_safety import MedicalSafetyGuard
from voice_assistant.speech_text_formatter import SpeechTextFormatter
from voice_assistant.tts_service import TTSService
from voice_assistant.translation_service import TranslationService
from voice_assistant.voice_manager import VoiceManager
from normalization.text_normalizer import (
    NormalizedPrescription,
    NormalizedMedication,
)
from normalization.medicine_label_normalizer import (
    NormalizedMedicineLabel,
    ActiveIngredient,
)


def test_medical_safety_extract_and_restore():
    text = "Take 1 tab Metoprolol 100 mg twice daily for 5 days."
    protected_map = MedicalSafetyGuard.extract_protected_tokens(text)
    assert len(protected_map) >= 2  # 1 tab, 100 mg, 5 days
    safe_text = MedicalSafetyGuard.protect_text(text, protected_map)
    assert "100 mg" not in safe_text

    restored = MedicalSafetyGuard.restore_protected_text(safe_text, protected_map)
    assert "100 mg" in restored
    assert MedicalSafetyGuard.verify_dosage_integrity(text, restored)


def test_speech_text_formatter_prescription():
    p = NormalizedPrescription(
        patient_name="John Doe",
        patient_age="34",
        date="10/11/2026",
        doctor_name="Dr. Rao",
        clinic_name="City Clinic",
        medications=[
            NormalizedMedication(
                original_text="Dolo 650",
                medicine_name="Paracetamol",
                formulation="TAB.",
                dosage="650 mg",
                frequency="twice daily",
                duration="5 days",
                instructions="after food",
                readable_sentence="Take one tablet twice daily after food.",
                verification_required=False,
                confidence_score=0.98,
                correction_reason="",
            )
        ],
    )
    summary = SpeechTextFormatter.format_prescription_summary(p)
    assert "John Doe" in summary
    assert "Paracetamol" in summary
    assert "650 mg" in summary

    fields = SpeechTextFormatter.format_prescription_field_scripts(p)
    assert len(fields) >= 2


def test_speech_text_formatter_label():
    lbl = NormalizedMedicineLabel(
        product_name="Cetirizine 10mg",
        brand_name="Cetirizine",
        generic_name="Cetirizine",
        formulation="Tablet",
        active_ingredients=[
            ActiveIngredient(name="Cetirizine", standard_name="Cetirizine", strength="10 mg", is_verified=True)
        ],
        dosage_and_administration=["As directed by physician."],
        storage_instructions=["Store below 30°C."],
        warnings_and_precautions=["Keep out of reach of children."],
    )
    summary = SpeechTextFormatter.format_medicine_label_summary(lbl)
    assert "Cetirizine" in summary
    assert "10 mg" in summary


def test_offline_tts_audio_generation():
    service = TTSService()
    # Force offline mode so test does not depend on internet
    res = service.synthesize("Hello world", lang_code="en", mode="offline")
    assert res.audio_bytes is not None
    assert len(res.audio_bytes) > 200
    assert "offline" in res.engine_used


def test_voice_manager_caching():
    vm = VoiceManager()
    resp1 = vm.process_speech("Take one tablet.", target_lang="en", mode="offline")
    resp2 = vm.process_speech("Take one tablet.", target_lang="en", mode="offline")
    assert resp1 is resp2  # identical cached object
