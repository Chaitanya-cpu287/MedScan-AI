"""End-to-end tests for multilingual translation, clinical formatting, and TTS speech generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import PrescriptionPipeline
from translations import LANGUAGES, TRANSLATIONS, t
from clinical_translator import (
    translate_formulation,
    translate_timing,
    translate_daily_intake,
    translate_dose,
    translate_clinical_instruction,
    translate_audit_reason,
    translate_safety_message,
    translate_status,
    format_audit_dataframe,
    build_localized_prescription_card,
    build_localized_prescription_text,
)
from tts_reader import (
    build_regimens_speech_text,
    build_audit_speech_text,
    build_label_composition_speech_text,
)


def test_all_languages_have_required_translation_keys():
    """Every language must return non-empty strings for all English UI keys."""
    en_keys = list(TRANSLATIONS["en"].keys())
    for lang_code in LANGUAGES.keys():
        for key in en_keys:
            val = t(key, lang_code)
            assert isinstance(val, str) and len(val) > 0, f"Missing/empty key '{key}' for '{lang_code}'"


def test_composable_clinical_instructions_across_languages():
    """Complex clinical instructions with food timing and duration translate in every Indian language."""
    samples = [
        "Take one tablet twice daily after food for 5 days.",
        "Take one capsule three times daily for 5 days.",
        "Take one tablet before food.",
        "Take one tablet as needed for fever.",
    ]
    for lang in [c for c in LANGUAGES.keys() if c != "en"]:
        for sent in samples:
            res = translate_clinical_instruction(sent, lang)
            assert res != sent, f"Sentence not translated for lang='{lang}': {sent}"
            assert len(res) > 3


def test_timing_translation_handles_ac_pc_hs_and_spurious_tokens():
    """Timing translator handles normalized AC/PC/HS strings and spurious numbers."""
    for lang in ["hi", "te", "ta", "bn", "mr", "kn", "gu", "ml", "pa"]:
        assert translate_timing("pre-prandial (before meals)", lang) != "pre-prandial (before meals)"
        assert translate_timing("post-prandial (after meals)", lang) != "post-prandial (after meals)"
        assert translate_timing("AC (pre-prandial (before meals))", lang) != "AC (pre-prandial (before meals))"
        assert translate_timing("2", lang) == translate_timing("unspecified", lang)


def test_end_to_end_prescription_multilingual_rendering():
    """Runs real normalization, validation, NLP labelling, card/table formatting, and TTS across all 10 languages."""
    raw_rx = """SRM Medical Centre
Dr. R. Sharma, M.D.
Date: 09/11/2026
Rx
1) Dol0-650 Tab
1 tab SOS for fever
2) Amoxicillin 500 mg Cap
1 cap TID for 5 days
3) Pantopra3ole 40 mg Tab
1 tab OD before food
Dr. R. Sharma"""

    pipeline = PrescriptionPipeline()
    norm = pipeline.normalizer.normalize(raw_rx, doc_type="prescription")
    val = pipeline.validator.validate(raw_rx, norm)
    nlp_data = pipeline.labeller.label_prescription(
        prescription_text=norm.to_formatted_text(),
        normalized_prescription_obj=norm,
    )

    groups = nlp_data.get("medication_groups", [])
    assert len(groups) == 3

    for lang in LANGUAGES.keys():
        # Card HTML must never contain 4+ leading spaces (Markdown code block trap)
        card = build_localized_prescription_card(norm, lang)
        if lang != "en":
            assert "<div" in card
            bad_lines = [line for line in card.splitlines() if line.startswith("    ") or line.startswith("\t")]
            assert len(bad_lines) == 0, f"Indented HTML detected for {lang}: {bad_lines}"

        # Plain text must populate all 3 medicines cleanly
        txt = build_localized_prescription_text(norm, lang)
        assert "PARACETAMOL" in txt.upper()
        assert "AMOXICILLIN" in txt.upper()
        assert "PANTOPRAZOLE" in txt.upper()

        # Table cell localization & timing dict extraction (no raw dicts, no shadowing 't')
        for g in groups:
            timing_obj = g.get("timing")
            t_raw = (timing_obj.get("timing_type") or timing_obj.get("normalized_value")) if isinstance(timing_obj, dict) else (timing_obj or "Unspecified")
            t_disp = translate_timing(t_raw, lang)
            assert "{" not in t_disp and "}" not in t_disp

            dose_val = g.get("dose", {}).get("normalized_value", "") if isinstance(g.get("dose"), dict) else ""
            assert isinstance(translate_dose(dose_val, lang), str)
            assert isinstance(translate_formulation(g.get("formulation", ""), lang), str)
            assert isinstance(translate_clinical_instruction(g.get("instructions", ""), lang), str)

        # Audit table & safety issues
        if norm.proposed_corrections:
            df_rows = format_audit_dataframe(norm.proposed_corrections, lang)
            assert len(df_rows) > 0
        for issue in val.issues:
            cat_txt, msg_txt = translate_safety_message(issue.message, issue.category, lang)
            assert cat_txt and msg_txt

        # TTS narratives
        reg_speech = build_regimens_speech_text(groups, lang=lang)
        aud_speech = build_audit_speech_text(norm.proposed_corrections, issues=val.issues, lang=lang)
        assert len(reg_speech) > 20
        assert len(aud_speech) > 20
        assert "{" not in reg_speech and "}" not in reg_speech


def test_end_to_end_medicine_label_multilingual_rendering():
    """Runs real medicine label pipeline and verifies multilingual composition speech & audit."""
    raw_lbl = """PARACETAMOL TABLETS IP 650 mg
Each tablet contains: Paracetamol IP 650 mg.
Dosage: As directed by the physician.
Schedule H Prescription Drug - Caution: Not to be sold by retail without prescription.
Keep out of reach of children. Store in a cool and dry place."""

    pipeline = PrescriptionPipeline()
    norm = pipeline.normalizer.normalize(raw_lbl, doc_type="medicine_label")
    val = pipeline.validator.validate(raw_lbl, norm)

    for lang in LANGUAGES.keys():
        comp_speech = build_label_composition_speech_text(norm.active_ingredients, lang=lang)
        assert len(comp_speech) > 10
        if lang != "en":
            assert "Active ingredients composition summary" not in comp_speech

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
            df_rows = format_audit_dataframe(raw_log, lang)
            assert len(df_rows) > 0
        for issue in val.issues:
            cat_txt, msg_txt = translate_safety_message(issue.message, issue.category, lang)
            assert cat_txt and msg_txt
