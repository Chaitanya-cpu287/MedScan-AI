"""Tests for prescription validation, hallucination detection, and audit logging."""

import pytest
from normalization.text_normalizer import TextNormalizer
from validation.prescription_validator import PrescriptionValidator


@pytest.fixture
def normalizer():
    return TextNormalizer()


def test_validation_detects_dosage_tampering(normalizer):
    raw_ocr = "Tab Paracetamol 500 mg 1-0-1"
    normalized = normalizer.normalize(raw_ocr)

    # Intentionally corrupt dosage to test validator sensitivity
    normalized.medications[0].dosage = "650 MG"

    report = PrescriptionValidator.validate(raw_ocr, normalized)
    assert len(report.dosage_discrepancies) > 0
    assert report.human_review_required is True
    assert any(i.category == "DOSAGE_MISMATCH" for i in report.issues)


def test_validation_detects_omission(normalizer):
    raw_ocr = "Tab Paracetamol 650 mg 1-0-1\nTab Cetirizine 10 mg at bedtime"
    normalized = normalizer.normalize("Tab Paracetamol 650 mg 1-0-1")  # Cetirizine omitted

    report = PrescriptionValidator.validate(raw_ocr, normalized)
    assert report.all_medications_retained is False
    assert len(report.omissions_detected) > 0
    assert any("10 mg" in om for om in report.omissions_detected)


def test_validation_detects_hallucination(normalizer):
    raw_ocr = "Tab Paracetamol 650 mg"
    normalized = normalizer.normalize(raw_ocr)

    # Manually append a completely hallucinated drug
    from normalization.text_normalizer import NormalizedMedication
    normalized.medications.append(NormalizedMedication(
        original_text="FictitiousDrug",
        medicine_name="FictitiousDrug",
        formulation="TAB.",
        dosage="100 MG",
        frequency="OD",
        duration="5 days",
        instructions="None",
        readable_sentence="Take one tablet once daily.",
        verification_required=True,
        confidence_score=10.0,
        correction_reason="None",
    ))

    report = PrescriptionValidator.validate(raw_ocr, normalized)
    assert len(report.potential_hallucinations) > 0
    assert "FictitiousDrug" in report.potential_hallucinations
    assert report.passed_safety_checks is False
