"""Tests for Member 2 NLP Medication Labelling Engine."""

import pytest
from nlp_labelling.labelling_engine import (
    MedicationLabeller,
    extract_prescription_information,
    extract_medicine_label_information,
    normalize_dose,
    normalize_frequency,
    normalize_strength,
    normalize_timing,
    normalize_duration,
)
from normalization.text_normalizer import TextNormalizer


@pytest.fixture
def labeller():
    return MedicationLabeller()


def test_labeller_loaded(labeller):
    """Verify that the spaCy model is loaded and ready."""
    assert labeller.is_available, "Member 2 spaCy model should be successfully loaded"


def test_normalize_helpers():
    """Verify clinical value normalizers."""
    # Strength
    assert normalize_strength("500 MG") == "500 mg"
    assert normalize_strength("10 mg") == "10 mg"

    # Dose
    dose_1 = normalize_dose("Take 2 tablets")
    assert dose_1["quantity"] == 2.0
    assert "2 tablets" in dose_1["normalized_value"]

    dose_half = normalize_dose("half tablet")
    assert dose_half["quantity"] == 0.5

    # Frequency
    freq_bid = normalize_frequency("twice daily")
    assert freq_bid["code"] == "BID"
    assert freq_bid["daily_count"] == 2

    freq_tid = normalize_frequency("three times a day")
    assert freq_tid["code"] == "TID"
    assert freq_tid["daily_count"] == 3

    # Timing
    t_pc = normalize_timing("after meals")
    assert t_pc["code"] == "PC"
    assert "after meals" in t_pc["normalized_value"].lower()

    # Duration
    dur = normalize_duration("for 5 days")
    assert dur["duration_days"] == 5


def test_extract_prescription_standalone():
    """Test extract_prescription_information function matching Member 2's signature."""
    prescription_sample = (
        "Take Paracetamol 500 mg 1 tablet twice daily after meals for 5 days."
    )
    res = extract_prescription_information(prescription_sample)

    assert res["document_type"] == "prescription"
    assert "Paracetamol" in res["medicines"] or "PARACETAMOL" in [m.upper() for m in res["medicines"]]
    assert len(res["strengths"]) > 0
    assert res["review_required"] is True
    assert "dose" in res
    assert "frequency" in res


def test_extract_medicine_label_standalone():
    """Test extract_medicine_label_information for medicine packaging / box labels."""
    label_sample = (
        "Paracetamol Tablets IP 500 mg. Dosage: As directed by the physician. "
        "Warning: To be sold by retail on the prescription of a Registered Medical Practitioner only. "
        "Keep out of reach of children. Store below 30°C."
    )
    res = extract_medicine_label_information(label_sample)

    assert res["document_type"] == "medicine_label"
    assert len(res["medicines"]) > 0
    assert len(res["administration"]) > 0
    assert any("physician" in a.lower() for a in res["administration"])
    assert len(res["warnings"]) > 0
    assert any("children" in w.lower() for w in res["warnings"])


def test_hybrid_integration_with_normalizer(labeller):
    """Test integration between Member 1's NormalizedPrescription and Member 2's NER."""
    raw_text = """
    RIVERSIDEMEDICALCENTRE
    NAME Joln Smitl
    AGE 34
    Betaloc 100mg - 1 tab BID
    Dorzolamidum 10 mg - 1 tab BID
    Cimetiline 50 mg - 2 tabs TTD
    Oxprelol 50mg - 1 tab QD
    """
    normalizer = TextNormalizer()
    normalized = normalizer.normalize(raw_text)

    labeled = labeller.label_prescription(
        prescription_text=normalized.to_formatted_text(),
        normalized_prescription_obj=normalized,
    )

    assert labeled["document_type"] == "prescription"
    assert len(labeled["medication_groups"]) == 4

    med_names = [g["medicine_name"] for g in labeled["medication_groups"]]
    assert "METOPROLOL" in med_names
    assert "DORZOLAMIDE" in med_names
    assert "CIMETIDINE" in med_names
    assert "OXPRENOLOL" in med_names

    # Check that grouped doses and frequencies are accurately normalized
    metoprolol_group = next(g for g in labeled["medication_groups"] if g["medicine_name"] == "METOPROLOL")
    assert metoprolol_group["strength"] == "100 MG"
    assert metoprolol_group["frequency"]["code"] == "BID"
    assert metoprolol_group["dose"]["quantity"] == 1.0

    cimetidine_group = next(g for g in labeled["medication_groups"] if g["medicine_name"] == "CIMETIDINE")
    assert cimetidine_group["strength"] == "50 MG"
    assert cimetidine_group["frequency"]["code"] == "TID"
    assert cimetidine_group["dose"]["quantity"] == 2.0
