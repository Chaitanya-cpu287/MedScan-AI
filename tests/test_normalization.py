"""Tests for Intelligent Text Normalization and Abbreviation Handling."""

import pytest
from normalization.text_normalizer import TextNormalizer
from normalization.abbreviation_handler import AbbreviationHandler
from normalization.medicine_matcher import MedicineMatcher


@pytest.fixture
def normalizer():
    return TextNormalizer()


@pytest.fixture
def matcher():
    return MedicineMatcher()


def test_medicine_name_corrections(matcher):
    # Prompt Section 6 examples:
    # paracetmol -> Paracetamol
    # amoxcilln -> Amoxicillin
    # pantaprazl -> Pantoprazole
    res_pcm = matcher.match_medicine_name("paracetmol 650mg")
    assert res_pcm.matched_name == "Paracetamol"
    assert res_pcm.is_corrected is True
    assert "650 MG" in res_pcm.extracted_dosage

    res_amox = matcher.match_medicine_name("cap amoxcilln 500")
    assert res_amox.matched_name == "Amoxicillin"
    assert res_amox.formulation == "CAP."

    res_panto = matcher.match_medicine_name("tab pantaprazl 40 mg")
    assert res_panto.matched_name == "Pantoprazole"
    assert res_panto.formulation == "TAB."


def test_brand_to_generic_aliases(matcher):
    res_dolo = matcher.match_medicine_name("dolo 650")
    assert res_dolo.matched_name == "Paracetamol"
    assert res_dolo.extracted_dosage == "650 MG"

    res_azithral = matcher.match_medicine_name("azithral 500")
    assert res_azithral.matched_name == "Azithromycin"


def test_abbreviation_handler():
    assert AbbreviationHandler.interpret_frequency("OD").expanded_english == "once daily"
    assert AbbreviationHandler.interpret_frequency("BD").expanded_english == "twice daily"
    assert AbbreviationHandler.interpret_frequency("TDS").expanded_english == "three times daily"
    assert AbbreviationHandler.interpret_frequency("QID").expanded_english == "four times daily"
    assert AbbreviationHandler.interpret_frequency("1-0-1").expanded_english == "twice daily (morning and night)"
    assert AbbreviationHandler.interpret_frequency("1-1-1").expanded_english == "three times daily (morning, afternoon, and night)"
    assert AbbreviationHandler.interpret_frequency("0-0-1").expanded_english == "once daily at bedtime"
    assert AbbreviationHandler.interpret_frequency("HS").expanded_english == "at bedtime"


def test_food_timing_handler():
    assert AbbreviationHandler.interpret_food_timing("take after food")[1] == "after food"
    assert AbbreviationHandler.interpret_food_timing("tab bbf")[1] == "before breakfast"
    assert AbbreviationHandler.interpret_food_timing("1 tab ac")[1] == "before meals"


def test_structured_json_normalization(normalizer):
    input_data = {
        "patient_name": "Priya Sharma",
        "date": "09/11/2026",
        "medications": [
            {
                "drug_name": "tab paracetmol 650mg",
                "dosage": "650 mg",
                "frequency": "TDS",
                "duration": "3 days",
                "instructions": "After food",
            },
            {
                "drug_name": "cap amoxcilln 500",
                "dosage": "500 mg",
                "frequency": "TDS",
                "duration": "7 days",
                "instructions": "",
            },
            {
                "drug_name": "tab pantaprazl 40 mg",
                "dosage": "40 mg",
                "frequency": "OD",
                "duration": "10 days",
                "instructions": "before breakfast",
            },
        ],
        "notes": "Take medicines with sufficient water. Do not exceed the prescribed dose.",
    }

    normalized = normalizer.normalize(input_data)
    formatted = normalized.to_formatted_text()

    assert "PATIENT: Priya Sharma" in formatted
    assert "DATE: 09/11/2026" in formatted
    assert "1. TAB. PARACETAMOL 650 MG" in formatted
    assert "2. CAP. AMOXICILLIN 500 MG" in formatted
    assert "3. TAB. PANTOPRAZOLE 40 MG" in formatted
    assert "three times daily" in formatted
    assert "for 3 days" in formatted
    assert "Take medicines with sufficient water" in formatted


def test_raw_unstructured_text_normalization(normalizer):
    raw_ocr = """
    Patient: Rajesh Kumar  Age: 45  Date: 12/04/2025
    Rx
    Tab Paracetmol 650mg
    1-0-1 after food for 5 days
    Cap Amoxcilln 500mg
    TDS for 7 days
    """
    normalized = normalizer.normalize(raw_ocr)
    assert len(normalized.medications) == 2
    assert normalized.medications[0].medicine_name == "Paracetamol"
    assert normalized.medications[0].dosage == "650 MG"
    assert normalized.medications[1].medicine_name == "Amoxicillin"
    assert normalized.medications[1].dosage == "500 MG"


def test_ambiguous_medicine_handling(matcher):
    # Completely nonsensical drug name should not be blindly replaced
    res = matcher.match_medicine_name("xyz123randomdrug 50mg")
    assert res.verification_required is True
    assert res.is_corrected is False
    assert "Preserving original" in res.correction_reason


def test_multidrug_handwriting_prescription(normalizer):
    raw_ocr = """RIVERSIDEMEDICALCENTRE
82414Street
NewYork,NY91743,USA
Joln Smitl
NAME
AGE
34
ADDREss_162ExampleS+,NY
DATE O9-11-12
尔
Betaloc 100mg - 1 tab BID
Dorzolamidum 10 mg - / +tab BD
Cimetiline 50 mg - 2 +abs TTD
Oxprelol 50mg -1 +ab QD
Dr. Steve Jolnson
signature"""

    normalized = normalizer.normalize(raw_ocr)
    assert normalized.patient_name == "John Smith"
    assert "2012" in normalized.date or "12" in normalized.date
    assert len(normalized.medications) == 4
    med_names = [m.medicine_name for m in normalized.medications]
    assert "Metoprolol" in med_names
    assert "Dorzolamide" in med_names
    assert "Cimetidine" in med_names
    assert "Oxprenolol" in med_names

    formatted = normalized.to_formatted_text()
    assert "PATIENT: John Smith" in formatted
    assert "1. TAB. METOPROLOL 100 MG" in formatted
    assert "Take one tablet twice daily." in formatted
    assert "Take two tablets three times daily." in formatted
    assert "Additional Instructions:" not in formatted


def test_noisy_multiline_prescription(normalizer):
    """Test noisy OCR with merged tokens, leet-speak typos, detached instructions, and doctor headers."""
    raw_ocr = """SRM Medical Centre
Dr.R.Sharma,M.D.
General Physician
Date: 0911/2026
R
1) Dol0-650 Tab
1 tab Sos for fever
2) Amoxicillin 5o0 mg Cap
1 capT1Dfor 5 days
3) Pantopra3ole 40 mg Tab
1 tabODbefore food
Dr. R.Sharma"""

    normalized = normalizer.normalize(raw_ocr)
    assert normalized.date == "09/11/2026"
    assert len(normalized.medications) == 3

    med_names = [m.medicine_name for m in normalized.medications]
    assert "Paracetamol" in med_names
    assert "Amoxicillin" in med_names
    assert "Pantoprazole" in med_names

    # Check Paracetamol (Dolo 650)
    p = next(m for m in normalized.medications if m.medicine_name == "Paracetamol")
    assert p.dosage == "650 MG"
    assert p.formulation == "TAB."
    assert "as needed for fever" in p.readable_sentence.lower()

    # Check Amoxicillin 500 mg
    a = next(m for m in normalized.medications if m.medicine_name == "Amoxicillin")
    assert a.dosage == "500 MG"
    assert a.formulation == "CAP."
    assert a.duration == "5 days"
    assert "three times daily" in a.readable_sentence.lower()

    # Check Pantoprazole 40 mg
    pan = next(m for m in normalized.medications if m.medicine_name == "Pantoprazole")
    assert pan.dosage == "40 MG"
    assert pan.formulation == "TAB."
    assert "before food" in pan.readable_sentence.lower()

    formatted = normalized.to_formatted_text()
    assert "1. TAB. PARACETAMOL 650 MG" in formatted
    assert "2. CAP. AMOXICILLIN 500 MG" in formatted
    assert "3. TAB. PANTOPRAZOLE 40 MG" in formatted
    # Ensure no headers, doctor names, or addresses leaked as medications
    assert "SRM" not in formatted
    assert "Sharma" not in formatted
    assert "Physician" not in formatted


def test_lab_investigations_and_vitals_filtering(normalizer):
    """Ensure lab tests (CBC, LFT) and vitals (BP, Pulse) are routed to instructions and never treated as medicines."""
    raw_ocr = """Apollo Clinic
Patient: Rahul Verma
Age: 45
Date: 11/11/2026
BP: 130/85 mmHg
Pulse: 74 bpm
Diagnosis: Hypertension and Type 2 Diabetes
Rx
1. Metformin 500 mg - 1 tab BD after meals
2. Amlodipine 5 mg - 1 tab OD morning
Advised Investigations:
CBC, LFT, KFT, HbA1c, Lipid Profile
Review after 15 days"""

    normalized = normalizer.normalize(raw_ocr)
    assert len(normalized.medications) == 2
    med_names = [m.medicine_name for m in normalized.medications]
    assert "Metformin" in med_names
    assert "Amlodipine" in med_names
    assert "CBC" not in med_names
    assert "LFT" not in med_names
    assert "BP" not in med_names

    formatted = normalized.to_formatted_text()
    assert "1. TAB. METFORMIN 500 MG" in formatted
    assert "2. TAB. AMLODIPINE 5 MG" in formatted
    assert "BP: 130/85 mmHg" in formatted or len(normalized.additional_instructions) > 0


def test_cursive_attached_dosage_and_instructions(normalizer):
    """Test cursive handwriting tokens with glued dosages and instructions."""
    raw_ocr = """Riverside Clinic
Date: 2026-11-11
Patient: Carlos Chavez
Aorastatin5mgTakenco duily
meitermin5mg-Afsdrecd
Aoicillin5Dmg-Afgnecdd tet pain"""

    normalized = normalizer.normalize(raw_ocr)
    assert len(normalized.medications) == 3
    med_names = [m.medicine_name for m in normalized.medications]
    assert "Atorvastatin" in med_names
    assert "Metformin" in med_names
    assert "Amoxicillin" in med_names
