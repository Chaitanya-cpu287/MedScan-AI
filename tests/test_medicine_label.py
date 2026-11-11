"""Tests for Medicine Packaging & Box Label Scanner and Normalizer."""

import pytest
from normalization.text_normalizer import TextNormalizer
from normalization.medicine_label_normalizer import MedicineLabelNormalizer, NormalizedMedicineLabel
from validation.prescription_validator import PrescriptionValidator
from utils.file_handler import FileHandler


@pytest.fixture
def label_normalizer():
    return MedicineLabelNormalizer()


@pytest.fixture
def text_normalizer():
    return TextNormalizer()


def test_detect_document_type(text_normalizer):
    """Verify automatic detection between Prescriptions and Medicine Labels."""
    prescription_text = """
    SRM Medical Centre
    Dr. R. Sharma, M.D.
    Patient: Rahul
    Date: 09/11/2026
    1) Dolo-650 Tab
    1 tab Sos for fever
    """
    assert text_normalizer.detect_document_type(prescription_text) == "prescription"

    label_text = """
    PARACETAMOL TABLETS IP 650 mg
    Each film coated tablet contains:
    Paracetamol IP 650 mg.
    Dosage: As directed by the physician.
    Swallow whole with water.
    Warning: Taking more than daily dose may cause serious liver damage.
    Keep out of reach of children.
    Store in a cool and dry place below 30°C.
    Schedule H Prescription Drug
    Mfg. Lic. No.: M/789/2021
    Batch No.: B102
    """
    assert text_normalizer.detect_document_type(label_text) == "medicine_label"


def test_normalize_paracetamol_label(label_normalizer):
    """Verify structured normalization of OTC Paracetamol label."""
    label_text = """
    PARACETAMOL TABLETS IP 650 mg
    Each tablet contains:
    Paracetamol IP 650 mg.
    Dosage: As directed by the physician.
    Swallow with water.
    Warning: Taking more than daily dose may cause serious liver damage.
    Keep out of reach of children.
    Store below 30°C.
    Schedule H Prescription Drug - Warning: To be sold by retail on prescription only.
    Batch No: B123
    Exp. Date: 12/2027
    """
    label: NormalizedMedicineLabel = label_normalizer.normalize(label_text)

    assert label.formulation == "Tablet"
    assert "Schedule H" in label.regulatory_schedule
    assert len(label.active_ingredients) > 0
    assert label.active_ingredients[0].standard_name == "Paracetamol"
    assert "650" in label.active_ingredients[0].strength

    assert any("physician" in a.lower() for a in label.dosage_and_administration)
    assert any("children" in w.lower() for w in label.warnings_and_precautions)
    assert any("below" in s.lower() or "30" in s for s in label.storage_instructions)

    formatted = label.to_formatted_text()
    assert "DOCUMENT TYPE: MEDICINE PACKAGING & REGULATORY LABEL" in formatted
    assert "ACTIVE INGREDIENTS & COMPOSITION" in formatted
    assert "WARNINGS & PRECAUTIONS" in formatted
    assert "STORAGE & HANDLING" in formatted

    export_json = label.to_export_json()
    assert export_json["document_type"] == "medicine_label"
    assert len(export_json["active_ingredients"]) > 0
    assert "warnings" in export_json


def test_normalize_amoxicillin_capsule_label(label_normalizer):
    """Verify antibiotic capsule packaging normalization."""
    label_text = """
    AMOXI-500 CAPSULES
    Each capsule contains:
    Amoxicillin Trihydrate IP equivalent to Amoxicillin 500 mg.
    Dosage: As directed by the physician.
    Capsules to be swallowed whole and not to be chewed, opened or crushed.
    Keep out of reach of children.
    Store in a cool, dry place protected from light and moisture.
    Schedule H Prescription Drug
    """
    label = label_normalizer.normalize(label_text)

    assert label.formulation == "Capsule"
    assert label.generic_name == "Amoxicillin"
    assert len(label.active_ingredients) > 0
    assert "500 mg" in label.active_ingredients[0].strength.lower()
    assert any("swallowed whole" in a.lower() for a in label.dosage_and_administration)
    assert any("children" in w.lower() for w in label.warnings_and_precautions)
    assert any("light" in s.lower() for s in label.storage_instructions)


def test_validate_medicine_label(label_normalizer):
    """Verify validation checks on medicine labels."""
    label_text = """
    PARACETAMOL TABLETS IP 650 mg
    Each tablet contains: Paracetamol IP 650 mg.
    Dosage: As directed by the physician.
    Schedule H Prescription Drug
    """
    label = label_normalizer.normalize(label_text)
    report = PrescriptionValidator.validate(label_text, label)

    assert report.total_medications_found == 1
    assert report.all_medications_retained is True
    assert report.passed_safety_checks is True


def test_file_handler_export_medicine_label(label_normalizer, tmp_path):
    """Verify that FileHandler exports distinct filenames for medicine labels."""
    label_text = "Paracetamol IP 500 mg. Each tablet contains 500 mg. Store below 30°C."
    label = label_normalizer.normalize(label_text)
    report = PrescriptionValidator.validate(label_text, label)

    exported = FileHandler.export_all(
        raw_ocr_text=label_text,
        normalized=label,
        validation_report=report,
        output_dir=tmp_path,
        prefix="test_label",
    )

    assert "normalized_txt" in exported
    assert "normalized_medicine_label.txt" in str(exported["normalized_txt"])
    assert "results_json" in exported
    assert "medicine_label_results.json" in str(exported["results_json"])
    assert exported["normalized_txt"].exists()
    assert exported["results_json"].exists()


def test_multicomponent_combination_medicine_label(label_normalizer):
    """Verify combination medicine label with multiple active salts (Montelukast + Levocetirizine)."""
    label_text = """
    MONTELUKAST & LEVOCETIRIZINE TABLETS
    Composition:
    Each film coated tablet contains:
    Montelukast Sodium IP equivalent to Montelukast 10 mg
    Levocetirizine Dihydrochloride IP 5 mg
    Excipients q.s.
    Dosage: As directed by the physician.
    Schedule H Prescription Drug - Warning: To be sold by retail on prescription only.
    Keep out of reach of children.
    Store protected from light and moisture at a temperature not exceeding 25°C.
    Batch No: ML-901
    Mfg. Lic. No: 28/UA/2021
    Exp. Date: 05/2028
    """
    label = label_normalizer.normalize(label_text)
    assert label.formulation == "Tablet"
    assert len(label.active_ingredients) >= 2

    ingredient_names = [ing.standard_name for ing in label.active_ingredients]
    assert "Montelukast" in ingredient_names
    assert "Levocetirizine" in ingredient_names
    assert "Schedule H" in label.regulatory_schedule
    assert any("25" in s or "light" in s.lower() for s in label.storage_instructions)
