import json
import re
import spacy
from pathlib import Path


# Load model relative to this Python file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model_v6"

nlp = spacy.load(MODEL_PATH)


def remove_duplicates(items):
    unique_items = []
    seen = set()

    for item in items:
        normalized = item.strip().casefold()

        if normalized not in seen:
            seen.add(normalized)
            unique_items.append(item.strip())

    return unique_items


def normalize_strength(strength):
    strength = strength.strip()
    parts = strength.split()

    if len(parts) == 2:
        value, unit = parts
        return f"{value} {unit.lower()}"

    return strength


def clean_medicine_names(medicines):
    generic_terms = {
        "capsule",
        "capsules",
        "tablet",
        "tablets",
        "injection",
        "syrup",
        "solution",
        "suspension"
    }

    cleaned = []

    for medicine in medicines:
        if medicine.strip().casefold() not in generic_terms:
            cleaned.append(medicine.strip())

    return remove_duplicates(cleaned)


def clean_strengths(strengths):
    valid_strengths = []

    strength_pattern = re.compile(
        r"^\d+(?:\.\d+)?\s*(?:mg|mcg|μg|g|ml|%|iu|units?)$",
        re.IGNORECASE
    )

    for strength in strengths:
        if strength_pattern.fullmatch(strength.strip()):
            valid_strengths.append(
                normalize_strength(strength)
            )

    return remove_duplicates(valid_strengths)


def extract_all_entities(ocr_text):
    doc = nlp(ocr_text)

    entities = {
        "medicines": [],
        "strengths": [],
        "dose": [],
        "frequency": [],
        "timing": [],
        "duration": [],
        "administration": [],
        "warnings": []
    }

    for ent in doc.ents:

        if ent.label_ == "MEDICINE":
            entities["medicines"].append(ent.text.strip())

        elif ent.label_ == "STRENGTH":
            entities["strengths"].append(
                normalize_strength(ent.text)
            )

        elif ent.label_ == "DOSE":
            entities["dose"].append({
                "original_text": ent.text,
                "normalized_value": None
            })

        elif ent.label_ == "FREQUENCY":
            entities["frequency"].append({
                "original_text": ent.text,
                "normalized_value": None
            })

        elif ent.label_ == "TIMING":
            entities["timing"].append({
                "original_text": ent.text,
                "normalized_value": None
            })

        elif ent.label_ == "DURATION":

            # Reject invalid duration predictions such as "2."
            if re.fullmatch(
                r"\d+(?:\.\d+)?\s*(?:days?|weeks?|months?)",
                ent.text.strip(),
                flags=re.IGNORECASE
            ):
                entities["duration"].append({
                    "original_text": ent.text,
                    "normalized_value": None
                })

        elif ent.label_ == "ADMINISTRATION":
            entities["administration"].append(ent.text.strip())

        elif ent.label_ == "WARNING":
            entities["warnings"].append(ent.text.strip())

    for key in [
        "medicines",
        "strengths",
        "administration",
        "warnings"
    ]:
        entities[key] = remove_duplicates(entities[key])

    return entities


def extract_prescription_information(ocr_text):
    entities = extract_all_entities(ocr_text)

    result = {
        "document_type": "prescription",
        "original_ocr_text": ocr_text.strip(),
        "medicines": entities["medicines"],
        "strengths": entities["strengths"],
        "dose": entities["dose"],
        "frequency": entities["frequency"],
        "timing": entities["timing"],
        "duration": entities["duration"],
        "review_required": True,
        "review_note": (
            "Verify extracted information against the original "
            "prescription. An empty field means the model did not "
            "extract a value."
        )
    }

    return result


def extract_medicine_label_information(ocr_text):
    entities = extract_all_entities(ocr_text)

    medicines = clean_medicine_names(
        entities["medicines"]
    )

    strengths = clean_strengths(
        entities["strengths"]
    )

    administration = entities["administration"].copy()
    warnings = entities["warnings"].copy()

    # Additional administration phrase detection
    administration_patterns = [
        r"As directed by the physician",
        r"swallowed whole",
        r"Shake well before use",
        r"Swallow with water"
    ]

    # Additional warning phrase detection
    warning_patterns = [
        r"not to be chewed, opened or crushed",
        r"Do not crush or chew",
        r"Keep out of reach of children",
        r"Do not exceed the recommended dosage"
    ]

    for pattern in administration_patterns:
        matches = re.findall(
            pattern,
            ocr_text,
            flags=re.IGNORECASE
        )

        administration.extend(matches)

    for pattern in warning_patterns:
        matches = re.findall(
            pattern,
            ocr_text,
            flags=re.IGNORECASE
        )

        warnings.extend(matches)

    administration = remove_duplicates(administration)
    warnings = remove_duplicates(warnings)

    result = {
        "document_type": "medicine_label",
        "original_ocr_text": ocr_text.strip(),
        "medicines": medicines,
        "strengths": strengths,
        "administration": administration,
        "warnings": warnings,
        "review_required": True,
        "review_note": (
            "Verify all extracted information against the original "
            "medicine label. Rule-based matches are not medical advice."
        )
    }

    return result


def save_output(data, filename):
    output_file = BASE_DIR / filename

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Output saved to: {output_file}")


if __name__ == "__main__":

    prescription_text = """
    AMOXICILLIN 500 MG CAPSULES

    Dosage:
    Take one capsule three times daily after meals for 7 days.
    """

    medicine_label_text = """
    PENTAB-DSR

    PANTOPRAZOLE GASTRO-RESISTANT &
    DOMPERIDONE PROLONGED RELEASE CAPSULES IP

    Each hard gelatin capsule contains:

    Pantoprazole Sodium Sesquihydrate IP
    equivalent to Pantoprazole 40 mg.

    Domperidone IP 30 mg.

    Dosage: As directed by the physician.

    Capsule should be swallowed whole and
    not to be chewed, opened or crushed.
    """

    prescription_data = extract_prescription_information(
        prescription_text
    )

    medicine_label_data = extract_medicine_label_information(
        medicine_label_text
    )

    save_output(
        prescription_data,
        "prescription_structured_output.json"
    )

    save_output(
        medicine_label_data,
        "medicine_label_structured_output.json"
    )

    print("\n========== PRESCRIPTION OUTPUT ==========")

    print(
        json.dumps(
            prescription_data,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n========== MEDICINE LABEL OUTPUT ==========")

    print(
        json.dumps(
            medicine_label_data,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nV6 structured extraction completed!")