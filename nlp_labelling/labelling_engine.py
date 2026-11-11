"""Enhanced NLP Medication Labelling Engine integrating Member 2's trained spaCy V6 model."""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import spacy

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "trained_model_v6"


def remove_duplicates(items: List[Any]) -> List[Any]:
    unique_items = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            normalized = item.strip().casefold()
            if normalized not in seen:
                seen.add(normalized)
                unique_items.append(item.strip())
        else:
            unique_items.append(item)
    return unique_items


def normalize_strength(strength: str) -> str:
    """Standardizes strength representations (e.g., '500 MG' -> '500 mg')."""
    strength = strength.strip()
    parts = strength.split()
    if len(parts) == 2:
        value, unit = parts
        return f"{value} {unit.lower()}"
    return strength


def normalize_dose(dose_text: str) -> Dict[str, Any]:
    """Normalizes dose text into numeric quantity and standard form."""
    cleaned = dose_text.strip().lower()
    qty = 1.0
    unit = "tablet"

    # Number parsing
    if "half" in cleaned or "1/2" in cleaned:
        qty = 0.5
    elif "two" in cleaned or re.search(r"\b2\b", cleaned):
        qty = 2.0
    elif "three" in cleaned or re.search(r"\b3\b", cleaned):
        qty = 3.0
    elif "four" in cleaned or re.search(r"\b4\b", cleaned):
        qty = 4.0
    elif "one" in cleaned or re.search(r"\b1\b", cleaned):
        qty = 1.0

    if "cap" in cleaned:
        unit = "capsule" if qty <= 1.0 else "capsules"
    elif "syp" in cleaned or "ml" in cleaned:
        unit = "ml"
    elif "puff" in cleaned:
        unit = "puff" if qty <= 1.0 else "puffs"
    elif "drop" in cleaned:
        unit = "drop" if qty <= 1.0 else "drops"
    else:
        unit = "tablet" if qty <= 1.0 else "tablets"

    display_qty = int(qty) if qty.is_integer() else qty
    return {
        "original_text": dose_text.strip(),
        "normalized_value": f"{display_qty} {unit}",
        "quantity": qty,
        "unit": unit,
    }


def normalize_frequency(freq_text: str) -> Dict[str, Any]:
    """Normalizes frequency phrases into standard clinical Latin codes and daily intake count."""
    cleaned = freq_text.strip().lower()
    code = "OD"
    daily_count = 1

    if "twice" in cleaned or "bid" in cleaned or "bd" in cleaned or "1-0-1" in cleaned:
        code = "BID"
        daily_count = 2
    elif "three" in cleaned or "thrice" in cleaned or "tid" in cleaned or "tds" in cleaned or "ttd" in cleaned or "1-1-1" in cleaned or "every 8 hours" in cleaned or "q8h" in cleaned:
        code = "TID"
        daily_count = 3
    elif "four" in cleaned or "qid" in cleaned or "1-1-1-1" in cleaned or "every 6 hours" in cleaned:
        code = "QID"
        daily_count = 4
    elif "bedtime" in cleaned or "hs" in cleaned or "0-0-1" in cleaned:
        code = "HS"
        daily_count = 1
    elif "once" in cleaned or "daily" in cleaned or "od" in cleaned or "qd" in cleaned or "1-0-0" in cleaned:
        code = "QD"
        daily_count = 1
    elif "sos" in cleaned or "prn" in cleaned or "needed" in cleaned:
        code = "PRN"
        daily_count = 0
    elif "every 12 hours" in cleaned or "q12h" in cleaned:
        code = "BID"
        daily_count = 2
    else:
        code = freq_text.strip().upper()
        daily_count = 1

    return {
        "original_text": freq_text.strip(),
        "normalized_value": code,
        "code": code,
        "daily_frequency": daily_count,
        "daily_count": daily_count,
    }


def normalize_timing(timing_text: str) -> Dict[str, Any]:
    """Normalizes food/timing relationships."""
    cleaned = timing_text.strip().lower()
    code = "AC"
    timing_type = "pre-prandial"

    if "after" in cleaned or "post" in cleaned or "pc" in cleaned or "af" in cleaned:
        code = "PC"
        timing_type = "post-prandial (after meals)"
    elif "before" in cleaned or "ante" in cleaned or "ac" in cleaned or "bbf" in cleaned or "empty stomach" in cleaned:
        code = "AC"
        timing_type = "pre-prandial (before meals)"
    elif "with food" in cleaned or "with meals" in cleaned:
        code = "With meals"
        timing_type = "with meals"
    elif "bedtime" in cleaned or "night" in cleaned or "hs" in cleaned:
        code = "HS"
        timing_type = "at bedtime"
    else:
        code = timing_text.strip()
        timing_type = "unspecified"

    return {
        "original_text": timing_text.strip(),
        "normalized_value": f"{code} ({timing_type})" if code in ["AC", "PC", "HS"] else code,
        "code": code,
        "timing_type": timing_type,
    }


def normalize_duration(duration_text: str) -> Dict[str, Any]:
    """Normalizes duration strings to standard day counts."""
    cleaned = duration_text.strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|d|w|m)\b", cleaned)
    days = None

    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("d"):
            days = int(val)
        elif unit.startswith("w"):
            days = int(val * 7)
        elif unit.startswith("m"):
            days = int(val * 30)

    return {
        "original_text": duration_text.strip(),
        "normalized_value": duration_text.strip(),
        "duration_days": days,
    }


class MedicationLabeller:
    """Singleton wrapper for Member 2's trained spaCy V6 NER model with enhanced entity resolution."""

    _instance = None
    _nlp = None

    def __new__(cls, model_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super(MedicationLabeller, cls).__new__(cls)
            target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            try:
                cls._instance._nlp = spacy.load(target_path)
                logger.info(f"Loaded Member 2 spaCy NER model from: {target_path}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model from {target_path}: {e}")
                cls._instance._nlp = None
        return cls._instance

    @property
    def is_available(self) -> bool:
        return self._nlp is not None

    def extract_entities(self, text: str) -> Dict[str, List[Any]]:
        """Runs the trained spaCy model to extract entities."""
        if not self._nlp:
            return {
                "medicines": [],
                "strengths": [],
                "dose": [],
                "frequency": [],
                "timing": [],
                "duration": [],
                "administration": [],
                "warnings": [],
            }

        doc = self._nlp(text)
        entities = {
            "medicines": [],
            "strengths": [],
            "dose": [],
            "frequency": [],
            "timing": [],
            "duration": [],
            "administration": [],
            "warnings": [],
        }

        non_med_stopwords = {
            "warning", "caution", "dosage", "schedule", "store", "mrp", "batch", "mfg",
            "excipients", "excipient", "composition", "contains", "prescription",
            "temperature", "direct", "physician", "doctor", "keep", "reach", "children"
        }
        for ent in doc.ents:
            txt = ent.text.strip()
            if ent.label_ == "MEDICINE":
                if txt.lower() not in non_med_stopwords:
                    entities["medicines"].append(txt)
            elif ent.label_ == "STRENGTH":
                entities["strengths"].append(normalize_strength(txt))
            elif ent.label_ == "DOSE":
                entities["dose"].append(normalize_dose(txt))
            elif ent.label_ == "FREQUENCY":
                entities["frequency"].append(normalize_frequency(txt))
            elif ent.label_ == "TIMING":
                entities["timing"].append(normalize_timing(txt))
            elif ent.label_ == "DURATION":
                if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|d|w|m)\b", txt, re.I):
                    entities["duration"].append(normalize_duration(txt))
            elif ent.label_ == "ADMINISTRATION":
                entities["administration"].append(txt)
            elif ent.label_ == "WARNING":
                entities["warnings"].append(txt)

        entities["medicines"] = remove_duplicates(entities["medicines"])
        entities["strengths"] = remove_duplicates(entities["strengths"])
        entities["administration"] = remove_duplicates(entities["administration"])
        entities["warnings"] = remove_duplicates(entities["warnings"])

        return entities

    def label_prescription(
        self,
        prescription_text: str,
        normalized_prescription_obj: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Extracts prescription entities and creates structured medication groups.

        Compatible with Member 2's extract_prescription_information format while adding
        coherent per-medication groupings and normalized values.
        """
        entities = self.extract_entities(prescription_text)

        # Build Per-Medication Groups
        # If we have normalized_prescription_obj (from Member 1), we can build 100% accurate groups!
        medication_groups = []
        if normalized_prescription_obj and hasattr(normalized_prescription_obj, "medications"):
            for idx, med in enumerate(normalized_prescription_obj.medications, 1):
                dose_norm = normalize_dose(med.readable_sentence)
                freq_norm = normalize_frequency(med.frequency)
                timing_norm = normalize_timing(med.instructions) if med.instructions and med.instructions != "Not specified" else None
                dur_norm = normalize_duration(med.duration) if med.duration and med.duration != "Not specified" else None

                medication_groups.append({
                    "item_number": idx,
                    "medicine_name": med.medicine_name.upper(),
                    "formulation": med.formulation,
                    "strength": med.dosage if med.dosage != "Not specified" else "",
                    "dose": dose_norm,
                    "frequency": freq_norm,
                    "timing": timing_norm,
                    "duration": dur_norm,
                    "instructions": med.readable_sentence,
                    "verification_required": med.verification_required,
                })

            # Ensure all medicines from Member 1 are present in the teammate's flat 'medicines' list
            for m in normalized_prescription_obj.medications:
                m_name = m.medicine_name.upper()
                if m_name not in [x.upper() for x in entities["medicines"]]:
                    entities["medicines"].append(m_name)
                if m.dosage and m.dosage != "Not specified" and m.dosage.lower() not in [s.lower() for s in entities["strengths"]]:
                    entities["strengths"].append(normalize_strength(m.dosage))

        return {
            "document_type": "prescription",
            "original_ocr_text": prescription_text.strip(),
            "medicines": entities["medicines"],
            "strengths": entities["strengths"],
            "dose": entities["dose"],
            "frequency": entities["frequency"],
            "timing": entities["timing"],
            "duration": entities["duration"],
            "medication_groups": medication_groups,
            "review_required": True,
            "review_note": (
                "Verify extracted information against the original prescription. "
                "Normalized values generated via hybrid clinical parsing."
            ),
        }

    def label_medicine_label(self, label_text: str) -> Dict[str, Any]:
        """Extracts medicine label information (medicines, strengths, administration, warnings).

        Compatible with Member 2's extract_medicine_label_information with expanded rule coverage.
        """
        entities = self.extract_entities(label_text)

        administration = entities["administration"].copy()
        warnings = entities["warnings"].copy()

        # Extended pattern matching for OTC / prescription medicine packaging
        admin_patterns = [
            r"As directed by the physician",
            r"swallowed whole",
            r"Shake well before use",
            r"Swallow with water",
            r"For external use only",
            r"Apply to the affected area",
            r"Dissolve in water before taking",
            r"Take with or after food",
        ]

        warning_patterns = [
            r"not to be chewed, opened or crushed",
            r"Do not crush or chew",
            r"Keep out of reach of children",
            r"Do not exceed the recommended dosage",
            r"Schedule H Prescription Drug",
            r"Warning: To be sold by retail on the prescription of a Registered Medical Practitioner only",
            r"Store in a cool and dry place",
            r"Protect from light and moisture",
            r"Not for injection",
            r"Store below \d+°C",
        ]

        for p in admin_patterns:
            matches = re.findall(p, label_text, re.IGNORECASE)
            administration.extend(matches)

        for p in warning_patterns:
            matches = re.findall(p, label_text, re.IGNORECASE)
            warnings.extend(matches)

        return {
            "document_type": "medicine_label",
            "original_ocr_text": label_text.strip(),
            "medicines": remove_duplicates(entities["medicines"]),
            "strengths": remove_duplicates(entities["strengths"]),
            "administration": remove_duplicates(administration),
            "warnings": remove_duplicates(warnings),
            "review_required": True,
            "review_note": (
                "Verify all extracted information against the original medicine label. "
                "Rule-based matches and NER extractions are safety aids, not medical advice."
            ),
        }


# Convenience standalone functions matching Member 2's API exactly
_labeller = None

def get_labeller() -> MedicationLabeller:
    global _labeller
    if _labeller is None:
        _labeller = MedicationLabeller()
    return _labeller

def extract_prescription_information(ocr_text: str) -> Dict[str, Any]:
    return get_labeller().label_prescription(ocr_text)

def extract_medicine_label_information(ocr_text: str) -> Dict[str, Any]:
    return get_labeller().label_medicine_label(ocr_text)
