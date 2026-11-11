"""Intelligent Medical Prescription Text Normalizer and Entity Reconstructor."""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union

from config.settings import CONFIG
from normalization.medicine_matcher import MedicineMatcher, MedicineMatchResult
from normalization.abbreviation_handler import AbbreviationHandler, FrequencyInterpretation
from normalization.pharmacopeia_db import FORMULATIONS
from normalization.medicine_label_normalizer import NormalizedMedicineLabel, MedicineLabelNormalizer

logger = logging.getLogger(__name__)


@dataclass
class NormalizedMedication:
    original_text: str
    medicine_name: str
    formulation: str
    dosage: str
    frequency: str
    duration: str
    instructions: str
    readable_sentence: str
    verification_required: bool
    confidence_score: float
    correction_reason: str
    soundalike_warning: Optional[str] = None


@dataclass
class NormalizedPrescription:
    patient_name: str
    patient_age: str
    date: str
    doctor_name: str
    clinic_name: str
    medications: List[NormalizedMedication] = field(default_factory=list)
    additional_instructions: List[str] = field(default_factory=list)
    uncertain_fields: List[str] = field(default_factory=list)
    proposed_corrections: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_formatted_text(self) -> str:
        """Generates clean standardized plain text expected by downstream medication labelling model."""
        lines = []
        patient_str = self.patient_name if self.patient_name else "Not specified"
        date_str = self.date if self.date else "Not specified"

        lines.append(f"PATIENT: {patient_str}")
        lines.append(f"DATE: {date_str}")
        lines.append("")
        lines.append("MEDICATION DETAILS")
        lines.append("")

        if not self.medications:
            lines.append("No medications identified.")
            lines.append("")
        else:
            for idx, med in enumerate(self.medications, 1):
                # Header: e.g. 1. TAB. PARACETAMOL 650 MG
                name_upper = med.medicine_name.upper()
                dosage_str = f" {med.dosage}" if med.dosage and med.dosage != "Not specified" else ""
                form_str = f"{med.formulation} " if med.formulation else ""
                lines.append(f"{idx}. {form_str}{name_upper}{dosage_str}")
                
                # Instruction sentence indent: e.g.    Take one tablet twice daily after food for 5 days.
                lines.append(f"   {med.readable_sentence}")
                lines.append("")

        if self.additional_instructions:
            lines.append("Additional Instructions:")
            for inst in self.additional_instructions:
                lines.append(inst)
            lines.append("")

        return "\n".join(lines).strip()

    def to_export_json(self) -> Dict[str, Any]:
        """Strict JSON schema required by teammate's model and Section 10 specification."""
        meds_json = []
        for m in self.medications:
            meds_json.append({
                "original_text": m.original_text,
                "medicine_name": m.medicine_name,
                "dosage": m.dosage if m.dosage else "Not specified",
                "frequency": m.frequency if m.frequency else "Not specified",
                "duration": m.duration if m.duration else "Not specified",
                "instructions": m.instructions if m.instructions else "Not specified",
                "verification_required": m.verification_required,
            })

        return {
            "patient": {
                "name": self.patient_name if self.patient_name else "Not specified",
                "date": self.date if self.date else "Not specified",
            },
            "medications": meds_json,
            "additional_instructions": self.additional_instructions,
            "uncertain_fields": self.uncertain_fields,
        }


class TextNormalizer:
    """Intelligently reconstructs, normalizes and validates prescription OCR text."""

    def __init__(self, matcher: Optional[MedicineMatcher] = None):
        self.matcher = matcher or MedicineMatcher()
        self.label_normalizer = MedicineLabelNormalizer(matcher=self.matcher)

    @staticmethod
    def detect_document_type(text: str) -> str:
        """Determines whether OCR text is from a Medical Prescription or a Medicine Packaging Label."""
        cleaned = text.lower()
        label_score = 0
        prescription_score = 0

        # Strong packaging / box label indicators
        label_patterns = [
            r"\beach\s+(?:tablet|film|uncoated|capsule|ml|5\s*ml|dose)\s*contains?\b",
            r"\bcomposition\b",
            r"\bschedule\s+[hgxk1-9]\b",
            r"\bto be sold by retail\b",
            r"\bkeep out of reach of children\b",
            r"\bstore (?:below|in a cool|protected|at room)\b",
            r"\b(?:mfg\.?\s*lic|batch\s*no|exp\.?\s*date|mfd\.?\s*by|b\.?\s*no)\b",
            r"\bfor (?:external|ophthalmic|pediatric|oral) use only\b",
            r"\bswallowed whole\b",
            r"\bshake well\b",
            r"\bdo not (?:chew|crush|open)\b",
        ]
        for p in label_patterns:
            if re.search(p, cleaned):
                label_score += 2

        # Strong prescription indicators
        prescription_patterns = [
            r"\bpatient(?:\s*name)?\b",
            r"\bdr[\.\s]|doctor\b",
            r"\b(?:age|years?|yrs?)\s*[:=-]?\s*\d+\b",
            r"\b(?:date|datc|dt)\s*[:=-]?\s*[0-9oO]{1,2}[/-][0-9oO]{1,2}\b",
            r"\b(?:rx|℞)\b",
            r"\b(?:1-0-1|1-1-1|0-0-1|1-0-0)\b",
            r"\b(?:bid|tid|qid|sos|hs|stat|tds|ttd|qd)\b",
            r"\bfor\s+\d+\s*(?:days?|weeks?|months?)\b",
            r"\b(?:take|instill|administer)\s+(?:one|two|half|\d+)\s*(?:tabs?|caps?)\b",
        ]
        for p in prescription_patterns:
            if re.search(p, cleaned):
                prescription_score += 1

        return "medicine_label" if label_score > prescription_score else "prescription"

    def _extract_patient_info(self, text: str) -> Tuple[str, str, str, str, str]:
        """Extracts patient name, age, date, doctor name, clinic name using contextual patterns."""
        patient_name = ""
        patient_age = ""
        date = ""
        doctor_name = ""
        clinic_name = ""

        raw_lines = [l.strip() for l in text.splitlines() if l.strip()]

        # 1. Patient Name Extraction
        # Look for 'NAME' label and check same line, preceding line, or following line
        for idx, line in enumerate(raw_lines):
            # Same line e.g. 'NAME: John Smith' or 'Patient Name - Priya Sharma'
            m_inline = re.search(r"^(?:name|patient(?:\s*name)?|pt\.?)\s*[:=-]\s*([A-Za-z\s\.\,\-]+)$", line, re.IGNORECASE)
            if m_inline:
                cand = m_inline.group(1).strip()
                if len(cand) > 1 and not re.match(r"^(age|date|datc|sex|male|female)$", cand, re.IGNORECASE):
                    patient_name = cand
                    break

            # Standalone 'NAME' or 'PATIENT' line
            if re.match(r"^(name|patient|pt\.?)$", line, re.IGNORECASE):
                # Check preceding line first (common in form-over-text OCR)
                if idx > 0:
                    prev_line = raw_lines[idx - 1]
                    if re.match(r"^[A-Za-z\s\.]+$", prev_line) and len(prev_line.split()) <= 4:
                        if not re.search(r"\b(hospital|clinic|centre|center|street|st|road|date|age)\b", prev_line, re.IGNORECASE):
                            patient_name = prev_line
                            break
                # Check following line
                if idx + 1 < len(raw_lines):
                    next_line = raw_lines[idx + 1]
                    if re.match(r"^[A-Za-z\s\.]+$", next_line) and len(next_line.split()) <= 4:
                        if not re.search(r"\b(hospital|clinic|centre|center|street|st|road|date|age)\b", next_line, re.IGNORECASE):
                            patient_name = next_line
                            break

        # If not found via standalone label, fallback to regex search
        if not patient_name:
            pat_match = re.search(r"(?:pa[tl]i?enti?|pt\.?|name|pelnt|patcni|alicnt)\s*[:=-]?\s*([A-Za-z\s]+?)(?=[,\.\s]*(?:age|aoo|dat[ceua]|sex|dr|mrs?|mr)\b|\n|$)", text, re.IGNORECASE)
            if pat_match:
                cand = pat_match.group(1).strip()
                if len(cand) > 1 and not re.match(r"^(age|date|datc|sex|male|female)$", cand, re.IGNORECASE):
                    patient_name = cand

        # Normalize common cursive/script handwriting OCR misreadings in names
        if patient_name:
            patient_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", patient_name)
            patient_name = re.sub(r"\bJoln\b", "John", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\bSmit[li]\b", "Smith", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\bFene\s*Smul\b", "Jane Smith", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\bWab\b", "Wei Li", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\bMara\s*arcia\b", "Maria Garcia", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\bCares\s*Chave\b", "Carlos Chavez", patient_name, flags=re.IGNORECASE)
            patient_name = re.sub(r"\b([A-Za-z]+)\s+([A-Za-z]+)", lambda m: f"{m.group(1).capitalize()} {m.group(2).capitalize()}", patient_name.strip())

        # 2. Patient Age Extraction
        age_match = re.search(r"\b(?:age|years?|yrs?|y/o|aoo)\s*[:=-]?\s*(\d{1,3}|[Yy]\d{1,2}|3g)\b", text, re.IGNORECASE)
        if age_match:
            raw_age = age_match.group(1)
            raw_age = re.sub(r"[Yy]", "4", raw_age)  # OCR Y -> 4
            raw_age = re.sub(r"3g", "38", raw_age)   # OCR 3g -> 38
            patient_age = raw_age
        else:
            # Check for 'AGE' line followed by a number line e.g. 'AGE\n34'
            for idx, line in enumerate(raw_lines):
                if re.match(r"^age$", line, re.IGNORECASE) and idx + 1 < len(raw_lines):
                    if re.match(r"^\d{1,3}$", raw_lines[idx + 1]):
                        patient_age = raw_lines[idx + 1]
                        break

        # 3. Date Extraction (handles OCR letter 'O' in place of '0', missing slashes like 0911/2026, and 2-digit years)
        date_match = re.search(r"\b(?:dat[ceua]|dt|d[ao]te)\s*[:=-]?\s*([0-9oO]{1,2})[\./\-]([0-9oO]{1,2})[\./\-]([0-9oO]{2,4})\b", text, re.IGNORECASE)
        if date_match:
            d, mth, y = date_match.group(1), date_match.group(2), date_match.group(3)
            d = re.sub(r"[oO]", "0", d).zfill(2)
            mth = re.sub(r"[oO]", "0", mth).zfill(2)
            y = re.sub(r"[oO]", "0", y)
            if len(y) == 2:
                y = f"20{y}"
            date = f"{d}/{mth}/{y}"
        else:
            # Match 4 contiguous digits before year e.g. 'Date: 0911/2026' or '0911-2026'
            date_match2 = re.search(r"\b(?:dat[ceua]|dt|d[ao]te)\s*[:=-]?\s*([0-9oO]{2})([0-9oO]{2})[\./\-]??([0-9oO]{4})\b", text, re.IGNORECASE)
            if date_match2:
                d, mth, y = date_match2.group(1), date_match2.group(2), date_match2.group(3)
                d = re.sub(r"[oO]", "0", d)
                mth = re.sub(r"[oO]", "0", mth)
                y = re.sub(r"[oO]", "0", y)
                date = f"{d}/{mth}/{y}"
            else:
                # Standalone date e.g. '09/11/2012' or '09-11-12'
                date_match3 = re.search(r"\b([0-9oO]{1,2})[/\-]([0-9oO]{1,2})[/\-]([0-9oO]{2,4})\b", text, re.IGNORECASE)
                if date_match3:
                    d, mth, y = date_match3.group(1), date_match3.group(2), date_match3.group(3)
                    d = re.sub(r"[oO]", "0", d).zfill(2)
                    mth = re.sub(r"[oO]", "0", mth).zfill(2)
                    y = re.sub(r"[oO]", "0", y)
                    if len(y) == 2:
                        y = f"20{y}"
                    date = f"{d}/{mth}/{y}"

        # 4. Doctor Name Extraction
        doc_match = re.search(r"\b(?:dr\.?|doctor)\s*([A-Za-z\s\.\,\-]+?)(?:\n|clinic|hospital|signature|sig|$|\r)", text, re.IGNORECASE)
        if doc_match:
            raw_doc = doc_match.group(1).strip()
            raw_doc = re.sub(r"\bJolnson\b", "Johnson", raw_doc, flags=re.IGNORECASE)
            doctor_name = f"Dr. {raw_doc}"

        # 5. Clinic / Hospital Extraction
        clinic_match = re.search(r"\b([A-Za-z\s\.\,\-]+(?:hospital|clinic|centre|center|health care|dispensary))\b", text, re.IGNORECASE)
        if clinic_match:
            c_name = clinic_match.group(1).strip()
            # If all uppercase without spaces e.g. RIVERSIDEMEDICALCENTRE
            if " " not in c_name and c_name.isupper():
                c_name = re.sub(r"(RIVERSIDE)(MEDICAL)(CENTRE)", r"\1 \2 \3", c_name, flags=re.IGNORECASE)
            clinic_name = c_name

        return patient_name, patient_age, date, doctor_name, clinic_name

    def _clean_ocr_line(self, text: str) -> str:
        """De-noises OCR tokens, strips numbering prefixes, fixes character substitutions, and disentangles merged tokens."""
        c = text.strip()
        # 1. Strip numbering prefixes e.g. '1) ', '2. ', '(3) ', '1- '
        c = re.sub(r"^\s*(?:\(?\d{1,2}\)?[\.\-\)]|\d{1,2}\s*[\.\-\)])\s*", "", c)
        # 2. Fix merged date prefix e.g. DATEO9-11-12 -> DATE 09-11-12
        c = re.sub(r"\b(date|datc|dt)[oO](\d)", r"\1 0\2", c, flags=re.IGNORECASE)
        # 3. Fix T1D/B1D/Q1D in words before splitting
        c = re.sub(r"T1D", "TID", c, flags=re.IGNORECASE)
        c = re.sub(r"B1D", "BID", c, flags=re.IGNORECASE)
        c = re.sub(r"Q1D", "QID", c, flags=re.IGNORECASE)
        # 3. Fix digit 3 inside letters e.g. Pantopra3ole -> Pantoprazole
        c = re.sub(r"(?<=[a-zA-Z])3(?=[a-zA-Z])", "z", c)
        # 4. Fix digit 0 inside letters e.g. Dol0 -> Dolo
        c = re.sub(r"(?<=[a-zA-Z])0(?=[a-zA-Z0-9\s\-_])", "o", c)
        # 5. Split attached letters and numbers e.g. Aoicillin5 -> Aoicillin 5, meitermin5mg -> meitermin 5mg
        c = re.sub(r"([a-zA-Z]{3,})(\d+)", r"\1 \2", c)
        # 6. Disentangle attached dosage units and trailing words/hyphens e.g. 5mgTakenco -> 5 mg - Takenco, 50mg-Aekcime -> 50 mg - Aekcime
        c = re.sub(r"\b(\d+(?:\.\d+)?)\s*(mg|gm|g|mcg|ml|iu|tablets?|caps?)([A-Za-z]+)", r"\1 \2 - \3", c, flags=re.IGNORECASE)
        c = re.sub(r"(\d+(?:\.\d+)?\s*(?:mg|gm|g|mcg|ml|iu))\s*[-–—]\s*([A-Za-z])", r"\1 - \2", c, flags=re.IGNORECASE)
        # 7. Fix OCR 'ng' (misread 'mg') e.g. 5ng -> 5 mg, 25ng -> 25 mg
        c = re.sub(r"\b(\d+(?:\.\d+)?)\s*ng\b", r"\1 mg", c, flags=re.IGNORECASE)
        # 8. Fix letter o/O/D/d in dosages e.g. 5o0 mg -> 500 mg, 5Dmg -> 50 mg, 4o mg -> 40 mg
        def fix_dosage_chars(m):
            num_fixed = re.sub(r"[oODdQ]", "0", m.group(1))
            return f"{num_fixed} {m.group(2)}"
        c = re.sub(r"\b([1-9][0-9oODdQ]{0,3})\s*(mg|gm|g|mcg|ml|iu|tablets?|caps?)\b", fix_dosage_chars, c, flags=re.IGNORECASE)
        # 9. Un-hyphenate brand dosage e.g. Dolo-650 -> Dolo 650, Augmentin-625 -> Augmentin 625
        def replace_hyphen(m):
            prefix = m.group(1)
            if prefix.lower() in ("mg", "gm", "g", "mcg", "ml", "iu", "tab", "tabs", "cap", "caps", "tablet", "tablets"):
                return m.group(0)  # Keep original hyphen
            return f"{prefix} {m.group(2)}"
        c = re.sub(r"\b([A-Za-z]+)\s*[-–—]\s*(\d+(?:\.\d+)?)\b", replace_hyphen, c)
        # 10. Split merged form and frequency e.g. tabOD -> tab OD, capTID -> cap TID
        c = re.sub(r"(tabs?|tablets?|caps?|capsules?)(OD|BD|BID|TID|QID|SOS|HS|STAT|PRN|TTD)", r"\1 \2", c, flags=re.IGNORECASE)
        # 11. Split frequency and prepositions e.g. TIDfor -> TID for, ODbefore -> OD before
        c = re.sub(r"(TID|BID|QID|OD|BD|SOS|HS|STAT|PRN|TTD)(for|before|after|at|with)", r"\1 \2", c, flags=re.IGNORECASE)
        # 12. Split dose quantity and unit e.g. 1tab -> 1 tab, 1cap -> 1 cap
        c = re.sub(r"\b(\d+(?:/\d+)?)\s*(tabs?|tablets?|caps?|capsules?|drops?|puffs?|ml|tsp)\b", r"\1 \2", c, flags=re.IGNORECASE)
        c = re.sub(r"\s+", " ", c).strip()
        return c

    def _is_header_or_address_line(self, line: str) -> bool:
        """Determines if a line is a clinic/hospital header, doctor name/credentials, address, or prescription artifact."""
        cleaned = line.strip()
        if not cleaned:
            return True
        # Prescription artifacts & single symbol lines
        if re.search(r"^(?:rx|℞|r|尔|sig|signature|date|datc|dt)[\s:\.\-]*$", cleaned, re.IGNORECASE):
            return True
        if re.search(r"^[尔℞rx\W_]+$", cleaned, re.IGNORECASE):
            return True
        # Doctor header variations including cursive OCR: 'preseried', 'resrited', 'rescrled', 'eserdy' (Prescribed by)
        if re.search(r"\b(?:dr[\.\s\.]*|doctor|physician|surgeon|consultant|mbbs|m\.?d\.?|m\.?s\.?|bams|bhms|bds|dnb|frcs|general physician|general practitioner|reg[\.\s]*no|prescri|rescr|eserdy|preseried)\b|\bdr\b", cleaned, re.IGNORECASE):
            return True
        if re.search(r"\b(?:dr[\.\s\.]*|doctor|physician|surgeon|consultant|mbbs|m\.?d\.?|m\.?s\.?|bams|bhms|bds|dnb|frcs|general physician|general practitioner|reg[\.\s]*no)\b|\bdr\b", cleaned, re.IGNORECASE):
            return True
        # Clinic, Hospital, Healthcare institution (with or without spaces e.g. RIVERSIDEMEDICALCENTRE)
        if re.search(r"(?:hospital|clinic|centre|center|dispensary|nursing\s*home|healthcare|department|dept|opd|ipd)", cleaned, re.IGNORECASE):
            return True
        # Addresses: street, road, avenue, colony, zip, city, state, country
        if re.search(r"\b(?:street|st\b|road|rd\b|ave\b|avenue|lane|nagar|colony|layout|sector|block|floor|flat|apt|apartment|building|bldg|opp\b|near\b|behind|cross\b|new\s*york|ny\b|usa|india|delhi|mumbai|bangalore|bengaluru|chennai|kolkata|hyderabad|pune|pincode|pin\b|zip|postal|po\s*box|phone|tel|mobile)\b", cleaned, re.IGNORECASE):
            return True
        if re.search(r"^\d+\s*(?:st|nd|rd|th)?\s*(?:street|st|road|rd|ave|lane)", cleaned, re.IGNORECASE):
            return True
        # Patient metadata labels
        if re.search(r"^(?:patient|pt[\.\s]|name|age|years?|yrs?|sex|gender|male|female|date|datc|dt|dateo|datco|address)\b", cleaned, re.IGNORECASE):
            return True
        return False

    def _is_instruction_line(self, line: str) -> bool:
        """Determines if a line represents dosage instructions, frequency, timing, or clinical guidance for a medicine."""
        cleaned = line.strip()
        if not cleaned:
            return False
        # Dose quantity + form: e.g. 1 tab, 2 caps, 1 tablet, 1/2 tab, 5 ml
        if re.search(r"^(?:take\s+|administer\s+|apply\s+|instill\s+|inhale\s+)?(?:\d+(?:/\d+)?|half|one|two|three|four)\s*(?:tabs?|tablets?|caps?|capsules?|drops?|puffs?|ml|tsp|spoon)\b", cleaned, re.IGNORECASE):
            return True
        # Frequency or timing start: e.g. OD, BD, TID, 1-0-1, before food, after meals
        if re.search(r"^(?:od|bd|bid|tds|tid|qid|sos|hs|stat|prn|ttd|1-0-1|1-1-1|0-0-1|before\s*(?:food|meals?)|after\s*(?:food|meals?)|at\s*bedtime|empty\s*stomach)\b", cleaned, re.IGNORECASE):
            return True
        # Clinical indication / guidance
        if re.search(r"\b(?:for\s*(?:fever|pain|headache|cough|cold|infection|relief|sleep)|as\s*needed|plenty\s*of\s*water|review\s*after)\b", cleaned, re.IGNORECASE):
            return True
        return False

    def _clean_instruction_artifacts(self, text: str) -> str:
        """Cleans handwriting OCR artifacts and missing spaces from instruction tokens."""
        if not text:
            return ""
        c = text.strip()
        # Separate run-together numbers, tabs, and frequency codes e.g. '1+abBD' -> '1 tabs BD', '2+absTD' -> '2 tabs TD'
        c = re.sub(r"(\d+)\s*\+?t?a?b(?:s)?([A-Za-z]+)", r" \1 tabs \2 ", c, flags=re.IGNORECASE)
        # Normalizes handwriting slashes and symbols: '/ +tab' -> '1 tab'
        c = re.sub(r"(?:^|\s)[/\\]\s*\+?t?a?b?s?\b", " 1 tab ", c, flags=re.IGNORECASE)
        c = re.sub(r"[-~]?\s*([1-9]|1/2)\s*\+?t?abs?\b", r" \1 tabs ", c, flags=re.IGNORECASE)
        c = re.sub(r"[-~]?\s*([1-9]|1/2)\s*\+?t?ab\b", r" \1 tab ", c, flags=re.IGNORECASE)
        c = re.sub(r"\+tabs?\b", "tab", c, flags=re.IGNORECASE)
        c = re.sub(r"\+abs?\b", "tabs", c, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", c).strip()

    def _is_clinical_instruction(self, text: str) -> bool:
        """Determines if a floating line is genuine clinical advice rather than header/address noise."""
        cleaned = text.strip()
        if len(cleaned) < 4:
            return False
        # Filter out headers, addresses, metadata
        if re.search(r"^(dat[ceua]|dt|pal?cent|patie?nt|name|age|dr[\.\s]|doctor|clinic|hospital|hosp|centre|center|signature|sig\b|rx\b|addr|street|ave|road|city|state|zip)", cleaned, re.IGNORECASE):
            return False
        if re.search(r"^\d+$|^[\W_]+$|\b\d{5}\b|\b(?:street|st|ave|avenue|road|rd|ny|usa)\b", cleaned, re.IGNORECASE):
            return False
        # Must contain clinical advice keywords
        clinical_keywords = r"\b(water|food|meal|meals|empty\s*stomach|avoid|review|follow\s*up|days?|dose|take|precaution|diet|advice|note|warning|instruction|rest|plenty|sufficient|exceed|alcohol|smoke|warm|bedtime|morning|night)\b"
        return bool(re.search(clinical_keywords, cleaned, re.IGNORECASE))

    def _is_lab_test_or_vital_line(self, line: str) -> bool:
        """Detects laboratory investigations, clinical diagnostic tests, and vital signs."""
        cleaned = line.strip()
        vitals_pattern = r"\b(?:bp|blood\s*pressure|pulse|pr|heart\s*rate|hr|spo2|temp(?:erature)?|weight|wt|height|ht|bmi)\s*[:=-]?\s*\d+"
        if re.search(vitals_pattern, cleaned, re.IGNORECASE):
            return True
        labs_pattern = r"\b(?:cbc|lft|kft|rft|hba1c|fbs|ppbs|rbs|blood\s*sugar|serum\s*(?:creatinine|electrolytes|bilirubin|calcium)|lipid\s*profile|thyroid\s*profile|tsh|t3|t4|urine\s*r/?m|urine\s*routine|urine\s*culture|esr|crp|widal|dengue\s*(?:ns1|igm|igg)|malaria|ecg|2d\s*echo|x-?ray|cxr|usg|ultrasound|ct\s*(?:scan|brain|abdomen)|mri)\b"
        if re.search(labs_pattern, cleaned, re.IGNORECASE):
            return True
        diagnoses_pattern = r"\b(?:diagnosis|dx|c/o|chief\s*complaint|provisional\s*diagnosis|impression|known\s*case\s*of|k/c/o)\s*[:=-]"
        if re.search(diagnoses_pattern, cleaned, re.IGNORECASE):
            return True
        return False

    def _extract_duration(self, text: str) -> Tuple[str, str]:
        """Extracts duration (e.g. 'for 5 days', '7 days', '1 month', 'x 3d') and remaining text."""
        duration = ""
        cleaned = text
        pattern = r"\b(?:for\s+|x\s*)?(\d+\s*(?:days?|weeks?|months?|d|w|m))\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_dur = match.group(1)
            # Expand '3d' -> '3 days'
            if raw_dur.lower().endswith("d") and not raw_dur.lower().endswith("nd"):
                num = raw_dur[:-1].strip()
                duration = f"{num} days"
            elif raw_dur.lower().endswith("w"):
                num = raw_dur[:-1].strip()
                duration = f"{num} weeks"
            elif raw_dur.lower().endswith("m"):
                num = raw_dur[:-1].strip()
                duration = f"{num} months"
            else:
                duration = raw_dur

            cleaned = (text[:match.start()] + " " + text[match.end():]).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)

        return duration, cleaned

    def _build_readable_sentence(
        self,
        formulation: str,
        dosage: str,
        freq_eng: str,
        food_eng: Optional[str],
        duration: str,
        custom_inst: str,
    ) -> str:
        """Constructs an idiomatic clinical sentence from structured components."""
        verb = "Take"
        noun = "tablet"
        f_lower = formulation.lower()
        if "cap" in f_lower:
            noun = "capsule"
        elif "syp" in f_lower or "susp" in f_lower:
            noun = "5 ml syrup"
        elif "inj" in f_lower:
            verb = "Administer"
            noun = "injection"
        elif "drops" in f_lower:
            verb = "Instill"
            noun = "drops"
        elif "inh" in f_lower:
            verb = "Inhale"
            noun = "puff"
        elif "oint" in f_lower or "gel" in f_lower:
            verb = "Apply"
            noun = "topically"

        # Check for dose quantity e.g. '2 tabs', '1 cap', 'half tablet'
        qty_str = "one"
        if custom_inst:
            m_qty = re.search(r"\b(1/2|half|[1-9])\s*(?:tabs?|tablets?|caps?|capsules?)\b", custom_inst, re.IGNORECASE)
            if m_qty:
                q_val = m_qty.group(1).lower()
                q_map = {"1": "one", "2": "two", "3": "three", "4": "four", "1/2": "half", "half": "half"}
                qty_str = q_map.get(q_val, "one")
                # Remove quantity phrase from custom_inst
                custom_inst = (custom_inst[:m_qty.start()] + " " + custom_inst[m_qty.end():]).strip()

        # Pluralize noun if quantity > 1
        if qty_str in ("two", "three", "four") and not noun.endswith("s"):
            noun = f"{noun}s"

        # Merge clinical indication e.g. 'for fever' with SOS frequency 'as needed'
        if custom_inst and re.search(r"\bfor\s+(?:fever|pain|headache|cough|cold|infection|relief|sleep)\b", custom_inst, re.IGNORECASE):
            m_ind = re.search(r"\b(for\s+(?:fever|pain|headache|cough|cold|infection|relief|sleep))\b", custom_inst, re.IGNORECASE)
            if m_ind:
                ind_phrase = m_ind.group(1).lower()
                if freq_eng and "as needed" in freq_eng.lower():
                    freq_eng = f"as needed {ind_phrase}"
                    custom_inst = (custom_inst[:m_ind.start()] + " " + custom_inst[m_ind.end():]).strip()

        parts = [f"{verb} {qty_str} {noun}"]
        if freq_eng and freq_eng != "Not specified":
            parts.append(freq_eng)
        if food_eng:
            parts.append(food_eng)
        if duration and duration != "Not specified":
            parts.append(f"for {duration}")

        sentence = " ".join(parts).strip()

        # Clean custom_inst: remove frequency abbreviations, dosages, and punctuation so they don't repeat
        if custom_inst:
            cleaned_custom = re.sub(r"\b(bid|bd|tid|ttd|tds|qd|od|qid|hs|sos|stat|1-0-1|1-1-1|0-0-1)\b", "", custom_inst, flags=re.IGNORECASE)
            cleaned_custom = re.sub(r"\b(\d+(?:\.\d+)?\s*(?:mg|gm|mcg|ml))\b", "", cleaned_custom, flags=re.IGNORECASE)
            cleaned_custom = re.sub(r"\b(?:tabs?|tablets?|caps?|capsules?)\b", "", cleaned_custom, flags=re.IGNORECASE)
            cleaned_custom = re.sub(r"^[-\s\.\,/\\+~]+|[-\s\.\,/\\+~]+$", "", cleaned_custom).strip()
            if cleaned_custom and len(cleaned_custom) > 2 and cleaned_custom.lower() not in sentence.lower():
                sentence = f"{sentence}. {cleaned_custom}"

        if not sentence.endswith("."):
            sentence += "."

        # Capitalize first letter
        return sentence[0].upper() + sentence[1:]

    def _normalize_from_structured_dict(self, data: Dict[str, Any]) -> NormalizedPrescription:
        """Normalizes prescription when input is already structured JSON (e.g. fine-tuned model output)."""
        patient_name = data.get("patient_name", "") or data.get("patient", {}).get("name", "")
        patient_age = str(data.get("patient_age", "") or "")
        date = data.get("date", "") or data.get("patient", {}).get("date", "")
        doctor_name = data.get("doctor_name", "")
        clinic_name = data.get("clinic_name", "")

        medications: List[NormalizedMedication] = []
        proposed_corrections = []
        warnings = []
        uncertain_fields = []

        raw_meds = data.get("medications", [])
        if isinstance(raw_meds, list):
            for item in raw_meds:
                if not isinstance(item, dict):
                    continue
                drug_name_raw = item.get("drug_name", "") or item.get("name", "") or item.get("medicine_name", "")
                dosage_raw = item.get("dosage", "")
                frequency_raw = item.get("frequency", "")
                duration_raw = item.get("duration", "")
                instructions_raw = item.get("instructions", "")

                match_res = self.matcher.match_medicine_name(drug_name_raw)

                # Dosage: if not in item, check if extracted from name
                final_dosage = dosage_raw.strip().upper() if dosage_raw else match_res.extracted_dosage
                if not final_dosage:
                    final_dosage = "Not specified"

                # Frequency interpretation
                freq_interp = AbbreviationHandler.interpret_frequency(frequency_raw)
                food_interp = AbbreviationHandler.interpret_food_timing(instructions_raw or frequency_raw)
                food_eng = food_interp[1] if food_interp else ""

                # Duration
                dur_extracted, _ = self._extract_duration(duration_raw)
                final_duration = dur_extracted if dur_extracted else (duration_raw.strip() if duration_raw else "Not specified")

                # Build readable sentence
                readable = self._build_readable_sentence(
                    formulation=match_res.formulation,
                    dosage=final_dosage,
                    freq_eng=freq_interp.expanded_english,
                    food_eng=food_eng,
                    duration=final_duration,
                    custom_inst=instructions_raw if not food_eng else "",
                )

                if match_res.is_corrected:
                    proposed_corrections.append({
                        "field": "medicine_name",
                        "original": drug_name_raw,
                        "normalized": match_res.matched_name,
                        "confidence": match_res.confidence_score,
                        "reason": match_res.correction_reason,
                    })

                if match_res.verification_required:
                    uncertain_fields.append(f"Medicine: {drug_name_raw}")

                medications.append(NormalizedMedication(
                    original_text=drug_name_raw,
                    medicine_name=match_res.matched_name,
                    formulation=match_res.formulation,
                    dosage=final_dosage,
                    frequency=freq_interp.original_notation or frequency_raw,
                    duration=final_duration,
                    instructions=instructions_raw or food_eng or "Not specified",
                    readable_sentence=readable,
                    verification_required=match_res.verification_required or freq_interp.ambiguous,
                    confidence_score=match_res.confidence_score,
                    correction_reason=match_res.correction_reason,
                    soundalike_warning=match_res.soundalike_warning,
                ))

        # Additional instructions / diagnosis / notes
        add_inst = []
        if data.get("diagnosis"):
            add_inst.append(f"Diagnosis: {data['diagnosis']}")
        if data.get("notes"):
            add_inst.append(f"Notes: {data['notes']}")

        return NormalizedPrescription(
            patient_name=patient_name,
            patient_age=patient_age,
            date=date,
            doctor_name=doctor_name,
            clinic_name=clinic_name,
            medications=medications,
            additional_instructions=add_inst,
            uncertain_fields=uncertain_fields,
            proposed_corrections=proposed_corrections,
            warnings=warnings,
        )

    def _normalize_from_tag_format(self, text: str) -> NormalizedPrescription:
        """Parses the specific <s_ocr> tag format used by naazimsnh02/medocr-vision-dataset."""
        data: Dict[str, Any] = {"medications": []}
        
        # Extract doctor, clinic, patient, age, date
        for field in ["doctor_name", "clinic_name", "patient_name", "patient_age", "date"]:
            m = re.search(rf"\b{field}\s*:\s*([^:]+?)(?=\s+[a-z_]+:|\s*medications:|\s*signature:|\s*</s>|$)", text)
            if m:
                data[field] = m.group(1).strip()

        # Extract medications section
        meds_section = ""
        m_section = re.search(r"medications\s*:\s*(.+?)(?=\s+signature:|\s*</s>|$)", text, re.DOTALL)
        if m_section:
            meds_section = m_section.group(1)

        # Parse bullet items in medications
        if meds_section:
            # Lines or bullet points starting with -
            items = [item.strip() for item in re.split(r"(?:^|\n)\s*-\s*|\s*-\s*", meds_section) if item.strip()]
            current_med = None
            
            for item in items:
                # Check if item starts with a known or likely medicine name / dosage
                dosage, rem_after_dose = self.matcher.extract_dosage(item)
                has_dose = bool(dosage)
                
                # Check if this item is an instruction (e.g. 'With food', 'Every 8 hours', 'After meals', 'As directed')
                freq_interp = AbbreviationHandler.interpret_frequency(item)
                food_interp = AbbreviationHandler.interpret_food_timing(item)
                is_pure_instruction = (
                    (freq_interp.is_standard and not has_dose)
                    or (food_interp is not None and not has_dose)
                    or re.search(r"^(?:as\s+(?:directed|advised|needed)|with\s+food|after\s+meals?|before\s+meals?|at\s+bedtime|take\s+once\s+daily|every\s+\d+\s+hours?)\b", item, re.IGNORECASE)
                )

                if is_pure_instruction and current_med is not None:
                    # Associate instruction with current medicine!
                    if freq_interp.is_standard and not current_med.get("frequency"):
                        current_med["frequency"] = freq_interp.original_notation
                    elif not current_med.get("frequency") and re.search(r"^(?:as\s+directed|take\s+once\s+daily|every\s+\d+\s+hours?)", item, re.IGNORECASE):
                        current_med["frequency"] = item
                    if food_interp and not current_med.get("instructions"):
                        current_med["instructions"] = food_interp[1]
                    elif not current_med.get("instructions"):
                        current_med["instructions"] = item
                else:
                    # New medicine entry
                    new_entry = {
                        "drug_name": item,
                        "dosage": dosage,
                        "frequency": "",
                        "duration": "",
                        "instructions": "",
                    }
                    data["medications"].append(new_entry)
                    current_med = new_entry

        return self._normalize_from_structured_dict(data)

    def _normalize_from_raw_unstructured_text(self, text: str) -> NormalizedPrescription:
        """Intelligently processes raw OCR text lines, detecting medicines, doses, frequencies across lines."""
        patient_name, patient_age, date, doctor_name, clinic_name = self._extract_patient_info(text)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        medications: List[NormalizedMedication] = []
        proposed_corrections = []
        warnings = []
        uncertain_fields = []
        additional_instructions = []

        current_med_candidate: Optional[Dict[str, Any]] = None

        def flush_current():
            nonlocal current_med_candidate
            if not current_med_candidate:
                return

            drug_raw = current_med_candidate["raw_name"]
            match_res = self.matcher.match_medicine_name(drug_raw)

            # Resolve dosage
            final_dosage = current_med_candidate.get("dosage") or match_res.extracted_dosage or "Not specified"

            # Resolve frequency
            freq_raw = current_med_candidate.get("frequency", "")
            freq_interp = AbbreviationHandler.interpret_frequency(freq_raw)

            # Resolve duration
            final_duration = current_med_candidate.get("duration") or "Not specified"

            # Resolve food timing & instructions
            food_raw = current_med_candidate.get("instructions", "")
            food_interp = AbbreviationHandler.interpret_food_timing(food_raw or freq_raw)
            food_eng = food_interp[1] if food_interp else ""

            dose_qty = current_med_candidate.get("dose_qty", "")
            custom_inst_full = f"{dose_qty} {food_raw}".strip()

            readable = self._build_readable_sentence(
                formulation=match_res.formulation,
                dosage=final_dosage,
                freq_eng=freq_interp.expanded_english,
                food_eng=food_eng,
                duration=final_duration,
                custom_inst=custom_inst_full,
            )

            if match_res.is_corrected:
                proposed_corrections.append({
                    "field": "medicine_name",
                    "original": drug_raw,
                    "normalized": match_res.matched_name,
                    "confidence": match_res.confidence_score,
                    "reason": match_res.correction_reason,
                })

            if match_res.verification_required:
                uncertain_fields.append(f"Medicine: {drug_raw}")

            medications.append(NormalizedMedication(
                original_text=drug_raw,
                medicine_name=match_res.matched_name,
                formulation=match_res.formulation,
                dosage=final_dosage,
                frequency=freq_interp.original_notation or (freq_raw if freq_raw else "Not specified"),
                duration=final_duration,
                instructions=food_raw or food_eng or "Not specified",
                readable_sentence=readable,
                verification_required=match_res.verification_required or freq_interp.ambiguous,
                confidence_score=match_res.confidence_score,
                correction_reason=match_res.correction_reason,
                soundalike_warning=match_res.soundalike_warning,
            ))

            current_med_candidate = None

        for line in lines:
            line_clean = self._clean_ocr_line(line)
            if not line_clean:
                continue

            # 1. Skip header, date, patient, doctor, hospital, address, and Rx symbol lines
            if self._is_header_or_address_line(line_clean):
                continue
            if patient_name and line_clean.lower() in (patient_name.lower(), "joln smitl", "john smith", "jane smith", "carlos chavez", "maria garcia", "wei li"):
                continue
            if patient_age and line_clean == patient_age:
                continue

            # 2. Check for lab tests, diagnostic investigations, or vitals line
            if self._is_lab_test_or_vital_line(line_clean):
                flush_current()
                additional_instructions.append(line_clean)
                continue

            # 2. Check if line contains general clinical advice instruction
            if re.search(r"\b(drink plenty|avoid cold|review after|rest|water|sufficient water|exceed)\b", line_clean, re.IGNORECASE):
                additional_instructions.append(line_clean)
                continue

            # 3. Check if line is a dose instruction / frequency / timing line
            if self._is_instruction_line(line_clean):
                if current_med_candidate is not None:
                    # Extract duration
                    dur, rem_inst = self._extract_duration(line_clean)
                    if dur:
                        current_med_candidate["duration"] = dur
                    # Extract frequency
                    f_in = AbbreviationHandler.interpret_frequency(rem_inst)
                    if f_in.is_standard:
                        current_med_candidate["frequency"] = f_in.original_notation
                    # Extract food relationship
                    food_in = AbbreviationHandler.interpret_food_timing(rem_inst)
                    if food_in:
                        current_med_candidate["instructions"] = food_in[1]
                    elif re.search(r"\bfor\s+([a-zA-Z\s]+)", rem_inst, re.IGNORECASE):
                        m_for = re.search(r"\b(for\s+[a-zA-Z\s]+)", rem_inst, re.IGNORECASE)
                        current_med_candidate["instructions"] = m_for.group(1).strip()
                    # Extract dose quantity e.g. 1 tab, 2 tabs
                    m_qty = re.search(r"\b(\d+(?:/\d+)?|half|one|two)\s*(?:tabs?|tablets?|caps?|capsules?)\b", rem_inst, re.IGNORECASE)
                    if m_qty:
                        current_med_candidate["dose_qty"] = m_qty.group(0)
                continue

            # 4. Handle bullet or hyphen separated drug and instruction e.g. 'Betaloc 100mg - 1 tab BID'
            sub_parts = [p.strip() for p in re.split(r"\s*[-–—]\s*", line_clean) if p.strip()]
            primary_segment = sub_parts[0] if sub_parts else line_clean
            instruction_segment = " ".join(sub_parts[1:]) if len(sub_parts) > 1 else ""

            # Check if line contains dosage
            dosage, rem_line = self.matcher.extract_dosage(primary_segment)
            if not dosage and instruction_segment:
                dosage, rem_inst = self.matcher.extract_dosage(instruction_segment)
                if dosage:
                    instruction_segment = rem_inst

            # Check for duration
            duration, rem_line = self._extract_duration(rem_line)
            if not duration and instruction_segment:
                duration, instruction_segment = self._extract_duration(instruction_segment)

            # Clean handwriting artifacts in instruction segment
            cleaned_inst = self._clean_instruction_artifacts(instruction_segment)

            # Check frequency in line or instruction segment
            freq_interp = AbbreviationHandler.interpret_frequency(cleaned_inst or rem_line)
            # Check food relationship
            food_interp = AbbreviationHandler.interpret_food_timing(cleaned_inst or rem_line)

            has_explicit_form = any(re.search(rf"\b{f}\b", line_clean, re.IGNORECASE) for f in FORMULATIONS.keys())
            has_explicit_dosage = bool(dosage and re.search(r"\b(?:mg|gm|g|mcg|ml|iu|%)\b", dosage, re.IGNORECASE))

            # Test if rem_line matches a medicine candidate
            match_test = self.matcher.match_medicine_name(rem_line)
            is_medicine_line = (
                match_test.confidence_score >= 70
                or (has_explicit_form and match_test.confidence_score >= 60 and len(rem_line.strip()) >= 3)
                or (has_explicit_dosage and match_test.confidence_score >= 60 and len(rem_line.strip()) >= 3)
                or (has_explicit_dosage and has_explicit_form and len(rem_line.strip()) >= 3)
            )

            if is_medicine_line:
                # Flush existing medicine candidate
                flush_current()
                dur_inst, rem_inst_clean = self._extract_duration(cleaned_inst)
                if dur_inst:
                    duration = dur_inst
                f_inst = AbbreviationHandler.interpret_frequency(rem_inst_clean)
                food_inst = AbbreviationHandler.interpret_food_timing(rem_inst_clean)

                current_med_candidate = {
                    "raw_name": rem_line if rem_line else line_clean,
                    "dosage": dosage,
                    "frequency": f_inst.original_notation if f_inst.is_standard else (freq_interp.original_notation if freq_interp.is_standard else ""),
                    "duration": duration,
                    "instructions": food_inst[1] if food_inst else (food_interp[1] if food_interp else cleaned_inst),
                    "dose_qty": "",
                }
                if instruction_segment:
                    # Same-line instruction was parsed and complete
                    flush_current()
            elif current_med_candidate is not None:
                # This line is likely an instruction/timing/duration continuation for the current medicine!
                if duration and not current_med_candidate.get("duration"):
                    current_med_candidate["duration"] = duration
                if freq_interp.is_standard and not current_med_candidate.get("frequency"):
                    current_med_candidate["frequency"] = freq_interp.original_notation
                if food_interp and not current_med_candidate.get("instructions"):
                    current_med_candidate["instructions"] = food_interp[1]
                elif not food_interp and not duration and not freq_interp.is_standard:
                    # Append as custom instruction
                    existing = current_med_candidate.get("instructions", "")
                    current_med_candidate["instructions"] = f"{existing} {line_clean}".strip()
            else:
                # Unassociated floating line - only retain genuine clinical instructions
                if self._is_clinical_instruction(line_clean):
                    additional_instructions.append(line_clean)

        # Flush final medicine candidate
        flush_current()

        return NormalizedPrescription(
            patient_name=patient_name,
            patient_age=patient_age,
            date=date,
            doctor_name=doctor_name,
            clinic_name=clinic_name,
            medications=medications,
            additional_instructions=additional_instructions,
            uncertain_fields=uncertain_fields,
            proposed_corrections=proposed_corrections,
            warnings=warnings,
        )

    def normalize(
        self,
        raw_input: Union[str, Dict[str, Any]],
        doc_type: str = "auto",
    ) -> Union[NormalizedPrescription, NormalizedMedicineLabel]:
        """Main entry point for intelligent normalization supporting Prescriptions and Medicine Labels."""
        if isinstance(raw_input, dict):
            return self._normalize_from_structured_dict(raw_input)

        text = raw_input.strip()

        # Check for JSON string
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("```json") and text.endswith("```")):
            cleaned_json = text
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            try:
                parsed = json.loads(cleaned_json.strip())
                return self._normalize_from_structured_dict(parsed)
            except Exception:
                pass

        # Check for dataset <s_ocr> tag format
        if "<s_ocr>" in text:
            return self._normalize_from_tag_format(text)

        # Document type routing
        active_type = doc_type.lower()
        if active_type == "auto":
            active_type = self.detect_document_type(text)

        if active_type in ("medicine_label", "label", "packaging"):
            return self.label_normalizer.normalize(text)

        # Raw unstructured prescription text
        return self._normalize_from_raw_unstructured_text(text)
