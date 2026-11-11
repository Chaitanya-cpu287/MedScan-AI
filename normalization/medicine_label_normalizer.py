"""Intelligent Normalizer and Entity Reconstructor for Medicine Packaging & Box Labels."""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union

from normalization.pharmacopeia_db import PHARMACOPEIA, ALIAS_TO_STANDARD, FORMULATIONS
from normalization.medicine_matcher import MedicineMatcher

logger = logging.getLogger(__name__)


@dataclass
class ActiveIngredient:
    name: str
    standard_name: str
    strength: str
    is_verified: bool


@dataclass
class NormalizedMedicineLabel:
    product_name: str
    brand_name: str
    generic_name: str
    formulation: str
    active_ingredients: List[ActiveIngredient] = field(default_factory=list)
    dosage_and_administration: List[str] = field(default_factory=list)
    warnings_and_precautions: List[str] = field(default_factory=list)
    storage_instructions: List[str] = field(default_factory=list)
    regulatory_schedule: str = "Unspecified"
    manufacturing_info: Dict[str, str] = field(default_factory=dict)
    dosing_safety_alerts: List[str] = field(default_factory=list)
    review_required: bool = True
    review_note: str = ""

    def to_formatted_text(self) -> str:
        """Generates clean standardized plain text output specifically for medicine labels."""
        lines = []
        lines.append("DOCUMENT TYPE: MEDICINE PACKAGING & REGULATORY LABEL")
        lines.append(f"PRODUCT NAME: {self.product_name or 'Not specified'}")
        lines.append(f"FORMULATION: {self.formulation or 'Not specified'}")
        if self.regulatory_schedule and self.regulatory_schedule != "Unspecified":
            lines.append(f"REGULATORY SCHEDULE: {self.regulatory_schedule}")
        lines.append("")

        lines.append("ACTIVE INGREDIENTS & COMPOSITION")
        if self.active_ingredients:
            for idx, ing in enumerate(self.active_ingredients, 1):
                std_note = f" (Standard: {ing.standard_name})" if ing.standard_name and ing.standard_name != ing.name else ""
                lines.append(f"{idx}. {ing.name.upper()}{std_note}: {ing.strength}")
        else:
            lines.append("- Not clearly specified")
        lines.append("")

        lines.append("DOSAGE & ADMINISTRATION")
        if self.dosage_and_administration:
            for admin in self.dosage_and_administration:
                lines.append(f"- {admin}")
        else:
            lines.append("- As directed by the physician.")
        lines.append("")

        lines.append("WARNINGS & PRECAUTIONS")
        if self.warnings_and_precautions:
            for warn in self.warnings_and_precautions:
                lines.append(f"- {warn}")
        else:
            lines.append("- Standard precautions apply. Consult physician before use.")
        lines.append("")

        lines.append("STORAGE & HANDLING")
        if self.storage_instructions:
            for stor in self.storage_instructions:
                lines.append(f"- {stor}")
        else:
            lines.append("- Store in a cool, dry place. Protect from light and moisture.")
        lines.append("")

        if self.manufacturing_info:
            lines.append("MANUFACTURING & REGULATORY METADATA")
            for k, v in self.manufacturing_info.items():
                lines.append(f"- {k.replace('_', ' ').title()}: {v}")
            lines.append("")

        if self.dosing_safety_alerts:
            lines.append("DOSING SAFETY & CLINICAL NOTES")
            for alert in self.dosing_safety_alerts:
                lines.append(f"⚠️ {alert}")
            lines.append("")

        return "\n".join(lines).strip()

    def to_export_json(self) -> Dict[str, Any]:
        """Strict JSON deliverable schema for medicine packaging labels."""
        return {
            "document_type": "medicine_label",
            "product_name": self.product_name,
            "brand_name": self.brand_name,
            "generic_name": self.generic_name,
            "formulation": self.formulation,
            "regulatory_schedule": self.regulatory_schedule,
            "active_ingredients": [
                {
                    "name": ing.name,
                    "standard_name": ing.standard_name,
                    "strength": ing.strength,
                    "verified": ing.is_verified,
                }
                for ing in self.active_ingredients
            ],
            "administration": self.dosage_and_administration,
            "warnings": self.warnings_and_precautions,
            "storage": self.storage_instructions,
            "manufacturing_info": self.manufacturing_info,
            "dosing_safety_alerts": self.dosing_safety_alerts,
            "review_required": self.review_required,
            "review_note": self.review_note or "Verify extracted packaging specifications against physical packaging.",
        }


class MedicineLabelNormalizer:
    """Normalizes and reconstructs raw OCR text from medicine packaging, blister packs, and bottles."""

    def __init__(self, matcher: Optional[MedicineMatcher] = None):
        self.matcher = matcher or MedicineMatcher()

    def normalize(self, text: str) -> NormalizedMedicineLabel:
        """Parses medicine label text into structured clinical components."""
        text = text.replace("\u2103", "°C")
        text = re.sub(r"°+", "°", text)
        raw_lines = [l.strip() for l in text.splitlines() if l.strip()]

        product_name = ""
        brand_name = ""
        generic_name = ""
        formulation = "Tablet"
        active_ingredients: List[ActiveIngredient] = []
        administration: List[str] = []
        warnings: List[str] = []
        storage: List[str] = []
        mfg_info: Dict[str, str] = {}
        safety_alerts: List[str] = []

        # 1. Detect formulation
        for form_key, form_val in FORMULATIONS.items():
            if re.search(rf"\b{form_key}\b", text, re.IGNORECASE):
                formulation = {
                    "TAB.": "Tablet",
                    "CAP.": "Capsule",
                    "SYP.": "Syrup / Oral Liquid",
                    "INJ.": "Injection",
                    "OINT.": "Ointment / Topical",
                    "GEL": "Gel",
                    "CREAM": "Cream",
                    "DROPS": "Drops",
                    "SUSP.": "Oral Suspension",
                }.get(form_val, form_val)
                break

        # 2. Regulatory Schedule
        sched_match = re.search(r"\bSchedule\s+([HhGgXxKk1-9]+)\s*(?:Prescription\s*Drug)?\b", text, re.IGNORECASE)
        sched = f"Schedule {sched_match.group(1).upper()}" if sched_match else "Unspecified"

        # 3. Product & Brand Name detection (usually the first dominant title lines)
        for line in raw_lines[:3]:
            # Skip pure numbers or metadata
            if re.search(r"^(?:each|dosage|warning|store|mfg|exp|batch)\b", line, re.IGNORECASE):
                continue
            # Remove symbols
            clean_title = re.sub(r"^[^\w]+|[^\w]+$", "", line).strip()
            if len(clean_title) >= 3 and not product_name:
                product_name = clean_title
                break

        # 4. Active Ingredients & Composition Extraction
        # Look for 'Each tablet contains: ...', 'Composition: ...' or explicit ingredient lines
        comp_pattern = r"(?:(?:each\s+(?:film\s*coated\s*tablet|uncoated\s*tablet|tablet|capsule|ml|5\s*ml|dose)\s*contains?|composition)\s*[:\s]*)(.*?)(?=\s*(?:dosage|warning|store|caution|mfg|batch|keep|$))"
        comp_match = re.search(comp_pattern, text, re.IGNORECASE | re.DOTALL)
        comp_text = comp_match.group(1) if comp_match else text

        # Match ingredient patterns: e.g. 'Paracetamol IP 650 mg', 'Amoxicillin Trihydrate IP equivalent to Amoxicillin 500 mg'
        ing_pattern = r"([A-Za-z\s\(\)\-]+?(?:IP|BP|USP)?)\s*(?:equivalent\s+to\s+([A-Za-z\s]+))?\s*(\d+(?:\.\d+)?\s*(?:mg|gm|g|mcg|ml|iu|%))\b"
        matches = re.finditer(ing_pattern, comp_text, re.IGNORECASE)
        found_any = False
        for m in matches:
            found_any = True
            raw_ing = m.group(1).strip()
            # Clean leading 'Each tablet contains' remnants
            raw_ing = re.sub(r"^(?:contains|each\s+tablet\s+contains|each\s+capsule\s+contains|each\s+ml\s+contains)\s*[:\s]*", "", raw_ing, flags=re.I).strip()
            # Clean IP/BP suffix for generic matching
            core_name = re.sub(r"\b(IP|BP|USP)\b", "", raw_ing, flags=re.I).strip()
            # Strip chemical salt suffixes for standard generic matching
            salt_pattern = r"\b(?:dihydrochloride|hydrochloride|sodium|potassium|trihydrate|monohydrate|maleate|tartrate|fumarate|mesylate|succinate|hcl|na|k)\b"
            base_drug = re.sub(salt_pattern, "", core_name, flags=re.I).strip()
            equiv = m.group(2).strip() if m.group(2) else ""
            strength = m.group(3).strip()

            target_to_match = equiv if equiv else (base_drug if len(base_drug) >= 3 else core_name)
            match_res = self.matcher.match_medicine_name(target_to_match)
            std_name = match_res.matched_name if match_res.confidence_score >= 70 else target_to_match

            active_ingredients.append(ActiveIngredient(
                name=raw_ing,
                standard_name=std_name,
                strength=strength,
                is_verified=match_res.confidence_score >= 70,
            ))
            if not generic_name:
                generic_name = std_name

        # Fallback ingredient extraction if 'contains' pattern missed
        if not found_any:
            for line in raw_lines:
                dosage_found, rem = self.matcher.extract_dosage(line)
                if dosage_found and len(rem) >= 3:
                    m_test = self.matcher.match_medicine_name(rem)
                    if m_test.confidence_score >= 70:
                        active_ingredients.append(ActiveIngredient(
                            name=rem,
                            standard_name=m_test.matched_name,
                            strength=dosage_found,
                            is_verified=True,
                        ))
                        if not generic_name:
                            generic_name = m_test.matched_name

        # If product name wasn't found, derive from generic name
        if not product_name and generic_name:
            product_name = f"{generic_name} {formulation}"

        # 5. Dosage & Administration Instructions
        admin_patterns = [
            r"Dosage\s*:\s*([^.\n]+)",
            r"As directed by the physician",
            r"(?:to be )?swallowed whole(?: and not to be (?:chewed|crushed|opened))?",
            r"Do not (?:crush|chew|open)",
            r"Shake well before use",
            r"Swallow with water",
            r"For external use only",
            r"Apply (?:thinly )?to the affected area",
            r"Dissolve in water before taking",
            r"Not for injection",
            r"For ophthalmic use only",
        ]
        for p in admin_patterns:
            for match in re.finditer(p, text, re.IGNORECASE):
                val = match.group(0).strip()
                val = re.sub(r"^(?:dosage\s*[:\s]*)", "", val, flags=re.I).strip()
                if val and val.lower() not in [a.lower() for a in administration]:
                    administration.append(val[0].upper() + val[1:])

        # 6. Warnings & Precautions
        warning_patterns = [
            r"Schedule\s+[HhGgXxKk1-9]+\s*Prescription\s*Drug[^\n.]*",
            r"Warning\s*:\s*To be sold by retail on the prescription of a Registered Medical Practitioner only",
            r"Caution\s*:\s*Not to be sold by retail without the prescription of a Registered Medical Practitioner",
            r"Warning\s*:\s*Taking more than daily dose may cause serious liver damage(?: or allergic reactions)?",
            r"Keep out of reach of children",
            r"Keep all medicines out of reach of children",
            r"Do not exceed the recommended (?:dose|dosage)",
            r"Not recommended for children under \d+ years?",
            r"Contraindicated in patients with [^\n.]+",
            r"Avoid alcohol while taking this medication",
        ]
        for p in warning_patterns:
            for match in re.finditer(p, text, re.IGNORECASE):
                val = match.group(0).strip()
                if val and val.lower() not in [w.lower() for w in warnings]:
                    warnings.append(val[0].upper() + val[1:])

        # 7. Storage Instructions
        storage_patterns = [
            r"Store in a cool(?: and|,) dry place[^\n.]*",
            r"Store protected from light(?: and moisture)?",
            r"Protect from light and moisture",
            r"Store below \d+\s*°\s*C",
            r"Do not freeze",
            r"Keep container tightly closed",
            r"Store at room temperature",
        ]
        for p in storage_patterns:
            for match in re.finditer(p, text, re.IGNORECASE):
                val = match.group(0).strip()
                if val and val.lower() not in [s.lower() for s in storage]:
                    storage.append(val[0].upper() + val[1:])

        # 8. Manufacturing / Batch Metadata
        batch_m = re.search(r"\b(?:batch\s*(?:no\.?|num\.?)|b\.?\s*no\.?)\s*[:=-]?\s*([A-Za-z0-9\-]+)\b", text, re.IGNORECASE)
        if batch_m:
            mfg_info["batch_number"] = batch_m.group(1).strip()

        lic_m = re.search(r"\b(?:mfg\.?\s*lic\.?\s*(?:no\.?|num\.?)|licence\s*no\.?)\s*[:=-]?\s*([A-Za-z0-9\-/\.]+)\b", text, re.IGNORECASE)
        if lic_m:
            mfg_info["manufacturing_license"] = lic_m.group(1).strip()

        exp_m = re.search(r"\b(?:exp\.?\s*date|expiry\s*date|exp\.?\s*dt\.?|use\s*before)\s*[:=-]?\s*([0-9A-Za-z/\-\.]+)\b", text, re.IGNORECASE)
        if exp_m:
            mfg_info["expiry_date"] = exp_m.group(1).strip()

        mfr_m = re.search(r"\b(?:mfd\.?\s*by|manufactured\s*by|marketed\s*by)\s*[:=-]?\s*([A-Za-z0-9\s\.\,\-]+?)(?:\n|mfg|lic|batch|$)", text, re.IGNORECASE)
        if mfr_m:
            mfg_info["manufacturer"] = mfr_m.group(1).strip()

        # 9. Dosing Safety Assessment Alerts
        if "paracetamol" in generic_name.lower():
            safety_alerts.append("Hepatotoxicity Risk: Ensure total Paracetamol intake from all sources does not exceed 4000 mg/day (4g).")
        if "amoxicillin" in generic_name.lower():
            safety_alerts.append("Hypersensitivity Precaution: Check for penicillin allergy before administration.")
        if "schedule h" in sched.lower() and not warnings:
            warnings.append("Schedule H Prescription Drug: Requires registered medical practitioner's prescription.")

        return NormalizedMedicineLabel(
            product_name=product_name,
            brand_name=brand_name or product_name,
            generic_name=generic_name or product_name,
            formulation=formulation,
            active_ingredients=active_ingredients,
            dosage_and_administration=administration,
            warnings_and_precautions=warnings,
            storage_instructions=storage,
            regulatory_schedule=sched,
            manufacturing_info=mfg_info,
            dosing_safety_alerts=safety_alerts,
            review_required=True,
            review_note="Verified packaging specification extracted via intelligent clinical entity normalization.",
        )
