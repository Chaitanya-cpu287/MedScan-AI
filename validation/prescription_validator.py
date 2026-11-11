"""Validation, safety audit logging, discrepancy detection, and consistency checking."""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from normalization.text_normalizer import NormalizedPrescription, NormalizedMedication
from normalization.medicine_label_normalizer import NormalizedMedicineLabel

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    severity: str  # "CRITICAL", "WARNING", "INFO"
    category: str  # "OMISSION", "HALLUCINATION", "DOSAGE_MISMATCH", "FREQUENCY_MISMATCH", "UNCERTAINTY"
    message: str
    medicine_name: Optional[str] = None
    original_value: Optional[str] = None
    normalized_value: Optional[str] = None


@dataclass
class CorrectionAuditEntry:
    field: str
    original_ocr_text: str
    corrected_text: str
    reason: str
    verification_status: str  # "VERIFIED", "REQUIRES_REVIEW", "UNCHANGED"
    confidence_score: float


@dataclass
class ValidationReport:
    total_medications_found: int
    all_medications_retained: bool
    omissions_detected: List[str]
    potential_hallucinations: List[str]
    dosage_discrepancies: List[Dict[str, Any]]
    frequency_discrepancies: List[Dict[str, Any]]
    duration_discrepancies: List[Dict[str, Any]]
    uncertain_associations: List[str]
    audit_log: List[CorrectionAuditEntry]
    issues: List[ValidationIssue]
    passed_safety_checks: bool
    human_review_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_medications_found": self.total_medications_found,
            "all_medications_retained": self.all_medications_retained,
            "passed_safety_checks": self.passed_safety_checks,
            "human_review_required": self.human_review_required,
            "omissions_detected": self.omissions_detected,
            "potential_hallucinations": self.potential_hallucinations,
            "dosage_discrepancies": self.dosage_discrepancies,
            "frequency_discrepancies": self.frequency_discrepancies,
            "duration_discrepancies": self.duration_discrepancies,
            "uncertain_associations": self.uncertain_associations,
            "audit_log": [
                {
                    "field": entry.field,
                    "original_ocr_text": entry.original_ocr_text,
                    "corrected_text": entry.corrected_text,
                    "reason": entry.reason,
                    "verification_status": entry.verification_status,
                    "confidence_score": entry.confidence_score,
                }
                for entry in self.audit_log
            ],
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "medicine_name": issue.medicine_name,
                    "original_value": issue.original_value,
                    "normalized_value": issue.normalized_value,
                }
                for issue in self.issues
            ],
        }


class PrescriptionValidator:
    """Rigorous clinical validator verifying preservation and flagging hallucinations or errors."""

    @classmethod
    def validate(
        cls,
        raw_ocr_text: str,
        normalized: Any,
    ) -> ValidationReport:
        if isinstance(normalized, NormalizedMedicineLabel):
            return cls.validate_medicine_label(raw_ocr_text, normalized)

        issues: List[ValidationIssue] = []
        audit_log: List[CorrectionAuditEntry] = []
        raw_lower = raw_ocr_text.lower()

        omissions: List[str] = []
        hallucinations: List[str] = []
        dosage_discrepancies: List[Dict[str, Any]] = []
        frequency_discrepancies: List[Dict[str, Any]] = []
        duration_discrepancies: List[Dict[str, Any]] = []
        uncertain_associations: List[str] = []

        # 1. Audit log and checks for each normalized medication
        for med in normalized.medications:
            orig_txt = med.original_text.strip()
            norm_name = med.medicine_name.strip()

            # Status determination
            if med.verification_required:
                status = "REQUIRES_REVIEW"
            elif norm_name.lower() != orig_txt.lower():
                status = "VERIFIED" if med.confidence_score >= 90 else "REQUIRES_REVIEW"
            else:
                status = "UNCHANGED"

            audit_log.append(CorrectionAuditEntry(
                field="medicine_name",
                original_ocr_text=orig_txt,
                corrected_text=norm_name,
                reason=med.correction_reason,
                verification_status=status,
                confidence_score=med.confidence_score,
            ))

            # Check: Hallucination check
            # Verify that either the original token or a close substring exists in raw OCR text
            first_3_chars = norm_name[:3].lower()
            if orig_txt.lower() not in raw_lower and first_3_chars not in raw_lower and med.confidence_score < 70:
                hallucinations.append(norm_name)
                issues.append(ValidationIssue(
                    severity="CRITICAL",
                    category="HALLUCINATION",
                    message=f"Possible hallucinated medicine: '{norm_name}' has no trace in raw OCR text.",
                    medicine_name=norm_name,
                ))

            # Check: Dosage preservation
            if med.dosage and med.dosage != "Not specified":
                # Extract digits
                dosage_nums = re.findall(r"\d+", med.dosage)
                if dosage_nums:
                    for num in dosage_nums:
                        if num not in raw_ocr_text:
                            dosage_discrepancies.append({
                                "medicine": norm_name,
                                "normalized_dosage": med.dosage,
                                "error": f"Dose number '{num}' not found in raw OCR text.",
                            })
                            issues.append(ValidationIssue(
                                severity="CRITICAL",
                                category="DOSAGE_MISMATCH",
                                message=f"Dose number '{num}' in '{med.dosage}' for {norm_name} does not appear in raw OCR text.",
                                medicine_name=norm_name,
                                original_value="Not found in OCR",
                                normalized_value=med.dosage,
                            ))

            # Check: Frequency preservation
            if med.frequency and med.frequency != "Not specified":
                freq_token = med.frequency.lower()
                # Check if frequency token appears in raw text or was derived
                if freq_token not in raw_lower and len(freq_token) <= 4:
                    frequency_discrepancies.append({
                        "medicine": norm_name,
                        "frequency": med.frequency,
                        "error": "Frequency token not explicitly recognized in raw OCR.",
                    })

            # Check: Uncertain associations
            if med.verification_required or med.readable_sentence.startswith("Unclear"):
                uncertain_associations.append(norm_name)
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="UNCERTAINTY",
                    message=f"Medicine '{norm_name}' requires clinician verification ({med.correction_reason}).",
                    medicine_name=norm_name,
                ))

            # Soundalike warning
            if med.soundalike_warning:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="SOUNDALIKE",
                    message=med.soundalike_warning,
                    medicine_name=norm_name,
                ))

        # 2. Check for Potential Omissions
        # Look for dosage patterns in raw OCR text that are not covered in any medication
        all_raw_dosages = re.findall(r"\b(\d+\s*(?:mg|gm|mcg|ml))\b", raw_ocr_text, re.IGNORECASE)
        normalized_dosages = [m.dosage.lower().replace(" ", "") for m in normalized.medications if m.dosage]
        
        for raw_dose in all_raw_dosages:
            cleaned_raw_dose = raw_dose.lower().replace(" ", "")
            if not any(cleaned_raw_dose in nd for nd in normalized_dosages):
                omissions.append(f"Unmapped dosage token: '{raw_dose}'")
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="OMISSION",
                    message=f"Possible omitted medicine with dosage '{raw_dose}' present in OCR text.",
                    original_value=raw_dose,
                ))

        all_retained = len(omissions) == 0
        has_critical = any(i.severity == "CRITICAL" for i in issues)
        requires_review = has_critical or len(uncertain_associations) > 0 or len(issues) > 0

        return ValidationReport(
            total_medications_found=len(normalized.medications),
            all_medications_retained=all_retained,
            omissions_detected=omissions,
            potential_hallucinations=hallucinations,
            dosage_discrepancies=dosage_discrepancies,
            frequency_discrepancies=frequency_discrepancies,
            duration_discrepancies=duration_discrepancies,
            uncertain_associations=uncertain_associations,
            audit_log=audit_log,
            issues=issues,
            passed_safety_checks=not has_critical,
            human_review_required=requires_review,
        )

    @classmethod
    def validate_medicine_label(
        cls,
        raw_ocr_text: str,
        label: NormalizedMedicineLabel,
    ) -> ValidationReport:
        """Validates medicine packaging label extraction against raw OCR text."""
        issues: List[ValidationIssue] = []
        audit_log: List[CorrectionAuditEntry] = []
        omissions: List[str] = []
        hallucinations: List[str] = []

        # 1. Audit active ingredients
        for ing in label.active_ingredients:
            audit_log.append(CorrectionAuditEntry(
                field="active_ingredient",
                original_ocr_text=ing.name,
                corrected_text=f"{ing.standard_name} ({ing.strength})",
                reason="Pharmacopeial generic verification",
                verification_status="VERIFIED" if ing.is_verified else "REQUIRES_REVIEW",
                confidence_score=95.0 if ing.is_verified else 70.0,
            ))

        # 2. Check for missing active ingredient
        if not label.active_ingredients:
            issues.append(ValidationIssue(
                severity="CRITICAL",
                category="OMISSION",
                message="No active pharmaceutical ingredient or strength identified from medicine label.",
            ))

        # 3. Check for Schedule H without prescription warning
        if "schedule h" in label.regulatory_schedule.lower():
            has_sched_warn = any("prescription" in w.lower() for w in label.warnings_and_precautions)
            if not has_sched_warn:
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="REGULATORY",
                    message="Schedule H medication label detected; statutory retail warning should be confirmed.",
                ))

        # 4. Check for storage instructions
        if not label.storage_instructions:
            issues.append(ValidationIssue(
                severity="INFO",
                category="STORAGE",
                message="No explicit storage conditions extracted; standard temperature precautions advised.",
            ))

        # 5. Check for dosage discrepancies in raw text vs label
        raw_dosages = re.findall(r"\b(\d+\s*(?:mg|gm|mcg|ml))\b", raw_ocr_text, re.IGNORECASE)
        label_dosages = [ing.strength.lower().replace(" ", "") for ing in label.active_ingredients if ing.strength]
        for rd in raw_dosages:
            cleaned_rd = rd.lower().replace(" ", "")
            if not any(cleaned_rd in ld for ld in label_dosages):
                omissions.append(f"Unmapped dosage token: '{rd}'")
                issues.append(ValidationIssue(
                    severity="WARNING",
                    category="OMISSION",
                    message=f"Possible dosage specification '{rd}' in raw OCR not attached to an active ingredient.",
                    original_value=rd,
                ))

        has_critical = any(i.severity == "CRITICAL" for i in issues)
        requires_review = has_critical or len(issues) > 0 or not all(ing.is_verified for ing in label.active_ingredients)

        return ValidationReport(
            total_medications_found=len(label.active_ingredients),
            all_medications_retained=len(omissions) == 0,
            omissions_detected=omissions,
            potential_hallucinations=hallucinations,
            dosage_discrepancies=[],
            frequency_discrepancies=[],
            duration_discrepancies=[],
            uncertain_associations=[],
            audit_log=audit_log,
            issues=issues,
            passed_safety_checks=not has_critical,
            human_review_required=requires_review,
        )
