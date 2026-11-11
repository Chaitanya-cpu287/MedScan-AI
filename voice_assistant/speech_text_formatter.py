"""
Speech text formatter for MediScan AI.
Transforms structured medical records (Prescriptions and Packaging Labels)
into natural, conversational spoken scripts for clinical voice synthesis.
"""

from typing import Dict, List, Optional
from normalization.text_normalizer import NormalizedPrescription
from normalization.medicine_label_normalizer import NormalizedMedicineLabel
from localization.i18n import t


class SpeechTextFormatter:
    """
    Constructs natural conversational spoken text from clinical records.
    Designed to translate accurately into all 8 languages without clinical distortion.
    """

    @staticmethod
    def format_prescription_summary(p: NormalizedPrescription) -> str:
        """
        Builds a comprehensive spoken prescription summary script.
        """
        parts: List[str] = []

        # Intro
        patient_str = p.patient_name if p.patient_name else "Not specified"
        parts.append(f"Medical Prescription Summary for patient {patient_str}.")

        if p.date:
            parts.append(f"Prescription date: {p.date}.")

        # Medications count
        med_count = len(p.medications)
        if med_count == 0:
            parts.append("No active medications identified.")
        elif med_count == 1:
            parts.append("There is one prescribed medication.")
        else:
            parts.append(f"There are {med_count} prescribed medications.")

        # Medication details
        for idx, med in enumerate(p.medications, 1):
            med_name = med.medicine_name.title() if med.medicine_name else "Medication"
            dosage_str = f" {med.dosage}" if med.dosage and med.dosage != "Not specified" else ""
            form_str = f" {med.formulation.title()}" if med.formulation else ""

            med_line = f"Item {idx}: {med_name}{form_str}{dosage_str}."

            sentence = med.readable_sentence if med.readable_sentence else ""
            if sentence:
                med_line += f" Directions: {sentence}"
            elif med.instructions:
                med_line += f" Directions: {med.instructions}."
            else:
                med_line += " Directions: As directed by the physician."

            parts.append(med_line)

        # Additional instructions / doctor
        if p.additional_instructions:
            advice = ". ".join(p.additional_instructions)
            parts.append(f"Additional doctor instructions: {advice}.")

        if p.doctor_name:
            parts.append(f"Prescribed by Dr. {p.doctor_name}.")

        parts.append("Please verify all dosages with your pharmacist or healthcare provider.")
        return " ".join(parts)

    @staticmethod
    def format_prescription_field_scripts(p: NormalizedPrescription) -> Dict[str, str]:
        """
        Builds individual spoken scripts for field-wise speech playback.
        """
        scripts: Dict[str, str] = {}

        # 1. Patient & Doctor
        p_name = p.patient_name if p.patient_name else "Not specified"
        d_name = f"Dr. {p.doctor_name}" if p.doctor_name else "Not specified"
        date_str = p.date if p.date else "Not specified"
        scripts["Patient & Clinician Info"] = (
            f"Patient Name: {p_name}. Prescription Date: {date_str}. Prescribing Doctor: {d_name}."
        )

        # 2. Individual medications
        for idx, med in enumerate(p.medications, 1):
            name = med.medicine_name.title() if med.medicine_name else f"Medicine {idx}"
            dosage = f" {med.dosage}" if med.dosage and med.dosage != "Not specified" else ""
            form = f" {med.formulation.title()}" if med.formulation else ""
            instr = med.readable_sentence if med.readable_sentence else (med.instructions if med.instructions else "As directed by physician.")
            scripts[f"Medication {idx}: {name}"] = (
                f"Medication {idx}: {name}{form}{dosage}. Directions: {instr}."
            )

        # 3. Additional Instructions
        if p.additional_instructions:
            scripts["Additional Clinical Notes"] = (
                "Additional clinical instructions: " + ". ".join(p.additional_instructions)
            )

        return scripts

    @staticmethod
    def format_medicine_label_summary(lbl: NormalizedMedicineLabel) -> str:
        """
        Builds a comprehensive spoken medicine packaging summary script.
        """
        parts: List[str] = ["Medicine Packaging and Label Summary."]

        # Active Ingredients
        if lbl.active_ingredients:
            ing_strs = [
                f"{ing.name.title()} {ing.strength or ''}".strip()
                for ing in lbl.active_ingredients
            ]
            parts.append(f"Active Pharmaceutical Ingredients: {', '.join(ing_strs)}.")
        else:
            parts.append("Active ingredients: Not clearly specified on the label.")

        # Dosage & Administration
        if lbl.dosage_and_administration:
            admin_str = ". ".join(lbl.dosage_and_administration)
            parts.append(f"Dosage and Administration: {admin_str}.")
        else:
            parts.append("Dosage: As directed by the physician.")

        # Storage
        if lbl.storage_instructions:
            storage_str = ". ".join(lbl.storage_instructions)
            parts.append(f"Storage instructions: {storage_str}.")
        else:
            parts.append("Storage: Store in a cool and dry place.")

        # Warnings & Precautions
        if lbl.warnings_and_precautions:
            warn_str = ". ".join(lbl.warnings_and_precautions)
            parts.append(f"Safety warnings: {warn_str}.")

        # Dosing Alerts
        if lbl.dosing_safety_alerts:
            alert_str = ". ".join(lbl.dosing_safety_alerts)
            parts.append(f"Caution: {alert_str}.")

        parts.append("Keep out of reach of children. Consult your doctor if symptoms persist.")
        return " ".join(parts)

    @staticmethod
    def format_medicine_label_field_scripts(lbl: NormalizedMedicineLabel) -> Dict[str, str]:
        """
        Builds individual spoken scripts for field-wise packaging speech playback.
        """
        scripts: Dict[str, str] = {}

        if lbl.active_ingredients:
            ing_strs = [
                f"{ing.name.title()} {ing.strength or ''}".strip()
                for ing in lbl.active_ingredients
            ]
            scripts["Active Ingredients"] = f"Active Ingredients: {', '.join(ing_strs)}."

        if lbl.dosage_and_administration:
            scripts["Dosage & Usage"] = (
                "Dosage and administration: " + ". ".join(lbl.dosage_and_administration)
            )

        if lbl.storage_instructions:
            scripts["Storage Instructions"] = (
                "Storage and handling: " + ". ".join(lbl.storage_instructions)
            )

        if lbl.warnings_and_precautions:
            scripts["Safety Warnings"] = (
                "Safety warnings and statutory precautions: " + ". ".join(lbl.warnings_and_precautions)
            )

        return scripts

    @staticmethod
    def format_navigation_guide(lang: str = "en") -> str:
        """
        Returns spoken navigation guide for user onboarding.
        """
        return t("voice_guide_script", lang=lang)
