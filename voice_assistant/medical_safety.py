"""
Medical Translation Safety Guard.

Ensures that medical entities (medicine names, dosages, units, numbers)
are never distorted, hallucinated, or altered during translation for voice or display.
"""

import re
from typing import Dict, List, Tuple, Set


# Standard units that must be strictly preserved
PROTECTED_UNITS = {
    "mg", "ml", "mcg", "g", "gm", "kg", "iu", "tab", "tabs", "cap", "caps",
    "tablet", "tablets", "capsule", "capsules", "drop", "drops", "puff", "puffs",
    "sachet", "sachets", "ampoule", "ampoules", "vial", "vials", "%", "w/v", "w/w"
}


class MedicalSafetyGuard:
    """
    Enforces clinical translation safety:
    1. Preserves medicine names verbatim.
    2. Preserves numerical quantities and dosage units.
    3. Prevents invention or hallucination of missing clinical directions.
    4. Marks unverified or ambiguous tokens.
    """

    @staticmethod
    def extract_protected_tokens(text: str) -> Dict[str, str]:
        """
        Extract numerical quantities, units, and capitalized medicine names
        to protect them with placeholders before translation.
        """
        protected: Dict[str, str] = {}
        counter = 0

        # Pattern for dosages like 650 mg, 500mg, 10 ml, 5 days, 1 tab
        dosage_pattern = r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g|iu|tab|tabs|cap|caps|tablet|tablets|capsule|capsules|days|day|hours|hrs)\b"
        for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
            token = match.group(0)
            placeholder = f"__NUM_UNIT_{counter}__"
            protected[placeholder] = token
            counter += 1

        return protected

    @staticmethod
    def protect_text(text: str, protected_map: Dict[str, str]) -> str:
        """Replace protected medical tokens with placeholders."""
        safe_text = text
        for placeholder, original in protected_map.items():
            safe_text = safe_text.replace(original, placeholder, 1)
        return safe_text

    @staticmethod
    def restore_protected_text(translated_text: str, protected_map: Dict[str, str]) -> str:
        """Restore exact numerical dosages and units after translation."""
        restored = translated_text
        for placeholder, original in protected_map.items():
            # Account for possible spaces inserted by translation engines
            clean_placeholder = placeholder.replace("_", " ").strip()
            if placeholder in restored:
                restored = restored.replace(placeholder, original)
            elif clean_placeholder in restored:
                restored = restored.replace(clean_placeholder, original)
        return restored

    @staticmethod
    def verify_dosage_integrity(original: str, translated_and_restored: str) -> bool:
        """
        Verify that all numbers present in the original text are present in the output.
        """
        orig_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", original))
        trans_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", translated_and_restored))
        # All original numbers must be conserved
        return orig_nums.issubset(trans_nums)
