"""Abbreviation interpretation and contextual clinical instruction expansion."""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class FrequencyInterpretation:
    original_notation: str
    expanded_english: str
    is_standard: bool
    ambiguous: bool
    notes: str = ""


class AbbreviationHandler:
    """Interprets Latin and Indian prescription frequency/timing abbreviations."""

    # Precise mapping of unambiguous abbreviations to standard English expansions
    FREQUENCY_MAP: Dict[str, Tuple[str, str]] = {
        # Latin standard abbreviations
        "od": ("OD", "once daily"),
        "qd": ("QD", "once daily"),
        "once daily": ("OD", "once daily"),
        "bd": ("BD", "twice daily"),
        "bid": ("BID", "twice daily"),
        "twice daily": ("BD", "twice daily"),
        "tds": ("TDS", "three times daily"),
        "tid": ("TID", "three times daily"),
        "ttd": ("TID", "three times daily"),
        "td": ("TID", "three times daily"),
        "thrice daily": ("TDS", "three times daily"),
        "qid": ("QID", "four times daily"),
        "four times daily": ("QID", "four times daily"),
        "sos": ("SOS", "as needed for symptoms"),
        "prn": ("PRN", "as needed"),
        "as needed": ("PRN", "as needed"),
        "as needed for pain": ("PRN", "as needed for pain"),
        "as needed for fever": ("SOS", "as needed for fever"),
        "as directed": ("As directed", "as directed by the physician"),
        "as advised": ("As advised", "as advised by the physician"),
        "take once daily": ("OD", "take once daily"),
        "take twice daily": ("BD", "take twice daily"),
        "take three times daily": ("TDS", "take three times daily"),
        "ud": ("UD", "as directed"),
        "ut dict": ("UD", "as directed"),
        "hs": ("HS", "at bedtime"),
        "qhs": ("QHS", "at bedtime"),
        "at bedtime": ("HS", "at bedtime"),
        "stat": ("STAT", "immediately as a single dose"),
        "weekly": ("Weekly", "once weekly"),
        "q4h": ("q4h", "every 4 hours"),
        "every 4 hours": ("q4h", "every 4 hours"),
        "q6h": ("q6h", "every 6 hours"),
        "every 6 hours": ("q6h", "every 6 hours"),
        "q8h": ("q8h", "every 8 hours"),
        "every 8 hours": ("q8h", "every 8 hours"),
        "q12h": ("q12h", "every 12 hours"),
        "every 12 hours": ("q12h", "every 12 hours"),

        # Indian numeric timing notations (Morning - Afternoon - Night)
        "1-0-0": ("1-0-0", "once daily in the morning"),
        "0-1-0": ("0-1-0", "once daily in the afternoon"),
        "0-0-1": ("0-0-1", "once daily at bedtime"),
        "1-0-1": ("1-0-1", "twice daily (morning and night)"),
        "1-1-0": ("1-1-0", "twice daily (morning and afternoon)"),
        "0-1-1": ("0-1-1", "twice daily (afternoon and night)"),
        "1-1-1": ("1-1-1", "three times daily (morning, afternoon, and night)"),
        "1-1-1-1": ("1-1-1-1", "four times daily"),
        "1/2-0-1/2": ("1/2-0-1/2", "half tablet morning and half tablet night"),
        "1/2-0-0": ("1/2-0-0", "half tablet in the morning"),
    }

    # Food relationship abbreviations
    FOOD_MAP: Dict[str, Tuple[str, str]] = {
        "ac": ("AC", "before meals"),
        "ante cibum": ("AC", "before meals"),
        "before meals": ("AC", "before meals"),
        "before food": ("AC", "before food"),
        "pc": ("PC", "after meals"),
        "post cibum": ("PC", "after meals"),
        "after meals": ("PC", "after meals"),
        "af": ("AF", "after food"),
        "after food": ("AF", "after food"),
        "with food": ("With food", "with food"),
        "with meals": ("With meals", "with meals"),
        "bbf": ("BBF", "before breakfast"),
        "empty stomach": ("Empty stomach", "on an empty stomach"),
    }

    @classmethod
    def interpret_frequency(cls, notation: str) -> FrequencyInterpretation:
        """Parses frequency notation preserving original token and generating readable English."""
        if not notation or not notation.strip():
            return FrequencyInterpretation(
                original_notation="",
                expanded_english="Not specified",
                is_standard=False,
                ambiguous=False,
            )

        cleaned = notation.strip().lower()

        # Direct match
        if cleaned in cls.FREQUENCY_MAP:
            orig, eng = cls.FREQUENCY_MAP[cleaned]
            return FrequencyInterpretation(
                original_notation=orig,
                expanded_english=eng,
                is_standard=True,
                ambiguous=False,
            )

        # Regex search for numeric pattern like 1-0-1 or 1-1-1
        pattern = re.search(r"\b([012]|1/2)-([012]|1/2)-([012]|1/2)(?:-([012]|1/2))?\b", cleaned)
        if pattern:
            matched = pattern.group(0)
            if matched in cls.FREQUENCY_MAP:
                orig, eng = cls.FREQUENCY_MAP[matched]
                return FrequencyInterpretation(
                    original_notation=matched,
                    expanded_english=eng,
                    is_standard=True,
                    ambiguous=False,
                )

        # Check for abbreviation embedded in text
        for abbr, (orig, eng) in cls.FREQUENCY_MAP.items():
            if re.search(rf"\b{re.escape(abbr)}\b", cleaned):
                return FrequencyInterpretation(
                    original_notation=orig,
                    expanded_english=eng,
                    is_standard=True,
                    ambiguous=False,
                )

        # Non-standard or ambiguous
        return FrequencyInterpretation(
            original_notation=notation.strip(),
            expanded_english=notation.strip(),
            is_standard=False,
            ambiguous=True,
            notes="Non-standard frequency notation preserved verbatim.",
        )

    @classmethod
    def interpret_food_timing(cls, text: str) -> Optional[Tuple[str, str]]:
        """Identifies food relationship (before food, after meals, empty stomach, etc.)."""
        if not text:
            return None

        cleaned = text.strip().lower()
        for key, (orig, eng) in cls.FOOD_MAP.items():
            if re.search(rf"\b{re.escape(key)}\b", cleaned):
                return orig, eng
        return None
