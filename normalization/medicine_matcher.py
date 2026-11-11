"""Contextual medicine name matching, fuzzy resolution, and ambiguity detection."""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from rapidfuzz import fuzz, process

from normalization.pharmacopeia_db import (
    PHARMACOPEIA,
    ALIAS_TO_STANDARD,
    FORMULATIONS,
    MedicineInfo,
)
from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class MedicineMatchResult:
    original_token: str
    matched_name: str
    formulation: str
    extracted_dosage: str
    confidence_score: float
    is_corrected: bool
    verification_required: bool
    ambiguous: bool
    correction_reason: str
    soundalike_warning: Optional[str] = None


class MedicineMatcher:
    """Performs contextual medicine matching against pharmacopeia with strict safety guards."""

    def __init__(self, fuzzy_threshold: Optional[int] = None):
        self.threshold = fuzzy_threshold or CONFIG.normalization.fuzzy_match_threshold
        # Precompile lookup candidate list for rapidfuzz
        self.all_candidates = list(ALIAS_TO_STANDARD.keys())

    def extract_formulation(self, text: str) -> Tuple[str, str]:
        """Extracts and strips formulation prefix (TAB., CAP., SYP., etc.) from text."""
        cleaned = text.strip()
        found_form = "TAB."  # Default assumption for Indian oral solid prescriptions if unspecified

        # Check for formulation patterns at word boundaries
        words = cleaned.split()
        remaining_words = []
        for w in words:
            w_lower = re.sub(r"[^a-zA-Z]", "", w.lower())
            if w_lower in FORMULATIONS:
                found_form = FORMULATIONS[w_lower]
            else:
                remaining_words.append(w)

        return found_form, " ".join(remaining_words).strip()

    def extract_dosage(self, text: str) -> Tuple[str, str]:
        """Extracts dosage expressions (e.g., 650 mg, 500mg, Lisinopril250mg, 5Dmg, 5ng, or bare numbers)."""
        # 1. Match number with explicit unit (mg, gm, g, mcg, ml, iu, %, ng), including OCR leet-speak like 5Dmg, 5omg
        pattern_unit = r"(\d+(?:[oODdQ\.]\d+)?|[1-9][oODdQ])\s*(mg|gm|g|mcg|ml|iu|%|ng|tablets?|caps?)\b"
        match = re.search(pattern_unit, text, re.IGNORECASE)
        if match:
            raw_num = match.group(1)
            raw_unit = match.group(2).upper()
            # De-noise digits e.g. 5D -> 50, 4o -> 40
            clean_num = re.sub(r"[oODdQ]", "0", raw_num)
            # Correct OCR 'NG' (nanograms in oral solid) to 'MG'
            if raw_unit == "NG":
                raw_unit = "MG"
            dosage = f"{clean_num} {raw_unit}".strip()
            remaining = (text[:match.start()] + " " + text[match.end():]).strip()
            remaining = re.sub(r"\s+", " ", remaining)
            return dosage, remaining

        # 2. Match bare dosage numbers (e.g., 'PCM 650' or 'Amox 500')
        # Safety guard: Never extract bare numbers from address lines, metadata, or non-medications
        address_or_meta = r"\b(street|st\b|road|rd\b|ave\b|avenue|lane|flat|apt|apartment|floor|block|sector|house|building|bldg|room|ward|bed|plot|near|opp|behind|pin|zip|phone|tel|mobile|age|date|datc|dt|dateo|datco|dr|doctor|hospital|clinic|centre|center|patient|name|rx)\b"
        if re.search(address_or_meta, text, re.IGNORECASE):
            return "", text

        pattern_bare = r"(\d{1,4}(?:\.\d+)?)\s*$"
        match_bare = re.search(pattern_bare, text.strip())
        if match_bare:
            num = match_bare.group(1)
            remaining = text.strip()[:match_bare.start()].strip()
            remaining = re.sub(r"[\s\-_]+$", "", remaining).strip()
            # Only accept bare number if remaining candidate is not a metadata term and looks like a medicine name (at least 2 letters)
            if re.match(r"^(?:date|dateo|datc|dt|age|name|patient|rx|sig|dr|doctor)$", remaining, re.IGNORECASE):
                return "", text
            if re.match(r"^[A-Za-z\s\.\-_]{2,}$", remaining):
                dosage = f"{num} MG"
                return dosage, remaining

        return "", text

    def match_medicine_name(self, raw_name: str) -> MedicineMatchResult:
        """Resolves a raw medicine name token using exact aliases and contextual fuzzy matching."""
        if not raw_name or not raw_name.strip():
            return MedicineMatchResult(
                original_token="",
                matched_name="Unclear - Requires verification",
                formulation="TAB.",
                extracted_dosage="",
                confidence_score=0.0,
                is_corrected=False,
                verification_required=True,
                ambiguous=True,
                correction_reason="Empty or unreadable medicine name.",
            )

        # Step 1: Extract formulation
        formulation, stripped_form = self.extract_formulation(raw_name)

        # Step 2: Extract embedded dosage
        dosage, cleaned_token = self.extract_dosage(stripped_form)
        cleaned_token = re.sub(r"[^\w\s-]", "", cleaned_token).strip().lower()

        if not cleaned_token:
            return MedicineMatchResult(
                original_token=raw_name,
                matched_name=raw_name,
                formulation=formulation,
                extracted_dosage=dosage,
                confidence_score=0.0,
                is_corrected=False,
                verification_required=True,
                ambiguous=True,
                correction_reason="Could not isolate medicine name after dosage/formulation stripping.",
            )

        # Step 3: Exact lookup (direct alias or standard name)
        if cleaned_token in ALIAS_TO_STANDARD:
            std_name = ALIAS_TO_STANDARD[cleaned_token]
            is_corrected = std_name.lower() != raw_name.strip().lower()
            return MedicineMatchResult(
                original_token=raw_name,
                matched_name=std_name,
                formulation=formulation,
                extracted_dosage=dosage,
                confidence_score=100.0,
                is_corrected=is_corrected,
                verification_required=False,
                ambiguous=False,
                correction_reason="Exact match in pharmacopeia database" if is_corrected else "Preserved exact spelling",
            )

        # Step 4: Contextual Fuzzy Matching using RapidFuzz
        matches = process.extract(
            cleaned_token,
            self.all_candidates,
            scorer=fuzz.ratio,
            limit=3,
        )

        if not matches:
            return MedicineMatchResult(
                original_token=raw_name,
                matched_name=raw_name,
                formulation=formulation,
                extracted_dosage=dosage,
                confidence_score=0.0,
                is_corrected=False,
                verification_required=True,
                ambiguous=True,
                correction_reason="No close match in pharmacopeia. Original text preserved for review.",
            )

        best_candidate, best_score, _ = matches[0]
        second_score = matches[1][1] if len(matches) > 1 else 0

        # Safety Check 1: Ambiguity if top 2 candidates are very close
        if len(matches) > 1 and (best_score - second_score) < 4 and best_score < 95:
            cand1_std = ALIAS_TO_STANDARD[best_candidate]
            cand2_std = ALIAS_TO_STANDARD[matches[1][0]]
            if cand1_std != cand2_std:
                return MedicineMatchResult(
                    original_token=raw_name,
                    matched_name=raw_name,
                    formulation=formulation,
                    extracted_dosage=dosage,
                    confidence_score=float(best_score),
                    is_corrected=False,
                    verification_required=True,
                    ambiguous=True,
                    correction_reason=(
                        f"Ambiguous match between {cand1_std} ({best_score}%) and {cand2_std} ({second_score}%). "
                        "Preserving original OCR text for clinical safety."
                    ),
                )

        # Safety Check 2: High confidence threshold
        # If best candidate is above threshold, OR >= 75% when single clear candidate or both top candidates map to same standard drug
        has_clear_margin = (
            len(matches) == 1
            or (best_score - second_score >= 8)
            or (len(matches) > 1 and ALIAS_TO_STANDARD.get(best_candidate) == ALIAS_TO_STANDARD.get(matches[1][0]))
        )
        effective_threshold = min(self.threshold, 75) if has_clear_margin else self.threshold

        if best_score >= effective_threshold:
            resolved_std = ALIAS_TO_STANDARD[best_candidate]
            info = PHARMACOPEIA.get(resolved_std.lower())
            soundalike = None
            if info and info.soundalike_warnings:
                soundalike = f"Look-alike / Sound-alike warning: {', '.join(info.soundalike_warnings)}"

            return MedicineMatchResult(
                original_token=raw_name,
                matched_name=resolved_std,
                formulation=formulation,
                extracted_dosage=dosage,
                confidence_score=float(best_score),
                is_corrected=True,
                verification_required=best_score < 90,  # Require human review if between 75-89%
                ambiguous=False,
                correction_reason=f"Fuzzy match to {resolved_std} ({best_score:.1f}% confidence)",
                soundalike_warning=soundalike,
            )

        # Score below threshold: Preserve original, flag for verification
        return MedicineMatchResult(
            original_token=raw_name,
            matched_name=raw_name,
            formulation=formulation,
            extracted_dosage=dosage,
            confidence_score=float(best_score),
            is_corrected=False,
            verification_required=True,
            ambiguous=True,
            correction_reason=(
                f"Confidence {best_score:.1f}% below safety threshold ({self.threshold}%). "
                "Preserving original text for human verification."
            ),
        )
