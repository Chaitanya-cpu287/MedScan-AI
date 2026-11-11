"""Evaluation metrics for OCR and Medical Normalization."""

import re
from typing import List, Dict, Any, Tuple
from rapidfuzz.distance import Levenshtein


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculates Character Error Rate (CER) = Levenshtein(ref, hyp) / len(ref)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    dist = Levenshtein.distance(reference, hypothesis)
    return round(float(dist) / max(1, len(reference)), 4)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate (WER) = Levenshtein_words(ref, hyp) / len(ref_words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = Levenshtein.distance(ref_words, hyp_words)
    return round(float(dist) / max(1, len(ref_words)), 4)


def evaluate_normalization_metrics(
    ground_truth_medications: List[Dict[str, str]],
    extracted_medications: List[Dict[str, str]],
) -> Dict[str, float]:
    """Calculates precision, recall, accuracy, and hallucination rate.

    Each dict expected to have: 'medicine_name', 'dosage', 'frequency', 'duration'.
    """
    if not ground_truth_medications:
        if not extracted_medications:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "name_accuracy": 1.0,
                "dosage_accuracy": 1.0,
                "frequency_accuracy": 1.0,
                "duration_accuracy": 1.0,
                "hallucination_rate": 0.0,
            }
        else:
            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1_score": 0.0,
                "name_accuracy": 0.0,
                "dosage_accuracy": 0.0,
                "frequency_accuracy": 0.0,
                "duration_accuracy": 0.0,
                "hallucination_rate": 1.0,
            }

    gt_names = [m.get("medicine_name", "").lower().strip() for m in ground_truth_medications]
    pred_names = [m.get("medicine_name", "").lower().strip() for m in extracted_medications]

    true_positives = 0
    name_correct = 0
    dosage_correct = 0
    frequency_correct = 0
    duration_correct = 0

    matched_gt = set()

    for pred in extracted_medications:
        p_name = pred.get("medicine_name", "").lower().strip()
        p_dose = pred.get("dosage", "").lower().replace(" ", "")
        p_freq = pred.get("frequency", "").lower().strip()
        p_dur = pred.get("duration", "").lower().strip()

        # Find closest ground truth match
        best_gt_idx = None
        for g_idx, gt in enumerate(ground_truth_medications):
            if g_idx in matched_gt:
                continue
            g_name = gt.get("medicine_name", "").lower().strip()
            # Check match by substring or Levenshtein distance
            if p_name == g_name or g_name in p_name or p_name in g_name or Levenshtein.distance(p_name, g_name) <= 3:
                best_gt_idx = g_idx
                break

        if best_gt_idx is not None:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            gt_item = ground_truth_medications[best_gt_idx]

            # Name accuracy
            if p_name == gt_item.get("medicine_name", "").lower().strip():
                name_correct += 1

            # Dosage accuracy
            gt_dose = gt_item.get("dosage", "").lower().replace(" ", "")
            if p_dose and (p_dose == gt_dose or gt_dose in p_dose or p_dose in gt_dose):
                dosage_correct += 1
            elif not p_dose and not gt_dose:
                dosage_correct += 1

            # Frequency accuracy
            gt_freq = gt_item.get("frequency", "").lower().strip()
            if p_freq and (p_freq == gt_freq or gt_freq in p_freq):
                frequency_correct += 1
            elif not p_freq and not gt_freq:
                frequency_correct += 1

            # Duration accuracy
            gt_dur = gt_item.get("duration", "").lower().strip()
            if p_dur and (p_dur == gt_dur or gt_dur in p_dur):
                duration_correct += 1
            elif not p_dur and not gt_dur:
                duration_correct += 1

    precision = true_positives / max(1, len(extracted_medications))
    recall = true_positives / max(1, len(ground_truth_medications))
    f1 = 2 * (precision * recall) / max(1e-6, (precision + recall))

    false_positives = len(extracted_medications) - true_positives
    hallucination_rate = false_positives / max(1, len(extracted_medications))

    tp_count = max(1, true_positives)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "name_accuracy": round(name_correct / tp_count, 4),
        "dosage_accuracy": round(dosage_correct / tp_count, 4),
        "frequency_accuracy": round(frequency_correct / tp_count, 4),
        "duration_accuracy": round(duration_correct / tp_count, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }
