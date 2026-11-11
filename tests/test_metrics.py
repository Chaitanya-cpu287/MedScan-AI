"""Tests for CER, WER, and Normalization Evaluation Metrics."""

from utils.metrics import calculate_cer, calculate_wer, evaluate_normalization_metrics


def test_cer_and_wer_exact_match():
    ref = "Tab. Paracetamol 650mg TDS"
    hyp = "Tab. Paracetamol 650mg TDS"
    assert calculate_cer(ref, hyp) == 0.0
    assert calculate_wer(ref, hyp) == 0.0


def test_cer_and_wer_with_errors():
    ref = "Tab Paracetamol 650 mg"
    hyp = "Tab Paracetmol 650 mg"
    cer = calculate_cer(ref, hyp)
    wer = calculate_wer(ref, hyp)
    assert 0.0 < cer < 0.2
    assert wer == 0.25  # 1 word wrong out of 4 words


def test_normalization_metrics_evaluation():
    ground_truth = [
        {"medicine_name": "Paracetamol", "dosage": "650 MG", "frequency": "TDS", "duration": "3 days"},
        {"medicine_name": "Amoxicillin", "dosage": "500 MG", "frequency": "TDS", "duration": "7 days"},
    ]
    extracted = [
        {"medicine_name": "Paracetamol", "dosage": "650 MG", "frequency": "TDS", "duration": "3 days"},
        {"medicine_name": "Amoxicillin", "dosage": "500 MG", "frequency": "TDS", "duration": "7 days"},
    ]
    metrics = evaluate_normalization_metrics(ground_truth, extracted)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["name_accuracy"] == 1.0
    assert metrics["dosage_accuracy"] == 1.0
    assert metrics["hallucination_rate"] == 0.0
