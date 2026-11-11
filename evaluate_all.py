"""Comprehensive Evaluation Suite comparing OCR and Normalization against Real Prescriptions Ground Truth."""

import json
import logging
from pathlib import Path
import pandas as pd

from config.settings import CONFIG
from main import PrescriptionPipeline
from utils.metrics import calculate_cer, calculate_wer, evaluate_normalization_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_evaluation():
    pipeline = PrescriptionPipeline()
    gt_path = CONFIG.sample_data_dir / "ground_truth.json"

    if not gt_path.exists():
        print("Ground truth file not found.")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_db = json.load(f)

    results = []

    print("\n" + "=" * 80)
    print("      MEDICAL PRESCRIPTION OCR & NORMALIZATION EVALUATION BENCHMARK")
    print("=" * 80)

    for img_name, data in ground_truth_db.items():
        img_path = CONFIG.sample_data_dir / img_name
        if not img_path.exists():
            continue

        raw_gt_text = data["ground_truth_text"]

        # Parse ground truth medications
        norm_gt = pipeline.normalizer.normalize(raw_gt_text)
        gt_meds = [
            {
                "medicine_name": m.medicine_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "duration": m.duration,
            }
            for m in norm_gt.medications
        ]

        # Run pipeline
        res = pipeline.process(
            image_source=img_path,
            ocr_mode="rapid_ocr",
            apply_contrast=True,
            save_outputs=True,
            output_prefix=Path(img_name).stem,
        )

        extracted_meds = [
            {
                "medicine_name": m.medicine_name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "duration": m.duration,
            }
            for m in res.normalized.medications
        ]

        # Calculate OCR CER and WER
        cer = calculate_cer(raw_gt_text, res.ocr_result.raw_text)
        wer = calculate_wer(raw_gt_text, res.ocr_result.raw_text)

        # Calculate Normalization Metrics
        norm_metrics = evaluate_normalization_metrics(gt_meds, extracted_meds)

        res_record = {
            "image": img_name,
            "gt_meds_count": len(gt_meds),
            "pred_meds_count": len(extracted_meds),
            "cer": cer,
            "wer": wer,
            "precision": norm_metrics["precision"],
            "recall": norm_metrics["recall"],
            "f1_score": norm_metrics["f1_score"],
            "name_accuracy": norm_metrics["name_accuracy"],
            "dosage_accuracy": norm_metrics["dosage_accuracy"],
            "hallucination_rate": norm_metrics["hallucination_rate"],
            "safety_passed": res.validation_report.passed_safety_checks,
        }
        results.append(res_record)

        print(f"\n[IMAGE: {img_name}]")
        print(f"  Ground Truth Meds ({len(gt_meds)}): {[m['medicine_name'] for m in gt_meds]}")
        print(f"  Extracted Meds    ({len(extracted_meds)}): {[m['medicine_name'] for m in extracted_meds]}")
        print(f"  OCR Metrics:       CER={cer:.4f}, WER={wer:.4f}")
        print(f"  Norm Precision:    {norm_metrics['precision']:.2%}")
        print(f"  Norm Recall:       {norm_metrics['recall']:.2%}")
        print(f"  Name Accuracy:     {norm_metrics['name_accuracy']:.2%}")
        print(f"  Dosage Accuracy:   {norm_metrics['dosage_accuracy']:.2%}")
        print(f"  Hallucination Rate:{norm_metrics['hallucination_rate']:.2%}")

    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("                      AGGREGATE BENCHMARK SUMMARY")
    print("=" * 80)
    avg_metrics = {
        "mean_cer": round(df["cer"].mean(), 4),
        "mean_wer": round(df["wer"].mean(), 4),
        "mean_precision": round(df["precision"].mean(), 4),
        "mean_recall": round(df["recall"].mean(), 4),
        "mean_f1": round(df["f1_score"].mean(), 4),
        "mean_name_accuracy": round(df["name_accuracy"].mean(), 4),
        "mean_dosage_accuracy": round(df["dosage_accuracy"].mean(), 4),
        "mean_hallucination_rate": round(df["hallucination_rate"].mean(), 4),
        "samples_evaluated": len(df),
    }

    for k, v in avg_metrics.items():
        print(f"  {k:25s}: {v}")

    eval_json_path = CONFIG.output_dir / "evaluation_benchmark_summary.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": avg_metrics, "per_sample": results}, f, indent=2)

    print(f"\nFull evaluation report saved to: {eval_json_path.resolve()}\n")


if __name__ == "__main__":
    run_evaluation()
