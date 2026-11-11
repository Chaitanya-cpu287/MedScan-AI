"""File management and export utility for prescription outputs."""

from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import CONFIG
from normalization.text_normalizer import NormalizedPrescription
from normalization.medicine_label_normalizer import NormalizedMedicineLabel
from validation.prescription_validator import ValidationReport


class FileHandler:
    """Handles saving exports and managing output directories."""

    @classmethod
    def export_all(
        cls,
        raw_ocr_text: str,
        normalized: Any,
        validation_report: ValidationReport,
        nlp_labels: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        prefix: str = "",
    ) -> Dict[str, Path]:
        """Saves all required deliverables for a processed prescription or medicine label."""
        out_dir = Path(output_dir) if output_dir else CONFIG.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        p = f"{prefix}_" if prefix else ""
        is_label = isinstance(normalized, NormalizedMedicineLabel)

        raw_path = out_dir / f"{p}raw_ocr.txt"
        norm_filename = f"{p}normalized_medicine_label.txt" if is_label else f"{p}normalized_prescription.txt"
        json_filename = f"{p}medicine_label_results.json" if is_label else f"{p}prescription_results.json"
        norm_path = out_dir / norm_filename
        json_path = out_dir / json_filename
        val_path = out_dir / f"{p}validation_report.json"
        nlp_path = out_dir / f"{p}nlp_medication_labels.json"

        # 1. raw_ocr.txt
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_ocr_text)

        # 2. normalized_prescription.txt
        with open(norm_path, "w", encoding="utf-8") as f:
            f.write(normalized.to_formatted_text())

        # 3. prescription_results.json
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(normalized.to_export_json(), f, indent=2, ensure_ascii=False)

        # 4. validation_report.json
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validation_report.to_dict(), f, indent=2, ensure_ascii=False)

        # 5. nlp_medication_labels.json (Member 2 deliverable)
        if nlp_labels is not None:
            with open(nlp_path, "w", encoding="utf-8") as f:
                json.dump(nlp_labels, f, indent=2, ensure_ascii=False)

        res = {
            "raw_ocr_txt": raw_path,
            "normalized_txt": norm_path,
            "results_json": json_path,
            "validation_json": val_path,
        }
        if nlp_labels is not None:
            res["nlp_labels_json"] = nlp_path

        return res
