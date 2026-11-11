"""Complete Medical Prescription OCR & Intelligent Text Normalization Pipeline."""

import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from PIL import Image

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config.settings import CONFIG
from models.memory_manager import MemoryManager, HardwareProfile
from preprocessing.image_preprocessor import ImagePreprocessor, PreprocessedResult
from ocr.ocr_engine import PrescriptionOCREngine, OCRResult, OCRMode
from normalization.text_normalizer import TextNormalizer, NormalizedPrescription
from normalization.medicine_label_normalizer import NormalizedMedicineLabel
from validation.prescription_validator import PrescriptionValidator, ValidationReport
from nlp_labelling.labelling_engine import MedicationLabeller
from utils.file_handler import FileHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PrescriptionPipeline")


@dataclass
class PipelineResult:
    preprocessed: PreprocessedResult
    ocr_result: OCRResult
    normalized: Union[NormalizedPrescription, NormalizedMedicineLabel]
    validation_report: ValidationReport
    document_type: str = "prescription"
    nlp_labels: Optional[Dict[str, Any]] = None
    exported_files: Optional[Dict[str, Path]] = None


class PrescriptionPipeline:
    """Master orchestrator connecting Image Preprocessing, OCR, Normalization, Validation and Export."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = PrescriptionOCREngine()
        self.normalizer = TextNormalizer()
        self.validator = PrescriptionValidator()
        self.labeller = MedicationLabeller()

    @staticmethod
    def get_hardware_status() -> HardwareProfile:
        return MemoryManager.get_hardware_profile()

    def process(
        self,
        image_source: Union[str, Path, Image.Image, bytes],
        ocr_mode: Optional[Union[OCRMode, str]] = OCRMode.AUTO,
        doc_type: str = "auto",
        apply_contrast: bool = True,
        apply_denoise: bool = False,
        apply_deskew: bool = False,
        hf_token: Optional[str] = None,
        allow_cpu_low_ram: bool = False,
        save_outputs: bool = True,
        output_prefix: str = "",
    ) -> PipelineResult:
        """Executes the complete end-to-end pipeline for Prescriptions and Medicine Labels."""
        logger.info("Step 1: Running image preprocessing...")
        prep_res = self.preprocessor.process(
            image_source=image_source,
            apply_contrast=apply_contrast,
            apply_denoise=apply_denoise,
            apply_deskew=apply_deskew,
        )

        logger.info(f"Step 2: Running OCR extraction (Mode: {ocr_mode})...")
        ocr_res = self.ocr_engine.extract(
            image=prep_res.processed_image,
            mode=ocr_mode,
            hf_token=hf_token,
            allow_cpu_low_ram=allow_cpu_low_ram,
        )

        logger.info(f"Step 3: Running intelligent text normalization (Doc Type: {doc_type})...")
        norm_input = ocr_res.parsed_json if (ocr_res.is_structured_json and ocr_res.parsed_json) else ocr_res.raw_text
        normalized = self.normalizer.normalize(norm_input, doc_type=doc_type)

        is_label = isinstance(normalized, NormalizedMedicineLabel)
        detected_doc_type = "medicine_label" if is_label else "prescription"

        logger.info(f"Step 4: Running validation & safety checks ({detected_doc_type})...")
        val_report = self.validator.validate(
            raw_ocr_text=ocr_res.raw_text,
            normalized=normalized,
        )

        logger.info(f"Step 5: Running Member 2 NLP medication labelling ({detected_doc_type})...")
        nlp_labels = None
        try:
            if is_label:
                nlp_labels = self.labeller.label_medicine_label(ocr_res.raw_text)
            else:
                nlp_labels = self.labeller.label_prescription(
                    prescription_text=normalized.to_formatted_text(),
                    normalized_prescription_obj=normalized,
                )
        except Exception as e:
            logger.error(f"NLP Labelling error: {e}")

        exported_files = None
        if save_outputs:
            logger.info("Step 6: Exporting deliverables...")
            exported_files = FileHandler.export_all(
                raw_ocr_text=ocr_res.raw_text,
                normalized=normalized,
                validation_report=val_report,
                nlp_labels=nlp_labels,
                prefix=output_prefix,
            )

        logger.info(f"Pipeline processing completed successfully for {detected_doc_type}.")
        return PipelineResult(
            preprocessed=prep_res,
            ocr_result=ocr_res,
            normalized=normalized,
            validation_report=val_report,
            document_type=detected_doc_type,
            nlp_labels=nlp_labels,
            exported_files=exported_files,
        )


def main():
    parser = argparse.ArgumentParser(description="Medical Prescription & Medicine Label Scanner Pipeline")
    parser.add_argument("--image", type=str, help="Path to prescription or medicine label image")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "local_qwen", "cloud_hf", "rapid_ocr"], help="OCR engine mode")
    parser.add_argument("--doc-type", type=str, default="auto", choices=["auto", "prescription", "medicine_label"], help="Document type (auto-detects between prescription and packaging label)")
    parser.add_argument("--check-hardware", action="store_true", help="Display hardware profile and memory diagnostics")
    parser.add_argument("--test-samples", action="store_true", help="Run pipeline on sample prescriptions in sample_data/")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for cloud execution")
    parser.add_argument("--contrast", action="store_true", default=True, help="Apply gentle CLAHE contrast enhancement")
    parser.add_argument("--no-contrast", action="store_false", dest="contrast", help="Disable contrast enhancement")

    args = parser.parse_args()

    pipeline = PrescriptionPipeline()

    if args.check_hardware:
        hw = pipeline.get_hardware_status()
        print("\n=== SYSTEM HARDWARE & MEMORY PROFILE ===")
        print(f"Device:           {hw.device_name}")
        print(f"CUDA Available:   {hw.has_cuda}")
        print(f"Total RAM:        {hw.total_ram_gb:.2f} GB")
        print(f"Available RAM:    {hw.available_ram_gb:.2f} GB ({hw.ram_usage_percent}% used)")
        print(f"Total VRAM:       {hw.total_vram_gb:.2f} GB")
        print(f"Available VRAM:   {hw.available_vram_gb:.2f} GB")
        print(f"Can Run Local 3B: {hw.can_run_local_3b}")
        print(f"Recommended Mode: {hw.recommended_backend}")
        print(f"Assessment:       {hw.reason}")
        print("=========================================\n")
        return

    if args.test_samples:
        import json
        gt_file = CONFIG.sample_data_dir / "ground_truth.json"
        if not gt_file.exists():
            print(f"No ground truth found at {gt_file}")
            return
        with open(gt_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        print(f"\nProcessing {len(samples)} sample prescriptions...")
        for img_name in samples.keys():
            img_path = CONFIG.sample_data_dir / img_name
            if not img_path.exists():
                continue
            print(f"\n---> Processing: {img_name}")
            res = pipeline.process(
                image_source=img_path,
                ocr_mode=args.mode,
                doc_type=args.doc_type,
                apply_contrast=args.contrast,
                hf_token=args.hf_token,
                output_prefix=Path(img_name).stem,
            )
            print("--- NORMALIZED OUTPUT ---")
            print(res.normalized.to_formatted_text())
            print(f"Safety Passed: {res.validation_report.passed_safety_checks}, Human Review: {res.validation_report.human_review_required}")
        return

    if args.image:
        res = pipeline.process(
            image_source=args.image,
            ocr_mode=args.mode,
            doc_type=args.doc_type,
            apply_contrast=args.contrast,
            hf_token=args.hf_token,
        )
        banner = "NORMALIZED MEDICINE LABEL" if res.document_type == "medicine_label" else "NORMALIZED PRESCRIPTION TEXT"
        print(f"\n================== {banner} ==================")
        print(res.normalized.to_formatted_text())
        print("==================================================================\n")
        print(f"Exported files saved in: {CONFIG.output_dir.resolve()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
