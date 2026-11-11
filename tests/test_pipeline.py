"""Integration test for the full end-to-end pipeline on real prescription images."""

import json
from pathlib import Path
import pytest

from main import PrescriptionPipeline
from config.settings import CONFIG


@pytest.fixture
def pipeline():
    return PrescriptionPipeline()


def test_pipeline_on_sample_prescription(pipeline, tmp_path):
    img_path = CONFIG.sample_data_dir / "prescription_1.png"
    if not img_path.exists():
        pytest.skip(f"Sample image {img_path} not found.")

    res = pipeline.process(
        image_source=img_path,
        ocr_mode="rapid_ocr",  # Safe fast local inference for testing
        apply_contrast=True,
        save_outputs=True,
        output_prefix="test_run",
    )

    # 1. Preprocessing assertion
    assert res.preprocessed.processed_image is not None
    assert res.preprocessed.original_image is not None

    # 2. OCR assertion
    assert len(res.ocr_result.raw_text) > 0
    assert res.ocr_result.backend_used != ""

    # 3. Normalization assertion
    formatted_text = res.normalized.to_formatted_text()
    assert "PATIENT:" in formatted_text
    assert "MEDICATION DETAILS" in formatted_text

    # 4. Deliverables assertion
    assert res.exported_files is not None
    assert res.exported_files["raw_ocr_txt"].exists()
    assert res.exported_files["normalized_txt"].exists()
    assert res.exported_files["results_json"].exists()
    assert res.exported_files["validation_json"].exists()

    # 5. Schema check on results JSON
    with open(res.exported_files["results_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "patient" in data
        assert "medications" in data
        assert "additional_instructions" in data
        assert "uncertain_fields" in data
        assert isinstance(data["medications"], list)

    # 6. Schema check on validation JSON
    with open(res.exported_files["validation_json"], "r", encoding="utf-8") as f:
        val_data = json.load(f)
        assert "total_medications_found" in val_data
        assert "passed_safety_checks" in val_data
        assert "audit_log" in val_data
