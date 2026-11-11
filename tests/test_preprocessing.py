"""Tests for image preprocessing module."""

import pytest
import numpy as np
from PIL import Image

from preprocessing.image_preprocessor import ImagePreprocessor
from config.settings import PreprocessingSettings


@pytest.fixture
def preprocessor():
    return ImagePreprocessor()


@pytest.fixture
def sample_rgb_image(tmp_path):
    img = Image.new("RGB", (400, 300), color=(240, 240, 240))
    path = tmp_path / "test_presc.png"
    img.save(path)
    return path


def test_load_valid_image(preprocessor, sample_rgb_image):
    img = preprocessor.load_image(sample_rgb_image)
    assert img.size == (400, 300)
    assert img.mode == "RGB"


def test_load_nonexistent_image(preprocessor):
    with pytest.raises(FileNotFoundError):
        preprocessor.load_image("nonexistent_image_123.jpg")


def test_load_unsupported_format(preprocessor, tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        preprocessor.load_image(path)


def test_rgb_conversion(preprocessor):
    grayscale = Image.new("L", (100, 100), color=128)
    converted = preprocessor.to_rgb(grayscale)
    assert converted.mode == "RGB"


def test_resolution_optimization(preprocessor):
    settings = PreprocessingSettings(max_dimension=500)
    proc = ImagePreprocessor(settings)
    huge_img = Image.new("RGB", (2000, 1000), color=(255, 255, 255))
    resized, did_resize = proc.optimize_resolution(huge_img)
    assert did_resize is True
    assert resized.size == (500, 250)


def test_clahe_contrast_enhancement(preprocessor, sample_rgb_image):
    img = Image.open(sample_rgb_image)
    enhanced = preprocessor.enhance_contrast(img)
    assert enhanced.size == img.size
    assert enhanced.mode == "RGB"


def test_complete_preprocessing_preserves_original(preprocessor, sample_rgb_image):
    res = preprocessor.process(sample_rgb_image, apply_contrast=True, apply_denoise=True)
    assert res.original_image is not None
    assert res.processed_image is not None
    assert res.contrast_enhanced is True
    assert res.denoised is True
