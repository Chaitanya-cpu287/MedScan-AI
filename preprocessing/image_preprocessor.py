"""Image preprocessing module for handwritten and printed medical prescriptions."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from config.settings import CONFIG, PreprocessingSettings


@dataclass
class PreprocessedResult:
    original_image: Image.Image
    processed_image: Image.Image
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    exif_corrected: bool
    resized: bool
    contrast_enhanced: bool
    denoised: bool
    deskewed: bool
    skew_angle_deg: float
    format_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """Handles loading, validation, orientation correction, and gentle enhancement of prescription images."""

    def __init__(self, settings: Optional[PreprocessingSettings] = None):
        self.settings = settings or CONFIG.preprocessing

    def load_image(self, image_source: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """Loads and validates an image from file path, PIL Image, or raw bytes."""
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if not path.exists():
                raise FileNotFoundError(f"Prescription image not found at: {path}")
            ext = path.suffix.lower()
            if ext not in self.settings.supported_extensions:
                raise ValueError(
                    f"Unsupported image format: {ext}. Supported formats are: {self.settings.supported_extensions}"
                )
            img = Image.open(path)
        elif isinstance(image_source, bytes):
            import io
            img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, Image.Image):
            img = image_source
        else:
            raise TypeError(f"Invalid image_source type: {type(image_source)}")

        # Validate dimensions
        if img.width == 0 or img.height == 0:
            raise ValueError("Invalid image dimensions (width or height is 0).")

        return img

    def correct_exif_orientation(self, img: Image.Image) -> Tuple[Image.Image, bool]:
        """Corrects image orientation based on EXIF metadata."""
        try:
            transposed = ImageOps.exif_transpose(img)
            if transposed is not None and transposed != img:
                return transposed, True
        except Exception:
            pass
        return img, False

    def to_rgb(self, img: Image.Image) -> Image.Image:
        """Ensures image is in RGB format."""
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def optimize_resolution(self, img: Image.Image) -> Tuple[Image.Image, bool]:
        """Resizes image if either dimension exceeds max_dimension while preserving aspect ratio."""
        max_dim = self.settings.max_dimension
        w, h = img.size
        if max(w, h) > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return resized, True
        return img, False

    def enhance_contrast(self, img: Image.Image) -> Image.Image:
        """Gentle CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel.

        Enhances faint pen ink without aggressively blowing out paper texture or eroding strokes.
        """
        np_img = np.array(img)
        # Convert RGB to LAB
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=self.settings.clahe_tile_grid_size,
        )
        cl = clahe.apply(l)

        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)

    def reduce_noise(self, img: Image.Image) -> Image.Image:
        """Applies edge-preserving bilateral filter to gently reduce paper grain while preserving sharp pen strokes."""
        np_img = np.array(img)
        d = self.settings.denoise_strength
        denoised = cv2.bilateralFilter(np_img, d=d, sigmaColor=35, sigmaSpace=35)
        return Image.fromarray(denoised)

    def detect_and_deskew(self, img: Image.Image) -> Tuple[Image.Image, bool, float]:
        """Detects skew angle and rotates the image if skew is significant but within safe bounds."""
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return img, False, 0.0

        angles = []
        for line in np.reshape(lines, (-1, 4)):
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Focus on near horizontal lines (-45 to 45 degrees)
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return img, False, 0.0

        median_angle = float(np.median(angles))
        max_safe_angle = self.settings.deskew_max_angle

        # Only deskew if clearly tilted (> 2.0 deg) and safe (<= max_safe_angle) to avoid blurring crumpled paper
        if 2.0 <= abs(median_angle) <= max_safe_angle:
            h, w = np_img.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                np_img,
                rot_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return Image.fromarray(rotated), True, median_angle

        return img, False, median_angle

    def process(
        self,
        image_source: Union[str, Path, Image.Image, bytes],
        apply_contrast: bool = True,
        apply_denoise: bool = False,
        apply_deskew: bool = False,
    ) -> PreprocessedResult:
        """Complete preprocessing pipeline. Always retains and returns the original image."""
        raw_img = self.load_image(image_source)
        format_name = getattr(raw_img, "format", "PNG") or "PNG"
        orig_w, orig_h = raw_img.size

        # Always clone original to guarantee untouched availability
        original_copy = raw_img.copy()

        # Step 1: EXIF orientation
        img, exif_corrected = self.correct_exif_orientation(raw_img)

        # Step 2: RGB conversion
        img = self.to_rgb(img)

        # Step 3: Resolution optimization
        img, resized = self.optimize_resolution(img)

        # Step 4: Optional gentle contrast enhancement
        contrast_enhanced = False
        if apply_contrast:
            img = self.enhance_contrast(img)
            contrast_enhanced = True

        # Step 5: Optional noise reduction
        denoised = False
        if apply_denoise:
            img = self.reduce_noise(img)
            denoised = True

        # Step 6: Optional deskewing
        deskewed = False
        skew_angle = 0.0
        if apply_deskew:
            img, deskewed, skew_angle = self.detect_and_deskew(img)

        return PreprocessedResult(
            original_image=original_copy,
            processed_image=img,
            original_size=(orig_w, orig_h),
            processed_size=img.size,
            exif_corrected=exif_corrected,
            resized=resized,
            contrast_enhanced=contrast_enhanced,
            denoised=denoised,
            deskewed=deskewed,
            skew_angle_deg=round(skew_angle, 2),
            format_name=format_name,
            metadata={
                "aspect_ratio": round(orig_w / max(1, orig_h), 3),
                "is_portrait": orig_h >= orig_w,
            },
        )
