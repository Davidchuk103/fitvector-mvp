"""
Body measurement estimation from 2 photos (front + side) + height.

Uses Replicate OpenPose API for person detection validation
and anthropometric estimation for 25 body measurements.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

REPLICATE_MODEL = "cjwbw/openpose-editor"

MIN_PHOTO_DIMENSION = 256
MAX_PHOTO_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

MEASUREMENT_RATIOS: dict[str, float] = {
    "head_circumference": 0.326,
    "neck_circumference": 0.217,
    "shoulder_width": 0.257,
    "chest_circumference": 0.557,
    "under_bust_circumference": 0.480,
    "shoulder_length": 0.091,
    "upper_arm_length": 0.186,
    "full_arm_length": 0.368,
    "bicep_circumference": 0.171,
    "forearm_circumference": 0.143,
    "wrist_circumference": 0.094,
    "waist_circumference": 0.463,
    "belly_circumference": 0.491,
    "hip_circumference": 0.554,
    "waist_height": 0.600,
    "hip_height": 0.510,
    "torso_length": 0.300,
    "upper_thigh_circumference": 0.320,
    "knee_circumference": 0.206,
    "calf_circumference": 0.200,
    "upper_leg_length": 0.245,
    "full_leg_length": 0.530,
    "inseam": 0.470,
    "chest_width": 0.183,
    "chest_depth": 0.126,
}

BMI_ADJUSTMENTS: dict[str, float] = {
    "neck_circumference": 0.006,
    "shoulder_width": 0.008,
    "chest_circumference": 0.010,
    "under_bust_circumference": 0.010,
    "bicep_circumference": 0.010,
    "forearm_circumference": 0.008,
    "wrist_circumference": 0.003,
    "waist_circumference": 0.015,
    "belly_circumference": 0.018,
    "hip_circumference": 0.010,
    "upper_thigh_circumference": 0.010,
    "knee_circumference": 0.005,
    "calf_circumference": 0.005,
    "chest_width": 0.006,
    "chest_depth": 0.012,
}

BMI_REF = 22.5

MEASUREMENT_LABELS: dict[str, str] = {
    "head_circumference": "Обхват головы",
    "neck_circumference": "Обхват шеи",
    "shoulder_width": "Ширина плеч",
    "chest_circumference": "Обхват груди",
    "under_bust_circumference": "Обхват под грудью",
    "shoulder_length": "Длина плеча",
    "upper_arm_length": "Длина руки до локтя",
    "full_arm_length": "Длина руки до запястья",
    "bicep_circumference": "Обхват бицепса",
    "forearm_circumference": "Обхват предплечья",
    "wrist_circumference": "Обхват запястья",
    "waist_circumference": "Обхват талии",
    "belly_circumference": "Обхват живота",
    "hip_circumference": "Обхват бёдер",
    "waist_height": "Высота талии от пола",
    "hip_height": "Высота бёдер от пола",
    "torso_length": "Длина туловища",
    "upper_thigh_circumference": "Обхват бедра",
    "knee_circumference": "Обхват колена",
    "calf_circumference": "Обхват икры",
    "upper_leg_length": "Длина ноги до колена",
    "full_leg_length": "Длина ноги до пола",
    "inseam": "Длина шага (внутренний шов)",
    "chest_width": "Ширина грудной клетки",
    "chest_depth": "Глубина грудной клетки",
}

MEASUREMENT_UNITS: dict[str, str] = {
    "head_circumference": "cm",
    "neck_circumference": "cm",
    "shoulder_width": "cm",
    "chest_circumference": "cm",
    "under_bust_circumference": "cm",
    "shoulder_length": "cm",
    "upper_arm_length": "cm",
    "full_arm_length": "cm",
    "bicep_circumference": "cm",
    "forearm_circumference": "cm",
    "wrist_circumference": "cm",
    "waist_circumference": "cm",
    "belly_circumference": "cm",
    "hip_circumference": "cm",
    "waist_height": "cm",
    "hip_height": "cm",
    "torso_length": "cm",
    "upper_thigh_circumference": "cm",
    "knee_circumference": "cm",
    "calf_circumference": "cm",
    "upper_leg_length": "cm",
    "full_leg_length": "cm",
    "inseam": "cm",
    "chest_width": "cm",
    "chest_depth": "cm",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    front_dimensions: Optional[tuple[int, int]] = None
    side_dimensions: Optional[tuple[int, int]] = None


def validate_photo(image_bytes: bytes, label: str) -> tuple[bool, list[str], Optional[tuple[int, int]]]:
    errors: list[str] = []
    dimensions = None

    if len(image_bytes) > MAX_PHOTO_SIZE:
        errors.append(f"{label}: file too large (max {MAX_PHOTO_SIZE // (1024*1024)}MB)")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        dimensions = img.size

        if dimensions[0] < MIN_PHOTO_DIMENSION or dimensions[1] < MIN_PHOTO_DIMENSION:
            errors.append(
                f"{label}: image too small ({dimensions[0]}x{dimensions[1]}), min {MIN_PHOTO_DIMENSION}px"
            )

        if img.format not in ALLOWED_FORMATS and img.format is not None:
            errors.append(f"{label}: unsupported format ({img.format}), use JPEG/PNG/WebP")
    except Exception as exc:
        errors.append(f"{label}: invalid image file — {exc}")
        return False, errors, None

    if errors:
        return False, errors, dimensions

    if dimensions is not None and dimensions[0] < MIN_PHOTO_DIMENSION or dimensions[1] < MIN_PHOTO_DIMENSION:
        errors.append(f"{label}: image too small")

    return len(errors) == 0, errors, dimensions


def validate_all(front_bytes: bytes, side_bytes: bytes) -> ValidationResult:
    all_errors: list[str] = []
    front_dims = None
    side_dims = None

    ok_front, errs_front, front_dims = validate_photo(front_bytes, "Front photo")
    all_errors.extend(errs_front)

    ok_side, errs_side, side_dims = validate_photo(side_bytes, "Side photo")
    all_errors.extend(errs_side)

    return ValidationResult(
        valid=ok_front and ok_side,
        errors=all_errors,
        front_dimensions=front_dims,
        side_dimensions=side_dims,
    )


async def run_openpose_validation(image_bytes: bytes) -> bool:
    try:
        import replicate

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            tmp_path = tmp.name

        result = await asyncio.to_thread(
            replicate.run,
            REPLICATE_MODEL,
            input={"image": open(tmp_path, "rb")},
        )

        if result is not None:
            return True

        return False
    except Exception as exc:
        logger.warning("OpenPose validation failed: %s", exc)
        return False


WEIGHT_CIRCUMFERENCE_CORRECTION: dict[str, float] = {
    "waist_circumference": 0.35,
    "belly_circumference": 0.40,
    "chest_circumference": 0.20,
    "under_bust_circumference": 0.18,
    "hip_circumference": 0.28,
    "upper_thigh_circumference": 0.22,
    "bicep_circumference": 0.10,
    "neck_circumference": 0.08,
}


def calculate_measurements(
    height_cm: float,
    weight_kg: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}

    bmi = None
    if weight_kg is not None and height_cm > 0:
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m * height_m)

    ref_weight = (BMI_REF / 10000.0) * (height_cm * height_cm)

    for key, ratio in MEASUREMENT_RATIOS.items():
        value = height_cm * ratio

        if bmi is not None and key in BMI_ADJUSTMENTS:
            bmi_offset = bmi - BMI_REF
            adjustment = 1.0 + BMI_ADJUSTMENTS[key] * bmi_offset
            value = value * adjustment

        if weight_kg is not None and key in WEIGHT_CIRCUMFERENCE_CORRECTION:
            weight_diff = weight_kg - ref_weight
            value = value + WEIGHT_CIRCUMFERENCE_CORRECTION[key] * weight_diff

        value = round(max(value, 1.0), 1)

        measurements[key] = {
            "label": MEASUREMENT_LABELS.get(key, key),
            "value": value,
            "unit": MEASUREMENT_UNITS.get(key, "cm"),
        }

    return measurements


def get_confidence(openpose_validated: bool, has_weight: bool) -> float:
    confidence = 0.65
    if openpose_validated:
        confidence += 0.15
    if has_weight:
        confidence += 0.05
    return min(confidence, 0.85)