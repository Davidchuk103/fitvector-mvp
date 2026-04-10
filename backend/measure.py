"""
Body measurement estimation from 2 photos (front + side) + height.

Pipeline:
1. Validate photos (format, size, dimensions)
2. Run MediaPipe Pose on front photo -> extract 33 landmarks
3. Run MediaPipe Pose on side photo -> extract 33 landmarks
4. Calibrate pixel-to-cm scale using known height
5. Calculate linear measurements from keypoint distances
6. Calculate circumferences using Ramanujan ellipse formula
   (front photo = width, side photo = depth)
7. Fall back to anthropometric formulas if CV fails
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp_pose_pkg

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp_pose_pkg = None

logger = logging.getLogger(__name__)

MIN_PHOTO_DIMENSION = 256
MAX_PHOTO_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_LANDMARKK_VISIBILITY = 0.5

HEAD_ABOVE_NOSE_CM = 12.0

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

MEASUREMENT_UNITS: dict[str, str] = {k: "cm" for k in MEASUREMENT_LABELS}

FORMULA_RATIOS: dict[str, float] = {
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


@dataclass
class Landmarks:
    points: dict[int, tuple[float, float, float]]
    img_w: int
    img_h: int

    def get(self, idx: int) -> Optional[tuple[float, float]]:
        p = self.points.get(idx)
        if p is None or p[2] < MIN_LANDMARKK_VISIBILITY:
            return None
        return (p[0] * self.img_w, p[1] * self.img_h)

    def pixel_dist(self, a: int, b: int) -> Optional[float]:
        pa = self.get(a)
        pb = self.get(b)
        if pa is None or pb is None:
            return None
        return math.dist(pa, pb)

    def vertical_dist(self, a: int, b: int) -> Optional[float]:
        pa = self.get(a)
        pb = self.get(b)
        if pa is None or pb is None:
            return None
        return abs(pa[1] - pb[1])

    def horizontal_dist(self, a: int, b: int) -> Optional[float]:
        pa = self.get(a)
        pb = self.get(b)
        if pa is None or pb is None:
            return None
        return abs(pa[0] - pb[0])

    def midpoint(self, a: int, b: int) -> Optional[tuple[float, float]]:
        pa = self.get(a)
        pb = self.get(b)
        if pa is None or pb is None:
            return None
        return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)

    def y_at_x_fraction(self, x_frac: float, y_top_idx: int, y_bot_idx: int) -> Optional[float]:
        pt = self.get(y_top_idx)
        pb = self.get(y_bot_idx)
        if pt is None or pb is None:
            return None
        return pt[1] + (pb[1] - pt[1]) * x_frac

    def width_at_level(self, left_idx: int, right_idx: int, y_ref: Optional[float] = None) -> Optional[float]:
        pl = self.get(left_idx)
        pr = self.get(right_idx)
        if pl is None or pr is None:
            return None
        return abs(pl[0] - pr[0])

    def depth_at_level(self, front_idx: int, back_idx: int) -> Optional[float]:
        pf = self.get(front_idx)
        pb = self.get(back_idx)
        if pf is None or pb is None:
            return None
        return abs(pf[0] - pb[0])


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
            errors.append(f"{label}: too small ({dimensions[0]}x{dimensions[1]}), min {MIN_PHOTO_DIMENSION}px")

        if img.format not in ALLOWED_FORMATS and img.format is not None:
            errors.append(f"{label}: unsupported format ({img.format}), use JPEG/PNG/WebP")
    except Exception as exc:
        errors.append(f"{label}: invalid image — {exc}")
        return False, errors, None

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


def _image_bytes_to_cv2(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception as exc:
        logger.warning("Failed to decode image for CV: %s", exc)
        return None


def run_mediapipe_pose(image_bytes: bytes) -> Optional[Landmarks]:
    if not _MP_AVAILABLE:
        logger.warning("MediaPipe not installed, skipping pose detection")
        return None

    img = _image_bytes_to_cv2(image_bytes)
    if img is None:
        return None

    h, w = img.shape[:2]

    try:
        mp_pose = mp_pose_pkg.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_pose.process(rgb)
        mp_pose.close()

        if result.pose_landmarks is None:
            logger.info("MediaPipe: no pose detected in image")
            return None

        points: dict[int, tuple[float, float, float]] = {}
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            points[idx] = (lm.x, lm.y, lm.visibility)

        return Landmarks(points=points, img_w=w, img_h=h)
    except Exception as exc:
        logger.warning("MediaPipe pose detection failed: %s", exc)
        return None


def calibrate_scale(landmarks: Landmarks, height_cm: float) -> Optional[float]:
    nose = landmarks.get(0)
    l_ankle = landmarks.get(27)
    r_ankle = landmarks.get(28)

    if nose is None:
        return None

    ankle_y = None
    if l_ankle is not None and r_ankle is not None:
        ankle_y = (l_ankle[1] + r_ankle[1]) / 2
    elif l_ankle is not None:
        ankle_y = l_ankle[1]
    elif r_ankle is not None:
        ankle_y = r_ankle[1]
    else:
        l_heel = landmarks.get(29)
        r_heel = landmarks.get(30)
        if l_heel is not None and r_heel is not None:
            ankle_y = (l_heel[1] + r_heel[1]) / 2
        elif l_heel is not None:
            ankle_y = l_heel[1]
        elif r_heel is not None:
            ankle_y = r_heel[1]

    if ankle_y is None:
        return None

    nose_to_ankle_px = abs(ankle_y - nose[1])
    if nose_to_ankle_px < 10:
        return None

    nose_to_ankle_cm = height_cm - HEAD_ABOVE_NOSE_CM
    scale = nose_to_ankle_cm / nose_to_ankle_px
    return scale


def ellipse_circumference(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return max(a, b) * math.pi * 2
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def calculate_cv_measurements(
    front: Landmarks,
    side: Landmarks,
    height_cm: float,
    weight_kg: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    front_scale = calibrate_scale(front, height_cm)
    side_scale = calibrate_scale(side, height_cm)

    if front_scale is None:
        logger.warning("Cannot calibrate front photo scale")
        return None

    if side_scale is None:
        side_scale = front_scale
        logger.info("Using front scale for side photo")

    measurements: dict[str, dict[str, Any]] = {}

    neck_mid = front.midpoint(11, 12)
    mid_hip = front.get(23) or front.get(24)

    shoulder_w_px = front.pixel_dist(11, 12)
    shoulder_width = shoulder_w_px * front_scale if shoulder_w_px else None

    neck_l = front.get(11)
    neck_r = front.get(12)
    neck_px = front.horizontal_dist(11, 12) if neck_l and neck_r else None
    neck_width = neck_px * front_scale * 0.5 if neck_px else None

    neck_depth_px = side.horizontal_dist(11, 12) if side.get(11) and side.get(12) else None
    neck_depth = neck_depth_px * side_scale * 0.5 if neck_depth_px else None

    if neck_width and neck_depth:
        neck_circ = ellipse_circumference(neck_width, neck_depth)
    elif neck_width:
        neck_circ = neck_width * math.pi * 1.15
    else:
        neck_circ = None

    shoulder_l_px = front.pixel_dist(11, 0) if neck_mid else front.pixel_dist(12, 0)
    if shoulder_l_px is None:
        shoulder_l_px = front.pixel_dist(0, 11) or front.pixel_dist(0, 12)
    shoulder_length = shoulder_l_px * front_scale * 0.35 if shoulder_l_px else None

    upper_arm_px = front.pixel_dist(11, 13) or front.pixel_dist(12, 14)
    upper_arm_length = upper_arm_px * front_scale if upper_arm_px else None

    full_arm_px = front.pixel_dist(11, 15) or front.pixel_dist(12, 16)
    full_arm_length = full_arm_px * front_scale if full_arm_px else None

    upper_leg_px = front.pixel_dist(23, 25) or front.pixel_dist(24, 26)
    upper_leg_length = upper_leg_px * front_scale if upper_leg_px else None

    full_leg_px = front.pixel_dist(23, 27) or front.pixel_dist(24, 28)
    full_leg_length = full_leg_px * front_scale if full_leg_px else None

    torso_px = front.pixel_dist(0, 23) or front.pixel_dist(0, 24)
    torso_length = torso_px * front_scale if torso_px else None

    l_hip = front.get(23)
    r_hip = front.get(24)
    hip_width_px = front.horizontal_dist(23, 24) if l_hip and r_hip else None
    hip_width = hip_width_px * front_scale * 0.5 if hip_width_px else None

    hip_depth_px = side.horizontal_dist(23, 24) if side.get(23) and side.get(24) else None
    hip_depth = hip_depth_px * side_scale * 0.5 if hip_depth_px else None

    if hip_width and hip_depth:
        hip_circ = ellipse_circumference(hip_width, hip_depth)
    elif hip_width:
        hip_circ = hip_width * math.pi * 1.1
    else:
        hip_circ = None

    if neck_mid and mid_hip:
        neck_y = neck_mid[1]
        hip_y = mid_hip[1]
        torso_h = hip_y - neck_y

        chest_frac = 0.25
        under_bust_frac = 0.38
        waist_frac = 0.55
        belly_frac = 0.50

        chest_level_y = neck_y + torso_h * chest_frac
        under_bust_level_y = neck_y + torso_h * under_bust_frac
        waist_level_y = neck_y + torso_h * waist_frac
        belly_level_y = neck_y + torso_h * belly_frac

        chest_width_px = shoulder_w_px * 0.72 if shoulder_w_px else None
        chest_width_cm = chest_width_px * front_scale * 0.5 if chest_width_px else None

        chest_depth_cm = hip_depth * 1.15 if hip_depth else None

        if chest_width_cm and chest_depth_cm:
            chest_circ = ellipse_circumference(chest_width_cm, chest_depth_cm)
        elif chest_width_cm:
            chest_circ = chest_width_cm * math.pi * 1.2
        else:
            chest_circ = None

        if chest_width_cm and chest_depth_cm:
            under_bust_circ = ellipse_circumference(chest_width_cm * 0.87, chest_depth_cm * 0.87)
        elif chest_circ:
            under_bust_circ = chest_circ * 0.86
        else:
            under_bust_circ = None

        waist_width_cm = hip_width * 0.82 if hip_width else None
        waist_depth_cm = hip_depth * 0.85 if hip_depth else None

        if waist_width_cm and waist_depth_cm:
            waist_circ = ellipse_circumference(waist_width_cm, waist_depth_cm)
        elif waist_width_cm:
            waist_circ = waist_width_cm * math.pi * 1.1
        else:
            waist_circ = None

        belly_width_cm = hip_width * 0.90 if hip_width else None
        belly_depth_cm = hip_depth * 0.92 if hip_depth else None

        if belly_width_cm and belly_depth_cm:
            belly_circ = ellipse_circumference(belly_width_cm, belly_depth_cm)
        elif belly_width_cm:
            belly_circ = belly_width_cm * math.pi * 1.1
        else:
            belly_circ = None

        ankle_y = (front.get(27)[1] + front.get(28)[1]) / 2 if front.get(27) and front.get(28) else None

        if ankle_y is not None:
            waist_height = abs(ankle_y - waist_level_y) * front_scale
            hip_height = abs(ankle_y - hip_y) * front_scale
        else:
            waist_height = None
            hip_height = None

        chest_width_val = chest_width_cm * 2 if chest_width_cm else None
        chest_depth_val = chest_depth_cm * 2 if chest_depth_cm else None
    else:
        chest_circ = None
        under_bust_circ = None
        waist_circ = None
        belly_circ = None
        waist_height = None
        hip_height = None
        chest_width_val = None
        chest_depth_val = None

    l_eye = front.get(2) or front.get(5)
    r_eye = front.get(5) or front.get(2)
    l_ear = front.get(7)
    r_ear = front.get(8)

    head_width_px = front.horizontal_dist(7, 8) if l_ear and r_ear else None
    head_width = head_width_px * front_scale * 0.5 if head_width_px else None

    head_depth_px = side.horizontal_dist(7, 8) if side.get(7) and side.get(8) else None
    head_depth = head_depth_px * side_scale * 0.5 if head_depth_px else None

    if head_width and head_depth:
        head_circ = ellipse_circumference(head_width, head_depth)
    elif head_width:
        head_circ = head_width * math.pi * 1.12
    else:
        head_circ = None

    bicep_width_px = front.horizontal_dist(13, 13) if front.get(13) else None
    if hip_width and shoulder_width:
        bicep_circ = (shoulder_width * 0.18 + hip_width * math.pi * 0.32) * 0.5
    else:
        bicep_circ = None

    forearm_circ = bicep_circ * 0.84 if bicep_circ else None
    wrist_circ = bicep_circ * 0.55 if bicep_circ else None

    if hip_width and hip_depth:
        upper_thigh_circ = ellipse_circumference(hip_width * 0.55, hip_depth * 0.65)
    elif hip_width:
        upper_thigh_circ = hip_width * math.pi * 1.2
    else:
        upper_thigh_circ = None

    knee_circ = upper_thigh_circ * 0.60 if upper_thigh_circ else None
    calf_circ = upper_thigh_circ * 0.63 if upper_thigh_circ else None

    inseam = full_leg_length * 0.88 if full_leg_length else None

    cv_data: dict[str, Optional[float]] = {
        "head_circumference": head_circ,
        "neck_circumference": neck_circ,
        "shoulder_width": shoulder_width,
        "chest_circumference": chest_circ,
        "under_bust_circumference": under_bust_circ,
        "shoulder_length": shoulder_length,
        "upper_arm_length": upper_arm_length,
        "full_arm_length": full_arm_length,
        "bicep_circumference": bicep_circ,
        "forearm_circumference": forearm_circ,
        "wrist_circumference": wrist_circ,
        "waist_circumference": waist_circ,
        "belly_circumference": belly_circ,
        "hip_circumference": hip_circ,
        "waist_height": waist_height,
        "hip_height": hip_height,
        "torso_length": torso_length,
        "upper_thigh_circumference": upper_thigh_circ,
        "knee_circumference": knee_circ,
        "calf_circumference": calf_circ,
        "upper_leg_length": upper_leg_length,
        "full_leg_length": full_leg_length,
        "inseam": inseam,
        "chest_width": chest_width_val,
        "chest_depth": chest_depth_val,
    }

    for key, val in cv_data.items():
        if val is not None:
            val = round(max(val, 1.0), 1)
        measurements[key] = {
            "label": MEASUREMENT_LABELS.get(key, key),
            "value": val,
            "unit": MEASUREMENT_UNITS.get(key, "cm"),
            "method": "cv",
        }

    return measurements


def calculate_formula_measurements(
    height_cm: float,
    weight_kg: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    measurements: dict[str, dict[str, Any]] = {}

    bmi = None
    if weight_kg is not None and height_cm > 0:
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m * height_m)

    ref_weight = (BMI_REF / 10000.0) * (height_cm * height_cm)

    for key, ratio in FORMULA_RATIOS.items():
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
            "method": "formula",
        }

    return measurements


def calculate_measurements(
    height_cm: float,
    weight_kg: Optional[float] = None,
    front_bytes: Optional[bytes] = None,
    side_bytes: Optional[bytes] = None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    if front_bytes and side_bytes and _MP_AVAILABLE:
        front_lm = run_mediapipe_pose(front_bytes)
        side_lm = run_mediapipe_pose(side_bytes)

        if front_lm is not None and side_lm is not None:
            cv_result = calculate_cv_measurements(front_lm, side_lm, height_cm, weight_kg)
            if cv_result is not None:
                formula = calculate_formula_measurements(height_cm, weight_kg)
                for key, cv_val in cv_result.items():
                    if cv_val["value"] is not None:
                        cv_val["method"] = "cv"
                    else:
                        cv_val["value"] = formula[key]["value"]
                        cv_val["method"] = "formula"
                return cv_result, True
            else:
                logger.info("CV measurement calculation incomplete, using formulas")

        if front_lm is None:
            logger.warning("MediaPipe: no pose detected in front photo")
        if side_lm is None:
            logger.warning("MediaPipe: no pose detected in side photo")

    measurements = calculate_formula_measurements(height_cm, weight_kg)
    return measurements, False


def get_confidence(cv_detected: bool, has_weight: bool) -> float:
    if cv_detected:
        confidence = 0.82
        if has_weight:
            confidence += 0.03
        return min(confidence, 0.85)
    else:
        confidence = 0.55
        if has_weight:
            confidence += 0.10
        return min(confidence, 0.65)