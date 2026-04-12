"""
Body measurement estimation from 2 photos (front + side) + height.

Pipeline:
1. Validate photos (format, size, dimensions)
2. Preprocess images (background removal + contrast normalization)
3. Run MediaPipe Pose on front photo -> extract 33 landmarks
4. Run MediaPipe Pose on side photo -> extract 33 landmarks
5. Validate pose quality (standing straight, arms at sides)
6. Calibrate pixel-to-cm scale using multi-point calibration
7. Calculate linear measurements from keypoint distances
8. Calculate circumferences using body contour tracing + Ramanujan ellipse
9. Apply gender-specific corrections
10. Fall back to anthropometric formulas if CV fails
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

try:
    from rembg import remove as rembg_remove

    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False
    rembg_remove = None

logger = logging.getLogger(__name__)

MIN_PHOTO_DIMENSION = 256
MAX_PHOTO_SIZE = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_LANDMARK_VISIBILITY = 0.5

HEAD_ABOVE_NOSE_CM = 12.0
EAR_TO_EAR_AVG_CM = 14.0
INTER_PUPILLARY_AVG_CM = 6.3

HEAD_RATIO_MIN = 0.048
HEAD_RATIO_MAX = 0.058
HEAD_RATIO_DEFAULT = 0.052

BMI_REF = 22.5

MALE_BODY_RATIOS = {
    "shoulder_to_hip_ratio": 1.15,
    "chest_to_waist_ratio": 1.12,
    "waist_to_hip_ratio": 0.85,
    "fat_distribution_upper": 0.60,
}

FEMALE_BODY_RATIOS = {
    "shoulder_to_hip_ratio": 0.92,
    "chest_to_waist_ratio": 1.05,
    "waist_to_hip_ratio": 0.72,
    "fat_distribution_upper": 0.45,
}

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


def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    img = _image_bytes_to_cv2(image_bytes)
    if img is None:
        return None

    if _REMBG_AVAILABLE:
        try:
            input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            output_pil = rembg_remove(input_pil)
            output_bytes = io.BytesIO()
            output_pil.save(output_bytes, format="PNG")
            output_bytes.seek(0)
            arr = np.frombuffer(output_bytes.read(), dtype=np.uint8)
            img_no_bg = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img_no_bg is not None:
                if img_no_bg.shape[2] == 4:
                    b, g, r, a = cv2.split(img_no_bg)
                    mask = (a > 128).astype(np.uint8) * 255
                    img_no_bg = cv2.merge([b, g, r])
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask_3ch = cv2.merge([mask, mask, mask])
                    img_no_bg = cv2.bitwise_and(img_no_bg, mask_3ch)
                    img = img_no_bg
        except Exception as exc:
            logger.warning("Background removal failed, using original: %s", exc)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    if img.shape[:2] != enhanced_bgr.shape[:2]:
        enhanced_bgr = cv2.resize(enhanced_bgr, (img.shape[1], img.shape[0]))

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([mask, mask, mask])
    result = cv2.bitwise_and(enhanced_bgr, mask_3ch)

    return result


def run_mediapipe_pose(image_bytes: bytes, preprocessed: Optional[np.ndarray] = None) -> Optional[Landmarks]:
    if not _MP_AVAILABLE:
        logger.warning("MediaPipe not installed, skipping pose detection")
        return None

    img = preprocessed if preprocessed is not None else _image_bytes_to_cv2(image_bytes)
    if img is None:
        return None

    h, w = img.shape[:2]

    try:
        mp_pose = mp_pose_pkg.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.6,
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


@dataclass
class PoseQualityReport:
    is_valid: bool
    score: float
    issues: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


def validate_pose_quality(landmarks: Landmarks, view: str = "front") -> PoseQualityReport:
    issues: list[str] = []
    details: dict[str, float] = {}
    score = 1.0

    nose = landmarks.get(0)
    l_shoulder = landmarks.get(11)
    r_shoulder = landmarks.get(12)
    l_elbow = landmarks.get(13)
    r_elbow = landmarks.get(14)
    l_wrist = landmarks.get(15)
    r_wrist = landmarks.get(16)
    l_hip = landmarks.get(23)
    r_hip = landmarks.get(24)
    l_knee = landmarks.get(25)
    r_knee = landmarks.get(26)
    l_ankle = landmarks.get(27)
    r_ankle = landmarks.get(28)

    if view == "front":
        if l_shoulder and r_shoulder and nose:
            shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
            x_offset = abs(shoulder_mid_x - nose[0])
            shoulder_w = abs(l_shoulder[0] - r_shoulder[0])
            if shoulder_w > 0:
                rotation_ratio = x_offset / shoulder_w
                details["body_rotation"] = rotation_ratio
                if rotation_ratio > 0.15:
                    issues.append("body is rotated (not facing camera directly)")
                    score -= 0.2

        if l_shoulder and l_wrist and l_hip:
            wrist_above_shoulder = l_wrist[1] < l_shoulder[1]
            details["left_arm_raised"] = 1.0 if wrist_above_shoulder else 0.0
            if wrist_above_shoulder:
                issues.append("left arm is raised (should be at sides)")
                score -= 0.15

        if r_shoulder and r_wrist and r_hip:
            wrist_above_shoulder = r_wrist[1] < r_shoulder[1]
            details["right_arm_raised"] = 1.0 if wrist_above_shoulder else 0.0
            if wrist_above_shoulder:
                issues.append("right arm is raised (should be at sides)")
                score -= 0.15

        if l_shoulder and l_wrist and l_elbow and l_hip:
            arm_across_body = l_wrist[0] > l_shoulder[0] + abs(l_shoulder[0] - r_shoulder[0]) * 0.3
            details["left_arm_across"] = 1.0 if arm_across_body else 0.0
            if arm_across_body:
                issues.append("left arm is across the body")
                score -= 0.15

        if r_shoulder and r_wrist and r_elbow and r_hip:
            arm_across_body = r_wrist[0] < r_shoulder[0] - abs(l_shoulder[0] - r_shoulder[0]) * 0.3
            details["right_arm_across"] = 1.0 if arm_across_body else 0.0
            if arm_across_body:
                issues.append("right arm is across the body")
                score -= 0.15

        if l_hip and r_hip and l_knee and r_knee:
            l_knee_bent = abs(l_knee[0] - l_hip[0]) > abs(r_hip[0] - l_hip[0]) * 0.15
            r_knee_bent = abs(r_knee[0] - r_hip[0]) > abs(r_hip[0] - l_hip[0]) * 0.15
            details["leg_straightness"] = 1.0 - (float(l_knee_bent) + float(r_knee_bent)) / 2
            if l_knee_bent:
                issues.append("left leg is bent")
                score -= 0.1
            if r_knee_bent:
                issues.append("right leg is bent")
                score -= 0.1

        if l_hip and r_hip and l_ankle and r_ankle:
            hip_width = abs(r_hip[0] - l_hip[0])
            l_ankle_offset = abs(l_ankle[0] - l_hip[0])
            r_ankle_offset = abs(r_ankle[0] - r_hip[0])
            if hip_width > 0:
                stance_ratio = max(l_ankle_offset, r_ankle_offset) / hip_width
                details["stance_width"] = stance_ratio
                if stance_ratio > 0.5:
                    issues.append("legs are spread wide (stand with feet together)")
                    score -= 0.1

    if l_shoulder and r_shoulder:
        shoulder_y_diff = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_x_dist = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_x_dist > 0:
            tilt = shoulder_y_diff / shoulder_x_dist
            details["shoulder_tilt"] = tilt
            if tilt > 0.1:
                issues.append("shoulders are tilted (stand straight)")
                score -= 0.1

    if l_hip and r_hip:
        hip_y_diff = abs(l_hip[1] - r_hip[1])
        hip_x_dist = abs(r_hip[0] - l_hip[0])
        if hip_x_dist > 0:
            hip_tilt = hip_y_diff / hip_x_dist
            details["hip_tilt"] = hip_tilt
            if hip_tilt > 0.1:
                issues.append("hips are tilted (stand straight)")
                score -= 0.1

    visible_count = sum(
        1 for idx in range(33)
        if landmarks.points.get(idx) and landmarks.points[idx][2] >= MIN_LANDMARK_VISIBILITY
    )
    visibility_ratio = visible_count / 33.0
    details["visibility_ratio"] = visibility_ratio
    if visibility_ratio < 0.7:
        issues.append(f"too many landmarks occluded ({visible_count}/33 visible)")
        score -= 0.2

    score = max(0.0, min(1.0, score))

    return PoseQualityReport(
        is_valid=len(issues) < 3 and score >= 0.5,
        score=round(score, 2),
        issues=issues,
        details=details,
    )


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

    nose_to_ankle_cm = height_cm * HEAD_RATIO_DEFAULT
    scale_primary = nose_to_ankle_cm / nose_to_ankle_px

    debug_info = {"primary_scale": scale_primary, "method": "nose_ankle_ratio"}
    
    l_ear = landmarks.get(7)
    r_ear = landmarks.get(8)
    if l_ear is not None and r_ear is not None:
        ear_dist_px = math.dist(l_ear, r_ear)
        if ear_dist_px > 5:
            scale_ears = EAR_TO_EAR_AVG_CM / ear_dist_px
            scale = scale_primary * 0.7 + scale_ears * 0.3
            debug_info = {"primary_scale": scale_primary, "ears_scale": scale_ears, "method": "nose_ankle_ears"}
            return scale, debug_info

    l_eye = landmarks.get(2)
    r_eye = landmarks.get(5)
    if l_eye is not None and r_eye is not None:
        eye_dist_px = math.dist(l_eye, r_eye)
        if eye_dist_px > 5:
            scale_eyes = INTER_PUPILLARY_AVG_CM / eye_dist_px
            scale = scale_primary * 0.8 + scale_eyes * 0.2
            debug_info = {"primary_scale": scale_primary, "eyes_scale": scale_eyes, "method": "nose_ankle_eyes"}
            return scale, debug_info

    return scale_primary, debug_info


def ellipse_circumference(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return max(a, b) * math.pi * 2
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def measure_body_contour_width_at_level(
    image_bytes: bytes,
    landmarks: Landmarks,
    y_level: float,
    scale: float,
    tolerance_px: int = 15,
) -> Optional[float]:
    img = _image_bytes_to_cv2(image_bytes)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    y_pixel = int(y_level * landmarks.img_h)
    y_pixel = max(tolerance_px, min(landmarks.img_h - tolerance_px - 1, y_pixel))
    
    search_start = y_pixel - tolerance_px
    search_end = y_pixel + tolerance_px
    
    method_used = "none"
    best_width = 0
    
    for y in range(search_start, search_end + 1):
        row = gray[y, :]
        
        edges = cv2.Canny(row, 30, 100)
        edge_points = np.where(edges > 0)[0]
        
        if len(edge_points) >= 2:
            left_edge = edge_points[0]
            right_edge = edge_points[-1]
            width = right_edge - left_edge
            if width > best_width:
                best_width = width
                method_used = "canny_row"
    
    if best_width < 15:
        blur = cv2.GaussianBlur(gray, (5, 5))
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        for y in range(search_start, search_end + 1):
            row = binary[y, :]
            body_pixels = np.where(row == 255)[0]
            if len(body_pixels) >= 2:
                width = body_pixels[-1] - body_pixels[0]
                if width > best_width:
                    best_width = width
                    method_used = "otsu_row"
    
    if best_width < 15:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        edges = cv2.Canny(blurred, 30, 100)
        filled = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        
        y_slice = filled[y_pixel, :]
        body_pixels = np.where(filled[y_pixel, :] > 0)[0]
        if len(body_pixels) >= 2:
            best_width = body_pixels[-1] - body_pixels[0]
            method_used = "canny_dilated"
    
    if best_width < 15:
        return None, "none"
    
    return best_width * scale, method_used


def calculate_cv_measurements(
    front: Landmarks,
    side: Landmarks,
    height_cm: float,
    weight_kg: Optional[float] = None,
    gender: str = "unknown",
    front_bytes: Optional[bytes] = None,
    side_bytes: Optional[bytes] = None,
) -> dict[str, dict[str, Any]]:
    front_scale_result = calibrate_scale(front, height_cm)
    side_scale_result = calibrate_scale(side, height_cm)

    if front_scale_result is None:
        logger.warning("Cannot calibrate front photo scale")
        return None

    if isinstance(front_scale_result, tuple):
        front_scale, front_debug = front_scale_result
    else:
        front_scale = front_scale_result
        front_debug = {"method": "nose_ankle_ratio"}

    if isinstance(side_scale_result, tuple):
        side_scale, side_debug = side_scale_result
    else:
        side_scale = side_scale_result
        side_debug = {"method": "nose_ankle_ratio"}

    if side_scale is None:
        side_scale = front_scale
        logger.info("Using front scale for side photo")

    measurements: dict[str, dict[str, Any]] = {}

    body_ratios = MALE_BODY_RATIOS if gender == "male" else (FEMALE_BODY_RATIOS if gender == "female" else None)

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
        belly_frac = 0.60

        chest_level_y = neck_y + torso_h * chest_frac
        under_bust_level_y = neck_y + torso_h * under_bust_frac
        waist_level_y = neck_y + torso_h * waist_frac
        belly_level_y = neck_y + torso_h * belly_frac

        if body_ratios:
            chest_width_factor = 0.78 if gender == "male" else 0.70
            chest_depth_factor = 1.10 if gender == "male" else 1.18
            waist_width_factor = 0.70 if gender == "male" else 0.78
            waist_depth_factor = 0.75 if gender == "male" else 0.82
            belly_width_factor = 0.78 if gender == "male" else 0.85
            belly_depth_factor = 0.80 if gender == "male" else 0.88
        else:
            chest_width_factor = 0.72
            chest_depth_factor = 1.15
            waist_width_factor = 0.82
            waist_depth_factor = 0.85
            belly_width_factor = 0.90
            belly_depth_factor = 0.92

        contour_method = "none"
        
        if front_bytes and side_bytes:
            cw, cw_method = measure_body_contour_width_at_level(
                front_bytes, front, chest_level_y, front_scale
            )
            contour_chest_width = cw if cw else None
            
            cd, cd_method = measure_body_contour_width_at_level(
                side_bytes, side, chest_level_y, side_scale
            )
            contour_chest_depth = cd if cd else None
            
            ww, ww_method = measure_body_contour_width_at_level(
                front_bytes, front, waist_level_y, front_scale
            )
            contour_waist_width = ww if ww else None
            
            wd, wd_method = measure_body_contour_width_at_level(
                side_bytes, side, waist_level_y, side_scale
            )
            contour_waist_depth = wd if wd else None
            
            contour_methods = [m for m in [cw_method, cd_method, ww_method, wd_method] if m != "none"]
            if contour_methods:
                contour_method = contour_methods[0]
        else:
            contour_chest_width = contour_chest_depth = None
            contour_waist_width = contour_waist_depth = None

        if contour_chest_width and contour_chest_width > 5:
            chest_width_cm = contour_chest_width / 2
        elif shoulder_w_px:
            chest_width_cm = shoulder_w_px * front_scale * 0.5 * chest_width_factor
        else:
            chest_width_cm = None

        if contour_chest_depth and contour_chest_depth > 5:
            chest_depth_cm = contour_chest_depth / 2
        elif hip_depth:
            chest_depth_cm = hip_depth * chest_depth_factor
        else:
            chest_depth_cm = None

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

        if contour_waist_width and contour_waist_width > 5:
            waist_width_cm = contour_waist_width / 2
        elif hip_width:
            waist_width_cm = hip_width * waist_width_factor
        else:
            waist_width_cm = None

        if contour_waist_depth and contour_waist_depth > 5:
            waist_depth_cm = contour_waist_depth / 2
        elif hip_depth:
            waist_depth_cm = hip_depth * waist_depth_factor
        else:
            waist_depth_cm = None

        if waist_width_cm and waist_depth_cm:
            waist_circ = ellipse_circumference(waist_width_cm, waist_depth_cm)
        elif waist_width_cm:
            waist_circ = waist_width_cm * math.pi * 1.1
        else:
            waist_circ = None

        if hip_width:
            belly_width_cm = hip_width * belly_width_factor
        else:
            belly_width_cm = None

        if hip_depth:
            belly_depth_cm = hip_depth * belly_depth_factor
        else:
            belly_depth_cm = None

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

    if hip_width and shoulder_width:
        if gender == "male":
            bicep_circ = shoulder_width * 0.22 + hip_width * 0.15
        elif gender == "female":
            bicep_circ = shoulder_width * 0.18 + hip_width * 0.18
        else:
            bicep_circ = (shoulder_width * 0.18 + hip_width * math.pi * 0.32) * 0.5
    else:
        bicep_circ = None

    if gender == "male":
        forearm_circ = bicep_circ * 0.78 if bicep_circ else None
        wrist_circ = bicep_circ * 0.52 if bicep_circ else None
    elif gender == "female":
        forearm_circ = bicep_circ * 0.82 if bicep_circ else None
        wrist_circ = bicep_circ * 0.58 if bicep_circ else None
    else:
        forearm_circ = bicep_circ * 0.84 if bicep_circ else None
        wrist_circ = bicep_circ * 0.55 if bicep_circ else None

    if hip_width and hip_depth:
        if gender == "male":
            upper_thigh_circ = ellipse_circumference(hip_width * 0.58, hip_depth * 0.62)
        elif gender == "female":
            upper_thigh_circ = ellipse_circumference(hip_width * 0.52, hip_depth * 0.68)
        else:
            upper_thigh_circ = ellipse_circumference(hip_width * 0.55, hip_depth * 0.65)
    elif hip_width:
        upper_thigh_circ = hip_width * math.pi * 1.2
    else:
        upper_thigh_circ = None

    if gender == "male":
        knee_circ = upper_thigh_circ * 0.55 if upper_thigh_circ else None
        calf_circ = upper_thigh_circ * 0.60 if upper_thigh_circ else None
    elif gender == "female":
        knee_circ = upper_thigh_circ * 0.52 if upper_thigh_circ else None
        calf_circ = upper_thigh_circ * 0.58 if upper_thigh_circ else None
    else:
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

    debug_output = {
        "scale_factor": round(front_scale, 4),
        "scale_method": front_debug.get("method", "unknown"),
        "contour_method": contour_method,
    }
    
    return measurements, debug_output


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
    gender: str = "unknown",
) -> tuple[dict[str, dict[str, Any]], bool, dict[str, Any]]:
    extra_info: dict[str, Any] = {"pose_issues": [], "pose_score": 0.0}

    if front_bytes and side_bytes and _MP_AVAILABLE:
        front_pre = preprocess_image(front_bytes)
        side_pre = preprocess_image(side_bytes)

        front_lm = run_mediapipe_pose(front_bytes, front_pre)
        side_lm = run_mediapipe_pose(side_bytes, side_pre)

        if front_lm is not None and side_lm is not None:
            front_quality = validate_pose_quality(front_lm, "front")
            side_quality = validate_pose_quality(side_lm, "side")

            extra_info["pose_score"] = round((front_quality.score + side_quality.score) / 2, 2)
            extra_info["pose_issues"] = front_quality.issues + side_quality.issues
            extra_info["front_visibility"] = front_quality.details.get("visibility_ratio", 0)
            extra_info["side_visibility"] = side_quality.details.get("visibility_ratio", 0)

            cv_result = calculate_cv_measurements(
                front_lm, side_lm, height_cm, weight_kg, gender, front_bytes, side_bytes
            )
            if cv_result is not None:
                if isinstance(cv_result, tuple):
                    measurements, debug_info = cv_result
                else:
                    measurements = cv_result
                    debug_info = {"error": "no debug info"}
                
                formula = calculate_formula_measurements(height_cm, weight_kg)
                for key, cv_val in measurements.items():
                    if cv_val["value"] is not None:
                        cv_val["method"] = "cv"
                    else:
                        cv_val["value"] = formula[key]["value"]
                        cv_val["method"] = "formula"
                extra_info.update(debug_info)
                return measurements, True, extra_info
            else:
                logger.info("CV measurement calculation incomplete, using formulas")

        if front_lm is None:
            logger.warning("MediaPipe: no pose detected in front photo")
        if side_lm is None:
            logger.warning("MediaPipe: no pose detected in side photo")

    measurements = calculate_formula_measurements(height_cm, weight_kg)
    return measurements, False, extra_info


def get_confidence(
    cv_detected: bool,
    has_weight: bool,
    pose_score: float = 0.0,
    visibility_ratio: float = 0.0,
    gender_known: bool = False,
) -> float:
    if cv_detected:
        base = 0.75
        base += min(pose_score * 0.08, 0.08)
        base += min(visibility_ratio * 0.07, 0.07)
        if has_weight:
            base += 0.03
        if gender_known:
            base += 0.02
        return round(min(base, 0.95), 2)
    else:
        base = 0.55
        if has_weight:
            base += 0.10
        return round(min(base, 0.65), 2)