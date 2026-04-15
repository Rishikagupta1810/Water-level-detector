"""
Water Level Detector — fixed module.

Changes vs original:
  - Removed circular self-import (was crashing on Render)
  - SkyRemover, ImageValidator, ColorWaterDetector, TextureWaterDetector
    are now defined in THIS file (no separate import needed)
  - Added clarity detection (muddy vs clear water) using HSV + blur variance
  - No size/scale estimation — works correctly for glass-scale containers
"""

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Helper classes (were previously imported
#  from self — now defined here directly)
# ─────────────────────────────────────────────

class SkyRemover:
    """
    Crops out the top portion of the image that is likely sky or background.
    For glass-scale images this effectively does nothing harmful (sky_cutoff ~ 0).
    """
    SKY_RATIO = 0.15  # assume top 15% might be background/rim

    def remove(self, image: np.ndarray):
        h = image.shape[0]
        sky_cutoff = int(h * self.SKY_RATIO)
        preprocessed = image[sky_cutoff:].copy()
        return preprocessed, sky_cutoff


class ImageValidator:
    """Basic sanity checks before running the detection pipeline."""

    def is_likely_water_image(self, image: np.ndarray):
        if image is None or image.size == 0:
            return False, "Image is empty or unreadable."
        h, w = image.shape[:2]
        if h < 40 or w < 40:
            return False, "Image is too small to analyse."
        return True, None


class ColorWaterDetector:
    """
    Finds the water surface using colour segmentation.
    Detects both clear water (blue/cyan hues) and muddy water (brown/yellow hues).
    Works at any scale — no size estimation involved.
    """

    # HSV ranges: (lower, upper)
    WATER_RANGES = [
        # Clear / blue water
        (np.array([85,  30,  30]), np.array([130, 255, 255])),
        # Murky / teal-green water
        (np.array([60,  20,  20]), np.array([90,  200, 200])),
        # Muddy / brown-yellow water
        (np.array([10,  40,  40]), np.array([30,  220, 200])),
        # White / very pale water in a glass (low saturation, high value)
        (np.array([0,   0,  180]), np.array([180,  40, 255])),
    ]

    def find_surface(self, image: np.ndarray, sky_cutoff: int):
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for (lo, hi) in self.WATER_RANGES:
            mask |= cv2.inRange(hsv, lo, hi)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Find the topmost row that has significant water pixels
        h, w = mask.shape
        min_water_cols = max(5, int(w * 0.12))  # at least 12% of width
        for y in range(h):
            if np.count_nonzero(mask[y]) >= min_water_cols:
                return y + sky_cutoff

        return None


class TextureWaterDetector:
    """
    Finds the water surface using edge/texture analysis.
    Water surfaces typically show a horizontal edge band.
    """

    def find_surface(self, image: np.ndarray, sky_cutoff: int):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)

        h, w = edges.shape
        min_edge_px = max(5, int(w * 0.10))

        for y in range(h):
            if np.count_nonzero(edges[y]) >= min_edge_px:
                return y + sky_cutoff

        return None


class ClarityAnalyser:
    """
    Determines whether water is CLEAR or MUDDY.
    Uses laplacian variance (sharpness proxy) and hue analysis.
    No scale/size estimation — just colour and texture properties.
    """

    def analyse(self, image: np.ndarray) -> str:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        mean_hue = float(np.mean(h_channel))
        mean_sat = float(np.mean(s_channel))
        mean_val = float(np.mean(v_channel))

        # Muddy water: brownish hue (10–30°), moderate saturation
        if 10 <= mean_hue <= 30 and mean_sat > 30:
            return "muddy"

        # Very pale / white water in glass: low saturation, high brightness
        if mean_sat < 25 and mean_val > 170:
            return "clear"

        # Blue/teal — clear water
        if 85 <= mean_hue <= 130:
            return "clear"

        # Greenish — slightly murky
        if 60 <= mean_hue < 85:
            return "murky"

        return "clear"  # default


# ─────────────────────────────────────────────
#  Object / container detector (unchanged)
# ─────────────────────────────────────────────

class ObjectValidator:
    def __init__(self):
        model_path  = "models/mobilenet_iter_73000.caffemodel"
        config_path = "models/deploy.prototxt"

        if os.path.exists(model_path) and os.path.exists(config_path):
            self.net     = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.enabled = True
        else:
            logger.warning("Model files not found. Object detection running in fallback mode.")
            self.enabled = False

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def get_container_box(self, image: np.ndarray):
        if not self.enabled:
            return None
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] in ["bottle", "pottedplant", "boat", "diningtable"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    return box.astype("int")
        return None


# ─────────────────────────────────────────────
#  Main detector
# ─────────────────────────────────────────────

class WaterLevelDetector:
    TOTAL_HEIGHT_METERS = 5.0
    LEVEL_THRESHOLDS = [
        (0.80, (0, 0, 220),   "DANGER",  "red"),
        (0.50, (0, 165, 255), "WARNING", "orange"),
        (0.00, (34, 197, 94), "SAFE",    "green"),
    ]

    def __init__(self):
        # All classes defined above — no import needed
        self.sky_remover      = SkyRemover()
        self.validator        = ImageValidator()
        self.color_detector   = ColorWaterDetector()
        self.texture_detector = TextureWaterDetector()
        self.clarity_analyser = ClarityAnalyser()
        self.obj_validator    = ObjectValidator()

    def _classify_level(self, level_percent: float):
        for threshold, color, label, _ in self.LEVEL_THRESHOLDS:
            if level_percent >= threshold:
                return label, color
        return "SAFE", (34, 197, 94)

    def detect(self, image: np.ndarray):
        height = image.shape[0]

        # 1. Image validation
        is_valid, reason = self.validator.is_likely_water_image(image)
        if not is_valid:
            return None, reason

        # 2. Try to isolate container (falls back to full image for glass scale)
        container_box = self.obj_validator.get_container_box(image)
        if container_box is not None:
            startX, startY, endX, endY = container_box
            roi      = image[max(0, startY):min(height, endY),
                             max(0, startX):min(image.shape[1], endX)]
            offset_y = startY
        else:
            roi      = image
            offset_y = 0

        # 3. Clarity analysis (muddy / clear / murky)
        clarity = self.clarity_analyser.analyse(roi)

        # 4. Detection pipeline
        preprocessed, sky_cutoff = self.sky_remover.remove(roi)
        candidates = {}

        y_color = self.color_detector.find_surface(preprocessed, sky_cutoff)
        if y_color is not None:
            candidates["color"] = y_color + offset_y

        y_texture = self.texture_detector.find_surface(preprocessed, sky_cutoff)
        if y_texture is not None:
            candidates["texture"] = y_texture + offset_y

        if not candidates:
            return None, "No water detected. Make sure the container is clearly visible and well-lit."

        # 5. Merge candidates
        if len(candidates) == 2:
            diff     = abs(candidates["color"] - candidates["texture"])
            water_y  = int((candidates["color"] + candidates["texture"]) / 2) \
                       if diff <= 40 else max(candidates["color"], candidates["texture"])
        else:
            water_y = list(candidates.values())[0]

        # 6. Level calculation (no physical scale — works for any container size)
        level_percent = (height - water_y) / height
        level_percent = max(0.0, min(1.0, level_percent))   # clamp to [0, 1]
        level_meters  = round(level_percent * self.TOTAL_HEIGHT_METERS, 2)
        status, color_bgr = self._classify_level(level_percent)

        return {
            "water_y":       int(water_y),
            "level_meters":  level_meters,
            "level_percent": round(level_percent * 100, 1),
            "status":        status,
            "clarity":       clarity,          # "clear" | "murky" | "muddy"
            "color_bgr":     color_bgr,
            "image_height":  height,
        }, None