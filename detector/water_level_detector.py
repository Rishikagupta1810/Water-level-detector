"""
River Water Level Detector — rewritten for real river images.

Strategy:
  - Detects the WATER-LAND BOUNDARY (waterline) using color segmentation
  - Waterline position in the image determines the level
  - Higher waterline (closer to top) = higher flood risk = DANGER
  - Lower waterline (closer to bottom) = normal = SAFE
  - Calibration: image height maps to TOTAL_HEIGHT_METERS (default 5m)
"""

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

TOTAL_HEIGHT_METERS = 5.0


class ImageValidator:
    def is_likely_water_image(self, image: np.ndarray):
        if image is None or image.size == 0:
            return False, "Image is empty or unreadable."
        h, w = image.shape[:2]
        if h < 40 or w < 40:
            return False, "Image is too small to analyse."
        return True, None


class RiverWaterlineDetector:
    """
    Detects the TOP edge of the water body in a river image.

    Key insight for rivers:
      - Water occupies the LOWER portion of the image
      - Land/banks/trees occupy the UPPER portion
      - The waterline is the TOPMOST boundary of the water region
      - If waterline is HIGH in image  → flooding → DANGER
      - If waterline is LOW in image   → normal level → SAFE
    """

    # HSV ranges tuned for river/lake water
    WATER_RANGES = [
        # Blue river water (deep/clear)
        (np.array([90,  25,  25]), np.array([130, 255, 255])),
        # Teal/green-blue river water
        (np.array([75,  20,  20]), np.array([100, 220, 220])),
        # Grey/silver water (overcast sky reflection)
        (np.array([0,    0,  80]), np.array([180,  30, 200])),
        # Muddy brown river water
        (np.array([8,   40,  40]), np.array([22,  200, 180])),
        # Dark murky water
        (np.array([85,  10,  10]), np.array([135,  80, 120])),
    ]

    def create_water_mask(self, image: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.WATER_RANGES:
            mask |= cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        return mask

    def find_waterline(self, image: np.ndarray) -> tuple:
        """
        Returns (waterline_y, water_mask, confidence).
        waterline_y = topmost row index of water body.
        """
        h, w = image.shape[:2]
        mask = self.create_water_mask(image)

        water_coverage = np.count_nonzero(mask) / (h * w)
        logger.info("Water pixel coverage: %.1f%%", water_coverage * 100)

        if water_coverage < 0.03:
            logger.warning("Very little water detected (%.1f%%)", water_coverage * 100)
            return None, mask, 0.0

        min_water_cols = max(10, int(w * 0.20))
        waterline_y = None

        # Find topmost row with continuous band of water pixels
        for y in range(h):
            if np.count_nonzero(mask[y]) >= min_water_cols:
                waterline_y = y
                break

        if waterline_y is None:
            row_sums = np.array([np.count_nonzero(mask[y]) for y in range(h)])
            waterline_y = int(np.argmax(row_sums))

        confidence = min(1.0, water_coverage * 3.0)
        logger.info("Waterline at y=%d (%.1f%% from top), confidence=%.2f",
                    waterline_y, (waterline_y / h) * 100, confidence)

        return waterline_y, mask, confidence


class ClarityAnalyser:
    def analyse(self, image: np.ndarray) -> str:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hue = float(np.mean(hsv[:, :, 0]))
        mean_sat = float(np.mean(hsv[:, :, 1]))
        mean_val = float(np.mean(hsv[:, :, 2]))

        if 8 <= mean_hue <= 25 and mean_sat > 35:
            return "muddy"
        if mean_sat < 25 and mean_val > 160:
            return "clear"
        if 85 <= mean_hue <= 130:
            return "clear"
        if 60 <= mean_hue < 85:
            return "murky"
        return "clear"


class WaterLevelDetector:
    """
    River water level detector.

    Level logic:
      waterline_y = top of water in image (row index from top)
      level_fraction = (image_height - waterline_y) / image_height

      waterline near TOP   → level_fraction ≈ 1.0 → DANGER (flooding)
      waterline at middle  → level_fraction ≈ 0.5 → WARNING
      waterline near BOTTOM → level_fraction ≈ 0.2 → SAFE (normal river)
    """

    LEVEL_THRESHOLDS = [
        (0.75, (0, 0, 220),   "DANGER"),
        (0.45, (0, 165, 255), "WARNING"),
        (0.00, (34, 197, 94), "SAFE"),
    ]

    def __init__(self):
        self.validator = ImageValidator()
        self.detector  = RiverWaterlineDetector()
        self.clarity   = ClarityAnalyser()

    def _classify(self, level_fraction: float):
        for threshold, color, label in self.LEVEL_THRESHOLDS:
            if level_fraction >= threshold:
                return label, color
        return "SAFE", (34, 197, 94)

    def detect(self, image: np.ndarray):
        is_valid, reason = self.validator.is_likely_water_image(image)
        if not is_valid:
            return None

        h = image.shape[0]
        waterline_y, water_mask, confidence = self.detector.find_waterline(image)

        if waterline_y is None:
            return None

        level_fraction = (h - waterline_y) / h
        level_fraction = max(0.0, min(1.0, level_fraction))

        level_meters = round(level_fraction * TOTAL_HEIGHT_METERS, 2)
        status, color_bgr = self._classify(level_fraction)
        clarity = self.clarity.analyse(image)

        logger.info(
            "Result: waterline_y=%d, level=%.1f%%, %.2fm, status=%s, clarity=%s",
            waterline_y, level_fraction * 100, level_meters, status, clarity
        )

        return {
            "water_y":       int(waterline_y),
            "level_meters":  level_meters,
            "level_percent": round(level_fraction * 100, 1),
            "status":        status,
            "clarity":       clarity,
            "color_bgr":     color_bgr,
            "image_height":  h,
            "confidence":    round(confidence * 100, 1),
        }