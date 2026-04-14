"""
WaterLevelDetector: Detects water surface using HSV color segmentation.

Preprocessing pipeline:
  Step 1 — Sky trim: blacks out top SKY_TRIM_RATIO of image (default 50%)
            Raised from 35% → 50% because residual sky-blue pixels in the
            upper-middle frame were being mis-detected as the water surface,
            producing falsely high water level readings.
  Step 2 — HSV masking: isolates blue-green water color range
  Step 3 — Morphological cleanup: removes noise specks
  Step 4 — Surface scan: finds topmost row with enough water pixels,
            validated across SURFACE_CONFIRM_ROWS consecutive rows to
            reject single-row noise hits (e.g. sky reflections, glare).
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SkyRemover:
    """
    Responsible ONLY for trimming the sky region from an image.
    Blacks out the top portion so sky-blue pixels don't
    get confused with water during color segmentation.

    SKY_TRIM_RATIO raised to 0.50 — in typical river/flood camera
    setups the sky occupies the top half of the frame. 0.35 was too
    shallow and left residual sky-blue pixels that were mis-classified
    as water, pulling the detected surface upward (falsely high reading).
    """

    # Fraction of image height to black out from the top.
    # Raised from 0.35 → 0.50 to eliminate residual sky bleed.
    SKY_TRIM_RATIO = 0.50

    def remove(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a copy of the image with the top SKY_TRIM_RATIO blacked out.
        Original image is NOT modified.
        """
        h = image.shape[0]
        sky_cutoff = int(h * self.SKY_TRIM_RATIO)

        trimmed = image.copy()
        trimmed[:sky_cutoff, :] = 0   # black out top rows

        logger.debug(
            "Sky removed | cutoff_y=%d (top %.0f%% of %dpx height)",
            sky_cutoff, self.SKY_TRIM_RATIO * 100, h
        )
        return trimmed, sky_cutoff


class WaterLevelDetector:
    TOTAL_HEIGHT_METERS = 5.0

    # HSV range for water (blue-green tones, hue 85-135)
    WATER_HSV_LOWER = np.array([85, 40, 40])
    WATER_HSV_UPPER = np.array([135, 255, 255])

    # Minimum % of row width that must be water-colored to count as water surface
    MIN_ROW_FILL = 0.15

    # How many consecutive qualifying rows must be found before accepting
    # a surface detection. Prevents single noisy rows (sky reflections,
    # glare, floating debris) from triggering a false surface hit.
    SURFACE_CONFIRM_ROWS = 3

    LEVEL_THRESHOLDS = [
        (0.80, (0, 0, 220),   "DANGER",  "red"),
        (0.50, (0, 165, 255), "WARNING", "orange"),
        (0.00, (34, 197, 94), "SAFE",    "green"),
    ]

    def __init__(self):
        self.sky_remover = SkyRemover()   # injected as dependency

    def detect(self, image: np.ndarray):
        logger.info("Starting detection | image shape=%s", image.shape)

        # ── Step 1: Remove sky ──────────────────────────────────
        preprocessed, sky_cutoff = self.sky_remover.remove(image)
        logger.info("Preprocessing done | sky blacked out above y=%d", sky_cutoff)

        # ── Step 2: HSV water color mask ────────────────────────
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        water_mask = cv2.inRange(hsv, self.WATER_HSV_LOWER, self.WATER_HSV_UPPER)

        # ── Step 3: Morphological cleanup (remove noise) ────────
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        logger.debug("Water mask | %d water pixels / %d total",
                     np.count_nonzero(water_mask), water_mask.size)

        # ── Step 4: Find water surface ──────────────────────────
        water_y = self._find_water_surface(water_mask, sky_cutoff)

        if water_y is None:
            logger.warning("No water surface found after sky removal.")
            return None

        height = image.shape[0]
        level_percent = (height - water_y) / height
        level_meters  = round(level_percent * self.TOTAL_HEIGHT_METERS, 2)
        status, color_bgr = self._classify_level(level_percent)

        logger.info("Water surface at y=%d | level=%.2fm (%.1f%%) | status=%s",
                    water_y, level_meters, level_percent * 100, status)

        return {
            "water_y":       water_y,
            "level_meters":  level_meters,
            "level_percent": round(level_percent * 100, 1),
            "status":        status,
            "color_bgr":     color_bgr,
            "image_height":  height,
            "sky_cutoff":    sky_cutoff,
            "water_mask":    water_mask,
        }

    def _find_water_surface(self, mask: np.ndarray, start_y: int):
        """
        Scans rows from sky_cutoff downward (skips blacked-out sky rows).

        A row qualifies when >= MIN_ROW_FILL of its pixels are water-colored.
        To avoid false triggers from isolated noisy rows (glare, reflections,
        floating debris), SURFACE_CONFIRM_ROWS consecutive qualifying rows
        must be found before the first one is accepted as the true surface.

        Returns the y-coordinate of the first qualifying row in the confirmed
        run, or None if no valid surface is found.
        """
        h, w = mask.shape
        min_pixels = int(w * self.MIN_ROW_FILL)
        consecutive = 0     # count of back-to-back qualifying rows
        candidate_y = None  # y of the first row in the current run

        for y in range(start_y, h):   # start AFTER sky cutoff
            if np.count_nonzero(mask[y, :]) >= min_pixels:
                if consecutive == 0:
                    candidate_y = y   # mark start of potential surface
                consecutive += 1

                if consecutive >= self.SURFACE_CONFIRM_ROWS:
                    logger.debug(
                        "Water surface confirmed at y=%d "
                        "(%d consecutive qualifying rows)",
                        candidate_y, consecutive
                    )
                    return candidate_y
            else:
                # Reset — isolated qualifying rows are noise
                if consecutive > 0:
                    logger.debug(
                        "Rejected noise run at y=%d (only %d consecutive rows, "
                        "need %d)",
                        candidate_y, consecutive, self.SURFACE_CONFIRM_ROWS
                    )
                consecutive = 0
                candidate_y = None

        return None

    def _classify_level(self, level_percent: float):
        for threshold, color, label, _ in self.LEVEL_THRESHOLDS:
            if level_percent >= threshold:
                return label, color
        return "SAFE", (34, 197, 94)