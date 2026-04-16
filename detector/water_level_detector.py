"""
WaterLevelDetector v6

Key fix: Object detection pre-check using MobileNet SSD.
  - If the image contains a book, laptop, phone, person, etc. → reject immediately
  - Only proceed to water detection if no non-water objects are detected
    with high confidence, OR if a container-like object (bottle, bowl) is found

Pipeline:
  Step 1 — Object detection pre-check (reject books, screens, people, etc.)
  Step 2 — Validate (reject blank/uniform images)
  Step 3 — Isolate container ROI if detected
  Step 4 — Remove sky
  Step 5 — Build water mask
  Step 6 — Find surface bottom-up
  Step 7 — Calculate level
"""

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Object Validator — rejects non-water images
# ─────────────────────────────────────────────

class ObjectValidator:
    # Objects that clearly mean this is NOT a water image
    REJECT_CLASSES = {
        "book", "tvmonitor", "laptop", "person", "car", "bus",
        "motorbike", "bicycle", "aeroplane", "train", "cat",
        "dog", "horse", "sheep", "cow", "bird", "sofa",
        "chair", "diningtable", "pottedplant"
    }

    # Objects that confirm a water container — always allow
    CONTAINER_CLASSES = {"bottle", "bowl", "boat"}

    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    def __init__(self):
        model_path  = "models/mobilenet_iter_73000.caffemodel"
        config_path = "models/deploy.prototxt"

        if os.path.exists(model_path) and os.path.exists(config_path):
            self.net     = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.enabled = True
            logger.info("ObjectValidator | MobileNet SSD loaded")
        else:
            logger.warning("ObjectValidator | model not found, running in fallback mode")
            self.enabled = False

    def check(self, image: np.ndarray) -> tuple:
        """
        Returns (allowed: bool, reason: str)
          allowed=True  → proceed with water detection
          allowed=False → reject with user-friendly message
        """
        if not self.enabled:
            return True, "OK"

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        found_container = False
        rejected_label  = None
        highest_conf    = 0.0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.45:
                continue

            idx   = int(detections[0, 0, i, 1])
            label = self.CLASSES[idx] if idx < len(self.CLASSES) else "unknown"
            logger.debug("ObjectValidator | detected=%s conf=%.2f", label, confidence)

            if label in self.CONTAINER_CLASSES:
                found_container = True

            if label in self.REJECT_CLASSES and confidence > highest_conf:
                highest_conf   = confidence
                rejected_label = label

        # Container detected → always allow
        if found_container:
            logger.info("ObjectValidator | container detected → allowed")
            return True, "OK"

        # Non-water object detected → reject
        if rejected_label:
            friendly = {
                "book":        "a book or document",
                "tvmonitor":   "a screen or monitor",
                "laptop":      "a laptop or screen",
                "person":      "a person",
                "car":         "a vehicle",
                "bus":         "a vehicle",
                "motorbike":   "a vehicle",
                "bicycle":     "a bicycle",
                "chair":       "furniture",
                "sofa":        "furniture",
                "diningtable": "furniture",
                "pottedplant": "a plant",
                "cat":         "an animal",
                "dog":         "an animal",
                "bird":        "an animal",
            }.get(rejected_label, f"a {rejected_label}")

            msg = (
                f"This image appears to contain {friendly}, not water. "
                f"Please upload a photo of a water body, tank, river, or glass of water."
            )
            logger.warning("ObjectValidator | rejected=%s conf=%.2f", rejected_label, highest_conf)
            return False, msg

        return True, "OK"

    def get_container_box(self, image: np.ndarray):
        """Returns bounding box of the best detected container, or None."""
        if not self.enabled:
            return None

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < 0.35:
                continue
            idx   = int(detections[0, 0, i, 1])
            label = self.CLASSES[idx] if idx < len(self.CLASSES) else "unknown"
            if label in self.CONTAINER_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                return box.astype("int")
        return None


# ─────────────────────────────────────────────
#  Sky Remover
# ─────────────────────────────────────────────

class SkyRemover:
    SKY_TRIM_RATIO = 0.30

    def remove(self, image: np.ndarray):
        h          = image.shape[0]
        sky_cutoff = int(h * self.SKY_TRIM_RATIO)
        trimmed    = image.copy()
        trimmed[:sky_cutoff, :] = 0
        logger.debug("SkyRemover | cutoff_y=%d", sky_cutoff)
        return trimmed, sky_cutoff


# ─────────────────────────────────────────────
#  Image Validator (basic sanity only)
# ─────────────────────────────────────────────

class ImageValidator:
    MIN_STD_DEV = 8.0

    def is_likely_water_image(self, image: np.ndarray) -> tuple:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        if std_dev < self.MIN_STD_DEV:
            return False, "Image is too uniform or blank. Please upload a real photo."
        return True, "OK"


# ─────────────────────────────────────────────
#  Water Mask Builder
# ─────────────────────────────────────────────

class WaterMaskBuilder:
    def build(self, image: np.ndarray) -> np.ndarray:
        hsv              = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Strategy A: Blue-green water
        mask_a = cv2.inRange(hsv, np.array([85, 35, 35]), np.array([135, 255, 255]))

        # Strategy B: Grey/white water (glass of water, clear liquid)
        mask_b = ((s_ch < 90) & (v_ch > 40) & (v_ch < 230)).astype(np.uint8) * 255

        # Strategy C: Dark/deep water
        mask_c = ((s_ch < 60) & (v_ch < 80)).astype(np.uint8) * 255

        combined = cv2.bitwise_or(mask_a, mask_b)
        combined = cv2.bitwise_or(combined, mask_c)

        kernel   = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        logger.debug("WaterMaskBuilder | water px=%d", np.count_nonzero(combined))
        return combined


# ─────────────────────────────────────────────
#  Bottom-Up Surface Finder
# ─────────────────────────────────────────────

class BottomUpSurfaceFinder:
    MIN_ROW_FILL = 0.10
    MAX_GAP_ROWS = 15

    def find_surface(self, mask: np.ndarray, sky_cutoff: int):
        h, w   = mask.shape
        min_px = int(w * self.MIN_ROW_FILL)

        bottom_zone = mask[int(h * 0.6):, :]
        if np.count_nonzero(bottom_zone) < (w * 0.05):
            logger.warning("BottomUpFinder | no water in bottom 40%% of image")
            return None

        in_water  = False
        gap_count = 0
        surface_y = None

        for y in range(h - 1, sky_cutoff, -1):
            is_water_row = np.count_nonzero(mask[y, :]) >= min_px

            if is_water_row:
                in_water  = True
                gap_count = 0
                surface_y = y
            elif in_water:
                gap_count += 1
                if gap_count > self.MAX_GAP_ROWS:
                    logger.debug("BottomUpFinder | gap exceeded at y=%d", y)
                    break

        logger.debug("BottomUpFinder | surface_y=%s", surface_y)
        return surface_y


# ─────────────────────────────────────────────
#  Main Detector
# ─────────────────────────────────────────────

class WaterLevelDetector:
    TOTAL_HEIGHT_METERS = 5.0

    LEVEL_THRESHOLDS = [
        (0.80, (0, 0, 220),   "DANGER",  "red"),
        (0.50, (0, 165, 255), "WARNING", "orange"),
        (0.00, (34, 197, 94), "SAFE",    "green"),
    ]

    def __init__(self):
        self.obj_validator  = ObjectValidator()
        self.validator      = ImageValidator()
        self.sky_remover    = SkyRemover()
        self.mask_builder   = WaterMaskBuilder()
        self.surface_finder = BottomUpSurfaceFinder()

    def detect(self, image: np.ndarray):
        logger.info("Detection start | shape=%s", image.shape)

        # Step 1 — Object detection pre-check
        allowed, reason = self.obj_validator.check(image)
        if not allowed:
            return None, reason

        # Step 2 — Basic sanity check
        is_valid, reason = self.validator.is_likely_water_image(image)
        if not is_valid:
            return None, reason

        # Step 3 — Isolate container ROI if detected
        height        = image.shape[0]
        container_box = self.obj_validator.get_container_box(image)

        if container_box is not None:
            startX, startY, endX, endY = container_box
            roi      = image[max(0, startY):min(height, endY),
                             max(0, startX):min(image.shape[1], endX)]
            offset_y = startY
            logger.info("Container ROI | box=%s", container_box)
        else:
            roi      = image
            offset_y = 0

        # Step 4 — Sky removal
        preprocessed, sky_cutoff = self.sky_remover.remove(roi)

        # Step 5 — Water mask
        water_mask = self.mask_builder.build(preprocessed)

        # Step 6 — Surface detection (bottom-up)
        water_y = self.surface_finder.find_surface(water_mask, sky_cutoff)

        if water_y is None:
            return None, (
                "No water surface detected. Please upload a clear photo of a "
                "water body, tank, river, or glass of water."
            )

        water_y += offset_y  # adjust for ROI crop

        # Step 7 — Calculate level
        level_percent = max(0.0, min(1.0, (height - water_y) / height))
        level_meters  = round(level_percent * self.TOTAL_HEIGHT_METERS, 2)
        status, color_bgr = self._classify_level(level_percent)

        logger.info(
            "Result | water_y=%d | level=%.2fm (%.1f%%) | status=%s",
            water_y, level_meters, level_percent * 100, status
        )

        return {
            "water_y":       int(water_y),
            "level_meters":  level_meters,
            "level_percent": round(level_percent * 100, 1),
            "status":        status,
            "color_bgr":     color_bgr,
            "image_height":  height,
            "sky_cutoff":    sky_cutoff,
        }, None

    def _classify_level(self, level_percent: float):
        for threshold, color, label, _ in self.LEVEL_THRESHOLDS:
            if level_percent >= threshold:
                return label, color
        return "SAFE", (34, 197, 94)