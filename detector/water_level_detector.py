import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class ObjectValidator:
    def __init__(self):
        # Paths to the pre-trained model files
        model_path = "models/mobilenet_iter_73000.caffemodel"
        config_path = "models/deploy.prototxt"
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self.enabled = True
        else:
            logger.warning("Model files not found. Object detection running in fallback mode.")
            self.enabled = False

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def get_container_box(self, image):
        if not self.enabled: return None
            
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                # Filter for container-like objects
                if self.CLASSES[idx] in ["bottle", "pottedplant", "boat", "diningtable"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    return box.astype("int")
        return None

class WaterLevelDetector:
    TOTAL_HEIGHT_METERS = 5.0
    LEVEL_THRESHOLDS = [
        (0.80, (0, 0, 220),   "DANGER",  "red"),
        (0.50, (0, 165, 255), "WARNING", "orange"),
        (0.00, (34, 197, 94), "SAFE",    "green"),
    ]

    def __init__(self):
        from .water_level_detector import SkyRemover, ImageValidator, ColorWaterDetector, TextureWaterDetector
        self.sky_remover      = SkyRemover()
        self.validator        = ImageValidator()
        self.color_detector   = ColorWaterDetector()
        self.texture_detector = TextureWaterDetector()
        self.obj_validator    = ObjectValidator()

    def _classify_level(self, level_percent: float):
        for threshold, color, label, _ in self.LEVEL_THRESHOLDS:
            if level_percent >= threshold:
                return label, color
        return "SAFE", (34, 197, 94)

    def detect(self, image: np.ndarray):
        height = image.shape[0]
        
        # 1. Image Validation
        is_valid, reason = self.validator.is_likely_water_image(image)
        if not is_valid: return None, reason

        # 2. Object detection to isolate the container
        container_box = self.obj_validator.get_container_box(image)
        
        # If no container, we use the full image but log a warning
        if container_box is not None:
            startX, startY, endX, endY = container_box
            roi = image[max(0, startY):min(height, endY), max(0, startX):min(image.shape[1], endX)]
            offset_y = startY
        else:
            roi = image
            offset_y = 0

        # 3. Pipeline
        preprocessed, sky_cutoff = self.sky_remover.remove(roi)
        candidates = {}

        y_color = self.color_detector.find_surface(preprocessed, sky_cutoff)
        if y_color is not None: candidates["color"] = y_color + offset_y

        y_texture = self.texture_detector.find_surface(preprocessed, sky_cutoff)
        if y_texture is not None: candidates["texture"] = y_texture + offset_y

        if not candidates:
            return None, "No water detected. Ensure the container is clearly visible."

        # 4. Strategy Merging
        if len(candidates) == 2:
            diff = abs(candidates["color"] - candidates["texture"])
            water_y = int((candidates["color"] + candidates["texture"]) / 2) if diff <= 40 else max(candidates["color"], candidates["texture"])
        else:
            water_y = list(candidates.values())[0]

        # 5. Final Calculations
        level_percent = (height - water_y) / height
        level_meters  = round(level_percent * self.TOTAL_HEIGHT_METERS, 2)
        status, color_bgr = self._classify_level(level_percent)

        return {
            "water_y":       int(water_y),
            "level_meters":  level_meters,
            "level_percent": round(level_percent * 100, 1),
            "status":        status,
            "color_bgr":     color_bgr,
            "image_height":  height
        }, None
        