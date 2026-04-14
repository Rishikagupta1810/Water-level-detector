"""
ImageProcessor: Handles image loading, resizing (fit-to-frame),
annotation drawing (water line, scale bar, legend), and saving.
Decoupled from detection logic and Flask.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Target frame size — all uploaded images are resized to this for consistency
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class ImageProcessor:
    """
    Responsible for image I/O and visual annotation.
    Does NOT perform detection — takes detection results as input.
    """

    def load_and_fit(self, img_path: str) -> np.ndarray | None:
        """
        Loads an image and resizes it to fit within FRAME dimensions
        while preserving aspect ratio (letterboxed on black background).
        This ensures scale consistency regardless of upload size.
        """
        image = cv2.imread(img_path)
        if image is None:
            logger.error("Failed to load image from path: %s", img_path)
            return None

        original_h, original_w = image.shape[:2]
        logger.info("Loaded image: path=%s, original_size=(%dx%d)", img_path, original_w, original_h)

        scale = min(FRAME_WIDTH / original_w, FRAME_HEIGHT / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact frame size (letterbox)
        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        x_offset = (FRAME_WIDTH - new_w) // 2
        y_offset = (FRAME_HEIGHT - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        logger.info("Resized to fit frame: scale=%.3f, padded_to=(%dx%d)", scale, FRAME_WIDTH, FRAME_HEIGHT)
        return canvas

    def annotate(self, image: np.ndarray, detection: dict) -> np.ndarray:
        """
        Draws:
        - Horizontal water level line (color-coded)
        - Scale bar on left edge (0m at bottom, 5m at top)
        - Level label near the line
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        water_y = detection["water_y"]
        level_meters = detection["level_meters"]
        color = detection["color_bgr"]
        status = detection["status"]

        # --- Water level line ---
        cv2.line(annotated, (0, water_y), (w, water_y), color, 3)
        cv2.putText(
            annotated,
            f"{level_meters}m  [{status}]",
            (60, max(water_y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        logger.debug("Drew water line at y=%d with color=%s", water_y, color)

        # --- Scale bar (left edge) ---
        self._draw_scale_bar(annotated, h, detection["level_meters"])

        return annotated

    def _draw_scale_bar(self, image: np.ndarray, img_height: int, level_meters: float):
        """
        Draws a vertical scale bar on the left side of the image.
        Ticks at every 0.5m. Red zone above 4m.
        """
        bar_x = 30
        top_y = 10
        bot_y = img_height - 10
        total_m = 5.0

        # Background strip
        cv2.rectangle(image, (bar_x - 10, top_y), (bar_x + 10, bot_y), (30, 30, 30), -1)

        # Filled level (color from 0 to current level)
        fill_y = int(bot_y - (level_meters / total_m) * (bot_y - top_y))
        cv2.rectangle(image, (bar_x - 8, fill_y), (bar_x + 8, bot_y), (34, 197, 94), -1)

        # Danger zone fill (above 4m)
        danger_y = int(bot_y - (4.0 / total_m) * (bot_y - top_y))
        if fill_y < danger_y:
            cv2.rectangle(image, (bar_x - 8, fill_y), (bar_x + 8, danger_y), (0, 0, 220), -1)

        # Tick marks and labels every 0.5m
        for i in range(11):  # 0.0 to 5.0 in 0.5 steps
            m = i * 0.5
            tick_y = int(bot_y - (m / total_m) * (bot_y - top_y))
            cv2.line(image, (bar_x - 12, tick_y), (bar_x + 12, tick_y), (200, 200, 200), 1)
            if i % 2 == 0:  # label every 1m
                cv2.putText(image, f"{m:.0f}m", (bar_x + 15, tick_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        logger.debug("Drew scale bar. fill up to %.2fm, danger zone above 4m", level_meters)

    def save(self, image: np.ndarray, path: str):
        success = cv2.imwrite(path, image)
        if success:
            logger.info("Saved annotated image to: %s", path)
        else:
            logger.error("Failed to save image to: %s", path)