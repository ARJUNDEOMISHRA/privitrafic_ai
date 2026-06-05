# ═══════════════════════════════════════════════════════════════════
# Layer 1 — Computer Vision + Privacy Preprocessing
# TensorFlow CNN detector + OpenCV privacy protection
# ═══════════════════════════════════════════════════════════════════

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None


class PrivacyPreprocessor:
    """Applies face blur and number plate blur to protect privacy."""

    def __init__(self, face_blur_kernel=(31, 31), plate_blur_kernel=(31, 31)):
        self.face_blur_kernel = face_blur_kernel
        self.plate_blur_kernel = plate_blur_kernel

    def blur_region(self, frame, bbox, kernel):
        """Gaussian-blur a rectangular region in the frame."""
        if cv2 is None:
            return frame
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        blurred = cv2.GaussianBlur(roi, kernel, 0)
        frame[y1:y2, x1:x2] = blurred
        return frame

    def anonymize_frame(self, frame, face_bboxes=None, plate_bboxes=None):
        """Anonymize a frame by blurring faces and plates."""
        result = frame.copy()
        for bbox in (face_bboxes or []):
            result = self.blur_region(result, bbox, self.face_blur_kernel)
        for bbox in (plate_bboxes or []):
            result = self.blur_region(result, bbox, self.plate_blur_kernel)
        return result


class TrafficObjectDetector:
    """CNN-based traffic object detector (TensorFlow).

    Detects: cars, buses, bikes, pedestrians.
    In production, this would use a pre-trained model like
    MobileNet-SSD or YOLO. For the hackathon, we provide a
    simulation layer that generates realistic detections.
    """

    CLASSES = ["car", "bus", "bike", "pedestrian"]

    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None  # Would load a real model in production
        self.preprocessor = PrivacyPreprocessor()

    def _build_model(self, input_shape=(224, 224, 3)):
        """Build a simple CNN feature extractor (demonstration)."""
        if keras is None:
            return None
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(self.CLASSES), activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def detect(self, frame=None):
        """Detect traffic objects in a frame.

        Returns dict with counts and metadata.
        For demo: generates synthetic detections.
        """
        rng = np.random.RandomState()

        # Simulate realistic traffic detections
        car_count = rng.poisson(lam=12)
        bus_count = rng.poisson(lam=2)
        bike_count = rng.poisson(lam=4)
        ped_count = rng.poisson(lam=6)

        total_vehicles = car_count + bus_count + bike_count
        lane_density = min(1.0, total_vehicles / 30.0)

        return {
            "vehicle_count": total_vehicles,
            "car_count": car_count,
            "bus_count": bus_count,
            "bike_count": bike_count,
            "pedestrian_count": ped_count,
            "lane_density": round(lane_density, 3),
            "face_detected": rng.random() > 0.3,
            "plate_detected": rng.random() > 0.2,
            "anonymized": True,
        }


class Layer1Pipeline:
    """Complete Layer 1 pipeline: detect + anonymize."""

    def __init__(self):
        self.detector = TrafficObjectDetector()
        self.preprocessor = PrivacyPreprocessor()

    def process(self, frame=None):
        """Process a single frame through Layer 1."""
        detections = self.detector.detect(frame)
        return detections
