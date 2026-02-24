import cv2
import torch
import time
import logging
import signal
import sys
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


# ==========================================================
# CONFIGURATION
# ==========================================================

@dataclass
class DetectionConfig:
    model_path: str = "yolov9c.pt"
    camera_index: int = 0
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    window_name: str = "Real-Time Object Detection"
    use_half_precision: bool = True
    enable_fps_display: bool = True
    resize_width: int = 1280
    resize_height: int = 720
    bbox_color: tuple = (0, 255, 0)
    bbox_thickness: int = 2
    text_scale: float = 0.5
    text_thickness: int = 1


# ==========================================================
# LOGGER SETUP
# ==========================================================

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("YOLOv9-Detector")


LOGGER = setup_logger()


# ==========================================================
# DEVICE MANAGEMENT
# ==========================================================

def get_device():
    if torch.cuda.is_available():
        LOGGER.info("CUDA detected — using GPU")
        return torch.device("cuda")
    LOGGER.info("CUDA not available — using CPU")
    return torch.device("cpu")


# ==========================================================
# FPS COUNTER
# ==========================================================

class FPSMeter:

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        current = time.time()
        delta = current - self.prev_time
        self.prev_time = current

        if delta > 0:
            self.fps = 1.0 / delta

        return self.fps


# ==========================================================
# VIDEO STREAM HANDLER
# ==========================================================

class VideoStream:

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Frame capture failed")
        return frame

    def release(self):
        self.cap.release()


# ==========================================================
# FRAME PREPROCESSOR
# ==========================================================

class FrameProcessor:

    @staticmethod
    def preprocess(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)
        return frame

    @staticmethod
    def postprocess(frame):
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# ==========================================================
# DETECTOR ENGINE
# ==========================================================

class YOLODetector:

    def __init__(self, config: DetectionConfig, device):
        self.config = config
        self.device = device

        LOGGER.info("Loading YOLO model...")
        self.model = YOLO(config.model_path)
        self.model.to(device)

        if device.type == "cuda" and config.use_half_precision:
            self.model.model.half()
            LOGGER.info("Half precision enabled")

        self.names = self.model.names

    def infer(self, frame):
        return self.model(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )


# ==========================================================
# RENDER ENGINE
# ==========================================================

class Renderer:

    def __init__(self, config: DetectionConfig):
        self.config = config

    def draw_detections(self, frame, results):

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                label = f"{result.names[cls_id]}: {conf:.2f}"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    self.config.bbox_color,
                    self.config.bbox_thickness
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.text_scale,
                    self.config.bbox_color,
                    self.config.text_thickness
                )

        return frame

    def draw_fps(self, frame, fps):
        text = f"FPS: {fps:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


# ==========================================================
# SHUTDOWN HANDLER
# ==========================================================

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    LOGGER.info("Shutdown signal received")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)


# ==========================================================
# MAIN APPLICATION
# ==========================================================

class ObjectDetectionApp:

    def __init__(self):
        self.config = DetectionConfig()
        self.device = get_device()

        self.stream = VideoStream(
            self.config.camera_index,
            self.config.resize_width,
            self.config.resize_height
        )

        self.detector = YOLODetector(self.config, self.device)
        self.renderer = Renderer(self.config)
        self.fps_meter = FPSMeter()

    def run(self):

        global RUNNING

        LOGGER.info("Starting detection loop")

        while RUNNING:

            try:
                frame = self.stream.read()

                processed = FrameProcessor.preprocess(frame)

                results = self.detector.infer(processed)

                frame = FrameProcessor.postprocess(processed)

                frame = self.renderer.draw_detections(
                    frame,
                    results
                )

                if self.config.enable_fps_display:
                    fps = self.fps_meter.update()
                    self.renderer.draw_fps(frame, fps)

                cv2.imshow(
                    self.config.window_name,
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                LOGGER.error(f"Runtime error: {e}")
                break

        self.cleanup()

    def cleanup(self):
        LOGGER.info("Cleaning resources")
        self.stream.release()
        cv2.destroyAllWindows()


# ==========================================================
# ENTRY POINT
# ==========================================================

def main():
    app = ObjectDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
