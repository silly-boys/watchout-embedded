import json
import logging
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config import AI_IMAGE_WIDTH, ALERTS_DIR, HARDHAT_EVERY_N_FRAMES, PERSON_CONF, PERSON_MODEL

logger = logging.getLogger(__name__)


def init_detectors():
    import torch
    from ultralytics import YOLO
    from detectors import (
        FallDetector,
        FireSmokeDetector,
        HardhatDetector,
        VirtualFenceDetector,
    )

    torch.set_num_threads(2)
    person_model_path = str(PERSON_MODEL) if PERSON_MODEL.exists() else "yolov8n.pt"
    person_model = YOLO(person_model_path)

    return {
        "fire_smoke": FireSmokeDetector(),
        "hardhat": HardhatDetector(),
        "virtual_fence": VirtualFenceDetector(model=person_model),
        "fall": FallDetector(model=person_model),
    }


def decode_analysis_image(image_bytes: bytes) -> tuple[np.ndarray, float]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(img)
    h, w = image_np.shape[:2]
    if AI_IMAGE_WIDTH <= 0 or w <= AI_IMAGE_WIDTH:
        return image_np, 1.0
    analysis_h = max(1, int(h * (AI_IMAGE_WIDTH / w)))
    resized = cv2.resize(image_np, (AI_IMAGE_WIDTH, analysis_h), interpolation=cv2.INTER_AREA)
    return resized, w / AI_IMAGE_WIDTH


def scale_result_boxes(result, scale: float) -> None:
    if scale == 1.0:
        return
    for detection in result.detections:
        if detection.bbox:
            detection.bbox = [int(v * scale) for v in detection.bbox]
        head_region = detection.metadata.get("head_region")
        if head_region:
            detection.metadata["head_region"] = [int(v * scale) for v in head_region]
        foot_point = detection.metadata.get("foot_point")
        if foot_point and not detection.metadata.get("coordinates_scaled"):
            detection.metadata["foot_point"] = [int(v * scale) for v in foot_point]


def serialize_result(result) -> dict:
    return {
        "triggered": result.triggered,
        "message": result.message,
        "error": result.error,
        "detections": [
            {
                "label": d.label,
                "confidence": round(float(d.confidence), 3),
                "bbox": d.bbox,
                "metadata": {
                    key: value
                    for key, value in d.metadata.items()
                    if key != "coordinates_scaled"
                },
            }
            for d in result.detections
        ],
    }


def analysis_loop(detectors, frames, analyses, stop_event, min_interval_sec: float) -> None:
    last_analyzed_seq = 0
    while not stop_event.is_set():
        started = time.time()
        frame, frame_seq = frames.get()

        if frame is None or frame_seq == last_analyzed_seq:
            stop_event.wait(0.005)
            continue

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        summary = {}
        alerts = []
        timings = {}

        try:
            image_np, image_scale = decode_analysis_image(frame)
        except Exception as exc:
            logger.exception("frame decode failed")
            analyses.set({
                "status": "error",
                "timestamp": timestamp,
                "frame_seq": frame_seq,
                "elapsed_ms": round((time.time() - started) * 1000, 1),
                "alerts": [],
                "summary": {},
                "error": str(exc),
            })
            last_analyzed_seq = frame_seq
            continue

        person_results = None
        if "virtual_fence" in detectors or "fall" in detectors:
            try:
                person_started = time.time()
                person_results = detectors["fall"]._model(image_np, conf=PERSON_CONF, verbose=False)[0]
                timings["person_yolo"] = round((time.time() - person_started) * 1000, 1)
            except Exception as exc:
                logger.exception("person YOLO failed")
                summary["person_yolo"] = {
                    "triggered": False,
                    "message": "사람 탐지 오류",
                    "error": str(exc),
                    "detections": [],
                }

        for name, detector in detectors.items():
            detector_started = time.time()
            try:
                if name == "hardhat":
                    if frame_seq % HARDHAT_EVERY_N_FRAMES != 0:
                        continue
                    result = detector.detect(frame, image_np=image_np)
                elif name in ("virtual_fence", "fall"):
                    if person_results is None:
                        continue
                    if name == "virtual_fence":
                        result = detector.detect(
                            frame,
                            image_np=image_np,
                            person_results=person_results,
                            coordinate_scale=image_scale,
                        )
                        for detection in result.detections:
                            detection.metadata["coordinates_scaled"] = True
                    else:
                        result = detector.detect(frame, image_np=image_np, person_results=person_results)
                else:
                    result = detector.detect(frame)
                scale_result_boxes(result, image_scale)
                serialized = serialize_result(result)
            except Exception as exc:
                logger.exception("%s detector failed", name)
                serialized = {
                    "triggered": False,
                    "message": "감지기 오류",
                    "error": str(exc),
                    "detections": [],
                }
            timings[name] = round((time.time() - detector_started) * 1000, 1)
            summary[name] = serialized
            if serialized["triggered"]:
                alerts.append(name)

        analysis = {
            "status": "ok",
            "timestamp": timestamp,
            "frame_seq": frame_seq,
            "elapsed_ms": round((time.time() - started) * 1000, 1),
            "timings_ms": timings,
            "alerts": alerts,
            "summary": summary,
        }
        analyses.set(analysis)
        last_analyzed_seq = frame_seq

        if alerts:
            print(f"[{timestamp}] 이상 감지: {', '.join(alerts)}")
            for name in alerts:
                print(f"  - {name}: {summary[name]['message']}")
            save_alert(timestamp, alerts, summary)
        else:
            print(f"[{timestamp}] 이상 없음 ({analysis['elapsed_ms']}ms)")

        remaining = max(0.0, min_interval_sec - (time.time() - started))
        stop_event.wait(remaining)


def save_alert(timestamp: str, alerts: list[str], summary: dict) -> None:
    alert_path = ALERTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    alert_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "alerts": alerts,
                "details": {name: summary[name] for name in alerts},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
