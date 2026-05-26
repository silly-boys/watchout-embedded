import json
import logging
import time

from config import ALERTS_DIR, PERSON_MODEL

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
                "metadata": d.metadata,
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

        for name, detector in detectors.items():
            try:
                result = detector.detect(frame)
                serialized = serialize_result(result)
            except Exception as exc:
                logger.exception("%s detector failed", name)
                serialized = {
                    "triggered": False,
                    "message": "감지기 오류",
                    "error": str(exc),
                    "detections": [],
                }
            summary[name] = serialized
            if serialized["triggered"]:
                alerts.append(name)

        analysis = {
            "status": "ok",
            "timestamp": timestamp,
            "frame_seq": frame_seq,
            "elapsed_ms": round((time.time() - started) * 1000, 1),
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
