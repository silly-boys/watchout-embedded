import threading
from typing import Any


class FrameStore:
    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._video_frame: Any | None = None
        self._seq = 0
        self._lock = threading.Lock()

    def set(self, jpeg: bytes, video_frame: Any | None = None) -> int:
        with self._lock:
            self._jpeg = jpeg
            self._video_frame = video_frame
            self._seq += 1
            return self._seq

    def get(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._seq

    def get_video(self) -> tuple[Any | None, int]:
        with self._lock:
            return self._video_frame, self._seq


class AnalysisStore:
    def __init__(self) -> None:
        self._analysis = {
            "status": "starting",
            "timestamp": None,
            "alerts": [],
            "summary": {},
        }
        self._lock = threading.Lock()

    def set(self, analysis: dict) -> None:
        with self._lock:
            self._analysis = analysis

    def get(self) -> dict:
        with self._lock:
            return self._analysis.copy()
