import threading


class FrameStore:
    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._seq = 0
        self._lock = threading.Lock()

    def set(self, frame: bytes) -> int:
        with self._lock:
            self._frame = frame
            self._seq += 1
            return self._seq

    def get(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._frame, self._seq


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
