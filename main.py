import os
import signal
import threading
import time

from runtime import MEMORY_LIMIT_GB, apply_memory_limit, configure_native_runtime


configure_native_runtime()

from ai_runner import analysis_loop, init_detectors
from camera import capture_loop, create_camera
from state import AnalysisStore, FrameStore
from stream_server import create_handler, create_server

STREAM_PORT = int(os.getenv("WATCHOUT_STREAM_PORT", "8080"))
FPS = int(os.getenv("WATCHOUT_FPS", "15"))
AI_MIN_INTERVAL = float(os.getenv("WATCHOUT_AI_INTERVAL", "0"))
CAMERA_SIZE = (
    int(os.getenv("WATCHOUT_CAMERA_WIDTH", "1280")),
    int(os.getenv("WATCHOUT_CAMERA_HEIGHT", "720")),
)


def main() -> None:
    apply_memory_limit()

    frames = FrameStore()
    analyses = AnalysisStore()
    stop_event = threading.Event()

    print(f"[{time.strftime('%H:%M:%S')}] AI 감지기 초기화 중...")
    detectors = init_detectors()
    print(f"[{time.strftime('%H:%M:%S')}] AI 감지기 초기화 완료")

    camera = create_camera(CAMERA_SIZE)
    camera.start()
    time.sleep(1)

    threading.Thread(
        target=capture_loop,
        args=(camera, frames, stop_event, FPS),
        daemon=True,
    ).start()
    threading.Thread(
        target=analysis_loop,
        args=(detectors, frames, analyses, stop_event, AI_MIN_INTERVAL),
        daemon=True,
    ).start()

    def request_shutdown(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    health = {
        "status": "ok",
        "memory_limit_gb": MEMORY_LIMIT_GB,
        "ai_min_interval_sec": AI_MIN_INTERVAL,
        "analysis_mode": "continuous_latest_frame",
        "stream": f"http://0.0.0.0:{STREAM_PORT}/stream",
    }
    handler = create_handler(frames, analyses, stop_event, FPS, health)
    server = create_server("0.0.0.0", STREAM_PORT, handler)

    print(f"[{time.strftime('%H:%M:%S')}] 시작 - 스트림: http://0.0.0.0:{STREAM_PORT}/stream")
    print(f"[{time.strftime('%H:%M:%S')}] 분석 결과: http://0.0.0.0:{STREAM_PORT}/analysis")

    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        camera.stop()
        print("종료")


if __name__ == "__main__":
    main()
