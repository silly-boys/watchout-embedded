import json
import ssl
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from config import FENCE_API_URL, FENCE_POLL_INTERVAL_SEC, FENCE_REMOTE_ZONE_NAME
from detectors.virtual_fence import replace_zones


def poll_fence_loop(camera_size: tuple[int, int], stop_event) -> None:
    while not stop_event.is_set():
        try:
            zones = fetch_remote_zones(camera_size)
            replace_zones(zones)
            print(f"[{time.strftime('%H:%M:%S')}] 위험구역 동기화 완료: {len(zones)}개")
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] 위험구역 동기화 실패: {exc}")

        stop_event.wait(FENCE_POLL_INTERVAL_SEC)


def fetch_remote_zones(camera_size: tuple[int, int]) -> dict[str, list[list[int]]]:
    req = Request(FENCE_API_URL, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10, context=_ssl_context()) as res:
            payload = json.loads(res.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"요청 실패: {exc}") from exc

    polygon = _extract_polygon(payload)
    if not polygon:
        return {}

    return {
        FENCE_REMOTE_ZONE_NAME: _normalized_polygon_to_pixels(polygon, camera_size),
    }


def _extract_polygon(payload: dict) -> list[list[float]]:
    if not payload.get("success", False):
        raise RuntimeError(f"API success=false: {payload}")

    data = payload.get("data") or {}
    polygon = data.get("polygon") or []
    if not isinstance(polygon, list):
        raise RuntimeError("polygon 필드가 배열이 아닙니다.")
    return polygon


def _normalized_polygon_to_pixels(polygon: list[list[float]], camera_size: tuple[int, int]) -> list[list[int]]:
    width, height = camera_size
    points: list[list[int]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        x = _clamp(float(point[0]), 0.0, 1.0)
        y = _clamp(float(point[1]), 0.0, 1.0)
        points.append([
            int(round(x * width)),
            int(round(y * height)),
        ])

    if 0 < len(points) < 3:
        raise RuntimeError("위험구역 polygon은 최소 3개 좌표가 필요합니다.")
    return points


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
