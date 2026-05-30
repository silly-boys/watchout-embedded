import json
import ssl
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import EVENT_API_URL, EVENT_POST_COOLDOWN_SEC


DETECTOR_EVENT_TYPES = {
    "fire_smoke": "FIRE",
    "hardhat": "NO_HELMET",
    "virtual_fence": "INTRUSION",
    "fall": "FALL",
    "anomaly": "EQUIPMENT_ANOMALY",
}


class EventReporter:
    def __init__(self) -> None:
        self._last_sent_at: dict[str, float] = {}

    def report_alerts(self, alerts: list[str]) -> None:
        for detector_name in alerts:
            event_type = DETECTOR_EVENT_TYPES.get(detector_name)
            if event_type is None:
                continue
            if not self._can_send(event_type):
                continue
            self._post_event(event_type)
            self._last_sent_at[event_type] = time.monotonic()

    def _can_send(self, event_type: str) -> bool:
        last_sent = self._last_sent_at.get(event_type)
        if last_sent is None:
            return True
        return time.monotonic() - last_sent >= EVENT_POST_COOLDOWN_SEC

    def _post_event(self, event_type: str) -> None:
        payload = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            EVENT_API_URL,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=10, context=_ssl_context()) as res:
                res.read()
            print(f"[{time.strftime('%H:%M:%S')}] 이벤트 전송 완료: {event_type}")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[{time.strftime('%H:%M:%S')}] 이벤트 전송 실패: {event_type} HTTP {exc.code} {detail}")
        except URLError as exc:
            print(f"[{time.strftime('%H:%M:%S')}] 이벤트 전송 실패: {event_type} {exc}")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
