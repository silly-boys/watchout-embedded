import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def create_handler(frames, analyses, stop_event, fps: int, health: dict):
    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

        def _send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self._send_cors()
                self.end_headers()
                self._stream_frames()
                return

            if self.path == "/analysis":
                self._send_json(analyses.get())
                return

            if self.path == "/health":
                self._send_json(health)
                return

            self.send_error(404)

        def _stream_frames(self):
            frame_delay = 1 / max(fps, 1)
            try:
                while not stop_event.is_set():
                    frame, _seq = frames.get()
                    if frame is None:
                        stop_event.wait(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    stop_event.wait(frame_delay)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return StreamHandler


def create_server(host: str, port: int, handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler)
    server.timeout = 0.5
    return server
