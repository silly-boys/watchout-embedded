import io
import time
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2

SERVER_URL = "https://84c4-221-168-22-205.ngrok-free.app/upload"
STREAM_PORT = 8080
FPS = 15
UPLOAD_INTERVAL = 5

latest_frame = None
frame_lock = threading.Lock()


def capture_loop(camera):
    global latest_frame
    while True:
        buf = io.BytesIO()
        camera.capture_file(buf, format="jpeg")
        buf.seek(0)
        with frame_lock:
            latest_frame = buf.read()
        time.sleep(1 / FPS)


def upload_loop():
    while True:
        time.sleep(UPLOAD_INTERVAL)
        with frame_lock:
            frame = latest_frame
        if frame is None:
            continue
        try:
            res = requests.post(
                SERVER_URL,
                files={"image": ("capture.jpg", io.BytesIO(frame), "image/jpeg")},
                timeout=15,
            )
            print(f"[{time.strftime('%H:%M:%S')}] 전송 완료 - {res.status_code}")
        except requests.RequestException as e:
            print(f"[{time.strftime('%H:%M:%S')}] 전송 실패 - {e}")


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(1 / FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_error(404)


if __name__ == "__main__":
    camera = Picamera2()
    camera.configure(
        camera.create_video_configuration(main={"size": (1280, 720)})
    )
    camera.start()
    time.sleep(1)

    threading.Thread(target=capture_loop, args=(camera,), daemon=True).start()
    threading.Thread(target=upload_loop, daemon=True).start()

    print(f"[{time.strftime('%H:%M:%S')}] 시작 - 스트림: http://0.0.0.0:{STREAM_PORT}")
    try:
        HTTPServer(("0.0.0.0", STREAM_PORT), StreamHandler).serve_forever()
    except KeyboardInterrupt:
        print("종료")
    finally:
        camera.stop()
