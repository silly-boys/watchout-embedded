import io
import time
import requests
from picamera2 import Picamera2

SERVER_URL = "https://8926-221-168-22-205.ngrok-free.app/upload"

camera = Picamera2()
camera.configure(camera.create_still_configuration(
    main={"size": (1280, 720)}
))
camera.start()
time.sleep(1)

try:
    while True:
        buf = io.BytesIO()
        camera.capture_file(buf, format='jpeg')
        buf.seek(0)

        try:
            res = requests.post(
                SERVER_URL,
                files={"image": ("capture.jpg", buf, "image/jpeg")},
                timeout=15
            )
            print(f"[{time.strftime('%H:%M:%S')}] 전송 완료 - {res.status_code}")
        except requests.RequestException as e:
            print(f"[{time.strftime('%H:%M:%S')}] 전송 실패 - {e}")

        time.sleep(5)
except KeyboardInterrupt:
    print("종료")
finally:
    camera.stop()