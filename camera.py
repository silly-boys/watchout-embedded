import io

from picamera2 import Picamera2
from PIL import Image as PILImage


def create_camera(size: tuple[int, int]) -> Picamera2:
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": size, "format": "RGB888"}))
    return camera


def capture_loop(camera: Picamera2, frames, stop_event, fps: int) -> None:
    frame_delay = 1 / max(fps, 1)
    while not stop_event.is_set():
        arr = camera.capture_array("main")
        buf = io.BytesIO()
        PILImage.fromarray(arr[:, :, ::-1]).save(buf, format="JPEG", quality=85)
        frames.set(buf.getvalue(), arr.copy())
        stop_event.wait(frame_delay)
