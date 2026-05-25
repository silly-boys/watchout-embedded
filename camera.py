import io

from picamera2 import Picamera2


def create_camera(size: tuple[int, int]) -> Picamera2:
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": size}))
    return camera


def capture_loop(camera: Picamera2, frames, stop_event, fps: int) -> None:
    frame_delay = 1 / max(fps, 1)
    while not stop_event.is_set():
        buf = io.BytesIO()
        camera.capture_file(buf, format="jpeg")
        frames.set(buf.getvalue())
        stop_event.wait(frame_delay)
