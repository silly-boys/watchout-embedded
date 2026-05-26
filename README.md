# WatchOut Embedded

Raspberry Pi camera application that streams live WebRTC video and analyzes the latest camera frame locally with WatchOut AI detectors. It does not upload frames to an external AI server.

## Runtime Flow

1. `main.py` applies the memory limit, initializes stores, loads detectors, starts the camera, and starts worker threads.
2. `camera.py` captures JPEG frames from `Picamera2` at `WATCHOUT_FPS` and writes only the newest frame into `FrameStore`.
3. `ai_runner.py` watches `FrameStore` for a new frame sequence number and analyzes the newest frame immediately.
4. `stream_server.py` exposes:
   - `GET /`: browser viewer
   - `POST /offer`: WebRTC signaling endpoint
5. Alerts are saved as JSON files under `alerts/`.

## Environment Variables

- `WATCHOUT_MEMORY_LIMIT_GB`: process memory cap, default `6`
- `WATCHOUT_STREAM_PORT`: HTTP port, default `8080`
- `WATCHOUT_FPS`: camera stream capture FPS, default `15`
- `WATCHOUT_AI_INTERVAL`: minimum seconds between analyses, default `0`
- `WATCHOUT_CAMERA_WIDTH`: camera width, default `1280`
- `WATCHOUT_CAMERA_HEIGHT`: camera height, default `720`

## Run

```bash
python3 main.py
```

Or after setup:

```bash
./start.sh
```

## Models

- `models/yolov8n.pt`: person detector shared by fall and virtual fence checks
- `models/yolov8n-pose.pt`: fallback pose model for hardhat checks
- `models/hardhat.pt`: optional dedicated helmet detector

`models/hardhat.pt` currently comes from `iam-tsr/yolov8n-helmet-detection` on Hugging Face. Its labels are `With Helmet` and `Without Helmet`, which are mapped by the app to helmet/no-helmet states.
