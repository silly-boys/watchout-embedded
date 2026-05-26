import asyncio
import json
from pathlib import Path

import av
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack


class PiCameraTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, frames, fallback_size: tuple[int, int]) -> None:
        super().__init__()
        self._frames = frames
        self._fallback_size = fallback_size

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        arr, _seq = self._frames.get_video()
        if arr is None:
            width, height = self._fallback_size
            arr = np.zeros((height, width, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRTCServer:
    def __init__(self, frames, stop_event, camera_size: tuple[int, int]) -> None:
        self._frames = frames
        self._stop_event = stop_event
        self._camera_size = camera_size
        self._pcs: set[RTCPeerConnection] = set()

    async def offer(self, request: web.Request) -> web.Response:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state():
            print(f"[WebRTC] {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self._pcs.discard(pc)

        pc.addTrack(PiCameraTrack(self._frames, self._camera_size))
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        for _ in range(200):
            if pc.iceGatheringState == "complete":
                break
            await asyncio.sleep(0.05)

        return web.Response(
            content_type="application/json",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }),
        )

    async def viewer(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(Path(__file__).parent / "viewer.html")

    async def options(self, _request: web.Request) -> web.Response:
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    async def shutdown(self) -> None:
        self._stop_event.set()
        await asyncio.gather(*(pc.close() for pc in list(self._pcs)), return_exceptions=True)
        self._pcs.clear()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.viewer)
        app.router.add_post("/offer", self.offer)
        app.router.add_route("OPTIONS", "/offer", self.options)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_shutdown(self, _app: web.Application) -> None:
        await self.shutdown()

async def run_server(host: str, port: int, frames, stop_event, camera_size: tuple[int, int]) -> None:
    server = WebRTCServer(frames, stop_event, camera_size)
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"WebRTC 서버 시작: http://{host}:{port}/offer")
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
    finally:
        await runner.cleanup()
