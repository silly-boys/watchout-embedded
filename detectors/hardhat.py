"""
안전모 미착용 감지기 — YOLOv8-Pose + 머리 영역 색상 분류
 - yolov8n-pose.pt (ultralytics 공식 자동 다운로드)로 17개 키포인트 탐지
 - nose(0), left_eye(1), right_eye(2) 키포인트로 머리 위치 특정
 - 눈 위~bbox 상단 영역의 HSV 색상 분포로 안전모 착용 판단
 - 안전모 색: 노랑, 흰색, 주황, 빨강, 파랑, 초록 (건설 현장 표준)
"""

from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

from config import HARDHAT_MIN_PERSON_HEIGHT, PERSON_CONF, POSE_MODEL
from .base import BaseDetector, Detection, DetectionResult

logger = logging.getLogger(__name__)

# ── 안전모 HSV 범위 ──────────────────────────────────────────
# 각 항목: (lower, upper) in HSV (H: 0-180, S: 0-255, V: 0-255)
HARDHAT_COLORS: list[tuple[np.ndarray, np.ndarray, str]] = [
    (np.array([15,  80,  80]), np.array([42, 255, 255]), "yellow"),
    (np.array([0,    0, 170]), np.array([180,  85, 255]), "white"),
    (np.array([3,   90,  80]), np.array([24, 255, 255]), "orange"),
    (np.array([0,   90,  70]), np.array([10, 255, 255]), "red_l"),
    (np.array([165, 90,  70]), np.array([180, 255, 255]), "red_h"),
    (np.array([92,  70,  55]), np.array([135, 255, 255]), "blue"),
    (np.array([45,  70,  55]), np.array([90, 255, 255]), "green"),
]
HELMET_PIXEL_RATIO = 0.055
WHITE_HELMET_PIXEL_RATIO = 0.16
MIN_HELMET_BLOB_RATIO = 0.035
MIN_HEAD_PIXELS = 120

# 키포인트 인덱스
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2


def _helmet_color_score(bgr: np.ndarray) -> tuple[bool, dict]:
    """BGR 이미지 패치에서 안전모 색상 여부를 판단합니다."""
    if bgr.size == 0 or bgr.shape[0] * bgr.shape[1] < MIN_HEAD_PIXELS:
        return False, {"reason": "small_head_region"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]

    ratios = {}
    colored_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    white_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi, _ in HARDHAT_COLORS:
        name = _ if _ not in ("red_l", "red_h") else "red"
        mask = cv2.inRange(hsv, lo, hi)
        ratio = np.count_nonzero(mask) / total
        ratios[name] = ratios.get(name, 0.0) + ratio
        if name == "white":
            white_mask = cv2.bitwise_or(white_mask, mask)
        else:
            colored_mask = cv2.bitwise_or(colored_mask, mask)

    colored_ratio = sum(
        ratios.get(name, 0.0)
        for name in ("yellow", "orange", "red", "blue", "green")
    )
    white_ratio = ratios.get("white", 0.0)
    colored_blob_ratio = _largest_blob_ratio(colored_mask)
    white_blob_ratio = _largest_blob_ratio(white_mask)

    has_colored_helmet = (
        colored_ratio >= HELMET_PIXEL_RATIO
        or colored_blob_ratio >= MIN_HELMET_BLOB_RATIO
    )
    has_white_helmet = (
        white_ratio >= WHITE_HELMET_PIXEL_RATIO
        or white_blob_ratio >= MIN_HELMET_BLOB_RATIO * 1.4
    )
    return has_colored_helmet or has_white_helmet, {
        "helmet_color_ratio": round(float(colored_ratio), 3),
        "helmet_blob_ratio": round(colored_blob_ratio, 3),
        "white_ratio": round(float(white_ratio), 3),
        "white_blob_ratio": round(white_blob_ratio, 3),
    }


def _largest_blob_ratio(mask: np.ndarray) -> float:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0
    _num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if len(stats) <= 1:
        return 0.0
    largest = max(int(s[cv2.CC_STAT_AREA]) for s in stats[1:])
    return largest / mask.size


class HardhatDetector(BaseDetector):
    name = "hardhat"

    def __init__(self) -> None:
        from ultralytics import YOLO

        model_path = str(POSE_MODEL) if POSE_MODEL.exists() else "yolov8n-pose.pt"
        self._model = YOLO(model_path)
        logger.info("HardhatDetector: 모델 로드 (%s)", model_path)

    def detect(self, image_bytes: bytes, image_np: np.ndarray | None = None) -> DetectionResult:
        try:
            if image_np is None:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_np = np.array(img)
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            h_img, w_img = bgr.shape[:2]

            results = self._model(bgr, conf=PERSON_CONF, verbose=False)[0]
            violations: list[Detection] = []

            if results.keypoints is None or len(results.boxes) == 0:
                return DetectionResult(
                    detector=self.name,
                    triggered=False,
                    message="사람 미탐지",
                )

            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_h = y2 - y1
                if person_h < HARDHAT_MIN_PERSON_HEIGHT:
                    continue

                kpts = results.keypoints.xy[i].cpu().numpy()  # (17, 2) — (x, y)
                conf_kpts = results.keypoints.conf[i].cpu().numpy()  # (17,)

                # 눈·코 키포인트 유효성 확인 (신뢰도 > 0.3)
                eye_y_vals = []
                for kp_idx in [KP_LEFT_EYE, KP_RIGHT_EYE]:
                    if conf_kpts[kp_idx] > 0.3:
                        eye_y_vals.append(int(kpts[kp_idx][1]))

                nose_y = int(kpts[KP_NOSE][1]) if conf_kpts[KP_NOSE] > 0.3 else None

                # 머리 영역 추정
                if eye_y_vals:
                    eye_level = min(eye_y_vals)
                elif nose_y is not None:
                    eye_level = nose_y - abs(nose_y - y1) // 3
                else:
                    # 키포인트 없으면 bbox 상단 30% 사용
                    eye_level = y1 + (y2 - y1) // 3

                # 안전모 영역: bbox 상단 35%와 keypoint 기반 영역 중 더 넓은 쪽을 사용
                head_h = max(1, y2 - y1)
                head_top = max(0, y1 - int(head_h * 0.03))
                keypoint_bot = eye_level + int(head_h * 0.10)
                fallback_bot = y1 + int(head_h * 0.35)
                head_bot = max(head_top + 5, min(y2, max(keypoint_bot, fallback_bot)))
                cx = (x1 + x2) // 2
                hw = int((x2 - x1) * 0.42)
                head_left = max(0, cx - hw)
                head_right = min(w_img, cx + hw)

                head_patch = bgr[head_top:head_bot, head_left:head_right]
                has_hardhat, color_meta = _helmet_color_score(head_patch)

                if not has_hardhat:
                    violations.append(
                        Detection(
                            label="no_helmet",
                            confidence=float(box.conf),
                            bbox=[x1, y1, x2, y2],
                            metadata={
                                "head_region": [head_left, head_top, head_right, head_bot],
                                **color_meta,
                            },
                        )
                    )

            triggered = len(violations) > 0
            return DetectionResult(
                detector=self.name,
                triggered=triggered,
                detections=violations,
                message=(
                    f"안전모 미착용 {len(violations)}명 감지"
                    if triggered
                    else "전원 안전모 착용 확인"
                ),
            )
        except Exception as e:
            logger.exception("HardhatDetector 오류")
            return DetectionResult(detector=self.name, triggered=False, error=str(e))
