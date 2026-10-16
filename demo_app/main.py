"""FastAPI demo for the local exterior-damage pipeline."""

from __future__ import annotations

import base64
import threading
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo_app" / "static"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

DAMAGE_MODEL_PATH = ROOT / "runs" / "damage" / "weights" / "best.pt"
PANEL_MODEL_PATH = ROOT / "runs" / "panel" / "weights" / "best.pt"

DAMAGE_VI = {
    "dent": "Móp",
    "scratch": "Trầy xước",
    "broken_or_cracked": "Nứt / vỡ",
}

PANEL_VI = {
    "back-bumper": "Cản sau",
    "back-door": "Cửa sau",
    "back-windshield": "Kính sau",
    "tail-light": "Đèn hậu",
    "front-bumper": "Cản trước",
    "front-door": "Cửa trước",
    "windshield": "Kính chắn gió",
    "headlight": "Đèn trước",
    "hood": "Nắp capo",
    "mirror": "Gương",
    "trunk": "Cốp sau",
    "front-wheel": "Bánh trước",
    "fender": "Vè trước",
    "quarter-panel": "Vè sau",
    "roof": "Nóc xe",
    "front-window": "Kính cửa trước",
    "back-window": "Kính cửa sau",
    "rocker-panel": "Ốp sườn",
    "back-wheel": "Bánh sau",
    "grille": "Lưới tản nhiệt",
    "license-plate": "Biển số",
}

COLORS = {
    "dent": (19, 138, 246),
    "scratch": (56, 197, 126),
    "broken_or_cracked": (67, 67, 239),
}


app = FastAPI(title="VinFast Exterior Check", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_damage_model: YOLO | None = None
_panel_model: YOLO | None = None
_model_lock = threading.Lock()


def get_models() -> tuple[YOLO, YOLO]:
    global _damage_model, _panel_model
    if not DAMAGE_MODEL_PATH.exists() or not PANEL_MODEL_PATH.exists():
        missing = [
            str(path.relative_to(ROOT))
            for path in (DAMAGE_MODEL_PATH, PANEL_MODEL_PATH)
            if not path.exists()
        ]
        raise RuntimeError("Missing model weights: " + ", ".join(missing))
    if _damage_model is None:
        _damage_model = YOLO(DAMAGE_MODEL_PATH)
    if _panel_model is None:
        _panel_model = YOLO(PANEL_MODEL_PATH)
    return _damage_model, _panel_model


def decode_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Tệp tải lên không phải ảnh hợp lệ.") from error
    if image.width * image.height > 24_000_000:
        raise HTTPException(status_code=413, detail="Ảnh có độ phân giải quá lớn.")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def masks_from_result(result, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if result.masks is None:
        return masks
    for polygon_group in result.masks.xy:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(polygon_group) >= 3:
            polygon = np.rint(polygon_group).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        masks.append(mask)
    return masks


def match_panel(
    box: tuple[int, int, int, int],
    panel_masks: list[np.ndarray],
    panel_names: list[str],
    panel_confidences: list[float],
) -> tuple[str | None, float]:
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    area = max(1, (x2 - x1) * (y2 - y1))
    best_name, best_score = None, 0.0

    for mask, name, confidence in zip(panel_masks, panel_names, panel_confidences):
        center_bonus = 0.35 if 0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and mask[center_y, center_x] else 0.0
        intersection = int(mask[y1:y2, x1:x2].sum())
        overlap = intersection / area
        score = overlap + center_bonus + 0.05 * confidence
        if score > best_score:
            best_name, best_score = name, score
    return best_name, min(1.0, best_score)


def severity_and_action(damage_name: str, area_ratio: float) -> tuple[str, str]:
    if area_ratio < 0.02:
        severity = "Nhẹ"
    elif area_ratio < 0.08:
        severity = "Trung bình"
    else:
        severity = "Nặng"

    if damage_name == "scratch":
        action = "Kiểm tra độ sâu; đánh bóng nếu chỉ xước lớp bóng."
    elif damage_name == "dent":
        action = "Kiểm tra sơn và cân nhắc sửa móp không sơn (PDR)."
    else:
        action = "Cần kỹ thuật viên kiểm tra; có thể phải sửa hoặc thay chi tiết."
    return severity, action


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - baseline - 8)
    cv2.rectangle(image, (x, top), (x + text_width + 10, y), color, -1)
    cv2.putText(image, text, (x + 5, y - baseline - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def diagnose(image: np.ndarray, confidence: float) -> dict:
    damage_model, panel_model = get_models()
    height, width = image.shape[:2]

    with _model_lock:
        damage_result = damage_model.predict(image, imgsz=640, conf=confidence, device=0, verbose=False)[0]
        panel_result = panel_model.predict(image, imgsz=640, conf=0.25, device=0, retina_masks=True, verbose=False)[0]

    panel_masks = masks_from_result(panel_result, height, width)
    panel_names: list[str] = []
    panel_confidences: list[float] = []
    if panel_result.boxes is not None:
        for class_id, panel_confidence in zip(
            panel_result.boxes.cls.cpu().numpy().astype(int),
            panel_result.boxes.conf.cpu().numpy(),
        ):
            panel_names.append(panel_result.names[int(class_id)])
            panel_confidences.append(float(panel_confidence))

    annotated = image.copy()
    diagnoses: list[dict] = []
    if damage_result.boxes is not None:
        boxes = damage_result.boxes.xyxy.cpu().numpy()
        classes = damage_result.boxes.cls.cpu().numpy().astype(int)
        confidences = damage_result.boxes.conf.cpu().numpy()
        for index, (coordinates, class_id, damage_confidence) in enumerate(zip(boxes, classes, confidences), start=1):
            x1, y1, x2, y2 = np.rint(coordinates).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            damage_name = damage_result.names[int(class_id)]
            panel_name, panel_score = match_panel(
                (x1, y1, x2, y2), panel_masks, panel_names, panel_confidences
            )
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
            severity, action = severity_and_action(damage_name, area_ratio)
            color = COLORS.get(damage_name, (255, 132, 0))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{index}. {damage_name} {float(damage_confidence):.0%}"
            draw_label(annotated, label, x1, max(24, y1), color)
            diagnoses.append(
                {
                    "id": index,
                    "damage": damage_name,
                    "damage_vi": DAMAGE_VI.get(damage_name, damage_name),
                    "confidence": round(float(damage_confidence), 4),
                    "panel": panel_name,
                    "panel_vi": PANEL_VI.get(panel_name, panel_name) if panel_name else "Chưa xác định",
                    "panel_match": round(panel_score, 4),
                    "severity": severity,
                    "action": action,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                }
            )

    success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Could not encode annotated image")
    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return {
        "count": len(diagnoses),
        "diagnoses": diagnoses,
        "annotated_image": f"data:image/jpeg;base64,{image_base64}",
        "image_size": {"width": width, "height": height},
        "disclaimer": "Kết quả AI chỉ dùng để sàng lọc ban đầu, không thay thế kiểm tra trực tiếp của kỹ thuật viên.",
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "damage_model": DAMAGE_MODEL_PATH.exists(),
        "panel_model": PANEL_MODEL_PATH.exists(),
        "device": "cuda:0",
    }


@app.post("/api/diagnose")
async def diagnose_upload(
    image: UploadFile = File(...),
    confidence: float = Form(0.20),
) -> dict:
    if not 0.05 <= confidence <= 0.90:
        raise HTTPException(status_code=400, detail="Confidence phải nằm trong khoảng 0.05–0.90.")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 12 MB.")
    decoded = decode_image(data)
    try:
        return diagnose(decoded, confidence)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
