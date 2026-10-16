"""Fast local verification of CUDA, weights, datasets and Ollama."""

from __future__ import annotations

import json
from pathlib import Path

import requests
import torch
from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    for weight in ("yolo26n.pt", "yolo26n-seg.pt"):
        path = root / "models" / weight
        model = YOLO(path)
        print(f"weight={weight} task={model.task} size_bytes={path.stat().st_size}")
    report = root / "data" / "processed" / "damage" / "build_report.json"
    print(json.dumps(json.loads(report.read_text(encoding="utf-8")), indent=2))
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [item["name"] for item in response.json().get("models", [])]
        print("ollama_ok=True")
        print("ollama_models=" + ",".join(models))
        assert "qwen3-vl:2b-instruct-q4_K_M" in models
    except requests.RequestException as error:
        raise SystemExit(f"ollama_ok=False: {error}") from error


if __name__ == "__main__":
    main()
