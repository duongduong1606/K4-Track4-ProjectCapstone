# Local training setup

Balanced stack:

- `yolo26n.pt`: three-class damage detector (`dent`, `scratch`, `broken_or_cracked`).
- `yolo26n-seg.pt`: 21-class exterior-panel segmenter.
- `qwen3-vl:2b-instruct-q4_K_M`: local report/reasoning model served by Ollama.

Verified machine setup: Python 3.11, PyTorch 2.6.0 + CUDA 12.4 and an RTX 3060 12 GB.

## Recreate the environment

```powershell
conda env create -f environment.yml
conda activate vinfast-damage
python scripts/prepare_damage_dataset.py
python scripts/verify_setup.py
```

If Ollama Desktop is not already running:

```powershell
ollama serve
```

## Train

The defaults are tuned for the local RTX 3060 12 GB:

| Task | Image | Batch | Epochs | Patience | Measured VRAM | Expected time |
|---|---:|---:|---:|---:|---:|---:|
| Damage detection | 640 | 16 | 60 | 12 | ~2.44 GB | 50–65 min |
| Panel segmentation | 640 | 8 | 100 | 20 | ~2.00 GB | 15–25 min |

Run the models sequentially. Stop/unload Ollama models first so they do not reserve GPU memory:

```powershell
$body = @{model='qwen3-vl:2b-instruct-q4_K_M'; keep_alive=0} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -ContentType application/json -Body $body
```

Then start training:

```powershell
conda activate vinfast-damage
python scripts/train.py damage
python scripts/train.py panel
```

The damage model runs first because it is the product's primary signal. Do not run both commands at the same time on one GPU. The defaults retain Ultralytics' pretrained-model augmentations and AdamW auto-tuning; measured learning rates were about 0.001429 for detection and 0.0004 for segmentation.

Monitor from a second PowerShell window:

```powershell
nvidia-smi -l 2
Get-Content runs\damage\results.csv -Tail 5 -Wait
```

Resume after an interruption:

```powershell
python scripts/train.py damage --resume
python scripts/train.py panel --resume
```

For a quick end-to-end smoke test, use a small fraction and one epoch:

```powershell
python scripts/train.py damage --epochs 1 --fraction 0.01 --batch 4 --name smoke-damage
python scripts/train.py panel --epochs 1 --fraction 0.1 --batch 4 --name smoke-panel
```

Training outputs are written to `runs/damage` and `runs/panel`. Generated data,
runs and weights are intentionally ignored by Git.

## Run the local demo

```powershell
conda activate vinfast-damage
uvicorn demo_app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The app loads `runs/damage/weights/best.pt` and
`runs/panel/weights/best.pt`, then matches each damage box to the panel mask with
the largest overlap.
