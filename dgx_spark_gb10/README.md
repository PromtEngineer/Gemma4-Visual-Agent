# Gemma Vision Agent — NVIDIA DGX Spark GB10 (CUDA / PyTorch)

This folder is a **parallel deployment** of the same agentic pipeline as the repository root (Falcon Perception + Gemma 4), built for **Linux on ARM64 with an NVIDIA Blackwell-class GPU** (DGX Spark GB10 class systems). The original MLX + macOS entry points in the repo root are **unchanged**.

## What matches the original system

- Same planning rules, tools (`DETECT`, `VLM`, `CROP`, `COMPARE`, `DETECT_EACH`, re-planning), and UI flow as `vision_studio.py` / `agent_studio.py` at the repo root.
- Falcon Perception uses the upstream **PyTorch** backend (`falcon-perception[torch]`).
- Gemma uses **`google/gemma-4-E4B-it`** via **Transformers** (`AutoModelForMultimodalLM`), equivalent role to the MLX `mlx-community/gemma-4-e4b-it-8bit` build.

## Prerequisites

- Python 3.10+ (3.12 recommended).
- NVIDIA driver and CUDA-compatible PyTorch wheels for your stack.
- Hugging Face account: accept the Gemma license for `google/gemma-4-E4B-it`, then `huggingface-cli login` (or set `HF_TOKEN`).

## Setup

```bash
cd /path/to/Gemma4-Visual-Agent

python3.12 -m venv .venv-dgx
source .venv-dgx/bin/activate

# Install PyTorch for your CUDA build (example: cu128 — adjust to your cluster image)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r dgx_spark_gb10/requirements.txt
pip install "falcon-perception[torch] @ git+https://github.com/tiiuae/falcon-perception.git"
```

**Gemma 4 and `transformers`:** `google/gemma-4-E4B-it` uses the `gemma4` architecture. Many PyPI releases (e.g. 4.57.x) do not yet expose `AutoModelForMultimodalLM` / Gemma4. The bundled `requirements.txt` installs **transformers from GitHub main** for that support. If you see `ImportError: AutoModelForMultimodalLM`, run:

```bash
pip install -U "git+https://github.com/huggingface/transformers.git"
```

## Run the main app

```bash
cd dgx_spark_gb10
python vision_studio.py
```

Open **http://localhost:7860**. Example images are read from **`../test_data`** (repository root).

## All entry points (mirrors repo root)

| Script | Port | Description |
|--------|------|-------------|
| `python vision_studio.py` | 7860 | FastAPI + SSE premium UI (Agent + Compare). |
| `python agent_studio.py` | 7860 | Gradio step-by-step agent. |
| `python agent.py` | 7861 | Gradio planning agent (DETECT / VLM / etc.). |
| `python app.py` | 7860 | Gradio image pipeline (Detect, Count, Q&A, Scene). |
| `python demo.py` | 7860 | Gradio unified Image + Video tabs. |
| `python video_tracker.py` | 7861 | Gradio video tracking only. |
| `python main.py` | 7860 | Combined Gradio: image tabs + video tracking. |

Always `cd dgx_spark_gb10` first so imports resolve. Example assets use **`../test_data`**.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `GEMMA_HF_MODEL_ID` | Override Gemma model id (default `google/gemma-4-E4B-it`). |
| `FALCON_HF_MODEL_ID` | Override Falcon checkpoint (default `tiiuae/Falcon-Perception`). |
| `FALCON_HF_REVISION` | Falcon revision on the Hub. |
| `FALCON_HF_LOCAL_DIR` | Load Falcon from a local directory instead of the Hub. |
| `CUDA_DEVICE` | Device string for Falcon, e.g. `cuda:0` (optional). |
| `FALCON_TORCH_COMPILE` | `0` to disable `torch.compile` for Falcon (faster cold start, slower steady state). |
| `FALCON_TORCH_DTYPE` | `bfloat16` (default), `float32`, or `float`. |
| `GEMMA_DO_SAMPLE` | `0` for greedy Gemma decoding; default uses sampling with temperature 0.1 like the MLX path. |

## Verification (smoke test)

With dependencies installed:

```bash
cd dgx_spark_gb10
python3 -m py_compile agent_studio.py app.py video_tracker.py demo.py agent.py main.py vision_studio.py
python3 -c "import agent_studio, app, video_tracker, demo, agent, main, vision_studio"
```

Start the main UI and check the server responds:

```bash
python vision_studio.py
# In another shell:
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/
curl -sS http://127.0.0.1:7860/api/examples | python3 -m json.tool | head
```

A full agent run loads Falcon + Gemma and requires GPU memory, HF auth for Gemma, and patience on the first Falcon compile.

## Notes

- **First Falcon run** can take noticeable time while PyTorch compiles and (if enabled) captures CUDA graphs; this is expected on upstream Falcon Perception.
- **VRAM**: Gemma 4 E4B-it plus Falcon is sized for a workstation GPU; reduce batch usage or use a smaller Gemma id via `GEMMA_HF_MODEL_ID` if you hit OOM.
- **`demo.py` video tab** re-encodes with **H.264** via `ffmpeg` when `ffmpeg` is on `PATH` (the macOS-only `/opt/homebrew/bin/ffmpeg` path is not used here).
