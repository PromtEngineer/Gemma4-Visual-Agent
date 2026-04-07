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
| `FALCON_TORCH_COMPILE` | `1` to enable `torch.compile` for Falcon (slower cold start, faster steady state). Default: disabled. |
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

## Video Object Tracking

The current `video_tracker.py` runs **Falcon Perception per-frame** with a greedy **IoU-based tracker** (`SimpleTracker`). This works for basic use cases but has limitations: per-frame detection is slow (~1-4s/frame), the IoU matcher has no motion model or appearance features, and each frame is treated independently with no temporal memory.

Below are model options evaluated for upgrading the tracking pipeline on CUDA hardware.

### Option 1: SAM 2 — Segment Anything Model 2 (recommended upgrade)

Meta's foundation model for video object segmentation. Prompt it on the first frame (with a point, box, or mask from Falcon), and it **propagates segmentation across all subsequent frames** using a streaming memory mechanism — no per-frame detection needed.

| Variant | Params | Use case |
|---------|--------|----------|
| `sam2.1-hiera-tiny` | 39M | Fastest, lightweight |
| `sam2.1-hiera-small` | 46M | Good balance |
| `sam2.1-hiera-base+` | 81M | Higher quality |
| `sam2.1-hiera-large` | 224M | Best quality |

- **Architecture:** Falcon detects on frame 1 → SAM 2 tracks masks across all frames → re-detect periodically for new objects
- **Advantage:** Eliminates per-frame detection; temporal consistency built in; supports `torch.compile` with `vos_optimized=True`
- **Repo:** [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (Apache 2.0)

### Option 2: Grounded SAM 2 (Grounding DINO + SAM 2)

Combines an open-vocabulary text-prompted detector (Grounding DINO 1.5/1.6 or Florence-2) with SAM 2 for detect + segment + track in one pipeline. Natural language prompts like the current system.

- **Advantage:** Text-prompted detection + SAM 2 tracking in a single pipeline; continuous ID tracking built in
- **Trade-off:** Adds Grounding DINO as a dependency; if keeping Falcon as detector, plain SAM 2 (Option 1) is cleaner
- **Repo:** [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

### Option 3: ByteTrack / BoT-SORT (easiest drop-in)

Production-grade multi-object tracking algorithms that work with **any detector**. Replace `SimpleTracker` with proper Kalman filtering, Hungarian matching, and appearance features.

- **ByteTrack:** Two-stage association (high + low confidence detections) handles occlusion well; 60.1 HOTA on MOT17
- **BoT-SORT:** Adds camera-motion compensation and appearance re-ID; 65.0 HOTA on MOT17
- **Install:** `pip install trackers` (Roboflow `trackers` v2.3.0, Apache 2.0)
- **Trade-off:** Still requires per-frame detection, so speed is limited by Falcon's inference time; but tracking quality improves significantly

### Option 4: CoTracker3 (dense point tracking)

Meta's transformer-based model that tracks **dense points** (up to 70k) jointly across video. Tracks motion trajectories rather than object bounding boxes.

- **Use case:** Motion analysis, object deformation, fine-grained trajectory tracking
- **Trade-off:** Tracks points, not objects — needs separate object-to-point association; best as an add-on, not a replacement
- **Repo:** [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker)

### Option 5: YOLO-World (fast open-vocabulary detection)

Real-time open-vocabulary detector (~52 FPS on V100) that understands text prompts. Could serve as the fast per-frame detector for video while keeping Falcon for high-quality single-image analysis.

- **Performance:** 35.4 AP on LVIS at 52 FPS
- **Trade-off:** Bounding boxes only (no segmentation); needs SAM 2 on top for masks
- **Repo:** [AILab-CVC/YOLO-World](https://github.com/AILAB-CVC/YOLO-World)

### Recommended combinations

| Goal | Approach | Speed | Quality |
|------|----------|-------|---------|
| Best quality with existing models | Falcon (frame 1) + **SAM 2** (propagate) | Fast (detect once) | Excellent |
| Best quality, fully text-prompted | **Grounded SAM 2** | Fast | Excellent |
| Quickest upgrade, minimal changes | Falcon + **ByteTrack/BoT-SORT** | Same as current | Much better ID consistency |
| Real-time capable | **YOLO-World** + ByteTrack + SAM 2 | ~30+ FPS detect | Good |
| Motion analysis add-on | Any above + **CoTracker3** | Real-time points | Dense trajectories |

## Notes

- **First Falcon run** can take noticeable time while PyTorch compiles and (if enabled) captures CUDA graphs; this is expected on upstream Falcon Perception.
- **VRAM**: Gemma 4 E4B-it plus Falcon is sized for a workstation GPU; reduce batch usage or use a smaller Gemma id via `GEMMA_HF_MODEL_ID` if you hit OOM.
- **`demo.py` video tab** re-encodes with **H.264** via `ffmpeg` when `ffmpeg` is on `PATH` (the macOS-only `/opt/homebrew/bin/ffmpeg` path is not used here).
