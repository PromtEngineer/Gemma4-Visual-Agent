# Vision Agent Studio

An agentic visual reasoning pipeline combining **Falcon Perception** (0.6B, instance segmentation) with **Gemma 4** (4B, visual language model) for object detection, counting, tracking, and scene understanding. Runs fully local on Apple Silicon via MLX.

## Quick Start

```bash
# Activate the environment
source .venv/bin/activate

# Launch the main app (recommended)
python vision_studio.py
# Open http://localhost:7860
```

## What It Does

Upload an image, type a natural language query, and the agent decides what to do:

| Query | What Happens |
|---|---|
| `dog` | Falcon segments all dogs, Gemma describes them |
| `How many cars?` | Falcon counts with exact bounding boxes + masks |
| `Are there more cars than people?` | Falcon detects both, compares counts |
| `Describe the largest dog` | Falcon detects, crops the largest, Gemma describes it |
| `What is happening here?` | Gemma analyzes, then re-plans (may trigger detection) |
| `Count everything` | Gemma lists object types, Falcon detects each one |

Every step shows which model is running (Falcon Perception or Gemma 4), the task type, execution time, and visual output.

## Applications

### Agent Pipeline (main tab)
Step-by-step agentic reasoning with visual output at each stage. The agent can re-plan after seeing results — if Gemma identifies objects, it can trigger Falcon to detect them, then reason over the annotated image.

### Compare Mode (second tab)
Runs the same query through two sub-pipelines side-by-side:
- **Gemma-only**: VLM reasoning without detection
- **Falcon + Gemma**: Full agent pipeline with segmentation

This demonstrates when segmentation is needed vs. when a VLM alone suffices.

### Video Tracking
Frame-by-frame detection with IoU-based tracking to maintain object identities across video. Available via `python demo.py` (Video tab) or `python video_tracker.py`.

## Models

| Model | HuggingFace ID | Params | Quant | Disk | Role |
|---|---|---|---|---|---|
| Falcon Perception | `tiiuae/Falcon-Perception` | 0.6B | float16 | ~1.2 GB | Detection + segmentation |
| Gemma 4 E4B | `mlx-community/gemma-4-e4b-it-8bit` | 4B effective | 8-bit | ~8 GB | Visual reasoning |

Models are downloaded automatically on first run. Total: ~9.2 GB. Requires 16 GB+ RAM (Apple Silicon M1+).

## How the Pipeline Works

```
User: "Are there more cars than people?"
                    |
              Plan Router
           (pattern matching)
            /              \
    DETECT 'cars'    DETECT 'people'
    Falcon (0.6B)    Falcon (0.6B)
    15 found          19 found
            \              /
             COMPARE counts
             cars: 15 | people: 19
                    |
                 ANSWER
    "More people (19) than cars (15)"
```

For open-ended questions, the agent enters a **re-planning loop** where Gemma decides the next action after each step (max 8 steps).

### Available Tools

| Tool | Model | What It Does |
|---|---|---|
| **DETECT** | Falcon Perception (0.6B) | Instance segmentation with bounding boxes + masks |
| **VLM** | Gemma 4 E4B (4B) | Visual reasoning, scene description, Q&A |
| **CROP** | (utility) | Zoom into a specific detection for closer inspection |
| **COMPARE** | (utility) | Compare counts between two object types |
| **DETECT_EACH** | Falcon Perception | Detect multiple object types identified by VLM |
| **VLM_PLAN** | Gemma 4 E4B | Re-planning — Gemma decides what to do next |

### Falcon Perception Internals

Falcon uses a single early-fusion Transformer (no separate vision encoder). For each detected object, it outputs a **Chain-of-Perception** sequence:

1. `<coord>` token — predicts center (x, y) normalized to 0-1
2. `<size>` token — predicts height, width normalized to 0-1
3. `<seg>` token — produces a mask embedding, dot-producted with upsampled image features for a full-resolution binary mask

Output is COCO RLE-encoded and decoded via `pycocotools`.

## Gemma-Only vs. Falcon+Gemma

Results from identical queries on the same images:

| Task | Gemma-Only | Falcon+Gemma |
|---|---|---|
| Count dogs (2 in image) | "two" (correct) | **2** exact + 2 masks |
| Count cars (dense scene) | "at least 10 taxis" (vague) | **16** exact + 16 masks |
| Count people (crowded) | "approximately 25-30" (range) | **31** exact + 31 masks |
| Spatial grounding | No bounding boxes or masks | Per-instance bbox + pixel mask |
| Scene understanding | Strong | Strong (with visual evidence) |
| Speed | 1-4s | 3-25s (depends on objects) |

**When to use Falcon**: Exact counting (>5 objects), spatial localization, instance separation, visual grounding.

**When Gemma alone suffices**: Scene description, mood/context, simple yes/no, breed/type identification.

## File Structure

```
Gemma-4/
├── vision_studio.py      # Main app — FastAPI + premium HTML/CSS/JS UI
│                          # Two tabs: Agent Pipeline + Compare
│
├── agent_studio.py       # Core pipeline logic (detection, VLM, planning,
│                          # re-planning, rendering, step metadata)
│
├── agent.py              # Standalone Gradio agent UI (older version)
├── demo.py               # Gradio unified UI (image + video tabs)
├── app.py                # Gradio image analysis app
├── video_tracker.py      # Video tracking with IoU tracker
├── main.py               # Combined Gradio launcher
│
├── ARCHITECTURE.md       # Detailed architecture docs with Mermaid diagrams
├── README.md             # This file
│
├── test_data/            # Example images + test videos
│   ├── dogs.jpg          # Two dogs (Corgi + Yorkshire Terrier)
│   ├── street.jpg        # NYC street scene (cars + people)
│   ├── kitchen.jpg       # Kitchen scene (people cooking)
│   ├── dogs_video.mp4    # Dogs zoom video (20 frames)
│   └── test_panning.mp4  # Street panning video (30 frames)
│
├── step_outputs/         # Temporary step images for UI rendering
└── .venv/                # Python 3.12 virtual environment
```

## All Entry Points

| Command | Port | Description |
|---|---|---|
| `python vision_studio.py` | 7860 | **Main app** — FastAPI, premium UI, Agent + Compare tabs |
| `python agent_studio.py` | 7860 | Gradio agent UI (step-by-step, older) |
| `python demo.py` | 7860 | Gradio unified UI (image + video tabs) |
| `python video_tracker.py` | 7861 | Gradio video tracking UI |
| `python app.py` | 7860 | Gradio image analysis (original) |

## Setup (from scratch)

```bash
# Create virtual environment
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install mlx-vlm gradio opencv-python-headless Pillow numpy pycocotools
pip install "falcon-perception[mlx] @ git+https://github.com/tiiuae/falcon-perception.git"
pip install fastapi uvicorn

# Launch
python vision_studio.py
```

### Requirements
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 16 GB+ RAM recommended
- ~7 GB disk for model weights (auto-downloaded)

## References

- [Falcon Perception](https://github.com/tiiuae/falcon-perception) — TII's open-vocabulary segmentation model
- [Falcon Perception Paper](https://arxiv.org/abs/2603.27365) — arXiv:2603.27365
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) — Vision language models on Apple Silicon
- [Gemma 4](https://ai.google.dev/gemma/docs/gemma-4) — Google's efficient multimodal model
- [mlx-vlm-falcon](https://github.com/korale77/mlx-vlm-falcon) — Inspiration for the combined pipeline
