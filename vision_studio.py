"""
Vision Agent Studio — Premium Web UI
======================================
FastAPI + SSE backend with inline HTML/CSS/JS frontend.
Two modes: Agent Pipeline (step-by-step) and Compare (Gemma-only vs Falcon+Gemma).
"""

import os, sys, io, time, json, base64, tempfile, re
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
import uvicorn

# ── Import pipeline functions from agent_studio ───────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_studio import (
    _ensure, _load_falcon, _load_gemma,
    _detect, _vlm,
    _render_detections, _render_comparison,
    _save_step_img, _img_to_b64,
    _ask_gemma_for_next_step,
    initial_plan, STEP_META, MAX_AGENT_STEPS,
    PALETTE, STEP_DIR,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA = os.path.join(BASE_DIR, "test_data")

EXAMPLES = [
    {"filename": "dogs.jpg",    "query": "How many dogs and what breeds?"},
    {"filename": "street.jpg",  "query": "Are there more cars than people?"},
    {"filename": "street.jpg",  "query": "Find all vehicles"},
    {"filename": "dogs.jpg",    "query": "Describe the largest dog"},
    {"filename": "kitchen.jpg", "query": "What is happening in this image?"},
]

app = FastAPI(title="Vision Agent Studio")

# ── Helpers ────────────────────────────────────────────────────────────

def decode_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")

def image_to_b64(img: Image.Image, max_w=800) -> str:
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=78)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def thumb_b64(path: str, w=160) -> str:
    img = Image.open(path)
    ratio = w / img.width
    img = img.resize((w, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

# ── Agent event generator ─────────────────────────────────────────────

def execute_agent_events(img: Image.Image, query: str):
    """Yields structured event dicts for each agent step."""
    steps = initial_plan(query)
    ctx = {"obj": query}
    detection_cache = {}
    current_img = img.copy()

    yield {"type": "plan", "steps": [
        {"tool": s["tool"], "label": s.get("label", s["tool"])} for s in steps
    ]}

    step_idx = 0
    executed = 0

    while step_idx < len(steps) and executed < MAX_AGENT_STEPS:
        step = steps[step_idx]
        tool = step["tool"]
        label = step.get("label", tool)
        meta = STEP_META.get(tool, STEP_META.get("ANSWER"))

        yield {"type": "step_start", "step_index": executed, "tool": tool,
               "label": label, "model": meta["model"], "model_size": meta["model_size"],
               "task": meta["task"], "color": meta["color"]}

        t0 = time.time()
        detail = ""
        img_b64 = None

        if tool == "DETECT":
            obj = step.get("object", ctx.get("obj", "object"))
            dets = _detect(img, obj)
            detection_cache[obj] = dets
            current_img = _render_detections(img, dets, obj)
            ctx.update(count=len(dets), obj=obj)
            ctx[f"{obj}_count"] = len(dets)
            detail = f"Found {len(dets)} instance(s) of '{obj}'"
            img_b64 = image_to_b64(current_img)

        elif tool == "DETECT_EACH":
            vlm_text = ctx.get("vlm_response", "")
            # Parse object list from various formats
            raw = vlm_text
            raw = re.sub(r'\*\*([^*]+)\*\*', r'\1', raw)
            raw = re.sub(r'^\s*[\d]+[.)]\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'^\s*[-*]\s*', '', raw, flags=re.MULTILINE)
            parts_raw = re.split(r'[,\n*]|\band\b', raw)
            candidates = []
            for p in parts_raw:
                c = p.strip().lower().rstrip('.').strip()
                if c and len(c) < 25 and len(c.split()) <= 3:
                    candidates.append(c)
            candidates = candidates[:8]
            parts, color_off = [], 0
            combined = img.copy()
            for obj in candidates:
                try:
                    dets = _detect(img, obj)
                    if dets:
                        detection_cache[obj] = dets
                        combined = _render_detections(combined, dets, obj, color_offset=color_off)
                        color_off += len(dets)
                        parts.append(f"{len(dets)} {obj}")
                except Exception:
                    pass
            ctx["detection_summary"] = ", ".join(parts) if parts else "none found"
            current_img = combined
            detail = f"Scanned {len(candidates)} types: {ctx['detection_summary']}"
            img_b64 = image_to_b64(current_img)

        elif tool == "CROP":
            obj = step.get("object", "")
            index = step.get("index", 0)
            dets = detection_cache.get(obj, [])
            if dets:
                if index == "largest":
                    det = max(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]) if "bbox" in d else 0)
                else:
                    det = dets[min(int(index) if str(index).isdigit() else 0, len(dets)-1)]
                if "bbox" in det:
                    pad, w, h = 15, img.width, img.height
                    x1, y1, x2, y2 = det["bbox"]
                    cropped = img.crop((max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)))
                    ctx["cropped"] = cropped
                    current_img = cropped
                    detail = f"Cropped region [{x1},{y1},{x2},{y2}]"
                    img_b64 = image_to_b64(cropped)
            else:
                detail = "No detections to crop"

        elif tool == "COMPARE":
            a, b = step.get("a", ""), step.get("b", "")
            ca, cb = ctx.get(f"{a}_count", 0), ctx.get(f"{b}_count", 0)
            dets_a, dets_b = detection_cache.get(a, []), detection_cache.get(b, [])
            if ca > cb: comp = f"More {a} ({ca}) than {b} ({cb})"
            elif cb > ca: comp = f"More {b} ({cb}) than {a} ({ca})"
            else: comp = f"Equal: {ca} each"
            ctx["comparison"] = comp
            current_img = _render_comparison(img, dets_a, dets_b, a, b)
            detail = f"{a}: {ca} | {b}: {cb} — {comp}"
            img_b64 = image_to_b64(current_img)

        elif tool == "VLM":
            question = step.get("question", None)
            if question is None:
                if detection_cache:
                    obj = ctx.get("obj", "object")
                    n = ctx.get("count", 0)
                    question = f"This image shows {n} detected '{obj}' highlighted with colored bounding boxes and segmentation masks. {query}"
                else:
                    question = query
            target = ctx.get("cropped", current_img)
            response = _vlm(target, question)
            ctx["vlm_response"] = response
            ctx["vlm_list"] = response
            detail = response

        elif tool == "VLM_PLAN":
            state = f"Detections: {', '.join(f'{k}: {len(v)}' for k,v in detection_cache.items()) or 'none'}\n"
            if "vlm_response" in ctx:
                state += f"VLM: {ctx['vlm_response'][:200]}\n"
            next_action = _ask_gemma_for_next_step(current_img, query, state)
            action = next_action.get("action", "DONE")
            detail = f"Agent decided: {action}"
            if action != "DONE":
                detail += f" — '{next_action.get('object', next_action.get('question', ''))}'"

            remaining = steps[step_idx+1:]
            if action == "DETECT":
                obj = next_action.get("object", "object")
                steps = steps[:step_idx+1] + [
                    {"tool": "DETECT", "label": f"Segment '{obj}'", "object": obj},
                    {"tool": "VLM", "label": "Analyze detections",
                     "question": f"Now that '{obj}' objects are highlighted, {query}"},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
            elif action == "CROP":
                obj = next_action.get("object", ctx.get("obj", ""))
                steps = steps[:step_idx+1] + [
                    {"tool": "CROP", "label": f"Crop {obj}", "object": obj,
                     "index": next_action.get("index", "largest")},
                    {"tool": "VLM", "label": "Describe cropped region",
                     "question": next_action.get("question", f"Describe this {obj}.")},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
            elif action == "VLM":
                steps = steps[:step_idx+1] + [
                    {"tool": "VLM", "label": "Follow-up analysis",
                     "question": next_action.get("question", query)},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
            else:
                ctx["final_answer"] = next_action.get("answer", ctx.get("vlm_response", ""))
                steps = steps[:step_idx+1] + [{"tool": "ANSWER", "label": "Final answer"}]

        elif tool == "ANSWER":
            answer = ctx.get("final_answer", "")
            if not answer:
                parts = []
                if "count" in ctx: parts.append(f"Found {ctx['count']} {ctx.get('obj', 'object')}(s).")
                if "comparison" in ctx: parts.append(ctx["comparison"])
                if "detection_summary" in ctx: parts.append(f"Scene: {ctx['detection_summary']}")
                if "vlm_response" in ctx: parts.append(ctx["vlm_response"])
                answer = " ".join(parts) if parts else "Analysis complete."
            detail = answer

        dt = time.time() - t0
        yield {"type": "step_complete", "step_index": executed, "tool": tool,
               "label": label, "duration_s": round(dt, 1), "detail": detail,
               "image_b64": img_b64, "model": meta["model"], "model_size": meta["model_size"],
               "task": meta["task"], "color": meta["color"]}

        step_idx += 1
        executed += 1

    # Build structured JSON output
    json_output = {
        "query": query,
        "image_size": {"width": img.width, "height": img.height},
        "detections": {},
        "answer": ctx.get("final_answer", detail),
    }
    for obj_name, dets in detection_cache.items():
        json_output["detections"][obj_name] = {
            "count": len(dets),
            "instances": []
        }
        for i, d in enumerate(dets):
            inst = {"id": i + 1}
            if "bbox" in d:
                inst["bbox"] = {"x1": d["bbox"][0], "y1": d["bbox"][1],
                                "x2": d["bbox"][2], "y2": d["bbox"][3]}
            if "cx" in d:
                inst["center"] = {"x": round(d["cx"], 4), "y": round(d["cy"], 4)}
            if "mask" in d:
                inst["mask_area_px"] = int(d["mask"].sum())
                inst["mask_coverage"] = round(float(d["mask"].sum()) / (img.width * img.height), 4)
            json_output["detections"][obj_name]["instances"].append(inst)
    if "comparison" in ctx:
        json_output["comparison"] = ctx["comparison"]
    if "vlm_response" in ctx:
        json_output["vlm_analysis"] = ctx["vlm_response"]

    yield {"type": "done", "json_output": json_output}


# ── API endpoints ─────────────────────────────────────────────────────

@app.get("/api/examples")
def get_examples():
    result = []
    for ex in EXAMPLES:
        path = os.path.join(TEST_DATA, ex["filename"])
        if os.path.exists(path):
            result.append({
                "filename": ex["filename"],
                "query": ex["query"],
                "thumbnail": thumb_b64(path),
            })
    return JSONResponse(result)

@app.get("/api/test-image/{filename}")
def get_test_image(filename: str):
    path = os.path.join(TEST_DATA, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, 404)

@app.post("/api/agent/stream")
async def agent_stream(request: Request):
    data = await request.json()
    img = decode_image(data["image_b64"])
    query = data["query"]

    def generate():
        yield sse({"type": "loading_models"})
        _ensure()
        yield sse({"type": "models_ready"})
        for event in execute_agent_events(img, query):
            yield sse(event)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/api/compare/stream")
async def compare_stream(request: Request):
    data = await request.json()
    img = decode_image(data["image_b64"])
    query = data["query"]

    def generate():
        yield sse({"type": "loading_models"})
        _ensure()
        yield sse({"type": "models_ready"})
        total_t0 = time.time()

        # Pipeline A: Gemma-only
        yield sse({"type": "pipeline_start", "pipeline": "gemma_only", "label": "Gemma 3n (VLM Only)"})
        meta = STEP_META["VLM"]
        yield sse({"type": "step_start", "pipeline": "gemma_only", "step_index": 0,
                   "tool": "VLM", "label": "Direct visual reasoning",
                   "model": meta["model"], "model_size": meta["model_size"],
                   "task": "Visual Q&A", "color": meta["color"]})
        t0 = time.time()
        answer = _vlm(img, query)
        dt = time.time() - t0
        yield sse({"type": "step_complete", "pipeline": "gemma_only", "step_index": 0,
                   "tool": "VLM", "label": "Direct visual reasoning",
                   "duration_s": round(dt, 1), "detail": answer, "image_b64": None,
                   "model": meta["model"], "model_size": meta["model_size"],
                   "task": "Visual Q&A", "color": meta["color"]})
        yield sse({"type": "pipeline_done", "pipeline": "gemma_only",
                   "total_duration_s": round(dt, 1)})

        # Pipeline B: Falcon + Gemma
        yield sse({"type": "pipeline_start", "pipeline": "falcon_gemma",
                   "label": "Falcon + Gemma (Full Agent)"})
        pipeline_t0 = time.time()
        for event in execute_agent_events(img, query):
            event["pipeline"] = "falcon_gemma"
            yield sse(event)
        pipeline_dt = time.time() - pipeline_t0
        yield sse({"type": "pipeline_done", "pipeline": "falcon_gemma",
                   "total_duration_s": round(pipeline_dt, 1)})

        yield sse({"type": "done", "total_duration_s": round(time.time() - total_t0, 1)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Frontend ──────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vision Agent Studio</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-root:#09090b;--bg-surface:#18181b;--bg-elevated:#27272a;--bg-overlay:#3f3f46;
  --text-1:#fafafa;--text-2:#a1a1aa;--text-3:#71717a;
  --indigo:#6366f1;--violet:#8b5cf6;--amber:#f59e0b;--cyan:#06b6d4;--emerald:#10b981;--rose:#f43f5e;
  --radius-sm:6px;--radius-md:8px;--radius-lg:12px;--radius-xl:16px;
  --font:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Roboto,sans-serif;
  --mono:"SF Mono","Fira Code",monospace;
}
html{background:var(--bg-root);color:var(--text-1);font-family:var(--font);font-size:14px;line-height:1.5}
body{min-height:100vh;display:flex;flex-direction:column;align-items:center}
a{color:var(--indigo);text-decoration:none}

/* Header */
.header{text-align:center;padding:32px 0 16px}
.header h1{font-size:24px;font-weight:700;letter-spacing:-0.03em;color:var(--text-1)}
.header p{font-size:13px;color:var(--text-3);margin-top:4px}
.header-badges{display:flex;gap:8px;justify-content:center;margin-top:10px}
.header-badge{font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500}
.header-badge.falcon{color:var(--indigo);background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25)}
.header-badge.gemma{color:var(--violet);background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.25)}

/* Tabs */
.tab-bar{display:flex;gap:4px;background:var(--bg-surface);border-radius:10px;padding:3px;margin:16px 0 24px;border:1px solid var(--bg-elevated)}
.tab-btn{padding:8px 24px;border:none;border-radius:8px;background:transparent;color:var(--text-3);font-size:13px;font-weight:600;cursor:pointer;transition:all .2s}
.tab-btn.active{background:var(--bg-elevated);color:var(--text-1)}
.tab-btn:hover:not(.active){color:var(--text-2)}

/* Layout */
.app{width:100%;max-width:1200px;padding:0 24px}
.tab-content{display:none}.tab-content.active{display:block}
.agent-layout{display:grid;grid-template-columns:380px 1fr;gap:24px;align-items:start}
.compare-layout{display:grid;grid-template-columns:320px 1fr 1fr;gap:20px;align-items:start}

/* Upload zone */
.upload-zone{border:2px dashed var(--bg-overlay);border-radius:var(--radius-lg);padding:40px 20px;text-align:center;cursor:pointer;transition:all .2s;position:relative;min-height:200px;display:flex;flex-direction:column;align-items:center;justify-content:center;overflow:hidden}
.upload-zone:hover{border-color:var(--text-3)}
.upload-zone.dragover{border-color:var(--indigo);background:rgba(99,102,241,.04)}
.upload-zone.has-image{padding:0;border-style:solid;border-color:var(--bg-overlay)}
.upload-zone img{width:100%;height:100%;object-fit:cover;border-radius:10px}
.upload-icon{width:32px;height:32px;color:var(--text-3);margin-bottom:8px}
.upload-text{font-size:13px;color:var(--text-3)}
.upload-clear{position:absolute;top:8px;right:8px;width:28px;height:28px;border-radius:50%;background:rgba(0,0,0,.7);border:none;color:var(--text-1);cursor:pointer;display:none;align-items:center;justify-content:center;font-size:16px;z-index:2}
.upload-zone.has-image .upload-clear{display:flex}
.upload-zone.has-image .upload-placeholder{display:none}

/* Query input */
.query-row{display:flex;gap:8px;margin-top:12px}
.query-input{flex:1;background:var(--bg-surface);border:1px solid var(--bg-elevated);color:var(--text-1);padding:10px 14px;border-radius:var(--radius-md);font-size:14px;font-family:var(--font);outline:none;transition:border .2s}
.query-input:focus{border-color:var(--indigo)}
.query-input::placeholder{color:var(--text-3)}
.run-btn{padding:10px 28px;border:none;border-radius:var(--radius-md);background:var(--indigo);color:#fff;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;white-space:nowrap;font-family:var(--font)}
.run-btn:hover:not(:disabled){filter:brightness(1.15);box-shadow:0 0 20px rgba(99,102,241,.25)}
.run-btn:disabled{opacity:.5;cursor:not-allowed}
.run-btn.processing{background:var(--bg-elevated);color:var(--text-2)}

/* Examples */
.examples{display:flex;gap:10px;overflow-x:auto;padding:16px 0 4px;scrollbar-width:none}
.examples::-webkit-scrollbar{display:none}
.example-card{flex-shrink:0;width:150px;background:var(--bg-surface);border:1px solid var(--bg-elevated);border-radius:var(--radius-md);cursor:pointer;transition:all .2s;overflow:hidden}
.example-card:hover{border-color:var(--text-3);transform:translateY(-2px)}
.example-card img{width:100%;height:75px;object-fit:cover}
.example-card .example-query{padding:8px 10px;font-size:12px;color:var(--text-2);line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}

/* Results panel */
.results-panel{min-height:300px;max-height:80vh;overflow-y:auto;position:relative;scrollbar-width:thin;scrollbar-color:var(--bg-overlay) transparent}
.results-empty{color:var(--text-3);text-align:center;padding:80px 20px;font-size:14px}
.progress-bar{height:3px;background:var(--bg-elevated);border-radius:2px;margin-bottom:16px;overflow:hidden}
.progress-fill{height:100%;background:var(--indigo);width:0%;transition:width .3s ease-out;border-radius:2px}

/* Step card */
.step-card{background:var(--bg-surface);border-left:3px solid var(--bg-overlay);border-radius:0 var(--radius-md) var(--radius-md) 0;padding:16px 20px;margin-bottom:14px;animation:slideUp .3s cubic-bezier(.16,1,.3,1) both}
@keyframes slideUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.step-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.step-icon{width:18px;height:18px;flex-shrink:0}
.step-label{font-size:14px;font-weight:600;color:var(--text-1)}
.step-time{font-size:12px;color:var(--text-3);margin-left:auto;font-family:var(--mono)}
.step-badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.step-badge{font-size:11px;padding:2px 8px;border-radius:4px;font-weight:500}
.step-detail{font-size:13px;color:var(--text-2);line-height:1.7}
.step-image{margin-top:10px;border-radius:var(--radius-md);overflow:hidden;border:1px solid var(--bg-elevated)}
.step-image img{width:100%;display:block}

/* Loading step */
.step-card.loading{opacity:.5;animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:.3}}

/* Compare panels */
.compare-panel{background:var(--bg-surface);border-radius:var(--radius-lg);padding:20px;border:1px solid var(--bg-elevated);min-height:200px;max-height:80vh;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--bg-overlay) transparent}
.compare-panel .panel-header{display:flex;align-items:center;gap:8px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--bg-elevated)}
.compare-panel .panel-title{font-size:15px;font-weight:600}
.compare-panel .panel-subtitle{font-size:11px;color:var(--text-3)}
.panel-time{font-size:12px;color:var(--text-3);margin-left:auto;font-family:var(--mono)}
.compare-panel.violet{border-top:2px solid var(--violet)}
.compare-panel.indigo{border-top:2px solid var(--indigo)}

/* Loading overlay */
.loading-models{display:flex;align-items:center;gap:10px;padding:16px 20px;background:var(--bg-surface);border-radius:var(--radius-md);color:var(--text-2);font-size:13px;margin-bottom:14px}
.spinner{width:16px;height:16px;border:2px solid var(--bg-overlay);border-top-color:var(--indigo);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* JSON output */
.json-toggle{background:var(--bg-surface);border:1px solid var(--bg-elevated);color:var(--text-2);padding:8px 16px;border-radius:var(--radius-md);font-size:13px;cursor:pointer;font-family:var(--font);transition:all .2s;width:100%}
.json-toggle:hover{border-color:var(--text-3);color:var(--text-1)}
.json-output{background:var(--bg-surface);border:1px solid var(--bg-elevated);border-radius:var(--radius-md);padding:16px;margin-top:8px;font-family:var(--mono);font-size:12px;color:var(--text-2);line-height:1.6;overflow-x:auto;max-height:500px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;scrollbar-width:thin;scrollbar-color:var(--bg-overlay) transparent}

/* Responsive */
@media(max-width:900px){
  .agent-layout,.compare-layout{grid-template-columns:1fr}
}
</style>
</head>
<body>

<div class="header">
  <h1>Vision Agent Studio</h1>
  <p>Step-by-step agentic visual reasoning, fully local via MLX</p>
  <div class="header-badges">
    <span class="header-badge falcon">Falcon Perception 0.6B</span>
    <span class="header-badge gemma">Gemma 3n 4B</span>
  </div>
</div>

<div class="app">
  <div style="display:flex;justify-content:center">
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('agent')">Agent Pipeline</button>
      <button class="tab-btn" onclick="switchTab('compare')">Compare</button>
    </div>
  </div>

  <!-- AGENT TAB -->
  <div id="tab-agent" class="tab-content active">
    <div class="agent-layout">
      <div class="input-col">
        <div class="upload-zone" id="agent-upload" onclick="document.getElementById('agent-file').click()">
          <input type="file" id="agent-file" accept="image/*" hidden>
          <div class="upload-placeholder">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 16V4m0 0L7 9m5-5l5 5M4 20h16"/></svg>
            <div class="upload-text">Drop image here or click to upload</div>
          </div>
          <button class="upload-clear" onclick="event.stopPropagation();clearImage('agent')">&times;</button>
        </div>
        <div class="query-row">
          <input type="text" class="query-input" id="agent-query" placeholder="e.g. Find all cars, How many dogs?">
          <button class="run-btn" id="agent-run" onclick="runAgent()">Run</button>
        </div>
        <div class="examples" id="agent-examples"></div>
      </div>
      <div class="results-panel" id="agent-results">
        <div class="results-empty">Results will appear here step by step</div>
      </div>
      <div id="agent-json-wrap" style="display:none;margin-top:12px;">
        <button onclick="toggleJson()" class="json-toggle" id="json-toggle-btn">Show JSON Output</button>
        <pre id="agent-json" class="json-output" style="display:none;"></pre>
      </div>
    </div>
  </div>

  <!-- COMPARE TAB -->
  <div id="tab-compare" class="tab-content">
    <div class="compare-layout">
      <div class="input-col">
        <div class="upload-zone" id="compare-upload" onclick="document.getElementById('compare-file').click()">
          <input type="file" id="compare-file" accept="image/*" hidden>
          <div class="upload-placeholder">
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 16V4m0 0L7 9m5-5l5 5M4 20h16"/></svg>
            <div class="upload-text">Drop image or click</div>
          </div>
          <button class="upload-clear" onclick="event.stopPropagation();clearImage('compare')">&times;</button>
        </div>
        <div class="query-row">
          <input type="text" class="query-input" id="compare-query" placeholder="e.g. How many dogs?">
          <button class="run-btn" id="compare-run" onclick="runCompare()">Compare</button>
        </div>
        <div class="examples" id="compare-examples"></div>
      </div>
      <div class="compare-panel violet" id="panel-gemma">
        <div class="panel-header">
          <div><div class="panel-title" style="color:var(--violet)">Gemma 3n Only</div>
          <div class="panel-subtitle">VLM reasoning without detection</div></div>
          <span class="panel-time" id="gemma-time"></span>
        </div>
        <div id="gemma-steps"><div class="results-empty">Waiting...</div></div>
      </div>
      <div class="compare-panel indigo" id="panel-falcon">
        <div class="panel-header">
          <div><div class="panel-title" style="color:var(--indigo)">Falcon + Gemma</div>
          <div class="panel-subtitle">Detection + segmentation + reasoning</div></div>
          <span class="panel-time" id="falcon-time"></span>
        </div>
        <div id="falcon-steps"><div class="results-empty">Waiting...</div></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──
const state = { agent: { image: null }, compare: { image: null } };

// ── SVG Icons ──
const ICONS = {
  DETECT: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="5"/><path d="M13 13l4 4"/></svg>',
  DETECT_EACH: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="7" height="7" rx="1"/><rect x="11" y="2" width="7" height="7" rx="1"/><rect x="2" y="11" width="7" height="7" rx="1"/><rect x="11" y="11" width="7" height="7" rx="1"/></svg>',
  VLM: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10 2a6 6 0 014 10.5V15a2 2 0 01-2 2H8a2 2 0 01-2-2v-2.5A6 6 0 0110 2z"/><path d="M8 18h4"/></svg>',
  VLM_PLAN: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10 2a6 6 0 014 10.5V15a2 2 0 01-2 2H8a2 2 0 01-2-2v-2.5A6 6 0 0110 2z"/><path d="M8 18h4"/><circle cx="10" cy="8" r="1" fill="currentColor"/></svg>',
  CROP: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 0v15h15M0 5h15v15"/></svg>',
  COMPARE: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M10 2v16M3 6l3-3 3 3M14 14l3 3 3-3"/></svg>',
  ANSWER: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="10" cy="10" r="8"/><path d="M6 10l3 3 5-6"/></svg>',
};

// ── Tabs ──
function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  document.querySelectorAll('.tab-btn')[tab === 'agent' ? 0 : 1].classList.add('active');
}

// ── Image upload ──
function setupUpload(prefix) {
  const zone = document.getElementById(prefix + '-upload');
  const input = document.getElementById(prefix + '-file');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('dragover'); if(e.dataTransfer.files[0]) handleFile(prefix, e.dataTransfer.files[0]); });
  input.addEventListener('change', () => { if(input.files[0]) handleFile(prefix, input.files[0]); });
}

function handleFile(prefix, file) {
  const reader = new FileReader();
  reader.onload = e => {
    state[prefix].image = e.target.result;
    const zone = document.getElementById(prefix + '-upload');
    zone.classList.add('has-image');
    let img = zone.querySelector('img.preview');
    if (!img) { img = document.createElement('img'); img.className = 'preview'; zone.insertBefore(img, zone.firstChild); }
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function clearImage(prefix) {
  state[prefix].image = null;
  const zone = document.getElementById(prefix + '-upload');
  zone.classList.remove('has-image');
  const img = zone.querySelector('img.preview');
  if (img) img.remove();
  document.getElementById(prefix + '-file').value = '';
}

// ── Examples ──
async function loadExamples() {
  const res = await fetch('/api/examples');
  const examples = await res.json();
  ['agent', 'compare'].forEach(prefix => {
    const container = document.getElementById(prefix + '-examples');
    container.innerHTML = examples.map((ex, i) =>
      '<div class="example-card" onclick="selectExample(\\'' + prefix + '\\',\\'' + ex.filename + '\\',\\'' + ex.query.replace(/'/g, "\\\\'") + '\\')">' +
      '<img src="' + ex.thumbnail + '" alt="">' +
      '<div class="example-query">' + ex.query + '</div></div>'
    ).join('');
  });
}

async function selectExample(prefix, filename, query) {
  const res = await fetch('/api/test-image/' + filename);
  const blob = await res.blob();
  const reader = new FileReader();
  reader.onload = e => {
    state[prefix].image = e.target.result;
    const zone = document.getElementById(prefix + '-upload');
    zone.classList.add('has-image');
    let img = zone.querySelector('img.preview');
    if (!img) { img = document.createElement('img'); img.className = 'preview'; zone.insertBefore(img, zone.firstChild); }
    img.src = e.target.result;
    document.getElementById(prefix + '-query').value = query;
  };
  reader.readAsDataURL(blob);
}

// ── SSE reader ──
async function readSSE(response, handler) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try { handler(JSON.parse(line.slice(6))); } catch(e) {}
      }
    }
  }
}

// ── Step card HTML ──
function renderStep(ev) {
  const icon = ICONS[ev.tool] || ICONS.ANSWER;
  const color = ev.color || '#71717a';
  let badges = '';
  if (ev.model && ev.model !== '—') {
    badges += '<span class="step-badge" style="color:' + color + ';background:' + color + '18;border:1px solid ' + color + '40">' + ev.model + ' (' + ev.model_size + ')</span>';
  }
  if (ev.task) {
    badges += '<span class="step-badge" style="color:var(--text-3);background:var(--bg-elevated);border:1px solid var(--bg-overlay)">' + ev.task + '</span>';
  }
  let imgHtml = '';
  if (ev.image_b64) {
    imgHtml = '<div class="step-image"><img src="' + ev.image_b64 + '"></div>';
  }
  return '<div class="step-card" style="border-left-color:' + color + '">' +
    '<div class="step-head"><span class="step-icon" style="color:' + color + '">' + icon + '</span>' +
    '<span class="step-label">' + ev.label + '</span>' +
    '<span class="step-time">' + (ev.duration_s != null ? ev.duration_s + 's' : '') + '</span></div>' +
    (badges ? '<div class="step-badges">' + badges + '</div>' : '') +
    '<div class="step-detail">' + (ev.detail || '') + '</div>' +
    imgHtml + '</div>';
}

function renderLoading(ev) {
  const icon = ICONS[ev.tool] || ICONS.ANSWER;
  const color = ev.color || '#71717a';
  return '<div class="step-card loading" id="loading-step" style="border-left-color:' + color + '">' +
    '<div class="step-head"><span class="step-icon" style="color:' + color + '">' + icon + '</span>' +
    '<span class="step-label">' + ev.label + '</span>' +
    '<span class="step-time"><span class="spinner"></span></span></div>' +
    '<div class="step-detail" style="color:var(--text-3)">Running...</div></div>';
}

// ── Agent Pipeline ──
async function runAgent() {
  if (!state.agent.image) return alert('Upload an image first');
  const query = document.getElementById('agent-query').value.trim();
  if (!query) return alert('Enter a query');
  const btn = document.getElementById('agent-run');
  const panel = document.getElementById('agent-results');
  btn.disabled = true; btn.textContent = 'Processing...'; btn.classList.add('processing');
  panel.innerHTML = '';

  const res = await fetch('/api/agent/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: state.agent.image, query })
  });

  await readSSE(res, ev => {
    if (ev.type === 'loading_models') {
      panel.innerHTML = '<div class="loading-models"><span class="spinner"></span>Loading models...</div>';
    } else if (ev.type === 'models_ready') {
      panel.innerHTML = '';
    } else if (ev.type === 'step_start') {
      const loader = document.getElementById('loading-step');
      if (loader) loader.remove();
      panel.innerHTML += renderLoading(ev);
    } else if (ev.type === 'step_complete') {
      const loader = document.getElementById('loading-step');
      if (loader) loader.remove();
      panel.innerHTML += renderStep(ev);
    } else if (ev.type === 'done') {
      btn.disabled = false; btn.textContent = 'Run'; btn.classList.remove('processing');
      if (ev.json_output) {
        const wrap = document.getElementById('agent-json-wrap');
        const pre = document.getElementById('agent-json');
        wrap.style.display = 'block';
        pre.textContent = JSON.stringify(ev.json_output, null, 2);
        pre.style.display = 'none';
        document.getElementById('json-toggle-btn').textContent = 'Show JSON Output';
      }
    }
  });
  btn.disabled = false; btn.textContent = 'Run'; btn.classList.remove('processing');
}

function toggleJson() {
  const pre = document.getElementById('agent-json');
  const btn = document.getElementById('json-toggle-btn');
  if (pre.style.display === 'none') {
    pre.style.display = 'block';
    btn.textContent = 'Hide JSON Output';
  } else {
    pre.style.display = 'none';
    btn.textContent = 'Show JSON Output';
  }
}

// ── Compare ──
async function runCompare() {
  if (!state.compare.image) return alert('Upload an image first');
  const query = document.getElementById('compare-query').value.trim();
  if (!query) return alert('Enter a query');
  const btn = document.getElementById('compare-run');
  btn.disabled = true; btn.textContent = 'Processing...'; btn.classList.add('processing');
  document.getElementById('gemma-steps').innerHTML = '';
  document.getElementById('falcon-steps').innerHTML = '';
  document.getElementById('gemma-time').textContent = '';
  document.getElementById('falcon-time').textContent = '';

  const res = await fetch('/api/compare/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_b64: state.compare.image, query })
  });

  await readSSE(res, ev => {
    const pipeline = ev.pipeline;
    const target = pipeline === 'gemma_only' ? 'gemma-steps' : (pipeline === 'falcon_gemma' ? 'falcon-steps' : null);

    if (ev.type === 'loading_models') {
      document.getElementById('gemma-steps').innerHTML = '<div class="loading-models"><span class="spinner"></span>Loading models...</div>';
    } else if (ev.type === 'models_ready') {
      document.getElementById('gemma-steps').innerHTML = '';
    } else if (ev.type === 'pipeline_start' && target) {
      document.getElementById(target).innerHTML = '';
    } else if (ev.type === 'step_start' && target) {
      const container = document.getElementById(target);
      const loader = container.querySelector('.loading');
      if (loader) loader.remove();
      container.innerHTML += renderLoading(ev);
    } else if (ev.type === 'step_complete' && target) {
      const container = document.getElementById(target);
      const loader = container.querySelector('.loading');
      if (loader) loader.remove();
      container.innerHTML += renderStep(ev);
    } else if (ev.type === 'pipeline_done') {
      const timeEl = ev.pipeline === 'gemma_only' ? 'gemma-time' : 'falcon-time';
      document.getElementById(timeEl).textContent = ev.total_duration_s + 's total';
    } else if (ev.type === 'done') {
      btn.disabled = false; btn.textContent = 'Compare'; btn.classList.remove('processing');
    }
  });
  btn.disabled = false; btn.textContent = 'Compare'; btn.classList.remove('processing');
}

// ── Init ──
setupUpload('agent');
setupUpload('compare');
loadExamples();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
