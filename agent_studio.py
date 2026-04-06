"""
Vision Agent Studio — Agentic Loop with Re-planning
=====================================================
Falcon Perception (segmentation/detection) + Gemma 3n (reasoning).
Each step shows which model and task is used.
The agent can re-plan after seeing results (multi-step reasoning loop).
"""

import os, time, re, tempfile, base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import gradio as gr

# ── Models ────────────────────────────────────────────────────────────

falcon_model = falcon_tokenizer = falcon_args = None
gemma_model = gemma_processor = None
FALCON_ID = "tiiuae/Falcon-Perception"
GEMMA_ID = "mlx-community/gemma-3n-E4B-it-4bit"

PALETTE = [
    (99, 102, 241), (16, 185, 129), (245, 158, 11), (239, 68, 68),
    (139, 92, 246), (6, 182, 212), (236, 72, 153), (34, 197, 94),
    (251, 146, 60), (168, 85, 247), (56, 189, 248), (251, 191, 36),
]

STEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step_outputs")
os.makedirs(STEP_DIR, exist_ok=True)


def _font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _load_falcon():
    global falcon_model, falcon_tokenizer, falcon_args
    if falcon_model is not None: return
    from falcon_perception import load_from_hf_export_mlx
    falcon_model, falcon_tokenizer, falcon_args = load_from_hf_export_mlx(
        hf_model_id=FALCON_ID, dtype="float16")

def _load_gemma():
    global gemma_model, gemma_processor
    if gemma_model is not None: return
    from mlx_vlm import load
    gemma_model, gemma_processor = load(GEMMA_ID)

def _ensure():
    _load_falcon(); _load_gemma()


# ── Core tools ────────────────────────────────────────────────────────

def _detect(img, query, task="segmentation"):
    from falcon_perception import build_prompt_for_task
    from falcon_perception.mlx.batch_inference import BatchInferenceEngine, process_batch_and_generate
    prompt = build_prompt_for_task(query, task)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); img.save(tmp.name)
    batch = process_batch_and_generate(falcon_tokenizer, [(tmp.name, prompt)],
        max_length=falcon_args.max_seq_len, min_dimension=256, max_dimension=1024,
        patch_size=falcon_args.spatial_patch_size)
    engine = BatchInferenceEngine(falcon_model, falcon_tokenizer)
    _, aux = engine.generate(tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
        max_new_tokens=100, temperature=0.0, task=task)
    os.unlink(tmp.name)
    w, h = img.size; bboxes = aux[0].bboxes_raw; masks_rle = aux[0].masks_rle
    dets, i, mi = [], 0, 0
    while i < len(bboxes):
        d = {}
        if "x" in bboxes[i]: d["cx"], d["cy"] = bboxes[i]["x"], bboxes[i]["y"]; i += 1
        if i < len(bboxes) and "h" in bboxes[i]: d["bh"], d["bw"] = bboxes[i]["h"], bboxes[i]["w"]; i += 1
        else: i += 1
        if "cx" in d and "bh" in d:
            cx, cy = d["cx"]*w, d["cy"]*h; bh, bw = d["bh"]*h, d["bw"]*w
            d["bbox"] = [max(0,int(cx-bw/2)), max(0,int(cy-bh/2)), min(w,int(cx+bw/2)), min(h,int(cy+bh/2))]
        if task == "segmentation" and mi < len(masks_rle):
            rle = masks_rle[mi]; m = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
            mask = mask_utils.decode(m).astype(bool)
            if mask.shape != (h, w):
                mask = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((w,h), Image.NEAREST)) > 127
            d["mask"] = mask
        if d: dets.append(d)
        mi += 1
    return dets


def _vlm(img, prompt):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    config = load_config(GEMMA_ID)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False); img.save(tmp.name)
    fmt = apply_chat_template(gemma_processor, config, prompt, num_images=1)
    res = generate(gemma_model, gemma_processor, fmt, [tmp.name], verbose=False, max_tokens=512, temperature=0.1)
    os.unlink(tmp.name)
    return res.text if hasattr(res, "text") else str(res)


# ── Rendering ─────────────────────────────────────────────────────────

def _render_detections(img, dets, query, color_offset=0):
    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    f12 = _font(12)
    for idx, d in enumerate(dets):
        c = PALETTE[(idx + color_offset) % len(PALETTE)]
        if "mask" in d:
            md = np.zeros((*d["mask"].shape, 4), dtype=np.uint8)
            md[d["mask"]] = (*c, 55)
            overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA"))
            draw = ImageDraw.Draw(overlay)
        if "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=(*c, 200), width=2)
            lbl = f"{query} {idx+1}"
            tb = draw.textbbox((0, 0), lbl, font=f12)
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            px, py = x1, max(0, y1-th-8)
            draw.rounded_rectangle([px, py, px+tw+12, py+th+6], radius=4, fill=(*c, 220))
            draw.text((px+6, py+3), lbl, fill=(255,255,255), font=f12)
    return overlay.convert("RGB")


def _render_comparison(img, dets_a, dets_b, qa, qb):
    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    f12 = _font(12)
    for d in dets_a:
        c = PALETTE[0]
        if "mask" in d:
            md = np.zeros((*d["mask"].shape, 4), dtype=np.uint8); md[d["mask"]] = (*c, 50)
            overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA")); draw = ImageDraw.Draw(overlay)
        if "bbox" in d: draw.rectangle(d["bbox"], outline=(*c, 180), width=2)
    for d in dets_b:
        c = PALETTE[1]
        if "mask" in d:
            md = np.zeros((*d["mask"].shape, 4), dtype=np.uint8); md[d["mask"]] = (*c, 50)
            overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA")); draw = ImageDraw.Draw(overlay)
        if "bbox" in d: draw.rectangle(d["bbox"], outline=(*c, 180), width=2)
    draw.rounded_rectangle([10,10,220,60], radius=8, fill=(0,0,0,180))
    draw.rounded_rectangle([14,16,28,30], radius=2, fill=(*PALETTE[0],255))
    draw.text((34,14), f"{qa}: {len(dets_a)}", fill=(255,255,255), font=f12)
    draw.rounded_rectangle([14,36,28,50], radius=2, fill=(*PALETTE[1],255))
    draw.text((34,34), f"{qb}: {len(dets_b)}", fill=(255,255,255), font=f12)
    return overlay.convert("RGB")


def _save_step_img(img):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=STEP_DIR)
    img.save(tmp.name); return tmp.name

def _img_to_b64(path, max_w=800):
    """Convert image to a compact base64 JPEG data URI."""
    img = Image.open(path)
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Agentic re-planning prompt ────────────────────────────────────────

REPLAN_PROMPT = """You are a vision analysis agent. You have already executed some steps and have intermediate results.

Available tools (respond with EXACTLY one JSON line):
- {{"action": "DETECT", "object": "..."}} — run Falcon Perception segmentation on the image
- {{"action": "CROP", "object": "...", "index": "largest"}} — crop a specific detection for closer look
- {{"action": "VLM", "question": "..."}} — ask a visual question about the current view
- {{"action": "DONE", "answer": "..."}} — provide the final answer

Current state:
{state_summary}

Original user query: "{query}"

What should the next step be? Respond with a single JSON object, nothing else."""


def _ask_gemma_for_next_step(img, query, state_summary):
    """Ask Gemma what to do next given current state."""
    prompt = REPLAN_PROMPT.format(state_summary=state_summary, query=query)
    raw = _vlm(img, prompt)
    # Extract JSON from response
    try:
        # Find first { ... } in the response
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            return eval(match.group())  # safe enough for our controlled format
    except Exception:
        pass
    return {"action": "DONE", "answer": raw}


# ── Step metadata ─────────────────────────────────────────────────────

STEP_META = {
    "DETECT":      {"icon": "🔍", "color": "#6366f1", "model": "Falcon Perception", "model_size": "0.6B", "task": "Instance Segmentation"},
    "DETECT_DET":  {"icon": "🔍", "color": "#6366f1", "model": "Falcon Perception", "model_size": "0.6B", "task": "Object Detection"},
    "DETECT_EACH": {"icon": "🔎", "color": "#818cf8", "model": "Falcon Perception", "model_size": "0.6B", "task": "Multi-class Segmentation"},
    "VLM":         {"icon": "🧠", "color": "#8b5cf6", "model": "Gemma 3n E4B",      "model_size": "4B",   "task": "Visual Reasoning"},
    "VLM_PLAN":    {"icon": "🤔", "color": "#a78bfa", "model": "Gemma 3n E4B",      "model_size": "4B",   "task": "Re-planning"},
    "CROP":        {"icon": "✂️", "color": "#f59e0b", "model": "—",                 "model_size": "",     "task": "Region Crop"},
    "COMPARE":     {"icon": "⚖️", "color": "#06b6d4", "model": "—",                 "model_size": "",     "task": "Count Comparison"},
    "ANSWER":      {"icon": "✅", "color": "#10b981", "model": "—",                 "model_size": "",     "task": "Final Answer"},
}


def _step_html(tool_key, label, dt, detail, img_path=None):
    """Render a single step as HTML."""
    meta = STEP_META.get(tool_key, STEP_META["ANSWER"])
    icon, color = meta["icon"], meta["color"]
    model, model_size, task = meta["model"], meta["model_size"], meta["task"]

    # Model badge
    if model != "—":
        badge = (
            f'<span style="display:inline-flex; align-items:center; gap:4px; '
            f'background:{color}22; border:1px solid {color}44; border-radius:4px; '
            f'padding:2px 8px; font-size:11px; color:{color};">'
            f'{model} ({model_size})'
            f'</span>'
            f'<span style="display:inline-flex; align-items:center; gap:4px; '
            f'background:#37415122; border:1px solid #37415188; border-radius:4px; '
            f'padding:2px 8px; font-size:11px; color:#9ca3af;">'
            f'{task}'
            f'</span>'
        )
    else:
        badge = (
            f'<span style="display:inline-flex; align-items:center; gap:4px; '
            f'background:#37415122; border:1px solid #37415188; border-radius:4px; '
            f'padding:2px 8px; font-size:11px; color:#9ca3af;">'
            f'{task}'
            f'</span>'
        )

    html = f'''
    <div style="margin-bottom:24px; border-left:3px solid {color}; padding-left:16px;">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <span style="font-size:18px;">{icon}</span>
        <span style="font-weight:600; font-size:15px; color:#e5e7eb;">{label}</span>
        <span style="font-size:12px; color:#6b7280; margin-left:auto;">{dt:.1f}s</span>
      </div>
      <div style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px;">
        {badge}
      </div>
      <div style="font-size:13px; color:#d1d5db; line-height:1.6;">
        {detail}
      </div>
    '''
    if img_path and os.path.exists(img_path):
        b64 = _img_to_b64(img_path)
        html += f'''
      <div style="margin-top:10px; border-radius:8px; overflow:hidden; border:1px solid #374151;">
        <img src="{b64}" style="width:100%; display:block;" />
      </div>
        '''
    html += '</div>'
    return html


def _pending_html(tool_key, label):
    meta = STEP_META.get(tool_key, STEP_META["ANSWER"])
    return f'''
    <div style="margin-bottom:24px; border-left:3px solid #374151; padding-left:16px; opacity:0.35;">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:18px;">{meta["icon"]}</span>
        <span style="font-weight:600; font-size:15px; color:#6b7280;">{label}</span>
        <span style="font-size:12px; color:#4b5563; margin-left:auto;">pending</span>
      </div>
    </div>
    '''


# ── Plan generation ───────────────────────────────────────────────────

def initial_plan(query):
    q = query.lower().strip().rstrip("?.!")

    # Count everything → scene scan
    if re.search(r"count\s+(?:everything|all|every\s+object)", q):
        return [
            {"tool": "VLM", "label": "Identify objects in scene"},
            {"tool": "DETECT_EACH", "label": "Detect & segment each type"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # How many X
    m = re.search(r"(?:count|how many)\s+(\w+)", q)
    if m:
        obj = m.group(1).strip()
        steps = [{"tool": "DETECT", "label": f"Segment '{obj}'", "object": obj}]
        if " and " in q or "what" in q or "breed" in q or "type" in q or "which" in q:
            steps.append({"tool": "VLM", "label": "Analyze detections"})
        steps.append({"tool": "ANSWER", "label": "Final answer"})
        return steps

    # Comparison
    m = re.search(r"more\s+(\w+)\s+than\s+(\w+)", q)
    if not m: m = re.search(r"(\w+)\s+(?:vs|versus)\s+(\w+)", q)
    if m:
        a, b = m.group(1), m.group(2)
        return [
            {"tool": "DETECT", "label": f"Segment '{a}'", "object": a},
            {"tool": "DETECT", "label": f"Segment '{b}'", "object": b},
            {"tool": "COMPARE", "label": "Compare counts", "a": a, "b": b},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Describe the [adj] X → detect + crop + VLM
    m = re.search(r"describe\s+the\s+(largest|biggest|smallest|left|right|first|last)\s+(.+)", q)
    if m:
        sel, obj = m.group(1), m.group(2).strip()
        return [
            {"tool": "DETECT", "label": f"Segment '{obj}'", "object": obj},
            {"tool": "CROP", "label": f"Crop {sel} {obj}", "object": obj, "index": "largest" if sel in ("largest","biggest") else 0},
            {"tool": "VLM", "label": f"Describe cropped {obj}"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Find / detect / segment
    m = re.search(r"(?:find|locate|show|detect|segment|where)\s+(?:the\s+|all\s+)?(.+)", q)
    if m:
        obj = m.group(1).strip()
        return [
            {"tool": "DETECT", "label": f"Segment '{obj}'", "object": obj},
            {"tool": "VLM", "label": "Describe detections"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Identify / list / add / name objects → always use detection
    if any(w in q for w in ("identify", "list", "name", "label", "add ", "different",
                             "items", "objects", "things", "stuff", "inventory",
                             "all the", "each", "every ")):
        return [
            {"tool": "VLM", "label": "Identify objects in scene",
             "question": "List the main distinct object types visible in this image as comma-separated single nouns. Be specific (e.g. 'apple' not 'fruit')."},
            {"tool": "DETECT_EACH", "label": "Detect & segment each type"},
            {"tool": "VLM", "label": "Analyze with segmentation results"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Question → VLM first, then detect what was found
    if any(q.startswith(w) for w in ("what", "describe", "tell", "explain", "analyze",
                                      "is ", "are ", "do ", "does ", "can ", "who", "why", "where")):
        return [
            {"tool": "VLM", "label": "Visual analysis"},
            {"tool": "DETECT_EACH", "label": "Verify with segmentation"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Short = object name
    if len(q.split()) <= 3:
        return [
            {"tool": "DETECT", "label": f"Segment '{q}'", "object": q},
            {"tool": "VLM", "label": "Describe detections"},
            {"tool": "ANSWER", "label": "Final answer"},
        ]

    # Default → identify then detect (always use both models)
    return [
        {"tool": "VLM", "label": "Identify objects",
         "question": "List the main distinct object types visible in this image as comma-separated single nouns. Be specific."},
        {"tool": "DETECT_EACH", "label": "Detect & segment each type"},
        {"tool": "VLM", "label": "Analyze results"},
        {"tool": "ANSWER", "label": "Final answer"},
    ]


# ── Agentic execution loop ───────────────────────────────────────────

MAX_AGENT_STEPS = 8  # safety limit


def execute_agent(image, query):
    """Run the agentic loop. Yields (html, current_img) after each step."""
    _ensure()
    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")

    steps = initial_plan(query)
    ctx = {"obj": query}
    detection_cache = {}
    current_img = pil.copy()
    step_htmls = []
    step_idx = 0

    while step_idx < len(steps) and len(step_htmls) < MAX_AGENT_STEPS:
        step = steps[step_idx]
        tool = step["tool"]
        label = step.get("label", tool)
        t0 = time.time()

        if tool == "DETECT":
            obj = step.get("object", ctx.get("obj", "object"))
            dets = _detect(pil, obj)
            dt = time.time() - t0
            detection_cache[obj] = dets
            current_img = _render_detections(pil, dets, obj)
            ctx["count"] = len(dets)
            ctx["obj"] = obj
            ctx[f"{obj}_count"] = len(dets)
            img_path = _save_step_img(current_img)
            step_htmls.append(_step_html("DETECT", label, dt,
                f"Found <b>{len(dets)}</b> instance(s) of '{obj}'", img_path))

        elif tool == "DETECT_EACH":
            vlm_text = ctx.get("vlm_response", "")
            # Parse object list from various formats: commas, bullets, newlines, numbered
            raw = vlm_text
            raw = re.sub(r'\*\*([^*]+)\*\*', r'\1', raw)  # strip **bold**
            raw = re.sub(r'^\s*[\d]+[.)]\s*', '', raw, flags=re.MULTILINE)  # strip "1. ", "2) "
            raw = re.sub(r'^\s*[-*]\s*', '', raw, flags=re.MULTILINE)  # strip "- " or "* "
            # Split on commas, newlines, "and", bullets
            parts_raw = re.split(r'[,\n*]|\band\b', raw)
            candidates = []
            for p in parts_raw:
                c = p.strip().lower().rstrip('.').strip()
                # Take only short noun-like tokens (skip sentences)
                if c and len(c) < 25 and len(c.split()) <= 3:
                    candidates.append(c)
            candidates = candidates[:8]
            parts, color_off = [], 0
            combined = pil.copy()
            for obj in candidates:
                try:
                    dets = _detect(pil, obj)
                    if dets:
                        detection_cache[obj] = dets
                        combined = _render_detections(combined, dets, obj, color_offset=color_off)
                        color_off += len(dets)
                        parts.append(f"{len(dets)} {obj}")
                except Exception: pass
            dt = time.time() - t0
            ctx["detection_summary"] = ", ".join(parts) if parts else "none found"
            current_img = combined
            img_path = _save_step_img(current_img)
            step_htmls.append(_step_html("DETECT_EACH", label, dt,
                f"Scanned {len(candidates)} types: <b>{ctx['detection_summary']}</b>", img_path))

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
                    pad = 15; w, h = pil.size
                    x1, y1, x2, y2 = det["bbox"]
                    cropped = pil.crop((max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)))
                    ctx["cropped"] = cropped
                    current_img = cropped
                    dt = time.time() - t0
                    img_path = _save_step_img(cropped)
                    step_htmls.append(_step_html("CROP", label, dt,
                        f"Cropped region [{x1},{y1},{x2},{y2}]", img_path))
            else:
                dt = time.time() - t0
                step_htmls.append(_step_html("CROP", label, dt, "No detections to crop"))

        elif tool == "COMPARE":
            a, b = step.get("a", ""), step.get("b", "")
            ca, cb = ctx.get(f"{a}_count", 0), ctx.get(f"{b}_count", 0)
            dets_a, dets_b = detection_cache.get(a, []), detection_cache.get(b, [])
            if ca > cb: comp = f"More <b>{a}</b> ({ca}) than <b>{b}</b> ({cb})"
            elif cb > ca: comp = f"More <b>{b}</b> ({cb}) than <b>{a}</b> ({ca})"
            else: comp = f"Equal: <b>{ca}</b> each"
            ctx["comparison"] = comp
            current_img = _render_comparison(pil, dets_a, dets_b, a, b)
            dt = time.time() - t0
            img_path = _save_step_img(current_img)
            step_htmls.append(_step_html("COMPARE", label, dt,
                f"{a}: <b>{ca}</b> &nbsp;|&nbsp; {b}: <b>{cb}</b> &nbsp;→&nbsp; {comp}", img_path))

        elif tool == "VLM":
            question = step.get("question", None)
            if question is None:
                # Auto-generate question based on context
                if detection_cache:
                    obj = ctx.get("obj", "object")
                    n = ctx.get("count", 0)
                    question = f"This image shows {n} detected '{obj}' highlighted with colored bounding boxes and segmentation masks. {query}"
                else:
                    question = query
            target = ctx.get("cropped", current_img)
            response = _vlm(target, question)
            dt = time.time() - t0
            ctx["vlm_response"] = response
            ctx["vlm_list"] = response
            step_htmls.append(_step_html("VLM", label, dt, response))

        elif tool == "VLM_PLAN":
            # === AGENTIC RE-PLANNING ===
            # Gemma decides what to do next based on current state
            state = f"Detections so far: {', '.join(f'{k}: {len(v)} found' for k,v in detection_cache.items()) or 'none'}\n"
            if "vlm_response" in ctx:
                state += f"VLM analysis: {ctx['vlm_response'][:200]}\n"

            next_action = _ask_gemma_for_next_step(current_img, query, state)
            dt = time.time() - t0

            action = next_action.get("action", "DONE")
            step_htmls.append(_step_html("VLM_PLAN", label, dt,
                f"Agent decided: <b>{action}</b>" +
                (f" — '{next_action.get('object', next_action.get('question', ''))}'" if action != "DONE" else "")))

            # Inject new steps based on the decision
            remaining = steps[step_idx+1:]
            if action == "DETECT":
                obj = next_action.get("object", "object")
                new_steps = [
                    {"tool": "DETECT", "label": f"Segment '{obj}'", "object": obj},
                    {"tool": "VLM", "label": "Analyze new detections",
                     "question": f"Now that '{obj}' objects are highlighted, {query}"},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
                steps = steps[:step_idx+1] + new_steps
            elif action == "CROP":
                obj = next_action.get("object", ctx.get("obj", ""))
                new_steps = [
                    {"tool": "CROP", "label": f"Crop {obj}", "object": obj,
                     "index": next_action.get("index", "largest")},
                    {"tool": "VLM", "label": "Describe cropped region",
                     "question": next_action.get("question", f"Describe this {obj} in detail.")},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
                steps = steps[:step_idx+1] + new_steps
            elif action == "VLM":
                new_steps = [
                    {"tool": "VLM", "label": "Follow-up analysis",
                     "question": next_action.get("question", query)},
                    {"tool": "ANSWER", "label": "Final answer"},
                ]
                steps = steps[:step_idx+1] + new_steps
            else:  # DONE
                ctx["final_answer"] = next_action.get("answer", ctx.get("vlm_response", ""))
                steps = steps[:step_idx+1] + [{"tool": "ANSWER", "label": "Final answer"}]

        elif tool == "ANSWER":
            dt = time.time() - t0
            answer = ctx.get("final_answer", "")
            if not answer:
                # Build answer from context
                parts = []
                if "count" in ctx:
                    parts.append(f"Found {ctx['count']} {ctx.get('obj', 'object')}(s).")
                if "comparison" in ctx:
                    parts.append(ctx["comparison"])
                if "detection_summary" in ctx:
                    parts.append(f"Scene: {ctx['detection_summary']}")
                if "vlm_response" in ctx:
                    parts.append(ctx["vlm_response"])
                answer = " ".join(parts) if parts else "Analysis complete."
            step_htmls.append(_step_html("ANSWER", label, dt, answer))

        # Yield current state
        yield "\n".join(step_htmls), current_img
        step_idx += 1

    yield "\n".join(step_htmls), current_img


# ── Gradio app ────────────────────────────────────────────────────────

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

def run_agent_studio(image, query):
    if image is None:
        return "<p style='color:#6b7280;'>Upload an image to get started.</p>", None
    if not query or not query.strip():
        return "<p style='color:#6b7280;'>Enter a query.</p>", None
    html, img = None, None
    for html, img in execute_agent(image, query.strip()):
        pass
    return html, img


def build_app():
    demo = gr.Blocks(title="Vision Agent Studio")

    with demo:
        gr.HTML("""
        <div style="text-align:center; padding:20px 0 8px;">
            <h1 style="font-size:28px; font-weight:700; letter-spacing:-0.5px; color:#f9fafb; margin:0;">
                Vision Agent Studio
            </h1>
            <p style="font-size:14px; color:#9ca3af; margin:4px 0 0;">
                Falcon Perception + Gemma 3n &mdash; agentic step-by-step visual reasoning, fully local via MLX
            </p>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                img_in = gr.Image(label="", type="numpy", height=360,
                                  container=False, show_label=False)
                query_in = gr.Textbox(
                    placeholder="'Find all cars'  ·  'Are there more dogs than cats?'  ·  'Describe the largest vehicle'  ·  'What is happening here?'",
                    show_label=False, container=False, lines=1,
                )
                run_btn = gr.Button("Run Agent", variant="primary", size="lg")

                gr.Examples(
                    examples=[
                        [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "How many dogs and what breeds?"],
                        [os.path.join(EXAMPLES_DIR, "street.jpg"), "Are there more cars than people?"],
                        [os.path.join(EXAMPLES_DIR, "street.jpg"), "Find all vehicles"],
                        [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "Describe the largest dog"],
                        [os.path.join(EXAMPLES_DIR, "kitchen.jpg"), "What is happening in this image?"],
                    ],
                    inputs=[img_in, query_in], label="",
                )

            with gr.Column(scale=3):
                steps_html = gr.HTML(
                    value='<p style="color:#6b7280; text-align:center; padding:80px 0;">Results will appear here step by step</p>',
                )
                final_img = gr.Image(label="", height=1, visible=False)

        run_btn.click(run_agent_studio, [img_in, query_in], [steps_html, final_img])

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True,
                allowed_paths=[STEP_DIR, EXAMPLES_DIR])
