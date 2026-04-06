"""
Vision Agent: Falcon Perception + Gemma VLM
=============================================
An agentic pipeline where Gemma 4 acts as the reasoning brain:
1. Parses user intent from natural language
2. Plans which objects to detect/segment
3. Calls Falcon Perception as a tool
4. Reasons over annotated results
5. Can chain multiple steps for complex queries

Examples:
  "Are there more cars than people?" → detect cars, detect people, compare
  "What's the dog near the red car doing?" → detect dog, detect car, crop, reason
  "Count everything in this scene" → identify objects, detect each, tally
"""

import os, time, json, re, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import gradio as gr

# ── Shared model state ────────────────────────────────────────────────

falcon_model = None
falcon_tokenizer = None
falcon_args = None
gemma_model = None
gemma_processor = None

FALCON_ID = "tiiuae/Falcon-Perception"
GEMMA_ID = "mlx-community/gemma-4-e4b-it-8bit"

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (128, 128, 0),
]


def _font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _load_falcon():
    global falcon_model, falcon_tokenizer, falcon_args
    if falcon_model is not None:
        return
    from falcon_perception import load_from_hf_export_mlx
    falcon_model, falcon_tokenizer, falcon_args = load_from_hf_export_mlx(
        hf_model_id=FALCON_ID, dtype="float16",
    )


def _load_gemma():
    global gemma_model, gemma_processor
    if gemma_model is not None:
        return
    from mlx_vlm import load
    gemma_model, gemma_processor = load(GEMMA_ID)


def _ensure_models():
    _load_falcon()
    _load_gemma()


# ── Tools the agent can use ───────────────────────────────────────────

def tool_detect(image: Image.Image, query: str, task: str = "segmentation") -> dict:
    """Run Falcon Perception. Returns count + list of detections."""
    from falcon_perception import build_prompt_for_task
    from falcon_perception.mlx.batch_inference import (
        BatchInferenceEngine, process_batch_and_generate,
    )
    prompt = build_prompt_for_task(query, task)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)

    batch = process_batch_and_generate(
        falcon_tokenizer, [(tmp.name, prompt)],
        max_length=falcon_args.max_seq_len,
        min_dimension=256, max_dimension=1024,
        patch_size=falcon_args.spatial_patch_size,
    )
    engine = BatchInferenceEngine(falcon_model, falcon_tokenizer)
    _, aux = engine.generate(
        tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
        max_new_tokens=100, temperature=0.0, task=task,
    )
    os.unlink(tmp.name)

    w, h = image.size
    bboxes = aux[0].bboxes_raw
    masks_rle = aux[0].masks_rle
    dets, i, mi = [], 0, 0
    while i < len(bboxes):
        d = {}
        if "x" in bboxes[i]:
            d["cx"], d["cy"] = bboxes[i]["x"], bboxes[i]["y"]
            i += 1
        if i < len(bboxes) and "h" in bboxes[i]:
            d["bh"], d["bw"] = bboxes[i]["h"], bboxes[i]["w"]
            i += 1
        else:
            i += 1
        if "cx" in d and "bh" in d:
            cx, cy = d["cx"] * w, d["cy"] * h
            bh, bw = d["bh"] * h, d["bw"] * w
            d["bbox"] = [max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
                         min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))]
        if task == "segmentation" and mi < len(masks_rle):
            rle = masks_rle[mi]
            m = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
            mask = mask_utils.decode(m).astype(bool)
            if mask.shape != (h, w):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                ) > 127
            d["mask"] = mask
        if d:
            dets.append(d)
        mi += 1

    return {"query": query, "count": len(dets), "detections": dets}


def tool_annotate(image: Image.Image, results: list[dict]) -> Image.Image:
    """Draw all detection results on the image."""
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    f14, f11 = _font(14), _font(11)

    global_idx = 0
    for result in results:
        query = result["query"]
        for det in result["detections"]:
            c = PALETTE[global_idx % len(PALETTE)]
            if "mask" in det:
                md = np.zeros((*det["mask"].shape, 4), dtype=np.uint8)
                md[det["mask"]] = (*c, 70)
                overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA"))
                draw = ImageDraw.Draw(overlay)
            if "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
                lbl = f"{query} #{global_idx + 1}"
                tb = draw.textbbox((x1, max(0, y1 - 17)), lbl, font=f11)
                draw.rectangle(tb, fill=(*c, 200))
                draw.text((x1, max(0, y1 - 17)), lbl, fill=(255, 255, 255), font=f11)
            global_idx += 1

    # Summary HUD
    parts = [f"{r['count']} {r['query']}" for r in results]
    hud = " | ".join(parts)
    tb = draw.textbbox((10, 10), hud, font=f14)
    draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4], fill=(0, 0, 0, 180))
    draw.text((10, 10), hud, fill=(255, 255, 255), font=f14)

    return overlay.convert("RGB")


def tool_crop(image: Image.Image, bbox: list, padding: int = 10) -> Image.Image:
    """Crop a region from the image."""
    w, h = image.size
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)
    return image.crop((x1, y1, x2, y2))


def tool_vlm(image: Image.Image, prompt: str) -> str:
    """Ask Gemma a question about an image."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    config = load_config(GEMMA_ID)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    fmt = apply_chat_template(gemma_processor, config, prompt, num_images=1)
    res = generate(gemma_model, gemma_processor, fmt, [tmp.name],
                   verbose=False, max_tokens=512, temperature=0.1)
    os.unlink(tmp.name)
    return res.text if hasattr(res, "text") else str(res)


# ── Agent: plan → execute → reason ───────────────────────────────────

PLAN_PROMPT = """You are a vision analysis agent. Given a user query about an image, create an action plan.

Available tools:
- DETECT(object): Detect and segment all instances of 'object' in the image. Returns count and bounding boxes.
- CROP(object, index): Crop the Nth detection of 'object' for closer inspection.
- VLM(question): Ask a visual question about the current image (or a cropped region).
- COMPARE(object1, object2): Compare counts of two detected object types.
- ANSWER(text): Provide the final answer to the user.

Respond with a JSON action plan. Example:

Query: "Are there more dogs than cats?"
Plan:
```json
[
  {"tool": "DETECT", "args": {"object": "dog"}},
  {"tool": "DETECT", "args": {"object": "cat"}},
  {"tool": "COMPARE", "args": {"object1": "dog", "object2": "cat"}},
  {"tool": "ANSWER", "args": {"template": "There are {dog_count} dogs and {cat_count} cats. {comparison}."}}
]
```

Query: "Describe the largest vehicle"
Plan:
```json
[
  {"tool": "DETECT", "args": {"object": "vehicle"}},
  {"tool": "CROP", "args": {"object": "vehicle", "index": "largest"}},
  {"tool": "VLM", "args": {"question": "Describe this vehicle in detail."}},
  {"tool": "ANSWER", "args": {"template": "{vlm_response}"}}
]
```

Query: "What objects are in this image?"
Plan:
```json
[
  {"tool": "VLM", "args": {"question": "List the main objects visible in this image, separated by commas."}},
  {"tool": "DETECT_EACH", "args": {"source": "vlm_list"}},
  {"tool": "ANSWER", "args": {"template": "Scene contains: {detection_summary}"}}
]
```

Now create a plan for this query. Return ONLY the JSON array, no other text.
Query: "{query}"
Plan:
"""


def agent_plan(query: str) -> list[dict]:
    """Use Gemma to create an action plan from a natural language query."""
    # For common patterns, use hardcoded plans (faster + more reliable)
    q = query.lower().strip().rstrip("?").rstrip(".")

    # Pattern: "count everything" / "count all objects" → scene scan
    if re.search(r"count\s+(?:everything|all|every\s+object|all\s+object)", q):
        return [
            {"tool": "VLM", "args": {"question": "List the main distinct object types visible in this image, as comma-separated single nouns only. Be concise."}},
            {"tool": "DETECT_EACH", "args": {"source": "vlm_list"}},
            {"tool": "ANSWER", "args": {"template": "Scene inventory: {detection_summary}"}}
        ]

    # Pattern: "count X" or "how many X" — add VLM if query is complex
    count_match = re.search(r"(?:count|how many)\s+(.+?)(?:\s+are|\s+in|\s+on|\s*$)", q)
    if count_match:
        obj = count_match.group(1).strip()
        # If query has extra parts (e.g. "and what breeds?"), also ask VLM
        if " and " in q or "what" in q or "which" in q or "describe" in q:
            return [
                {"tool": "DETECT", "args": {"object": obj}},
                {"tool": "VLM", "args": {"question": query}},
                {"tool": "ANSWER", "args": {"template": "Detected {count} " + obj + "(s). {vlm_response}"}}
            ]
        return [
            {"tool": "DETECT", "args": {"object": obj}},
            {"tool": "ANSWER", "args": {"template": f"I detected {{count}} {obj}(s)."}}
        ]

    # Pattern: "more X than Y" or "X vs Y"
    compare_match = re.search(r"more\s+(\w+)\s+than\s+(\w+)", q)
    if not compare_match:
        compare_match = re.search(r"(\w+)\s+(?:vs|versus|or)\s+(\w+)", q)
    if compare_match:
        obj1, obj2 = compare_match.group(1), compare_match.group(2)
        return [
            {"tool": "DETECT", "args": {"object": obj1}},
            {"tool": "DETECT", "args": {"object": obj2}},
            {"tool": "COMPARE", "args": {"object1": obj1, "object2": obj2}},
            {"tool": "ANSWER", "args": {"template": "{comparison}"}}
        ]

    # Pattern: "describe the largest/smaller/Nth X" → detect + crop + describe
    describe_obj = re.search(
        r"describe\s+the\s+(largest|biggest|smallest|smaller|larger|left|right|first|last|\d+(?:st|nd|rd|th)?)\s+(.+)", q
    )
    if describe_obj:
        selector, obj = describe_obj.group(1), describe_obj.group(2).strip()
        idx = "largest" if selector in ("largest", "biggest", "larger") else 0
        return [
            {"tool": "DETECT", "args": {"object": obj}},
            {"tool": "CROP", "args": {"object": obj, "index": idx}},
            {"tool": "VLM", "args": {"question": f"Describe this {obj} in detail."}},
            {"tool": "ANSWER", "args": {"template": "{vlm_response}"}}
        ]

    # Pattern: "what is/are" or "describe" → scene understanding
    if any(q.startswith(w) for w in ("what is", "what are", "what's", "describe", "tell me about",
                                      "explain", "analyze", "what can you see")):
        return [
            {"tool": "VLM", "args": {"question": query}},
            {"tool": "ANSWER", "args": {"template": "{vlm_response}"}}
        ]

    # Pattern: "find X" or just an object name
    find_match = re.search(r"(?:find|locate|show|detect|segment|where)\s+(?:the\s+|all\s+)?(.+)", q)
    if find_match:
        obj = find_match.group(1).strip()
        return [
            {"tool": "DETECT", "args": {"object": obj}},
            {"tool": "VLM", "args": {"question": f"Describe the detected {obj}(s) in this annotated image."}},
            {"tool": "ANSWER", "args": {"template": "Found {count} {obj}(s). {vlm_response}"}}
        ]

    # Pattern: "everything" or "all objects" → scene scan
    if any(w in q for w in ("everything", "all objects", "all items", "scene", "whole image")):
        return [
            {"tool": "VLM", "args": {"question": "List the main distinct object types visible, as comma-separated nouns."}},
            {"tool": "DETECT_EACH", "args": {"source": "vlm_list"}},
            {"tool": "ANSWER", "args": {"template": "{detection_summary}"}}
        ]

    # Default: treat the whole query as an object to detect + describe
    # Try to extract a noun, or use full query for VLM
    words = q.split()
    if len(words) <= 3:
        # Short query = probably an object name
        return [
            {"tool": "DETECT", "args": {"object": q}},
            {"tool": "VLM", "args": {"question": f"Describe the {q}(s) detected in this image."}},
            {"tool": "ANSWER", "args": {"template": "Detected {count} {obj}(s). {vlm_response}"}}
        ]
    else:
        # Long query = complex question
        return [
            {"tool": "VLM", "args": {"question": query}},
            {"tool": "ANSWER", "args": {"template": "{vlm_response}"}}
        ]


def agent_execute(image: Image.Image, query: str, progress=None):
    """Execute the full agent pipeline: plan → detect → reason → answer."""
    _ensure_models()

    plan = agent_plan(query)
    log = []
    detection_results = {}  # object_name → result dict
    all_results = []  # for annotation
    annotated = image.copy()
    context = {}  # shared state between steps

    log.append(f"**Query:** {query}")
    log.append(f"**Plan:** {len(plan)} step(s)")

    for step_idx, step in enumerate(plan):
        tool = step["tool"]
        args = step.get("args", {})

        if progress:
            progress((step_idx) / len(plan), desc=f"Step {step_idx+1}/{len(plan)}: {tool}...")

        if tool == "DETECT":
            obj = args["object"]
            log.append(f"\n**Step {step_idx+1}: DETECT** '{obj}'")
            t0 = time.time()
            result = tool_detect(image, obj)
            dt = time.time() - t0
            detection_results[obj] = result
            all_results.append(result)
            annotated = tool_annotate(image, all_results)
            context["count"] = result["count"]
            context["obj"] = obj
            context[f"{obj}_count"] = result["count"]
            log.append(f"  → Found **{result['count']}** '{obj}' in {dt:.1f}s")

        elif tool == "DETECT_EACH":
            # Detect multiple object types from VLM's list
            vlm_list = context.get("vlm_list", "")
            if not vlm_list and "vlm_response" in context:
                vlm_list = context["vlm_response"]
            candidates = [c.strip().lower().rstrip('.') for c in vlm_list.split(",")]
            candidates = [c for c in candidates if c and len(c) < 25][:6]
            log.append(f"\n**Step {step_idx+1}: DETECT_EACH** [{', '.join(candidates)}]")

            summary_parts = []
            for obj in candidates:
                try:
                    result = tool_detect(image, obj)
                    if result["count"] > 0:
                        detection_results[obj] = result
                        all_results.append(result)
                        summary_parts.append(f"{result['count']} {obj}")
                        log.append(f"  → {obj}: **{result['count']}**")
                except Exception as e:
                    log.append(f"  → {obj}: failed ({e})")

            annotated = tool_annotate(image, all_results)
            context["detection_summary"] = ", ".join(summary_parts) if summary_parts else "no objects detected"

        elif tool == "CROP":
            obj = args.get("object", "")
            index = args.get("index", 0)
            log.append(f"\n**Step {step_idx+1}: CROP** '{obj}' #{index}")
            if obj in detection_results and detection_results[obj]["detections"]:
                dets = detection_results[obj]["detections"]
                if index == "largest":
                    # Pick the detection with largest bbox area
                    det = max(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]) if "bbox" in d else 0)
                else:
                    idx = int(index) if isinstance(index, (int, str)) and str(index).isdigit() else 0
                    det = dets[min(idx, len(dets)-1)]
                if "bbox" in det:
                    cropped = tool_crop(image, det["bbox"])
                    context["cropped"] = cropped
                    log.append(f"  → Cropped region {det['bbox']}")

        elif tool == "VLM":
            question = args.get("question", query)
            log.append(f"\n**Step {step_idx+1}: VLM** '{question[:60]}...'")
            # Use cropped image if available, otherwise annotated
            target = context.get("cropped", annotated)
            t0 = time.time()
            response = tool_vlm(target, question)
            dt = time.time() - t0
            context["vlm_response"] = response
            context["vlm_list"] = response
            log.append(f"  → Response ({dt:.1f}s): {response[:150]}...")

        elif tool == "COMPARE":
            obj1 = args.get("object1", "")
            obj2 = args.get("object2", "")
            c1 = context.get(f"{obj1}_count", 0)
            c2 = context.get(f"{obj2}_count", 0)
            if c1 > c2:
                comp = f"Yes, there are more {obj1} ({c1}) than {obj2} ({c2})."
            elif c2 > c1:
                comp = f"No, there are more {obj2} ({c2}) than {obj1} ({c1})."
            else:
                comp = f"They're equal: {c1} {obj1} and {c2} {obj2}."
            context["comparison"] = comp
            log.append(f"\n**Step {step_idx+1}: COMPARE** → {comp}")

        elif tool == "ANSWER":
            template = args.get("template", "{vlm_response}")
            try:
                answer = template.format(**context)
            except KeyError:
                answer = template
                for k, v in context.items():
                    answer = answer.replace(f"{{{k}}}", str(v))
            context["final_answer"] = answer
            log.append(f"\n**Answer:** {answer}")

    if progress:
        progress(1.0, desc="Done!")

    final_answer = context.get("final_answer", context.get("vlm_response", "No answer generated."))
    agent_log = "\n".join(log)

    return annotated, final_answer, agent_log


# ── Gradio UI ─────────────────────────────────────────────────────────

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def run_agent(image, query, progress=gr.Progress()):
    if image is None:
        return None, "Upload an image first.", ""
    if not query or not query.strip():
        return None, "Enter a query.", ""

    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image
    annotated, answer, log = agent_execute(pil, query.strip(), progress=progress)
    return annotated, answer, log


def build_agent_ui():
    demo = gr.Blocks(title="Vision Agent")

    with demo:
        gr.Markdown(
            "# Vision Agent\n"
            "An agentic pipeline that **plans**, **detects**, and **reasons** over images.\n\n"
            "Ask natural language questions — the agent decides which tools to use:\n"
            "- **DETECT**: Find & segment objects with Falcon Perception\n"
            "- **VLM**: Visual reasoning with Gemma 4\n"
            "- **CROP**: Zoom into specific detections\n"
            "- **COMPARE**: Count comparisons between object types\n"
            "- **DETECT_EACH**: Auto-discover and detect all object types\n\n"
            "**Falcon Perception** (0.6B) + **Gemma 4** (4B) — fully local via MLX"
        )

        with gr.Row(equal_height=True):
            with gr.Column():
                img_in = gr.Image(label="Upload Image", type="numpy", height=350)
                query_in = gr.Textbox(
                    label="Query",
                    placeholder="e.g. 'Are there more cars than people?' or 'Find all dogs' or 'Describe everything'",
                    lines=2,
                )
                run_btn = gr.Button("Ask Agent", variant="primary", size="lg")
            with gr.Column():
                img_out = gr.Image(label="Annotated Result", height=350)
                answer_out = gr.Textbox(label="Answer", lines=4)

        with gr.Accordion("Agent Log (step-by-step reasoning)", open=False):
            log_out = gr.Markdown(label="Agent Log")

        run_btn.click(run_agent, [img_in, query_in], [img_out, answer_out, log_out])

        gr.Examples(
            examples=[
                [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "How many dogs are there and what breeds?"],
                [os.path.join(EXAMPLES_DIR, "street.jpg"), "Are there more cars than people?"],
                [os.path.join(EXAMPLES_DIR, "street.jpg"), "Find all vehicles"],
                [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "Describe the larger dog"],
                [os.path.join(EXAMPLES_DIR, "street.jpg"), "Count everything in this scene"],
                [os.path.join(EXAMPLES_DIR, "kitchen.jpg"), "What objects are on the counter?"],
            ],
            inputs=[img_in, query_in],
            label="Try these examples",
        )

    return demo


if __name__ == "__main__":
    demo = build_agent_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
