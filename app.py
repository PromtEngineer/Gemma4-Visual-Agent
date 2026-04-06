"""
Falcon Perception + Gemma VLM: Object Detection, Segmentation & Visual Reasoning
==================================================================================
A Gradio-based application combining:
- Falcon Perception (MLX) for object detection & instance segmentation
- Gemma 3n (MLX) for visual reasoning & natural language understanding

Applications: counting, identification, scene analysis, visual Q&A
"""

import os
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import gradio as gr

# ── Global state ──────────────────────────────────────────────────────

falcon_model = None
falcon_tokenizer = None
falcon_args = None
gemma_model = None
gemma_processor = None

FALCON_MODEL_ID = "tiiuae/Falcon-Perception"
GEMMA_MODEL_ID = "mlx-community/gemma-3n-E4B-it-4bit"

# Color palette for segmentation masks
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (150, 200, 50), (200, 50, 150),
]


def load_falcon():
    """Load Falcon Perception model (MLX)."""
    global falcon_model, falcon_tokenizer, falcon_args
    if falcon_model is not None:
        return
    from falcon_perception import load_from_hf_export_mlx
    falcon_model, falcon_tokenizer, falcon_args = load_from_hf_export_mlx(
        hf_model_id=FALCON_MODEL_ID, dtype="float16"
    )
    print("Falcon Perception loaded.")


def load_gemma():
    """Load Gemma 3n VLM (MLX)."""
    global gemma_model, gemma_processor
    if gemma_model is not None:
        return
    from mlx_vlm import load
    gemma_model, gemma_processor = load(GEMMA_MODEL_ID)
    print("Gemma 3n VLM loaded.")


def load_all_models():
    """Load both models."""
    load_falcon()
    load_gemma()
    return "Models loaded successfully."


# ── Falcon Perception inference ──────────────────────────────────────

def run_falcon_perception(image: Image.Image, query: str, task: str = "segmentation"):
    """Run Falcon Perception on an image with a text query.

    Returns: list of predictions, each with xy, hw, and optionally mask.
    """
    import mlx.core as mx
    from falcon_perception import build_prompt_for_task
    from falcon_perception.mlx.batch_inference import (
        BatchInferenceEngine, process_batch_and_generate,
    )

    prompt = build_prompt_for_task(query, task)

    # Save image temporarily for the data loader
    tmp_path = "/tmp/_falcon_input.png"
    image.save(tmp_path)

    batch = process_batch_and_generate(
        falcon_tokenizer,
        [(tmp_path, prompt)],
        max_length=falcon_args.max_seq_len,
        min_dimension=256,
        max_dimension=1024,
        patch_size=falcon_args.spatial_patch_size,
    )

    engine = BatchInferenceEngine(falcon_model, falcon_tokenizer)
    padded_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=100,
        temperature=0.0,
        task=task,
    )

    aux = aux_outputs[0]
    predictions = parse_aux_output(aux, image.size, task)
    return predictions


def parse_aux_output(aux, image_size, task="segmentation"):
    """Parse AuxOutput into structured predictions."""
    w_img, h_img = image_size
    results = []

    bboxes = aux.bboxes_raw
    masks_rle = aux.masks_rle

    # Pair up coordinates and sizes
    i = 0
    detection_idx = 0
    while i < len(bboxes):
        pred = {}
        if "x" in bboxes[i]:
            pred["center_x"] = bboxes[i]["x"]
            pred["center_y"] = bboxes[i]["y"]
            i += 1
        if i < len(bboxes) and "h" in bboxes[i]:
            pred["h"] = bboxes[i]["h"]
            pred["w"] = bboxes[i]["w"]
            i += 1
        else:
            i += 1

        # Compute bounding box in pixel coords
        if "center_x" in pred and "h" in pred:
            cx, cy = pred["center_x"] * w_img, pred["center_y"] * h_img
            bh, bw = pred["h"] * h_img, pred["w"] * w_img
            pred["bbox"] = [
                max(0, int(cx - bw / 2)),
                max(0, int(cy - bh / 2)),
                min(w_img, int(cx + bw / 2)),
                min(h_img, int(cy + bh / 2)),
            ]

        # Attach mask if available
        if task == "segmentation" and detection_idx < len(masks_rle):
            rle = masks_rle[detection_idx]
            m = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
            mask_np = mask_utils.decode(m).astype(bool)
            # Resize mask to original image dimensions if needed
            if mask_np.shape != (h_img, w_img):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask_np.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((w_img, h_img), PILImage.NEAREST)
                mask_np = np.array(mask_pil) > 127
            pred["mask"] = mask_np

        if pred:
            results.append(pred)
        detection_idx += 1

    return results


# ── Visualization ─────────────────────────────────────────────────────

def visualize_detections(image: Image.Image, predictions: list, query: str):
    """Draw bounding boxes, masks, and labels on the image."""
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    for idx, pred in enumerate(predictions):
        color = COLORS[idx % len(COLORS)]
        label = f"{query} #{idx + 1}"

        # Draw mask with transparency
        if "mask" in pred:
            mask = pred["mask"]
            mask_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            mask_data = np.zeros((*mask.shape, 4), dtype=np.uint8)
            mask_data[mask] = (*color, 80)  # semi-transparent
            mask_overlay = Image.fromarray(mask_data, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_overlay)
            draw = ImageDraw.Draw(overlay)

        # Draw bounding box
        if "bbox" in pred:
            x1, y1, x2, y2 = pred["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # Label background
            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font_small)
            draw.rectangle(text_bbox, fill=(*color, 200))
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font_small)

    # Count annotation
    count_text = f"Found {len(predictions)} '{query}' object(s)"
    text_bbox = draw.textbbox((10, 10), count_text, font=font)
    draw.rectangle(
        [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
        fill=(0, 0, 0, 180),
    )
    draw.text((10, 10), count_text, fill=(255, 255, 255), font=font)

    return overlay.convert("RGB")


# ── Gemma VLM reasoning ──────────────────────────────────────────────

def run_gemma_reasoning(image: Image.Image, prompt: str):
    """Run Gemma VLM on an image with a text prompt."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    config = load_config(GEMMA_MODEL_ID)

    # Save image temporarily
    tmp_path = "/tmp/_gemma_input.png"
    image.save(tmp_path)

    formatted_prompt = apply_chat_template(
        gemma_processor, config, prompt, num_images=1
    )
    result = generate(
        gemma_model, gemma_processor, formatted_prompt,
        [tmp_path], verbose=False, max_tokens=512, temperature=0.1,
    )
    return result.text if hasattr(result, 'text') else str(result)


# ── Combined pipeline ─────────────────────────────────────────────────

def detect_and_analyze(image, query, task_type, analysis_prompt):
    """Full pipeline: Falcon detect → Visualize → Gemma analyze."""
    if image is None:
        return None, "Please upload an image.", ""

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    # Phase 1: Extract target class from query using simple heuristics
    target_class = query.strip()
    if not target_class:
        target_class = "object"

    # Phase 2: Run Falcon Perception
    task = "segmentation" if task_type == "Segmentation" else "detection"
    t0 = time.time()
    predictions = run_falcon_perception(image_pil, target_class, task=task)
    detect_time = time.time() - t0

    # Phase 3: Visualize
    annotated = visualize_detections(image_pil, predictions, target_class)

    # Phase 4: Generate analysis
    detection_summary = f"Detected {len(predictions)} instance(s) of '{target_class}'.\n"
    detection_summary += f"Detection time: {detect_time:.2f}s\n\n"

    for idx, pred in enumerate(predictions):
        if "bbox" in pred:
            x1, y1, x2, y2 = pred["bbox"]
            detection_summary += f"  #{idx+1}: bbox=({x1},{y1})-({x2},{y2})"
            if "center_x" in pred:
                detection_summary += f" center=({pred['center_x']:.3f},{pred['center_y']:.3f})"
            detection_summary += "\n"

    # Phase 5: Gemma VLM reasoning (if analysis prompt given)
    vlm_response = ""
    if analysis_prompt and analysis_prompt.strip():
        t1 = time.time()
        # Feed the annotated image to Gemma for reasoning
        full_prompt = (
            f"This image has been annotated with object detection results. "
            f"The detected objects are labeled with colored bounding boxes and masks. "
            f"Query was: '{target_class}'. {len(predictions)} objects were found.\n\n"
            f"User question: {analysis_prompt}"
        )
        vlm_response = run_gemma_reasoning(annotated, full_prompt)
        vlm_time = time.time() - t1
        vlm_response = f"[Gemma analysis in {vlm_time:.1f}s]\n{vlm_response}"

    return annotated, detection_summary, vlm_response


def quick_count(image, object_class):
    """Quick counting mode - just detect and count."""
    if image is None:
        return None, "Please upload an image."

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    if not object_class.strip():
        return None, "Please specify what to count."

    predictions = run_falcon_perception(image_pil, object_class.strip())
    annotated = visualize_detections(image_pil, predictions, object_class.strip())

    count = len(predictions)
    return annotated, f"Count: {count} '{object_class.strip()}' object(s) detected"


def visual_qa(image, question):
    """Visual Q&A mode - just use Gemma VLM."""
    if image is None:
        return "Please upload an image."

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    if not question.strip():
        return "Please ask a question."

    response = run_gemma_reasoning(image_pil, question)
    return response


def scene_understanding(image):
    """Full scene understanding - detect multiple common objects and describe."""
    if image is None:
        return None, "Please upload an image."

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    # First, ask Gemma to identify objects in the scene
    description = run_gemma_reasoning(
        image_pil,
        "List the main visible objects in this image. Be brief and specific. "
        "Just list the object types, separated by commas."
    )

    # Parse object classes from Gemma's response
    candidates = [c.strip().lower().rstrip('.') for c in description.split(",")]
    candidates = [c for c in candidates if c and len(c) < 30][:6]

    all_predictions = {}
    annotated = image_pil.copy()

    for obj_class in candidates:
        try:
            preds = run_falcon_perception(image_pil, obj_class, task="segmentation")
            if preds:
                all_predictions[obj_class] = preds
                annotated = visualize_detections(annotated, preds, obj_class)
        except Exception as e:
            print(f"Warning: failed to detect '{obj_class}': {e}")

    summary = "Scene Analysis:\n"
    summary += f"Objects identified by VLM: {', '.join(candidates)}\n\n"
    for obj_class, preds in all_predictions.items():
        summary += f"  {obj_class}: {len(preds)} instance(s)\n"

    if not all_predictions:
        summary += "  No objects detected via segmentation.\n"

    return annotated, summary


# ── Gradio UI ─────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="Falcon + Gemma: Vision Pipeline",
    ) as demo:
        gr.Markdown(
            "# Falcon Perception + Gemma VLM Pipeline\n"
            "Combine **Falcon Perception** (detection/segmentation) with "
            "**Gemma 3n** (visual reasoning) for practical vision applications. "
            "All inference runs locally on Apple Silicon via MLX."
        )

        with gr.Tab("Detect & Analyze"):
            gr.Markdown("### Full Pipeline: Detect → Segment → Reason")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image", type="numpy")
                    query_input = gr.Textbox(
                        label="Object to detect",
                        placeholder="e.g., car, person, dog, chair...",
                    )
                    task_type = gr.Radio(
                        ["Segmentation", "Detection"],
                        value="Segmentation",
                        label="Task",
                    )
                    analysis_input = gr.Textbox(
                        label="Analysis question (optional, uses Gemma VLM)",
                        placeholder="e.g., How many are facing left? Which is the largest?",
                    )
                    run_btn = gr.Button("Run Pipeline", variant="primary")
                with gr.Column():
                    img_output = gr.Image(label="Annotated Result")
                    det_output = gr.Textbox(label="Detection Summary", lines=6)
                    vlm_output = gr.Textbox(label="VLM Analysis", lines=4)

            run_btn.click(
                detect_and_analyze,
                inputs=[img_input, query_input, task_type, analysis_input],
                outputs=[img_output, det_output, vlm_output],
            )

        with gr.Tab("Quick Count"):
            gr.Markdown("### Count specific objects in an image")
            with gr.Row():
                with gr.Column():
                    count_img = gr.Image(label="Upload Image", type="numpy")
                    count_class = gr.Textbox(
                        label="What to count",
                        placeholder="e.g., people, cars, trees...",
                    )
                    count_btn = gr.Button("Count", variant="primary")
                with gr.Column():
                    count_result_img = gr.Image(label="Detection Result")
                    count_result_txt = gr.Textbox(label="Count Result")

            count_btn.click(
                quick_count,
                inputs=[count_img, count_class],
                outputs=[count_result_img, count_result_txt],
            )

        with gr.Tab("Visual Q&A"):
            gr.Markdown("### Ask questions about an image (Gemma VLM)")
            with gr.Row():
                with gr.Column():
                    qa_img = gr.Image(label="Upload Image", type="numpy")
                    qa_question = gr.Textbox(
                        label="Your question",
                        placeholder="What is happening in this image?",
                    )
                    qa_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    qa_answer = gr.Textbox(label="Answer", lines=8)

            qa_btn.click(
                visual_qa,
                inputs=[qa_img, qa_question],
                outputs=[qa_answer],
            )

        with gr.Tab("Scene Understanding"):
            gr.Markdown(
                "### Automatic scene analysis\n"
                "Gemma identifies objects → Falcon detects & segments each one"
            )
            with gr.Row():
                with gr.Column():
                    scene_img = gr.Image(label="Upload Image", type="numpy")
                    scene_btn = gr.Button("Analyze Scene", variant="primary")
                with gr.Column():
                    scene_result_img = gr.Image(label="Annotated Scene")
                    scene_result_txt = gr.Textbox(label="Scene Summary", lines=8)

            scene_btn.click(
                scene_understanding,
                inputs=[scene_img],
                outputs=[scene_result_img, scene_result_txt],
            )

        gr.Markdown(
            "---\n"
            "**Models**: Falcon Perception (0.6B) + Gemma 3n E4B IT (4-bit) | "
            "**Backend**: MLX (Apple Silicon) | **Fully Local**"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
