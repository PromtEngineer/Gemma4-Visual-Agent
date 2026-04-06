"""
Falcon Perception + Gemma VLM (PyTorch / CUDA — DGX Spark GB10 profile)
========================================================================
Same Gradio flows as repo-root ``app.py``; models load via ``agent_studio``.
"""

import time
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

from agent_studio import (
    _font,
    load_all_models,
    run_falcon_perception,
    run_gemma_reasoning,
)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (150, 200, 50), (200, 50, 150),
]


def visualize_detections(image: Image.Image, predictions: list, query: str):
    overlay = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    font = _font(16)
    font_small = _font(12)

    for idx, pred in enumerate(predictions):
        color = COLORS[idx % len(COLORS)]
        label = f"{query} #{idx + 1}"

        if "mask" in pred:
            mask = pred["mask"]
            mask_data = np.zeros((*mask.shape, 4), dtype=np.uint8)
            mask_data[mask] = (*color, 80)
            mask_overlay = Image.fromarray(mask_data, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_overlay)
            draw = ImageDraw.Draw(overlay)

        if "bbox" in pred:
            x1, y1, x2, y2 = pred["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text_bbox = draw.textbbox((x1, y1 - 20), label, font=font_small)
            draw.rectangle(text_bbox, fill=(*color, 200))
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font_small)

    count_text = f"Found {len(predictions)} '{query}' object(s)"
    text_bbox = draw.textbbox((10, 10), count_text, font=font)
    draw.rectangle(
        [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5],
        fill=(0, 0, 0, 180),
    )
    draw.text((10, 10), count_text, fill=(255, 255, 255), font=font)

    return overlay.convert("RGB")


def detect_and_analyze(image, query, task_type, analysis_prompt):
    if image is None:
        return None, "Please upload an image.", ""

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    target_class = query.strip() or "object"
    task = "segmentation" if task_type == "Segmentation" else "detection"
    t0 = time.time()
    predictions = run_falcon_perception(image_pil, target_class, task=task)
    detect_time = time.time() - t0

    annotated = visualize_detections(image_pil, predictions, target_class)

    detection_summary = f"Detected {len(predictions)} instance(s) of '{target_class}'.\n"
    detection_summary += f"Detection time: {detect_time:.2f}s\n\n"

    for idx, pred in enumerate(predictions):
        if "bbox" in pred:
            x1, y1, x2, y2 = pred["bbox"]
            detection_summary += f"  #{idx+1}: bbox=({x1},{y1})-({x2},{y2})"
            if "center_x" in pred:
                detection_summary += f" center=({pred['center_x']:.3f},{pred['center_y']:.3f})"
            detection_summary += "\n"

    vlm_response = ""
    if analysis_prompt and analysis_prompt.strip():
        t1 = time.time()
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
    if image is None:
        return "Please upload an image."

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    if not question.strip():
        return "Please ask a question."

    return run_gemma_reasoning(image_pil, question)


def scene_understanding(image):
    if image is None:
        return None, "Please upload an image."

    load_all_models()

    image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_pil = image_pil.convert("RGB")

    description = run_gemma_reasoning(
        image_pil,
        "List the main visible objects in this image. Be brief and specific. "
        "Just list the object types, separated by commas.",
    )

    candidates = [c.strip().lower().rstrip(".") for c in description.split(",")]
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


def build_ui():
    with gr.Blocks(title="Falcon + Gemma: Vision Pipeline (CUDA)") as demo:
        gr.Markdown(
            "# Falcon Perception + Gemma VLM Pipeline\n"
            "Combine **Falcon Perception** (detection/segmentation) with "
            "**Gemma 4 E4B-it** (visual reasoning). Runs on **NVIDIA GPU** via PyTorch."
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

            qa_btn.click(visual_qa, inputs=[qa_img, qa_question], outputs=[qa_answer])

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
            "**Models**: Falcon Perception (0.6B) + Gemma 4 E4B-it | "
            "**Backend**: PyTorch / CUDA | **DGX Spark GB10 profile**"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
