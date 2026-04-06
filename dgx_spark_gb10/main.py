"""
Combined Application: Falcon Perception + Gemma VLM (PyTorch / CUDA)
======================================================================
Launch Image Analysis and Video Tracking UIs in one Gradio app.
Run from this directory: ``python main.py``
"""

import gradio as gr

from app import (
    detect_and_analyze,
    quick_count,
    visual_qa,
    scene_understanding,
)
from video_tracker import process_video_gradio


def build_combined_ui():
    with gr.Blocks(title="Falcon + Gemma Vision Pipeline (CUDA)") as demo:
        gr.Markdown(
            "# Falcon Perception + Gemma VLM Vision Pipeline\n"
            "**Falcon Perception** (0.6B) for detection & segmentation + "
            "**Gemma 4 E4B-it** for visual reasoning.\n"
            "Inference on **NVIDIA GPU** via PyTorch (DGX Spark GB10 profile)."
        )

        with gr.Tab("Detect & Analyze"):
            gr.Markdown("### Detect → Segment → Reason")
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
                        placeholder="e.g., How many are facing left?",
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
            gr.Markdown("### Count objects in an image")
            with gr.Row():
                with gr.Column():
                    count_img = gr.Image(label="Upload Image", type="numpy")
                    count_class = gr.Textbox(
                        label="What to count",
                        placeholder="e.g., people, cars...",
                    )
                    count_btn = gr.Button("Count", variant="primary")
                with gr.Column():
                    count_result_img = gr.Image(label="Result")
                    count_result_txt = gr.Textbox(label="Count")
            count_btn.click(
                quick_count,
                inputs=[count_img, count_class],
                outputs=[count_result_img, count_result_txt],
            )

        with gr.Tab("Visual Q&A"):
            gr.Markdown("### Ask questions about an image")
            with gr.Row():
                with gr.Column():
                    qa_img = gr.Image(label="Upload Image", type="numpy")
                    qa_question = gr.Textbox(label="Question", placeholder="What is happening?")
                    qa_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    qa_answer = gr.Textbox(label="Answer", lines=8)
            qa_btn.click(visual_qa, inputs=[qa_img, qa_question], outputs=[qa_answer])

        with gr.Tab("Scene Understanding"):
            gr.Markdown("### Auto-detect and segment all objects")
            with gr.Row():
                with gr.Column():
                    scene_img = gr.Image(label="Upload Image", type="numpy")
                    scene_btn = gr.Button("Analyze Scene", variant="primary")
                with gr.Column():
                    scene_result_img = gr.Image(label="Annotated Scene")
                    scene_result_txt = gr.Textbox(label="Summary", lines=8)
            scene_btn.click(
                scene_understanding,
                inputs=[scene_img],
                outputs=[scene_result_img, scene_result_txt],
            )

        with gr.Tab("Video Tracking"):
            gr.Markdown(
                "### Track objects across video frames\n"
                "Uses Falcon Perception + IoU-based tracking"
            )
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Video")
                    vid_query = gr.Textbox(
                        label="Object to track",
                        placeholder="e.g., person, car...",
                    )
                    vid_task = gr.Radio(
                        ["Segmentation", "Detection"],
                        value="Segmentation",
                        label="Task",
                    )
                    frame_skip = gr.Slider(
                        1, 10, value=2, step=1,
                        label="Frame skip (process every Nth frame)",
                    )
                    max_frames = gr.Number(value=60, label="Max frames (0 = all)")
                    vid_btn = gr.Button("Process Video", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Tracked Video")
                    vid_summary = gr.Textbox(label="Tracking Summary", lines=12)
            vid_btn.click(
                process_video_gradio,
                inputs=[video_input, vid_query, vid_task, frame_skip, max_frames],
                outputs=[video_output, vid_summary],
            )

        gr.Markdown(
            "---\n"
            "**Models**: Falcon Perception (0.6B) + Gemma 4 E4B-it | "
            "**Backend**: PyTorch / CUDA | **DGX Spark GB10 profile**"
        )

    return demo


if __name__ == "__main__":
    demo = build_combined_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
