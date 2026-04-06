"""
Falcon Perception + Gemma VLM — Unified Demo (PyTorch / CUDA)
==============================================================
Same tabs as repo-root ``demo.py``; inference via ``agent_studio``.
"""

import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw
import cv2
import gradio as gr

from agent_studio import _ensure, _detect, _vlm, _font

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def _ensure_models():
    _ensure()


def _falcon_detect(pil_img, query, task="segmentation"):
    return _detect(pil_img, query, task=task)


def _gemma_answer(pil_img, prompt):
    return _vlm(pil_img, prompt)


def _draw(pil_img, dets, query, extra_hud=""):
    overlay = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    f14, f11 = _font(14), _font(11)

    for idx, d in enumerate(dets):
        c = PALETTE[idx % len(PALETTE)]
        if "mask" in d:
            md = np.zeros((*d["mask"].shape, 4), dtype=np.uint8)
            md[d["mask"]] = (*c, 70)
            overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA"))
            draw = ImageDraw.Draw(overlay)
        if "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
            lbl = f"{query} #{idx+1}"
            tb = draw.textbbox((x1, max(0, y1 - 17)), lbl, font=f11)
            draw.rectangle(tb, fill=(*c, 200))
            draw.text((x1, max(0, y1 - 17)), lbl, fill=(255, 255, 255), font=f11)

    hud = f"{len(dets)} '{query}' detected"
    if extra_hud:
        hud += f"  |  {extra_hud}"
    tb = draw.textbbox((10, 10), hud, font=f14)
    draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4], fill=(0, 0, 0, 180))
    draw.text((10, 10), hud, fill=(255, 255, 255), font=f14)
    return overlay.convert("RGB")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0


class _Tracker:
    def __init__(self, iou_thr=0.3, max_lost=5):
        self.iou_thr, self.max_lost = iou_thr, max_lost
        self.nxt, self.trks = 1, {}

    def update(self, dets, fi):
        bboxes = [d["bbox"] for d in dets if "bbox" in d]
        if not bboxes:
            for t in list(self.trks):
                self.trks[t]["lost"] += 1
                if self.trks[t]["lost"] > self.max_lost:
                    del self.trks[t]
            return []
        tids = list(self.trks.keys())
        matched_t, matched_d, assigns = set(), set(), []
        if tids:
            M = np.array([[_iou(self.trks[t]["bbox"], b) for b in bboxes] for t in tids])
            while M.size:
                v = M.max()
                if v < self.iou_thr:
                    break
                i, j = np.unravel_index(M.argmax(), M.shape)
                matched_t.add(tids[i])
                matched_d.add(j)
                assigns.append((tids[i], j))
                M[i, :] = 0
                M[:, j] = 0
        out = []
        for tid, j in assigns:
            self.trks[tid].update(bbox=bboxes[j], lost=0)
            self.trks[tid]["hist"].append(fi)
            out.append((tid, dets[j]))
        for j, d in enumerate(dets):
            if j not in matched_d and "bbox" in d:
                tid = self.nxt
                self.nxt += 1
                self.trks[tid] = {"bbox": d["bbox"], "lost": 0, "hist": [fi]}
                out.append((tid, d))
        for t in tids:
            if t not in matched_t:
                self.trks[t]["lost"] += 1
                if self.trks[t]["lost"] > self.max_lost:
                    del self.trks[t]
        return out


def _draw_tracked(frame_bgr, tracked, query, fi, total):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(rgb).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    f14, f11 = _font(14), _font(11)
    for tid, d in tracked:
        c = PALETTE[(tid - 1) % len(PALETTE)]
        if "mask" in d:
            md = np.zeros((*d["mask"].shape, 4), dtype=np.uint8)
            md[d["mask"]] = (*c, 60)
            overlay = Image.alpha_composite(overlay, Image.fromarray(md, "RGBA"))
            draw = ImageDraw.Draw(overlay)
        if "bbox" in d:
            x1, y1, x2, y2 = d["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=c, width=2)
            lbl = f"#{tid}"
            tb = draw.textbbox((x1, max(0, y1 - 15)), lbl, font=f11)
            draw.rectangle(tb, fill=(*c, 200))
            draw.text((x1, max(0, y1 - 15)), lbl, fill=(255, 255, 255), font=f11)
    hud = f"Frame {fi+1}/{total} | {query}: {len(tracked)} tracked"
    tb = draw.textbbox((10, 10), hud, font=f14)
    draw.rectangle([tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4], fill=(0, 0, 0, 180))
    draw.text((10, 10), hud, fill=(255, 255, 255), font=f14)
    return cv2.cvtColor(np.array(overlay.convert("RGB")), cv2.COLOR_RGB2BGR)


def run_image(image, prompt, progress=gr.Progress()):
    if image is None:
        return None, "Upload an image first."
    if not prompt or not prompt.strip():
        return None, "Enter a prompt."

    prompt = prompt.strip()
    progress(0, desc="Loading models...")
    _ensure_models()

    pil = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.convert("RGB")

    is_question = any(
        prompt.lower().startswith(w)
        for w in (
            "how", "what", "where", "why", "who", "is ", "are ", "do ", "does ",
            "can ", "describe", "explain", "tell", "count how",
        )
    )

    if is_question:
        progress(0.3, desc="Asking Gemma...")
        vlm_answer = _gemma_answer(pil, prompt)
        det_query = None
        for word in prompt.lower().replace("?", "").replace(",", "").split():
            if word in (
                "person", "people", "car", "cars", "dog", "dogs", "cat", "cats",
                "bird", "tree", "chair", "bottle", "cup", "phone", "book",
                "bus", "truck", "bicycle", "motorcycle", "horse", "sheep", "cow",
                "airplane", "boat", "bench", "backpack", "umbrella", "handbag",
                "ball", "food", "plant", "flower", "building", "window", "door",
            ):
                det_query = word
                break
        if det_query:
            progress(0.6, desc=f"Detecting '{det_query}'...")
            dets = _falcon_detect(pil, det_query)
            annotated = _draw(pil, dets, det_query)
            return annotated, f"Detected {len(dets)} '{det_query}'\n\nGemma:\n{vlm_answer}"
        return pil, f"Gemma:\n{vlm_answer}"
    progress(0.3, desc=f"Detecting '{prompt}'...")
    t0 = time.time()
    dets = _falcon_detect(pil, prompt)
    dt = time.time() - t0
    annotated = _draw(pil, dets, prompt, f"{dt:.1f}s")

    progress(0.7, desc="Analyzing with Gemma...")
    vlm = _gemma_answer(
        annotated,
        f"{len(dets)} instance(s) of '{prompt}' were detected. Briefly describe them.",
    )
    return annotated, f"Found {len(dets)} '{prompt}' in {dt:.1f}s\n\nGemma:\n{vlm}"


def run_video(video, prompt, progress=gr.Progress()):
    if video is None:
        return None, "Upload a video first."
    if not prompt or not prompt.strip():
        return None, "Enter a prompt."

    prompt = prompt.strip()
    progress(0, desc="Loading models...")
    _ensure_models()

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "Cannot open video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    skip = max(1, total // 30)
    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / skip, (w, h))

    tracker = _Tracker(iou_thr=0.3)
    all_ids = set()
    fi, processed = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % skip == 0:
            progress(fi / max(total, 1), desc=f"Frame {fi+1}/{total}...")
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = _falcon_detect(pil, prompt)
            tracked = tracker.update(dets, fi)
            for tid, _ in tracked:
                all_ids.add(tid)
            ann = _draw_tracked(frame, tracked, prompt, fi, total)
            writer.write(ann)
            processed += 1
        fi += 1

    cap.release()
    writer.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        h264_path = out_path.replace(".mp4", "_h264.mp4")
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-loglevel", "error", "-i", out_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                h264_path,
            ],
            check=False,
        )
        if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            os.unlink(out_path)
            out_path = h264_path

    progress(1.0, desc="Done!")
    info = (
        f"Processed {processed} frames (skip={skip})\n"
        f"Unique '{prompt}' tracked: {len(all_ids)}\n\n"
    )
    for tid, t in sorted(tracker.trks.items()):
        info += f"  #{tid}: {len(t['hist'])} frames\n"
    return out_path, info


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "test_data")


def build_demo():
    demo = gr.Blocks(title="Falcon + Gemma Vision (CUDA)")

    with demo:
        gr.Markdown(
            "# Falcon Perception + Gemma VLM\n"
            "Object detection, segmentation, counting, tracking & visual reasoning. "
            "**Falcon Perception** (0.6B) + **Gemma 4 E4B-it** — PyTorch / CUDA."
        )

        with gr.Tabs():
            with gr.TabItem("Image"):
                with gr.Row(equal_height=True):
                    with gr.Column():
                        img_in = gr.Image(label="Upload Image", type="numpy", height=300)
                        img_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Object name (dog, car) or question (How many people?)",
                        )
                        img_btn = gr.Button("Run", variant="primary", size="lg")
                    with gr.Column():
                        img_out = gr.Image(label="Result", height=300)
                        img_text = gr.Textbox(label="Analysis", lines=6)

                img_btn.click(run_image, [img_in, img_prompt], [img_out, img_text])

                gr.Examples(
                    examples=[
                        [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "dog"],
                        [os.path.join(EXAMPLES_DIR, "street.jpg"), "car"],
                        [os.path.join(EXAMPLES_DIR, "street.jpg"), "person"],
                        [os.path.join(EXAMPLES_DIR, "kitchen.jpg"), "bottle"],
                        [os.path.join(EXAMPLES_DIR, "dogs.jpg"), "How many dogs are there and what breeds?"],
                    ],
                    inputs=[img_in, img_prompt],
                    label="Examples (click to load)",
                )

            with gr.TabItem("Video"):
                with gr.Row(equal_height=True):
                    with gr.Column():
                        vid_in = gr.Video(label="Upload Video", height=300)
                        vid_prompt = gr.Textbox(
                            label="Object to track",
                            placeholder="e.g. person, car, dog...",
                        )
                        vid_btn = gr.Button("Track", variant="primary", size="lg")
                    with gr.Column():
                        vid_out = gr.Video(label="Tracked Video", height=300)
                        vid_text = gr.Textbox(label="Summary", lines=6)

                vid_btn.click(run_video, [vid_in, vid_prompt], [vid_out, vid_text])

                gr.Examples(
                    examples=[
                        [os.path.join(EXAMPLES_DIR, "dogs_video.mp4"), "dog"],
                        [os.path.join(EXAMPLES_DIR, "test_panning.mp4"), "car"],
                    ],
                    inputs=[vid_in, vid_prompt],
                    label="Examples (click to load)",
                )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
