"""
Video Object Tracking — PyTorch / CUDA (DGX Spark GB10 profile)
================================================================
Falcon Perception per-frame detection + IoU tracking; same logic as repo-root
``video_tracker.py``, using ``agent_studio`` for inference.
"""

import os
import time
import numpy as np
from PIL import Image, ImageDraw
import cv2

from agent_studio import _ensure, _detect, _font

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (200, 100, 50), (50, 100, 200), (150, 200, 50), (200, 50, 150),
    (100, 200, 200), (200, 200, 100), (100, 100, 200), (200, 100, 200),
]


def load_falcon_model():
    _ensure()


def detect_objects_in_frame(frame_pil: Image.Image, query: str, task: str = "segmentation"):
    _ensure()
    return _detect(frame_pil, query, task=task)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_idx):
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["lost_count"] += 1
                if self.tracks[tid]["lost_count"] > self.max_lost:
                    del self.tracks[tid]
            return []

        det_bboxes = [d["bbox"] for d in detections if "bbox" in d]
        if not det_bboxes:
            return []

        track_ids = list(self.tracks.keys())
        matched_tracks = set()
        matched_dets = set()
        assignments = []

        if track_ids:
            iou_matrix = np.zeros((len(track_ids), len(det_bboxes)))
            for i, tid in enumerate(track_ids):
                for j, dbox in enumerate(det_bboxes):
                    iou_matrix[i, j] = compute_iou(self.tracks[tid]["bbox"], dbox)

            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = track_ids[i]
                matched_tracks.add(tid)
                matched_dets.add(j)
                assignments.append((tid, j))
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        results = []
        for tid, det_j in assignments:
            self.tracks[tid]["bbox"] = det_bboxes[det_j]
            self.tracks[tid]["lost_count"] = 0
            self.tracks[tid]["history"].append({
                "frame": frame_idx,
                "bbox": det_bboxes[det_j],
            })
            results.append((tid, detections[det_j]))

        for j in range(len(detections)):
            if j not in matched_dets and "bbox" in detections[j]:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": detections[j]["bbox"],
                    "lost_count": 0,
                    "history": [{"frame": frame_idx, "bbox": detections[j]["bbox"]}],
                }
                results.append((tid, detections[j]))

        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost_count"] += 1
                if self.tracks[tid]["lost_count"] > self.max_lost:
                    del self.tracks[tid]

        return results


def annotate_frame(frame_np, tracked_objects, query, frame_idx, total_frames):
    frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    overlay = Image.fromarray(frame_rgb).convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    font = _font(14)
    font_small = _font(11)

    for track_id, det in tracked_objects:
        color = COLORS[(track_id - 1) % len(COLORS)]
        label = f"#{track_id}"

        if "mask" in det:
            mask = det["mask"]
            mask_data = np.zeros((*mask.shape, 4), dtype=np.uint8)
            mask_data[mask] = (*color, 60)
            mask_overlay = Image.fromarray(mask_data, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_overlay)
            draw = ImageDraw.Draw(overlay)

        if "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            text_bbox = draw.textbbox((x1, y1 - 16), label, font=font_small)
            draw.rectangle(text_bbox, fill=(*color, 200))
            draw.text((x1, y1 - 16), label, fill=(255, 255, 255), font=font_small)

    active_count = len(tracked_objects)
    hud = f"Frame {frame_idx+1}/{total_frames} | {query}: {active_count} active"
    text_bbox = draw.textbbox((10, 10), hud, font=font)
    draw.rectangle(
        [text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4],
        fill=(0, 0, 0, 180),
    )
    draw.text((10, 10), hud, fill=(255, 255, 255), font=font)

    result = np.array(overlay.convert("RGB"))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def process_video(
    video_path: str,
    query: str,
    output_path: str = None,
    task: str = "segmentation",
    frame_skip: int = 1,
    max_frames: int = 0,
    iou_threshold: float = 0.3,
    progress_callback=None,
):
    load_falcon_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    if output_path is None:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_tracked.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / frame_skip
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    tracker = SimpleTracker(iou_threshold=iou_threshold)
    frame_stats = []
    all_track_ids = set()

    frame_idx = 0
    processed = 0

    print(f"Processing video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"  Query: '{query}', Task: {task}, Frame skip: {frame_skip}")
    print(f"  Output: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        if frame_idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            t0 = time.time()
            detections = detect_objects_in_frame(frame_pil, query, task=task)
            detect_time = time.time() - t0

            tracked = tracker.update(detections, frame_idx)
            for tid, _ in tracked:
                all_track_ids.add(tid)

            annotated = annotate_frame(frame, tracked, query, frame_idx, total_frames)
            writer.write(annotated)

            stat = {
                "frame": frame_idx,
                "detections": len(detections),
                "active_tracks": len(tracked),
                "detect_time": detect_time,
            }
            frame_stats.append(stat)
            processed += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames, stat)

            if processed % 5 == 0 or frame_idx == 0:
                print(
                    f"  Frame {frame_idx+1}/{total_frames}: "
                    f"{len(detections)} detections, "
                    f"{len(tracked)} tracked, "
                    f"{detect_time:.2f}s"
                )

        frame_idx += 1

    cap.release()
    writer.release()

    total_unique = len(all_track_ids)
    avg_detect_time = np.mean([s["detect_time"] for s in frame_stats]) if frame_stats else 0
    max_concurrent = max([s["active_tracks"] for s in frame_stats]) if frame_stats else 0

    track_summary = {}
    for tid, track in tracker.tracks.items():
        track_summary[tid] = {
            "total_appearances": len(track["history"]),
            "first_frame": track["history"][0]["frame"],
            "last_frame": track["history"][-1]["frame"],
        }

    result = {
        "output_path": output_path,
        "total_frames_processed": processed,
        "total_unique_objects": total_unique,
        "max_concurrent_objects": max_concurrent,
        "avg_detect_time": avg_detect_time,
        "track_summary": track_summary,
        "frame_stats": frame_stats,
    }

    print(f"\nDone! Processed {processed} frames")
    print(f"  Unique objects tracked: {total_unique}")
    print(f"  Max concurrent: {max_concurrent}")
    print(f"  Avg detection time: {avg_detect_time:.2f}s/frame")
    print(f"  Output saved: {output_path}")

    return result


def process_video_gradio(video_file, query, task_type, frame_skip, max_frames):
    if video_file is None:
        return None, "Please upload a video."

    if not query.strip():
        return None, "Please specify what to track."

    task = "segmentation" if task_type == "Segmentation" else "detection"
    frame_skip = max(1, int(frame_skip))
    max_frames = int(max_frames) if max_frames and int(max_frames) > 0 else 0

    output_path = video_file.replace(".", "_tracked.")
    if not output_path.endswith(".mp4"):
        output_path = os.path.splitext(video_file)[0] + "_tracked.mp4"

    try:
        result = process_video(
            video_path=video_file,
            query=query.strip(),
            output_path=output_path,
            task=task,
            frame_skip=frame_skip,
            max_frames=max_frames,
        )

        summary = (
            f"Video Processing Complete\n"
            f"{'='*40}\n"
            f"Frames processed: {result['total_frames_processed']}\n"
            f"Unique objects tracked: {result['total_unique_objects']}\n"
            f"Max concurrent objects: {result['max_concurrent_objects']}\n"
            f"Avg detection time: {result['avg_detect_time']:.2f}s/frame\n\n"
            f"Track Details:\n"
        )
        for tid, info in result["track_summary"].items():
            summary += (
                f"  Track #{tid}: "
                f"appeared in {info['total_appearances']} frames "
                f"(frames {info['first_frame']}-{info['last_frame']})\n"
            )

        return result["output_path"], summary

    except Exception as e:
        return None, f"Error: {str(e)}"


def build_video_ui():
    import gradio as gr

    with gr.Blocks(title="Falcon Video Tracker (CUDA)") as demo:
        gr.Markdown(
            "# Video Object Tracking with Falcon Perception\n"
            "Upload a video, specify objects to track. "
            "Uses Falcon Perception (PyTorch/CUDA) + IoU-based tracking."
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                query_input = gr.Textbox(
                    label="Object to track",
                    placeholder="e.g., person, car, dog...",
                )
                task_type = gr.Radio(
                    ["Segmentation", "Detection"],
                    value="Segmentation",
                    label="Task",
                )
                frame_skip = gr.Slider(
                    1, 10, value=2, step=1,
                    label="Frame skip (process every Nth frame)",
                )
                max_frames = gr.Number(
                    value=0,
                    label="Max frames (0 = all)",
                )
                run_btn = gr.Button("Process Video", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Tracked Video")
                summary_output = gr.Textbox(label="Tracking Summary", lines=12)

        run_btn.click(
            process_video_gradio,
            inputs=[video_input, query_input, task_type, frame_skip, max_frames],
            outputs=[video_output, summary_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_video_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
