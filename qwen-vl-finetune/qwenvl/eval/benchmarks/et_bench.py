from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark
from qwenvl.eval.timestamp_audit import TimestampAuditor

logger = logging.getLogger(__name__)

ET_BENCH_PATH = os.environ.get(
    "ET_BENCH_PATH",
    "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/et_bench",
)

# When True, prompts ask for frame numbers instead of seconds
USE_FRAME_NUMBERS = os.environ.get("EVAL_USE_FRAMES", "0") == "1"
VIDEO_FPS = 2.0  # sampling fps used in runner.py
MAX_FRAMES = 2048


def _get_frame_count(duration: float) -> int:
    """Calculate number of frames the model sees for a given video duration."""
    return min(int(duration * VIDEO_FPS), MAX_FRAMES)


def _frames_to_seconds(frame_num: float, duration: float) -> float:
    """Convert a frame number back to seconds."""
    n_frames = _get_frame_count(duration)
    if n_frames <= 0:
        return 0.0
    return frame_num * duration / n_frames


def parse_frame_events(text: str, duration: float) -> list[tuple[float, float, str]]:
    """Parse model output that uses frame numbers, convert to seconds."""
    events = []
    n_frames = _get_frame_count(duration)
    if n_frames <= 0:
        return events

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Patterns for frame-based output:
        # <frame 0 - frame 100> description
        # Frame 0-100: description
        # <0 - 100> description (same as timestamp but we know it's frames)
        patterns = [
            r"<?[Ff]rame\s*(\d+)\s*[-–]\s*[Ff]rame\s*(\d+)>?\s*[:]*\s*(.+)",
            r"<?(\d+)\s*[-–]\s*(\d+)>?\s*[:]*\s*(.+)",
        ]
        for pattern in patterns:
            m = re.match(pattern, line)
            if m:
                frame_start = float(m.group(1))
                frame_end = float(m.group(2))
                # Convert frames to seconds
                start_sec = frame_start * duration / n_frames
                end_sec = frame_end * duration / n_frames
                events.append((start_sec, end_sec, m.group(3).strip()))
                break
    return events


def _parse_frame_span(text: str, duration: float) -> tuple[float, float]:
    """Parse a frame span from model output, convert to seconds."""
    n_frames = _get_frame_count(duration)
    if n_frames <= 0:
        return (0, 0)

    # Try "frame X - frame Y" or "frames X-Y"
    m = re.search(r"[Ff]rames?\s*(\d+)\s*[-–]\s*[Ff]rames?\s*(\d+)", text)
    if m:
        f_start = float(m.group(1))
        f_end = float(m.group(2))
        return (f_start * duration / n_frames, f_end * duration / n_frames)

    # Try bare numbers (when we know output is frames)
    pair = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if pair:
        f_start = float(pair.group(1))
        f_end = float(pair.group(2))
        # Only treat as frames if values are in frame range
        if f_end <= n_frames * 1.1:
            return (f_start * duration / n_frames, f_end * duration / n_frames)

    return (0, 0)


def compute_iou(pred_span: tuple, gt_span: tuple) -> float:
    inter_start = max(pred_span[0], gt_span[0])
    inter_end = min(pred_span[1], gt_span[1])
    inter = max(0, inter_end - inter_start)
    union = (pred_span[1] - pred_span[0]) + (gt_span[1] - gt_span[0]) - inter
    return inter / (union + 1e-8)


def parse_timestamp_events(text: str) -> list[tuple[float, float, str]]:
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try MM:SS format first: <0:07 - 0:09> description or 0:07 - 0:09: description
        mmss_match = re.match(
            r"<?(\d+:\d+(?:\.\d+)?)\s*[-–]\s*(\d+:\d+(?:\.\d+)?)>?\s*[:]*\s*(.+)", line
        )
        if mmss_match:
            start = _parse_mmss_to_seconds(mmss_match.group(1))
            end = _parse_mmss_to_seconds(mmss_match.group(2))
            events.append((start, end, mmss_match.group(3).strip()))
            continue

        # Try decimal formats
        patterns = [
            # <0.0 - 3.5> description  (Qwen3-VL format)
            r"<\s*(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*>\s*(.+)",
            # 0.0-3.5: description
            r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*s?\s*[:]\s*(.+)",
            # [0.0, 3.5]: description
            r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]\s*[:]\s*(.+)",
            # (0.0, 3.5) description
            r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)\s*[:]*\s*(.+)",
            # 0.0s - 3.5s: description
            r"(\d+\.?\d*)\s*s\s*[-–]\s*(\d+\.?\d*)\s*s\s*[:]\s*(.+)",
        ]
        for pattern in patterns:
            m = re.match(pattern, line)
            if m:
                events.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
                break
    return events


def _parse_mmss_to_seconds(time_str: str) -> float:
    """Parse MM:SS or H:MM:SS to seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def _parse_time_span(text: str) -> tuple[float, float]:
    """Parse a time span from model output. Handles:
    - '0.0 - 4.9 seconds'
    - '0:07 - 0:09'
    - 'The event happens in 0.0 - 1.0 seconds.'
    - '<0.0 - 3.5>'
    """
    # Try MM:SS format first: 0:07 - 0:09
    mmss = re.findall(r"(\d+:\d+(?:\.\d+)?)", text)
    if len(mmss) >= 2:
        return (_parse_mmss_to_seconds(mmss[0]), _parse_mmss_to_seconds(mmss[1]))

    # Try decimal seconds: 0.0 - 4.9
    # Match pairs like "0.0 - 4.9" or "<0.0 - 4.9>"
    pair = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", text)
    if pair:
        return (float(pair.group(1)), float(pair.group(2)))

    # Fallback: extract any two numbers
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))

    return (0, 0)


class ETBenchDVC(Benchmark):
    name = "et_bench_dvc"
    requires_judge = True
    max_new_tokens = 2048
    repetition_penalty = 1.1

    def load_all(self) -> list[dict]:
        ann_path = Path(ET_BENCH_PATH) / "annotations" / "etbench_vid_v1.0.json"
        if not ann_path.exists():
            logger.warning(f"ET-Bench annotations not found at {ann_path}")
            return []
        with open(ann_path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            if item.get("task") != "dvc":
                continue
            video_path = Path(ET_BENCH_PATH) / "videos" / item.get("video", "")
            events = []
            tgt = item.get("tgt", [])
            captions = item.get("g", [])
            for span, cap in zip(tgt, captions):
                events.append({"start": span[0], "end": span[1], "caption": cap})
            samples.append({
                "id": str(item.get("idx", "")),
                "video": str(video_path),
                "query": item.get("q", ""),
                "events": events,
                "duration": item.get("duration", 0),
                "source": item.get("source", ""),
                "metadata": item,
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        duration = sample.get("duration", 0)

        if USE_FRAME_NUMBERS and duration > 0:
            n_frames = _get_frame_count(duration)
            prompt = (
                f"You are watching a video with {n_frames} frames. "
                f"Describe all events in the video with their frame ranges. "
                f"Format each event as: <start_frame - end_frame> description"
            )
        else:
            # Use the original query from ET-Bench if available
            query = sample.get("query", "")
            if query:
                prompt = query
            else:
                prompt = ("Describe all events in the video with their timestamps. "
                          "Format each event as: start_time-end_time: description")
        return {
            "video": sample["video"],
            "prompt": prompt,
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        defaults = {
            "mean_iou": 0.0,
            "iou_at_0.5": 0.0,
            "caption_f1_temporal": 0.0,
            "caption_f1_ordinal": 0.0,
        }
        if not predictions:
            return defaults

        auditor = TimestampAuditor("et_bench_dvc")
        iou_scores = []
        all_pred_events = []
        all_gt_events = []

        for pred in predictions:
            duration = pred["sample"].get("duration", 0)
            if USE_FRAME_NUMBERS and duration > 0:
                pred_events = parse_frame_events(pred["prediction"], duration)
            else:
                pred_events = parse_timestamp_events(pred["prediction"])
            gt_events = pred["sample"]["events"]
            all_pred_events.append(pred_events)
            all_gt_events.append(gt_events)

            if not pred_events or not gt_events:
                iou_scores.append(0.0)
                continue

            matched_ious = []
            for gt in gt_events:
                gt_span = (gt["start"], gt["end"])
                best_iou = 0
                for pe in pred_events:
                    iou = compute_iou((pe[0], pe[1]), gt_span)
                    best_iou = max(best_iou, iou)
                matched_ious.append(best_iou)

            avg_iou = sum(matched_ious) / len(matched_ious)
            iou_scores.append(avg_iou)

            # Audit: compare predicted event range vs GT event range
            pred_range = (pred_events[0][0], pred_events[-1][1])
            gt_range = (gt_events[0]["start"], gt_events[-1]["end"])
            duration = pred["sample"].get("duration", 0)
            auditor.check_prediction(
                sample_id=pred["sample"].get("id", ""),
                video_path=pred["sample"].get("video", ""),
                pred_span=pred_range,
                gt_span=gt_range,
                duration=duration,
                raw_text=pred["prediction"][:300],
                query=pred["sample"].get("query", ""),
            )

        # Batch-level audit
        all_pred_spans = []
        all_gt_spans = []
        for pe_list, ge_list in zip(all_pred_events, all_gt_events):
            if pe_list:
                all_pred_spans.append((pe_list[0][0], pe_list[-1][1]))
            if ge_list:
                all_gt_spans.append((ge_list[0]["start"], ge_list[-1]["end"]))
        auditor.check_batch_summary(all_pred_spans, all_gt_spans)
        auditor.flush()

        mean_iou = sum(iou_scores) / len(iou_scores)
        iou_at_05 = sum(1 for s in iou_scores if s > 0.5) / len(iou_scores)

        # Judge-based caption scoring
        caption_f1_temporal_scores = []
        caption_f1_ordinal_scores = []

        if judge is not None:
            for pred_events, gt_events in zip(all_pred_events, all_gt_events):
                if not pred_events or not gt_events:
                    caption_f1_temporal_scores.append(0.0)
                    caption_f1_ordinal_scores.append(0.0)
                    continue
                result = judge.dvc_caption_score(pred_events, gt_events)
                caption_f1_temporal_scores.append(result.get("caption_f1_temporal", 0.0))
                caption_f1_ordinal_scores.append(result.get("caption_f1_ordinal", 0.0))

        n = len(predictions)
        return {
            "mean_iou": round(mean_iou, 4),
            "iou_at_0.5": round(iou_at_05, 4),
            "caption_f1_temporal": round(sum(caption_f1_temporal_scores) / n, 4) if caption_f1_temporal_scores else 0.0,
            "caption_f1_ordinal": round(sum(caption_f1_ordinal_scores) / n, 4) if caption_f1_ordinal_scores else 0.0,
        }

    def metric_names(self) -> list[str]:
        return ["mean_iou", "iou_at_0.5", "caption_f1_temporal", "caption_f1_ordinal"]


class ETBenchTVG(Benchmark):
    name = "et_bench_tvg"
    requires_judge = True
    max_new_tokens = 1024

    def load_all(self) -> list[dict]:
        ann_path = Path(ET_BENCH_PATH) / "annotations" / "etbench_vid_v1.0.json"
        if not ann_path.exists():
            logger.warning(f"ET-Bench annotations not found at {ann_path}")
            return []
        with open(ann_path) as f:
            data = json.load(f)

        samples = []
        for item in data:
            if item.get("task") != "tvg":
                continue
            video_path = Path(ET_BENCH_PATH) / "videos" / item.get("video", "")
            query = item.get("q", "")
            tgt = item.get("tgt", [[0, 0]])
            # Handle both nested [[start, end]] and flat [start, end] formats
            if tgt and isinstance(tgt[0], (list, tuple)):
                gt_span = (tgt[0][0], tgt[0][1])
            elif tgt and len(tgt) >= 2:
                gt_span = (tgt[0], tgt[1])
            else:
                gt_span = (0, 0)
            samples.append({
                "id": str(item.get("idx", "")),
                "video": str(video_path),
                "query": query,
                "gt_span": gt_span,
                "duration": item.get("duration", 0),
                "source": item.get("source", ""),
                "metadata": item,
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        duration = sample.get("duration", 0)

        if USE_FRAME_NUMBERS and duration > 0:
            n_frames = _get_frame_count(duration)
            # Extract the event description from the original query
            query = sample.get("query", "")
            # Original query format: "...find a visual event described by the sentence: '...'..."
            event_match = re.search(r"sentence:\s*'([^']+)'", query)
            event_desc = event_match.group(1) if event_match else query
            prompt = (
                f"You are watching a video with {n_frames} frames. "
                f"Find when this event happens: '{event_desc}'. "
                f"The format of your response should be: 'The event happens at frame <start_frame> - frame <end_frame>'."
            )
        else:
            # Use the original query from ET-Bench which already includes instructions
            prompt = sample["query"]

        return {
            "video": sample["video"],
            "prompt": prompt,
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"f1_iou_0.3": 0.0, "f1_iou_0.5": 0.0, "f1_iou_0.7": 0.0, "response_quality": 0.0}

        auditor = TimestampAuditor("et_bench_tvg")
        thresholds = [0.3, 0.5, 0.7]
        hits = {t: 0 for t in thresholds}
        response_qualities = []
        all_pred_spans = []
        all_gt_spans = []

        # Batch TVG judge calls (groups of 10)
        judge_results = {}
        if judge is not None:
            batch_size = 10
            for batch_start in range(0, len(predictions), batch_size):
                batch = predictions[batch_start:batch_start + batch_size]
                items = [
                    {"response": p["prediction"], "query": p["sample"].get("query", "")}
                    for p in batch
                ]
                results = judge.tvg_score_batch(items)
                for j, r in enumerate(results):
                    judge_results[batch_start + j] = r

        for i, pred in enumerate(predictions):
            text = pred["prediction"]
            duration = pred["sample"].get("duration", 0)

            if USE_FRAME_NUMBERS and duration > 0:
                pred_span = _parse_frame_span(text, duration)
            else:
                pred_span = _parse_time_span(text)

            # If regex failed and judge extracted timestamps, use judge's
            if pred_span == (0, 0) and i in judge_results:
                jr = judge_results[i]
                if jr["start"] != 0.0 or jr["end"] != 0.0:
                    pred_span = (jr["start"], jr["end"])

            gt_span = pred["sample"]["gt_span"]
            iou = compute_iou(pred_span, gt_span)

            for t in thresholds:
                if iou >= t:
                    hits[t] += 1

            if i in judge_results:
                response_qualities.append(judge_results[i]["response_quality"])

            # Audit each prediction
            all_pred_spans.append(pred_span)
            all_gt_spans.append(gt_span)
            auditor.check_prediction(
                sample_id=pred["sample"].get("id", ""),
                video_path=pred["sample"].get("video", ""),
                pred_span=pred_span,
                gt_span=gt_span,
                duration=pred["sample"].get("duration", 0),
                raw_text=text,
                query=pred["sample"].get("query", ""),
            )

        auditor.check_batch_summary(all_pred_spans, all_gt_spans)
        auditor.flush()

        n = len(predictions)
        result = {f"f1_iou_{t}": round(hits[t] / n, 4) for t in thresholds}
        result["response_quality"] = round(
            sum(response_qualities) / len(response_qualities), 4
        ) if response_qualities else 0.0
        return result

    def metric_names(self) -> list[str]:
        return ["f1_iou_0.3", "f1_iou_0.5", "f1_iou_0.7", "response_quality"]
