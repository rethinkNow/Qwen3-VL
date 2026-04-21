from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark
from qwenvl.eval.benchmarks.et_bench import (
    compute_iou, parse_timestamp_events, parse_frame_events,
    USE_FRAME_NUMBERS, _get_frame_count,
)
from qwenvl.eval.timestamp_audit import TimestampAuditor

logger = logging.getLogger(__name__)

ACTIVITYNET_PATH = os.environ.get(
    "ACTIVITYNET_PATH",
    "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/activitynet",
)


class ActivityNetCaptions(Benchmark):
    name = "activitynet"
    requires_judge = True
    max_new_tokens = 2048
    repetition_penalty = 1.1

    def load_all(self) -> list[dict]:
        # Try val1 first, then val2
        for fname in ["activitynet_captions_val1.json", "activitynet_captions_val2.json"]:
            ann_path = Path(ACTIVITYNET_PATH) / fname
            if ann_path.exists():
                break
        else:
            logger.warning(f"ActivityNet annotations not found at {ACTIVITYNET_PATH}")
            return []

        with open(ann_path) as f:
            data = json.load(f)

        # Handle both dict format {video_id: info} and list format [{video_id, ...}]
        if isinstance(data, dict):
            items = [(vid, info) for vid, info in data.items()]
        else:
            items = [(item.get("video_id", ""), item) for item in data]

        samples = []
        for video_id, info in items:
            video_name = info.get("video", f"{video_id}.mp4")
            video_path = Path(ACTIVITYNET_PATH) / "Activity_Videos" / video_name
            if not video_path.exists():
                video_path = Path(ACTIVITYNET_PATH) / "Activity_Videos" / f"v_{video_id}.mp4"

            events = []
            timestamps = info.get("timestamps", [])
            sentences = info.get("sentences", [])
            if not sentences and info.get("caption"):
                sentences = [info["caption"]]
                timestamps = timestamps or [[0, info.get("duration", 0)]]
            for ts, sent in zip(timestamps, sentences):
                events.append({"start": ts[0], "end": ts[1], "caption": sent})
            samples.append({
                "id": video_id,
                "video": str(video_path),
                "events": events,
                "duration": info.get("duration", 0),
                "metadata": info,
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

        auditor = TimestampAuditor("activitynet")
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

            # Audit
            pred_range = (pred_events[0][0], pred_events[-1][1])
            gt_range = (gt_events[0]["start"], gt_events[-1]["end"])
            auditor.check_prediction(
                sample_id=pred["sample"].get("id", ""),
                video_path=pred["sample"].get("video", ""),
                pred_span=pred_range,
                gt_span=gt_range,
                duration=pred["sample"].get("duration", 0),
                raw_text=pred["prediction"][:300],
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
