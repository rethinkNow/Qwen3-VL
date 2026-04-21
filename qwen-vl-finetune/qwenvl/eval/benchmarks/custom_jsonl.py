"""Custom JSONL benchmark — evaluate model on any JSONL file (training data, held-out, etc).

Reads ShareGPT-format JSONL, runs model on each video with the prompt,
scores with LLM judge against ground truth.

Usage:
    python run_eval.py --benchmarks custom_jsonl --jsonl /path/to/data.jsonl --samples 10
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark
from qwenvl.eval.benchmarks.et_bench import parse_timestamp_events

logger = logging.getLogger(__name__)


def _parse_scene_and_events(gt_text: str) -> tuple[str, list[dict]]:
    """Parse ground truth into scene description and structured events.

    GT format:
        Scene: The setting is...

        Events:
        <0.0 - 3.5> man stands eating
        <3.5 - 8.0> boy reaches up

    Returns (scene_text, events_list) where events_list is
    [{"start": float, "end": float, "caption": str}, ...]
    """
    scene = ""
    events = []

    if "Events:" in gt_text:
        parts = gt_text.split("Events:", 1)
        scene = parts[0].replace("Scene:", "").strip()
        events_text = parts[1].strip()

        # Parse <start - end> lines
        parsed = parse_timestamp_events(events_text)
        for start, end, caption in parsed:
            events.append({"start": start, "end": end, "caption": caption})
    else:
        # No Events section — treat entire text as scene/reference
        scene = gt_text.replace("Scene:", "").strip()

    return scene, events


def _strip_video_tag(prompt: str) -> str:
    """Strip <video> tag from prompt — runner handles video embedding separately."""
    prompt = re.sub(r"<video>\s*\n?", "", prompt).strip()
    return prompt


class CustomJsonlBench(Benchmark):
    name = "custom_jsonl"
    requires_judge = True
    max_new_tokens = 2048
    repetition_penalty = 1.1

    def load_all(self) -> list[dict]:
        jsonl_path = os.environ.get("CUSTOM_JSONL_PATH", "")
        if not jsonl_path or not os.path.exists(jsonl_path):
            logger.warning(
                f"Custom JSONL not found at '{jsonl_path}'. "
                f"Set CUSTOM_JSONL_PATH env var or pass --jsonl to run_eval.py"
            )
            return []

        samples = []
        with open(jsonl_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {idx + 1}")
                    continue

                video_path = record.get("video", "")
                conversations = record.get("conversations", [])

                if len(conversations) < 2:
                    continue

                human_turn = conversations[0].get("value", "")
                gpt_turn = conversations[1].get("value", "")

                prompt = _strip_video_tag(human_turn)
                scene, events = _parse_scene_and_events(gpt_turn)

                samples.append({
                    "id": str(idx),
                    "video": video_path,
                    "prompt": prompt,
                    "reference": gpt_turn,
                    "scene": scene,
                    "events": events,
                })

        logger.info(f"Loaded {len(samples)} samples from {jsonl_path}")
        return samples

    def format_prompt(self, sample: dict) -> dict:
        return {
            "video": sample["video"],
            "prompt": sample["prompt"],
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        defaults = {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "spatial_f1": 0.0,
            "caption_f1_temporal": 0.0,
            "caption_f1_ordinal": 0.0,
        }
        if not predictions:
            return defaults
        if judge is None:
            logger.warning("custom_jsonl requires a judge. Returning 0.")
            return defaults

        overall_scores = []
        spatial_scores = []
        caption_temporal_scores = []
        caption_ordinal_scores = []

        for pred in predictions:
            prediction = pred["prediction"]
            sample = pred["sample"]
            gt_reference = sample["reference"]
            gt_scene = sample.get("scene", "")
            gt_events = sample.get("events", [])

            # 1. Overall F1 (autodq_score) — event coverage
            result = judge.autodq_score(prediction, gt_reference)
            overall_scores.append(result)

            # 2. Spatial F1 (capst_score) — scene quality
            if gt_scene:
                # Extract scene from model output (before "Events:" if present)
                pred_scene = prediction
                if "Events:" in prediction:
                    pred_scene = prediction.split("Events:")[0]
                pred_scene = pred_scene.replace("Scene:", "").strip()

                spatial_result = judge.capst_score(pred_scene, gt_scene, category="spatial")
                spatial_scores.append(spatial_result["f1"])
            else:
                spatial_scores.append(0.0)

            # 3. Temporal caption F1 (dvc_caption_score) — timestamp + caption quality
            if gt_events:
                pred_events = parse_timestamp_events(prediction)
                if pred_events:
                    dvc_result = judge.dvc_caption_score(pred_events, gt_events)
                    caption_temporal_scores.append(dvc_result.get("caption_f1_temporal", 0.0))
                    caption_ordinal_scores.append(dvc_result.get("caption_f1_ordinal", 0.0))
                else:
                    caption_temporal_scores.append(0.0)
                    caption_ordinal_scores.append(0.0)
            else:
                caption_temporal_scores.append(0.0)
                caption_ordinal_scores.append(0.0)

        n = len(predictions)
        precision = sum(s["precision"] for s in overall_scores) / n
        recall = sum(s["recall"] for s in overall_scores) / n
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "spatial_f1": round(sum(spatial_scores) / n, 4),
            "caption_f1_temporal": round(sum(caption_temporal_scores) / n, 4),
            "caption_f1_ordinal": round(sum(caption_ordinal_scores) / n, 4),
        }

    def metric_names(self) -> list[str]:
        return ["f1", "precision", "recall", "spatial_f1", "caption_f1_temporal", "caption_f1_ordinal"]
