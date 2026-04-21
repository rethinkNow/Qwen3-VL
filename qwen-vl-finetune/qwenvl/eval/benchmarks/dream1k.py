from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)

DREAM1K_PATH = os.environ.get(
    "DREAM1K_PATH",
    "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/dream1k",
)


class Dream1K(Benchmark):
    name = "dream1k"
    requires_judge = True
    max_new_tokens = 1024

    def load_all(self) -> list[dict]:
        ann_path = Path(DREAM1K_PATH) / "json" / "metadata.json"
        if not ann_path.exists():
            logger.warning(f"DREAM-1K annotations not found at {ann_path}")
            return []
        with open(ann_path) as f:
            data = json.load(f)
        samples = []
        for item in data:
            # video_file is like "video/1.mp4" but actual path is "video/DREAM-1K_videos/1.mp4"
            video_file = item.get("video_file", "")
            video_name = Path(video_file).name
            video_path = Path(DREAM1K_PATH) / "video" / "DREAM-1K_videos" / video_name
            if not video_path.exists():
                video_path = Path(DREAM1K_PATH) / video_file
            samples.append({
                "id": str(item.get("idx", "")),
                "video": str(video_path),
                "reference": item.get("description", ""),
                "events": item.get("events", []),
                "source": item.get("source", ""),
                "metadata": item,
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        return {
            "video": sample["video"],
            "prompt": "Describe this video in detail.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"f1": 0.0}
        if judge is None:
            logger.warning("DREAM-1K requires a GPT-4o judge. Returning 0.")
            return {"f1": 0.0}

        scores = []
        for pred in predictions:
            reference = pred["sample"]["reference"]
            prediction = pred["prediction"]
            result = judge.autodq_score(prediction, reference)
            scores.append(result)

        precision = sum(s["precision"] for s in scores) / len(scores)
        recall = sum(s["recall"] for s in scores) / len(scores)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    def metric_names(self) -> list[str]:
        return ["f1", "precision", "recall"]
