from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)

CAREBENCH_PATH = os.environ.get(
    "CAREBENCH_PATH",
    "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/carebench",
)


class CaReBench(Benchmark):
    name = "carebench"
    requires_judge = True
    max_new_tokens = 1024

    def load_all(self) -> list[dict]:
        ann_path = Path(CAREBENCH_PATH) / "json" / "metadata.json"
        if not ann_path.exists():
            logger.warning(f"CaReBench annotations not found at {ann_path}")
            return []
        with open(ann_path) as f:
            data = json.load(f)
        samples = []
        for item in data:
            video_name = item.get("video", "")
            video_path = Path(CAREBENCH_PATH) / "videos" / video_name
            samples.append({
                "id": video_name,
                "video": str(video_path),
                "reference": item.get("caption", ""),
                "spatial_reference": item.get("spatial_caption", ""),
                "temporal_reference": item.get("temporal_caption", ""),
                "events": item.get("events", []),
                "objects": item.get("objects", []),
                "metadata": item,
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        return {
            "video": sample["video"],
            "prompt": "Describe this video in detail, covering both spatial and temporal aspects.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"capst_f1": 0.0, "spatial_f1": 0.0, "temporal_f1": 0.0}
        if judge is None:
            logger.warning("CaReBench requires a GPT-4o judge. Returning 0.")
            return {"capst_f1": 0.0, "spatial_f1": 0.0, "temporal_f1": 0.0}

        spatial_scores = []
        temporal_scores = []

        for pred in predictions:
            prediction = pred["prediction"]
            sample = pred["sample"]

            spatial_result = judge.capst_score(
                prediction, sample.get("spatial_reference", sample["reference"]), category="spatial"
            )
            temporal_result = judge.capst_score(
                prediction, sample.get("temporal_reference", sample["reference"]), category="temporal"
            )
            spatial_scores.append(spatial_result["f1"])
            temporal_scores.append(temporal_result["f1"])

        spatial_f1 = sum(spatial_scores) / len(spatial_scores)
        temporal_f1 = sum(temporal_scores) / len(temporal_scores)
        capst_f1 = (spatial_f1 + temporal_f1) / 2

        return {
            "capst_f1": round(capst_f1, 4),
            "spatial_f1": round(spatial_f1, 4),
            "temporal_f1": round(temporal_f1, 4),
        }

    def metric_names(self) -> list[str]:
        return ["capst_f1", "spatial_f1", "temporal_f1"]
