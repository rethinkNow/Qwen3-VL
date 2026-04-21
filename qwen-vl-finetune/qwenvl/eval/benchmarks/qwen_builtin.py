"""Wrapper benchmarks for Qwen3-VL's built-in evaluation scripts.

These benchmarks (VideoMME, MMMU, RealWorldQA, MathVision, ODinW-13)
serve as sanity/regression checks during training. They use MCQ format
so scoring doesn't need GPT-4o judge calls.

Since the original scripts use vLLM, we re-implement lightweight versions
that work with the HuggingFace model directly during training.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark

logger = logging.getLogger(__name__)


def _extract_mcq_answer(text: str) -> str:
    """Extract MCQ answer letter from model output.

    Handles various formats:
    - "The answer is B"
    - "B. Candles"
    - "**B**"
    - Long reasoning ending with "B"
    - Just "B"
    """
    text = text.strip().upper()

    # Pattern 1: "the answer is X" or "correct answer is X"
    match = re.search(r"(?:THE\s+)?ANSWER\s+IS\s*:?\s*\(?([A-E])\)?", text)
    if match:
        return match.group(1)

    # Pattern 2: "**X**" (bold markdown)
    match = re.search(r"\*\*([A-E])\*\*", text)
    if match:
        return match.group(1)

    # Pattern 3: Last standalone letter A-E in the text
    matches = re.findall(r"\b([A-E])\b", text)
    if matches:
        return matches[-1]

    # Pattern 4: First letter if text starts with A-E
    if text and text[0] in "ABCDE":
        return text[0]

    return ""


class VideoMME(Benchmark):
    name = "videomme"
    requires_judge = False

    def __init__(self):
        self.data_dir = os.environ.get(
            "VIDEOMME_PATH",
            "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/videomme",
        )

    def load_all(self) -> list[dict]:
        # Try multiple paths for the parquet file
        parquet_paths = [
            Path(self.data_dir) / "videomme" / "test-00000-of-00001.parquet",
            Path(self.data_dir) / "test-00000-of-00001.parquet",
        ]
        ann_path = None
        for p in parquet_paths:
            if p.exists():
                ann_path = p
                break

        if ann_path is None:
            logger.warning(f"VideoMME data not found at {self.data_dir}")
            return []

        try:
            import pandas as pd
            df = pd.read_parquet(ann_path)
            data = df.to_dict("records")
        except ImportError:
            logger.warning("pandas needed for parquet. pip install pandas pyarrow")
            return []

        samples = []
        for item in data:
            video_id = item.get("videoID", item.get("video_id", ""))
            # Try multiple paths
            video_path = Path(self.data_dir) / "videos" / "data" / (video_id + ".mp4")
            if not video_path.exists():
                video_path = Path(self.data_dir) / "videos" / (video_id + ".mp4")
            # Options come as numpy array from parquet, convert to list
            options = item.get("options", [])
            if hasattr(options, "tolist"):
                options = options.tolist()

            samples.append({
                "id": item.get("question_id", video_id),
                "video": str(video_path),
                "question": item.get("question", ""),
                "choices": options,
                "answer": item.get("answer", ""),
                "duration": item.get("duration", "short"),
                "metadata": {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in item.items()},
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        choices = sample["choices"]
        if isinstance(choices, list):
            choices_str = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
        elif isinstance(choices, dict):
            choices_str = "\n".join(f"{k}) {v}" for k, v in choices.items())
        else:
            choices_str = str(choices)

        return {
            "video": sample["video"],
            "prompt": f"{sample['question']}\n\n{choices_str}\n\nAnswer with the letter only.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"accuracy": 0.0}

        correct = 0
        for pred in predictions:
            pred_answer = _extract_mcq_answer(pred["prediction"])
            gt = pred["sample"]["answer"].strip().upper()
            if pred_answer == gt:
                correct += 1

        return {"accuracy": round(correct / len(predictions), 4)}

    def metric_names(self) -> list[str]:
        return ["accuracy"]


class MMMU(Benchmark):
    name = "mmmu"
    requires_judge = False

    def __init__(self):
        self.data_dir = os.environ.get(
            "MMMU_PATH",
            "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/mmmu",
        )

    def load_all(self) -> list[dict]:
        # MMMU has per-subject directories with parquet files
        import glob
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas needed. pip install pandas pyarrow")
            return []

        parquet_files = glob.glob(str(Path(self.data_dir) / "*/validation-*.parquet"))
        if not parquet_files:
            logger.warning(f"MMMU data not found at {self.data_dir}")
            return []

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        # Save images from parquet to disk
        img_dir = Path(self.data_dir) / "_images"
        img_dir.mkdir(exist_ok=True)

        samples = []
        for _, item in df.iterrows():
            item_id = item.get("id", "")
            img_path = img_dir / f"{item_id}.jpg"

            # Extract first image from parquet
            if not img_path.exists():
                img_data = item.get("image_1")
                if img_data is not None:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_path.write_bytes(img_data["bytes"])
                    elif isinstance(img_data, bytes):
                        img_path.write_bytes(img_data)

            # Parse options string to list
            options = item.get("options", "[]")
            if isinstance(options, str):
                try:
                    options = json.loads(options.replace("'", '"'))
                except Exception:
                    options = [options]

            samples.append({
                "id": str(item_id),
                "video": str(img_path),
                "question": item.get("question", "").replace("<image 1>", ""),
                "choices": options if isinstance(options, list) else [options],
                "answer": item.get("answer", ""),
                "metadata": {"id": item_id, "subfield": item.get("subfield", "")},
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        choices = sample["choices"]
        if isinstance(choices, list):
            choices_str = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
        else:
            choices_str = str(choices)
        return {
            "video": sample["video"],
            "prompt": f"{sample['question']}\n\n{choices_str}\n\nAnswer with the letter only.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"accuracy": 0.0}
        correct = 0
        for pred in predictions:
            pred_answer = _extract_mcq_answer(pred["prediction"])
            if pred_answer == pred["sample"]["answer"].strip().upper():
                correct += 1
        return {"accuracy": round(correct / len(predictions), 4)}

    def metric_names(self) -> list[str]:
        return ["accuracy"]


class MathVision(Benchmark):
    name = "mathvision"
    requires_judge = False

    def __init__(self):
        self.data_dir = os.environ.get(
            "MATHVISION_PATH",
            "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/mathvision",
        )

    def load_all(self) -> list[dict]:
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas needed. pip install pandas pyarrow")
            return []

        # Find parquet files
        parquet_files = list(Path(self.data_dir).rglob("*.parquet"))
        # Prefer testmini for speed
        testmini = [f for f in parquet_files if "testmini" in f.name]
        target = testmini[0] if testmini else (parquet_files[0] if parquet_files else None)

        if target is None:
            logger.warning(f"MathVision data not found at {self.data_dir}")
            return []

        df = pd.read_parquet(target)

        # Save images from parquet
        img_dir = Path(self.data_dir) / "_images"
        img_dir.mkdir(exist_ok=True)

        samples = []
        for _, item in df.iterrows():
            item_id = str(item.get("id", ""))
            img_path = img_dir / f"{item_id}.jpg"

            if not img_path.exists():
                img_data = item.get("decoded_image") or item.get("image")
                if img_data is not None:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_path.write_bytes(img_data["bytes"])
                    elif isinstance(img_data, bytes):
                        img_path.write_bytes(img_data)

            options = item.get("options", "[]")
            if isinstance(options, str):
                try:
                    options = json.loads(options.replace("'", '"'))
                except Exception:
                    options = [options]

            samples.append({
                "id": item_id,
                "video": str(img_path),
                "question": item.get("question", ""),
                "choices": options if isinstance(options, list) else [options],
                "answer": item.get("answer", ""),
                "metadata": {"id": item_id, "subject": item.get("subject", "")},
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        choices = sample["choices"]
        if isinstance(choices, list):
            choices_str = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
        else:
            choices_str = str(choices)
        return {
            "video": sample["video"],
            "prompt": f"{sample['question']}\n\n{choices_str}\n\nAnswer with the letter only.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"accuracy": 0.0}
        correct = 0
        for pred in predictions:
            pred_answer = _extract_mcq_answer(pred["prediction"])
            if pred_answer == pred["sample"]["answer"].strip().upper():
                correct += 1
        return {"accuracy": round(correct / len(predictions), 4)}

    def metric_names(self) -> list[str]:
        return ["accuracy"]


class ODinW13(Benchmark):
    name = "odinw13"
    requires_judge = False

    def __init__(self):
        self.data_dir = os.environ.get(
            "ODINW13_PATH",
            "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/odinw13",
        )

    def load_all(self) -> list[dict]:
        ann_path = Path(self.data_dir) / "annotations.json"
        if not ann_path.exists():
            logger.warning(f"ODinW-13 data not found at {ann_path}")
            return []
        with open(ann_path) as f:
            data = json.load(f)
        return [{
            "id": item.get("id", ""),
            "video": str(Path(self.data_dir) / "images" / item.get("image", "")),
            "question": item.get("question", "Detect all objects in this image."),
            "categories": item.get("categories", []),
            "ground_truth": item.get("annotations", []),
            "metadata": item,
        } for item in data]

    def format_prompt(self, sample: dict) -> dict:
        categories = ", ".join(sample.get("categories", []))
        return {
            "video": sample["video"],
            "prompt": f"Detect all objects in this image. Categories: {categories}. "
                      "List each detected object with its category.",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"accuracy": 0.0}
        # Simplified scoring: check if predicted categories match ground truth
        correct = 0
        total_gt = 0
        for pred in predictions:
            gt_cats = set(a.get("category", "") for a in pred["sample"].get("ground_truth", []))
            pred_text = pred["prediction"].lower()
            matched = sum(1 for c in gt_cats if c.lower() in pred_text)
            total_gt += len(gt_cats)
            correct += matched
        recall = correct / (total_gt + 1e-8)
        return {"recall": round(recall, 4)}

    def metric_names(self) -> list[str]:
        return ["recall"]


class RealWorldQA(Benchmark):
    name = "realworldqa"
    requires_judge = False

    def __init__(self):
        self.data_dir = os.environ.get(
            "REALWORLDQA_PATH",
            "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/realworldqa",
        )

    def load_all(self) -> list[dict]:
        data_dir = Path(self.data_dir) / "data"
        parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []

        if not parquet_files:
            logger.warning(f"RealWorldQA data not found at {data_dir}")
            return []

        try:
            import pandas as pd
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            data = df.to_dict("records")
        except ImportError:
            logger.warning("pandas needed for parquet. pip install pandas pyarrow")
            return []

        img_dir = Path(self.data_dir) / "images"
        img_dir.mkdir(exist_ok=True)

        samples = []
        for i, item in enumerate(data):
            img_path = img_dir / f"{i}.jpg"
            if not img_path.exists() and item.get("image") is not None:
                img_data = item["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_path.write_bytes(img_data["bytes"])
                elif isinstance(img_data, bytes):
                    img_path.write_bytes(img_data)

            answer = str(item.get("answer", ""))

            samples.append({
                "id": str(i),
                "video": str(img_path),
                "question": item.get("question", ""),
                "choices": [],
                "answer": answer,
                "metadata": {k: v for k, v in item.items() if k != "image"},
            })
        return samples

    def format_prompt(self, sample: dict) -> dict:
        question = sample["question"]
        if "answer directly" not in question.lower():
            question = f"{question}\nPlease answer directly with a single word or number."
        return {
            "video": sample["video"],
            "prompt": question,
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"accuracy": 0.0}
        correct = 0
        for pred in predictions:
            pred_text = pred["prediction"].strip().lower()
            gt_text = pred["sample"]["answer"].strip().lower()
            if pred_text == gt_text or gt_text in pred_text or pred_text in gt_text:
                correct += 1
        return {"accuracy": round(correct / len(predictions), 4)}

    def metric_names(self) -> list[str]:
        return ["accuracy"]
