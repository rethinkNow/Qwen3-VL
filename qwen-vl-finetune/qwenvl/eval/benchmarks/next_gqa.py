from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path

from qwenvl.eval.benchmarks.base import Benchmark
from qwenvl.eval.benchmarks.et_bench import (
    compute_iou, _parse_time_span, _parse_frame_span,
    USE_FRAME_NUMBERS, _get_frame_count,
)
from qwenvl.eval.timestamp_audit import TimestampAuditor

logger = logging.getLogger(__name__)

NEXT_GQA_PATH = os.environ.get(
    "NEXT_GQA_PATH",
    "/home/mars_rover/model_disk/Qwen3-VL/evaluation_set/next_gqa",
)


class NExTGQA(Benchmark):
    name = "next_gqa"
    requires_judge = True

    def load_all(self) -> list[dict]:
        base = Path(NEXT_GQA_PATH) / "datasets" / "nextgqa"

        # Load QA from val.csv (has answers + matches gsub_val.json grounding annotations)
        csv_path = base / "val.csv"
        if not csv_path.exists():
            logger.warning(f"NExT-GQA CSV not found at {csv_path}")
            return []

        # Load grounding annotations
        gsub_path = base / "gsub_val.json"
        gsub = {}
        if gsub_path.exists():
            with open(gsub_path) as f:
                gsub = json.load(f)

        # Load video ID mapping
        map_path = base / "map_vid_vidorID.json"
        vid_map = {}
        if map_path.exists():
            with open(map_path) as f:
                vid_map = json.load(f)

        # Video directories - check multiple locations
        video_dirs = [
            Path(NEXT_GQA_PATH) / "NExTVideo",
            Path(NEXT_GQA_PATH) / "videos",
        ]

        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row.get("video_id", "")
                qid = row.get("qid", "")

                # Find video file
                video_path = None
                for vdir in video_dirs:
                    for ext in [".mp4", ".webm", ".mkv"]:
                        candidate = vdir / (video_id + ext)
                        if candidate.exists():
                            video_path = str(candidate)
                            break
                    if video_path:
                        break

                if video_path is None:
                    # Try mapped ID
                    mapped_id = vid_map.get(video_id, video_id)
                    for vdir in video_dirs:
                        for ext in [".mp4", ".webm", ".mkv"]:
                            candidate = vdir / (mapped_id + ext)
                            if candidate.exists():
                                video_path = str(candidate)
                                break
                        if video_path:
                            break

                if video_path is None:
                    continue

                # Build choices
                choices = {}
                for i in range(5):
                    key = f"a{i}"
                    if key in row and row[key]:
                        choices[chr(65 + i)] = row[key]

                # Get grounding span if available
                gt_span = (0, 0)
                if video_id in gsub and qid in gsub[video_id].get("location", {}):
                    spans = gsub[video_id]["location"][qid]
                    if spans:
                        gt_span = (spans[0][0], spans[0][1])

                # Find correct answer letter
                answer_text = row.get("answer", "")
                answer_letter = ""
                for letter, choice_text in choices.items():
                    if choice_text.strip().lower() == answer_text.strip().lower():
                        answer_letter = letter
                        break

                samples.append({
                    "id": f"{video_id}_{qid}",
                    "video": video_path,
                    "question": row.get("question", ""),
                    "choices": choices,
                    "answer": answer_letter,
                    "gt_span": gt_span,
                    "metadata": row,
                })

        return samples

    def format_prompt(self, sample: dict) -> dict:
        choices_str = "\n".join(
            f"{k}) {v}" for k, v in sample["choices"].items()
        )

        if USE_FRAME_NUMBERS:
            # NExT-GQA videos are typically short, but use frame numbers anyway
            time_instruction = (
                "Select the correct answer and provide the frame range "
                "(start and end frame numbers) where the evidence is found."
            )
        else:
            time_instruction = (
                "Select the correct answer and provide the time span "
                "(start and end seconds) where the evidence is found."
            )

        return {
            "video": sample["video"],
            "prompt": f"{sample['question']}\n\n{choices_str}\n\n{time_instruction}",
        }

    def score(self, predictions: list[dict], judge=None) -> dict[str, float]:
        if not predictions:
            return {"acc_gqa": 0.0, "acc": 0.0, "iou_0.5": 0.0, "response_quality": 0.0}

        auditor = TimestampAuditor("next_gqa")
        correct_answer = 0
        correct_gqa = 0
        iou_hits = 0
        response_qualities = []
        all_pred_spans = []
        all_gt_spans = []

        # Batch TVG judge calls for temporal grounding quality
        judge_results = {}
        if judge is not None:
            batch_size = 10
            for batch_start in range(0, len(predictions), batch_size):
                batch = predictions[batch_start:batch_start + batch_size]
                items = [
                    {"response": p["prediction"], "query": p["sample"].get("question", "")}
                    for p in batch
                ]
                results = judge.tvg_score_batch(items)
                for j, r in enumerate(results):
                    judge_results[batch_start + j] = r

        for i, pred in enumerate(predictions):
            text = pred["prediction"].strip()
            gt_answer = pred["sample"]["answer"].strip().upper()
            gt_span = pred["sample"]["gt_span"]

            # Extract predicted answer
            answer_match = re.search(r"\b([A-E])\b", text.upper())
            pred_answer = answer_match.group(1) if answer_match else ""

            answer_correct = pred_answer == gt_answer
            if answer_correct:
                correct_answer += 1

            # Extract predicted span using shared parser
            if USE_FRAME_NUMBERS:
                # NExT-GQA doesn't store duration in sample, estimate from gt_span
                # For frame mode, try frame parser first, fall back to time parser
                pred_span = _parse_time_span(text)  # still works for small numbers
            else:
                pred_span = _parse_time_span(text)

            # Fallback to judge-extracted timestamps if regex failed
            if pred_span == (0, 0) and i in judge_results:
                jr = judge_results[i]
                if jr["start"] != 0.0 or jr["end"] != 0.0:
                    pred_span = (jr["start"], jr["end"])

            iou = compute_iou(pred_span, gt_span)

            if iou >= 0.5:
                iou_hits += 1

            if answer_correct and iou >= 0.5:
                correct_gqa += 1

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
                raw_text=text,
                query=pred["sample"].get("question", ""),
            )

        auditor.check_batch_summary(all_pred_spans, all_gt_spans)
        auditor.flush()

        n = len(predictions)
        return {
            "acc_gqa": round(correct_gqa / n, 4),
            "acc": round(correct_answer / n, 4),
            "iou_0.5": round(iou_hits / n, 4),
            "response_quality": round(
                sum(response_qualities) / len(response_qualities), 4
            ) if response_qualities else 0.0,
        }

    def metric_names(self) -> list[str]:
        return ["acc_gqa", "acc", "iou_0.5", "response_quality"]
