from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class GPT4oJudge:
    """GPT-4o judge for DREAM-1K (AutoDQ) and CaReBench (CapST) scoring."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.endpoint = endpoint or os.environ.get(
            "OPENAI_ENDPOINT",
            "https://banim.cognitiveservices.azure.com/openai/v1/",
        )
        self.deployment = deployment or os.environ.get(
            "OPENAI_DEPLOYMENT",
            "gpt-4o",
        )

        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
        )
        logger.info(f"GPT4o judge initialized: endpoint={self.endpoint}, deployment={self.deployment}")

    def _call(self, system: str, user: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=1024,
                timeout=300,
            )
            return response.choices[0].message.content or "{}"
        except Exception as e:
            logger.warning(f"Judge API call failed: {e}")
            return "{}"

    def autodq_score(self, prediction: str, reference: str) -> dict[str, float]:
        """DREAM-1K AutoDQ: extract events from both, compute precision/recall."""
        system = (
            "You are an evaluation judge for video descriptions. "
            "Extract distinct atomic events from both the prediction and reference descriptions. "
            "An event is a single action or state change. "
            "Then compute how many predicted events match reference events (precision) "
            "and how many reference events are covered by predictions (recall). "
            "Return JSON only: {\"pred_events\": [...], \"ref_events\": [...], "
            "\"matched\": int, \"precision\": float, \"recall\": float}"
        )
        user = (
            f"Reference description:\n{reference}\n\n"
            f"Predicted description:\n{prediction}\n\n"
            "Extract events and compute precision/recall. Return JSON only."
        )
        try:
            result = self._call(system, user)
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return {
                "precision": float(data.get("precision", 0)),
                "recall": float(data.get("recall", 0)),
            }
        except Exception as e:
            logger.warning(f"AutoDQ judge parse failed: {e}")
            return {"precision": 0.0, "recall": 0.0}

    def dvc_caption_score(
        self,
        pred_events: list[tuple],
        gt_events: list[dict],
    ) -> dict[str, float]:
        """Score dense video captioning via temporal + ordinal matching.

        Args:
            pred_events: [(start, end, caption), ...] sorted by start time
            gt_events: [{"start": float, "end": float, "caption": str}, ...]

        Returns dict with caption_f1_temporal, caption_f1_ordinal, and their P/R.
        """
        from qwenvl.eval.benchmarks.et_bench import compute_iou

        if not pred_events or not gt_events:
            return {
                "caption_f1_temporal": 0.0,
                "caption_f1_ordinal": 0.0,
            }

        # Sort pred events by start time
        pred_events = sorted(pred_events, key=lambda x: x[0])

        # --- Temporal matching (IoU >= 0.3) ---
        temporal_pairs = []
        for gt in gt_events:
            gt_span = (gt["start"], gt["end"])
            best_iou = 0
            best_pred = None
            for pe in pred_events:
                iou = compute_iou((pe[0], pe[1]), gt_span)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pe
            if best_iou >= 0.3 and best_pred is not None:
                temporal_pairs.append((best_pred[2], gt["caption"]))

        # --- Ordinal matching (by position) ---
        n = min(len(pred_events), len(gt_events))
        ordinal_pairs = [
            (pred_events[i][2], gt_events[i]["caption"]) for i in range(n)
        ]

        # Combine all pairs into a single API call
        all_pairs = temporal_pairs + ordinal_pairs
        if not all_pairs:
            return {
                "caption_f1_temporal": 0.0,
                "caption_f1_ordinal": 0.0,
            }

        # Cap at 15 pairs per section
        temporal_pairs = temporal_pairs[:15]
        ordinal_pairs = ordinal_pairs[:15]
        all_pairs = temporal_pairs + ordinal_pairs

        system = (
            "You are an evaluation judge for dense video captioning. "
            "Score each predicted/reference caption pair from 0.0 to 1.0 based on semantic similarity. "
            "0 = completely unrelated, 1 = semantically equivalent. "
            'Return JSON only: {"scores": [float, ...]} with one score per pair, in order.'
        )
        lines = []
        for i, (pred_cap, gt_cap) in enumerate(all_pairs, 1):
            lines.append(f'{i}. Predicted: "{pred_cap}" | Reference: "{gt_cap}"')
        user = "Matched event pairs:\n" + "\n".join(lines) + "\n\nScore each pair 0.0-1.0. Return JSON only."

        try:
            result = self._call(system, user)
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            scores = [float(s) for s in data.get("scores", [])]
        except Exception as e:
            logger.warning(f"DVC caption judge parse failed: {e}")
            scores = []

        # Split scores back into temporal and ordinal
        n_temporal = len(temporal_pairs)
        n_ordinal = len(ordinal_pairs)
        temporal_scores = scores[:n_temporal] if len(scores) >= n_temporal else [0.0] * n_temporal
        ordinal_scores = scores[n_temporal:n_temporal + n_ordinal] if len(scores) >= n_temporal + n_ordinal else [0.0] * n_ordinal

        def _f1(precision: float, recall: float) -> float:
            return 2 * precision * recall / (precision + recall + 1e-8)

        # Temporal F1
        if temporal_pairs and pred_events and gt_events:
            t_precision = sum(temporal_scores) / len(pred_events)
            t_recall = sum(temporal_scores) / len(gt_events)
            temporal_f1 = _f1(t_precision, t_recall)
        else:
            temporal_f1 = 0.0

        # Ordinal F1
        if ordinal_pairs and pred_events and gt_events:
            o_precision = sum(ordinal_scores) / len(pred_events)
            o_recall = sum(ordinal_scores) / len(gt_events)
            ordinal_f1 = _f1(o_precision, o_recall)
        else:
            ordinal_f1 = 0.0

        return {
            "caption_f1_temporal": round(temporal_f1, 4),
            "caption_f1_ordinal": round(ordinal_f1, 4),
        }

    def tvg_score(self, response: str, query: str) -> dict[str, float]:
        """Evaluate a temporal video grounding response.

        Returns {"start": float, "end": float, "response_quality": float}
        """
        system = (
            "You are an evaluation judge for temporal video grounding. "
            "Given a query and a model's response, extract the predicted time span and assess response quality.\n"
            "- Extract start and end timestamps (in seconds) from the response text.\n"
            "- Score response_quality from 0.0 to 1.0:\n"
            "  1.0 = clearly specific, indicates the model understood the temporal location\n"
            "  0.5 = plausible but generic (e.g., 'beginning of video' for any query)\n"
            "  0.0 = degenerate, empty, or clearly wrong format\n"
            'Return JSON only: {"start": float, "end": float, "response_quality": float}'
        )
        user = (
            f'Query: "{query}"\n'
            f'Model response: "{response}"\n\n'
            "Extract timestamps and score quality. Return JSON only."
        )
        try:
            result = self._call(system, user)
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return {
                "start": float(data.get("start", 0)),
                "end": float(data.get("end", 0)),
                "response_quality": float(data.get("response_quality", 0)),
            }
        except Exception as e:
            logger.warning(f"TVG judge parse failed: {e}")
            return {"start": 0.0, "end": 0.0, "response_quality": 0.0}

    def tvg_score_batch(self, items: list[dict]) -> list[dict]:
        """Batch evaluate multiple TVG responses in a single API call.

        Args:
            items: [{"response": str, "query": str}, ...]

        Returns list of {"start": float, "end": float, "response_quality": float}
        """
        if not items:
            return []

        system = (
            "You are an evaluation judge for temporal video grounding. "
            "For each query-response pair, extract the predicted time span (in seconds) and assess response quality.\n"
            "- Score response_quality from 0.0 to 1.0:\n"
            "  1.0 = clearly specific, indicates the model understood the temporal location\n"
            "  0.5 = plausible but generic (e.g., 'beginning of video' for any query)\n"
            "  0.0 = degenerate, empty, or clearly wrong format\n"
            'Return JSON only: {"results": [{"start": float, "end": float, "response_quality": float}, ...]}'
        )
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f'{i}. Query: "{item["query"]}" | Response: "{item["response"]}"')
        user = "Evaluate these temporal grounding responses:\n" + "\n".join(lines) + "\n\nReturn JSON only."

        try:
            result = self._call(system, user)
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            results = []
            for r in data.get("results", []):
                results.append({
                    "start": float(r.get("start", 0)),
                    "end": float(r.get("end", 0)),
                    "response_quality": float(r.get("response_quality", 0)),
                })
            # Pad if judge returned fewer results
            while len(results) < len(items):
                results.append({"start": 0.0, "end": 0.0, "response_quality": 0.0})
            return results
        except Exception as e:
            logger.warning(f"TVG batch judge parse failed: {e}")
            return [{"start": 0.0, "end": 0.0, "response_quality": 0.0}] * len(items)

    def capst_score(self, prediction: str, reference: str, category: str = "spatial") -> dict[str, float]:
        """CaReBench CapST: score spatial or temporal aspects of caption."""
        system = (
            f"You are an evaluation judge for video descriptions. "
            f"Focus on {category} aspects of the descriptions. "
            f"{'Spatial aspects include: objects, their attributes, positions, and relationships.' if category == 'spatial' else 'Temporal aspects include: actions, events, their ordering, and timing.'} "
            f"Extract {category} phrases from both the prediction and reference. "
            f"Compute precision (correct {category} phrases / total predicted {category} phrases) "
            f"and recall (matched reference {category} phrases / total reference {category} phrases). "
            f"Return JSON only: {{\"pred_phrases\": [...], \"ref_phrases\": [...], "
            f"\"matched\": int, \"precision\": float, \"recall\": float, \"f1\": float}}"
        )
        user = (
            f"Reference:\n{reference}\n\n"
            f"Prediction:\n{prediction}\n\n"
            f"Extract {category} phrases and compute scores. Return JSON only."
        )
        try:
            result = self._call(system, user)
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            data = json.loads(json_str.strip())
            return {"f1": float(data.get("f1", 0))}
        except Exception as e:
            logger.warning(f"CapST judge parse failed: {e}")
            return {"f1": 0.0}
