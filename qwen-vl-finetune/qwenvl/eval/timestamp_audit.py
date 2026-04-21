"""Timestamp audit logger for flagging degenerate temporal predictions.

Logs issues to a file for manual review. Detects:
- Predictions always clustering at video start (degenerate)
- Timestamp range much smaller than video duration (scale mismatch)
- Parser failures (couldn't extract timestamps)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = os.environ.get(
    "EVAL_AUDIT_LOG",
    "eval_timestamp_audit.jsonl",
)


class TimestampAuditor:
    """Collects timestamp issues during scoring and writes them to a log file."""

    def __init__(self, benchmark_name: str, log_path: str | None = None):
        self.benchmark_name = benchmark_name
        self.log_path = log_path or AUDIT_LOG_PATH
        self.issues: list[dict] = []

    def check_prediction(
        self,
        sample_id: str,
        video_path: str,
        pred_span: tuple[float, float],
        gt_span: tuple[float, float],
        duration: float = 0,
        raw_text: str = "",
        query: str = "",
    ):
        """Check a single prediction for timestamp issues."""
        issue_types = []

        # Parser failure: couldn't extract any timestamps
        if pred_span == (0, 0) and raw_text.strip():
            issue_types.append("parser_failure")

        # Degenerate: prediction is always near start (0-5s) regardless of GT
        pred_end = pred_span[1]
        if pred_end <= 5.0 and gt_span[1] > 10.0:
            issue_types.append("degenerate_start")

        # Scale mismatch: predicted range is <10% of video duration
        if duration > 0:
            pred_range = pred_span[1] - pred_span[0]
            if pred_range > 0 and pred_range < duration * 0.1 and duration > 30:
                issue_types.append("scale_mismatch")

        # GT is far from prediction (large temporal gap)
        if pred_span != (0, 0) and gt_span != (0, 0):
            gap = min(abs(pred_span[0] - gt_span[0]), abs(pred_span[1] - gt_span[1]))
            if gap > 30 and duration > 60:
                issue_types.append("large_temporal_gap")

        if issue_types:
            self.issues.append({
                "benchmark": self.benchmark_name,
                "sample_id": sample_id,
                "video": os.path.basename(video_path),
                "query": query[:200] if query else "",
                "pred_span": list(pred_span),
                "gt_span": list(gt_span),
                "duration": duration,
                "raw_text": raw_text[:300] if raw_text else "",
                "issues": issue_types,
            })

    def check_batch_summary(
        self,
        all_pred_spans: list[tuple[float, float]],
        all_gt_spans: list[tuple[float, float]],
    ):
        """Check batch-level patterns (e.g., all predictions cluster together)."""
        if len(all_pred_spans) < 3:
            return

        # Check if all predictions have the same or very similar end time
        pred_ends = [s[1] for s in all_pred_spans if s != (0, 0)]
        if pred_ends:
            max_end = max(pred_ends)
            if max_end <= 15.0 and len(pred_ends) >= 3:
                gt_ends = [s[1] for s in all_gt_spans if s != (0, 0)]
                max_gt = max(gt_ends) if gt_ends else 0
                if max_gt > 30:
                    self.issues.append({
                        "benchmark": self.benchmark_name,
                        "sample_id": "_batch_summary_",
                        "video": "",
                        "query": "",
                        "pred_span": [0, max_end],
                        "gt_span": [0, max_gt],
                        "duration": 0,
                        "raw_text": f"All {len(pred_ends)} predictions end before {max_end}s, GT spans up to {max_gt}s",
                        "issues": ["batch_degenerate_clustering"],
                    })

    def flush(self):
        """Write accumulated issues to the log file."""
        if not self.issues:
            return

        log_path = Path(self.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as f:
            header = {
                "timestamp": datetime.now().isoformat(),
                "benchmark": self.benchmark_name,
                "total_issues": len(self.issues),
            }
            f.write(json.dumps(header) + "\n")
            for issue in self.issues:
                f.write(json.dumps(issue) + "\n")

        logger.info(
            f"Timestamp audit: {len(self.issues)} issues logged for {self.benchmark_name} → {self.log_path}"
        )
        self.issues = []
