from __future__ import annotations

import logging
import time

import torch
import wandb
from transformers import TrainerCallback

from qwenvl.eval.benchmarks import BENCHMARK_REGISTRY
from qwenvl.eval.judge import GPT4oJudge
from qwenvl.eval.runner import run_inference

logger = logging.getLogger(__name__)


class EvalCallback(TrainerCallback):
    """HuggingFace TrainerCallback for periodic benchmark evaluation.

    Runs eval every `eval_steps` training steps. Supports subset (fast)
    and full evaluation modes. Logs metrics to wandb.
    """

    def __init__(
        self,
        eval_steps: int,
        eval_mode: str = "subset",
        eval_subset_size: int = 200,
        eval_benchmarks: str = "dream1k,carebench",
        processor=None,
        verbose: bool = False,
    ):
        self.eval_steps = eval_steps
        self.eval_mode = eval_mode
        self.eval_subset_size = eval_subset_size
        self.benchmark_names = [b.strip() for b in eval_benchmarks.split(",") if b.strip()]
        self.processor = processor
        self.verbose = verbose

        # Initialize judge lazily (only when needed)
        self._judge = None

        # Import all benchmark modules to trigger registration
        import qwenvl.eval.benchmarks.dream1k  # noqa
        import qwenvl.eval.benchmarks.carebench  # noqa
        import qwenvl.eval.benchmarks.et_bench  # noqa
        import qwenvl.eval.benchmarks.activitynet  # noqa
        import qwenvl.eval.benchmarks.next_gqa  # noqa
        import qwenvl.eval.benchmarks.qwen_builtin  # noqa

        # Validate benchmark names
        available = set(BENCHMARK_REGISTRY.keys())
        requested = set(self.benchmark_names)
        missing = requested - available
        if missing:
            logger.warning(f"Unknown benchmarks: {missing}. Available: {available}")
            self.benchmark_names = [b for b in self.benchmark_names if b in available]

        logger.info(
            f"EvalCallback initialized: steps={eval_steps}, mode={eval_mode}, "
            f"subset_size={eval_subset_size}, benchmarks={self.benchmark_names}"
        )

    @property
    def judge(self):
        if self._judge is None:
            self._judge = GPT4oJudge()
        return self._judge

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.eval_steps != 0:
            return

        # Only run on rank 0
        if args.local_rank not in (-1, 0):
            return

        logger.info(f"=== Running evaluation at step {state.global_step} ===")
        t0 = time.time()

        metrics = self._run_eval(model, state.global_step)

        elapsed = time.time() - t0
        logger.info(f"=== Evaluation complete in {elapsed:.0f}s ===")

        # Log to wandb
        if wandb.run is not None:
            flat_metrics = {}
            for bench_name, scores in metrics.items():
                for metric_name, value in scores.items():
                    key = f"eval/{bench_name}/{metric_name}"
                    flat_metrics[key] = value
                    logger.info(f"  {key}: {value}")
            flat_metrics["eval/time_sec"] = elapsed
            wandb.log(flat_metrics, step=state.global_step)

    def _run_eval(self, model, step: int) -> dict[str, dict[str, float]]:
        metrics = {}

        for bench_name in self.benchmark_names:
            bench = BENCHMARK_REGISTRY[bench_name]

            # Load samples
            if self.eval_mode == "subset":
                samples = bench.load_subset(self.eval_subset_size, seed=step)
            else:
                samples = bench.load_all()

            if not samples:
                logger.warning(f"No samples loaded for {bench_name}, skipping")
                metrics[bench_name] = {m: 0.0 for m in bench.metric_names()}
                continue

            logger.info(f"Evaluating {bench_name}: {len(samples)} samples")

            # Run inference
            predictions = run_inference(
                model=model,
                processor=self.processor,
                samples=samples,
                benchmark=bench,
                verbose=self.verbose,
            )

            # Score with verbose logging
            if self.verbose:
                print(f"\n  [SCORING] {bench_name}: comparing {len(predictions)} predictions", flush=True)
                for p in predictions:
                    sample = p["sample"]
                    pred = p["prediction"]
                    # Log what's being compared
                    if "reference" in sample:
                        print(f"    [GT]   {sample['reference'][:150]}...", flush=True)
                    elif "events" in sample:
                        print(f"    [GT]   Events: {sample['events'][:3]}", flush=True)
                    elif "gt_span" in sample:
                        print(f"    [GT]   Span: {sample['gt_span']}", flush=True)
                    elif "answer" in sample:
                        print(f"    [GT]   Answer: {sample['answer']}", flush=True)
                    print(f"    [PRED] {pred[:150]}{'...' if len(pred) > 150 else ''}", flush=True)

            judge = self.judge if bench.requires_judge else None
            scores = bench.score(predictions, judge=judge)

            if self.verbose:
                print(f"  [SCORES] {bench_name}: {scores}", flush=True)

            metrics[bench_name] = scores

        return metrics
