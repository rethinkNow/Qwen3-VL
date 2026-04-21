"""Wandb logging callback for training: mini eval via subprocess + training data logging."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading

import wandb
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class WandbTrainingCallback(TrainerCallback):
    """Logs mini eval predictions + training data samples to wandb.

    Mini eval runs as two parallel subprocesses (one per benchmark, one per GPU)
    so it doesn't interfere with training. Each subprocess loads a clean model
    from a temp checkpoint, generates predictions, scores with judge, and returns.
    """

    def __init__(
        self,
        processor,
        sample_log_steps: int = 500,
        num_samples_per_bench: int = 3,
        num_workers: int = 4,
    ):
        self.processor = processor
        self.sample_log_steps = sample_log_steps
        self.num_samples_per_bench = num_samples_per_bench
        self.num_workers = num_workers
        self._eval_thread = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Log training data samples at the start."""
        if args.local_rank not in (-1, 0):
            return
        if wandb.run is None:
            return
        self._log_training_data_samples(kwargs.get("train_dataloader"))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.sample_log_steps != 0:
            return
        if args.local_rank not in (-1, 0):
            return
        if wandb.run is None:
            return

        # Mini eval is now tied to checkpoint saves (on_save), not steps.
        # This hook is kept for future use if needed.
        pass

    def on_train_end(self, args, state, control, **kwargs):
        """Wait for any running mini eval to finish before training exits."""
        if self._eval_thread is not None and self._eval_thread.is_alive():
            logger.info("Waiting for mini eval to finish...")
            self._eval_thread.join()

    def on_save(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        if wandb.run is None:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        wandb.log({"checkpoint/step": state.global_step}, step=state.global_step)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

        # Launch mini eval on the just-saved checkpoint
        if self._eval_thread is not None and self._eval_thread.is_alive():
            logger.info("Previous mini eval still running, skipping this checkpoint")
            return

        self._eval_thread = threading.Thread(
            target=self._run_parallel_eval,
            args=(checkpoint_dir, state.global_step),
            daemon=True,
        )
        self._eval_thread.start()
        logger.info(f"Mini eval launched on {checkpoint_dir}")


    def _run_parallel_eval(self, checkpoint_dir, global_step):
        """Run dream1k on GPU 0 and carebench on GPU 1 in parallel."""
        benchmarks = [
            ("dream1k", "0"),
            ("carebench", "1"),
        ]

        threads = []
        results = {}

        def _run_one(bench_name, gpu_id):
            output_md = os.path.join(checkpoint_dir, f"mini_eval_{bench_name}.md")
            cmd = [
                "python", "run_eval.py",
                "--model", checkpoint_dir,
                "--samples", str(self.num_samples_per_bench),
                "--benchmarks", bench_name,
                "--batch-size", "1",
                "--verbose",
                "--output", output_md,
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            logger.info(f"[GPU {gpu_id}] {bench_name}: {' '.join(cmd)}")
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=None, env=env,
                )
                if proc.returncode != 0:
                    logger.warning(f"[GPU {gpu_id}] {bench_name} failed:\n{proc.stderr[-300:]}")
                    return

                # Parse scores from stdout
                for line in proc.stdout.split("\n"):
                    line = line.strip()
                    if f"{bench_name}:" in line:
                        try:
                            score_str = line.split(":", 1)[1].strip()
                            results[bench_name] = eval(score_str)
                        except Exception:
                            pass

            except Exception as e:
                logger.warning(f"[GPU {gpu_id}] {bench_name} error: {e}")

        # Launch both in parallel
        for bench_name, gpu_id in benchmarks:
            t = threading.Thread(target=_run_one, args=(bench_name, gpu_id))
            threads.append(t)
            t.start()

        # Wait for both
        for t in threads:
            t.join()

        # Log scores to wandb
        if results and wandb.run is not None:
            flat = {}
            for bench, scores in results.items():
                for metric, value in scores.items():
                    flat[f"mini_eval/{bench}/{metric}"] = value
            wandb.log(flat, step=global_step)
            logger.info(f"Mini eval step {global_step}: {results}")

        # Log predictions table from both markdown files
        self._log_eval_table(checkpoint_dir, global_step, results)

        # Clean up eval markdown files (keep the checkpoint itself)
        for bench in ["dream1k", "carebench"]:
            md = os.path.join(checkpoint_dir, f"mini_eval_{bench}.md")
            if os.path.exists(md):
                os.remove(md)

    def _log_eval_table(self, checkpoint_dir, global_step, scores):
        """Parse mini eval markdown files and log as wandb table with scores."""
        try:
            table_data = []

            for bench_name in ["dream1k", "carebench"]:
                md_path = os.path.join(checkpoint_dir, f"mini_eval_{bench_name}.md")
                if not os.path.exists(md_path):
                    continue

                with open(md_path) as f:
                    content = f.read()

                # Parse scores line
                bench_score = ""
                if bench_name in scores:
                    score_parts = [f"{k}={v}" for k, v in scores[bench_name].items()]
                    bench_score = ", ".join(score_parts)

                # Parse individual samples
                current_file = ""
                current_prompt = ""
                current_output = ""
                current_gt = ""
                section = None

                for line in content.split("\n"):
                    if line.startswith("**File:**"):
                        # Save previous sample
                        if current_file and current_output:
                            table_data.append([
                                global_step, bench_name, current_file,
                                current_prompt.strip(), current_gt.strip(),
                                current_output.strip(), bench_score,
                            ])
                        current_file = line.split("`")[1] if "`" in line else ""
                        current_prompt = ""
                        current_output = ""
                        current_gt = ""
                        section = None
                    elif line.startswith("**Prompt:**"):
                        section = "prompt"
                    elif line.startswith("**Model Output:**"):
                        section = "output"
                    elif line.startswith("**GT "):
                        section = "gt"
                    elif line.startswith("**Scores:**"):
                        section = None
                    elif line.startswith("---"):
                        section = None
                    elif line.startswith("```"):
                        continue
                    elif section == "prompt":
                        current_prompt += line + "\n"
                    elif section == "output":
                        current_output += line + "\n"
                    elif section == "gt":
                        current_gt += line + "\n"

                # Last sample
                if current_file and current_output:
                    table_data.append([
                        global_step, bench_name, current_file,
                        current_prompt.strip(), current_gt.strip(),
                        current_output.strip(), bench_score,
                    ])

            if table_data and wandb.run is not None:
                table = wandb.Table(
                    columns=["step", "benchmark", "file", "prompt", "ground_truth", "prediction", "scores"],
                    data=table_data,
                )
                wandb.log({"mini_eval/predictions": table}, step=global_step)
                logger.info(f"Logged {len(table_data)} predictions to wandb table at step {global_step}")

        except Exception as e:
            logger.warning(f"Failed to log eval table: {e}")
            import traceback
            traceback.print_exc()

    def _log_training_data_samples(self, train_dataloader):
        """Log a batch of training data to wandb with clean formatting."""
        try:
            if train_dataloader is None:
                return

            batch_iter = iter(train_dataloader)
            batch = next(batch_iter)

            input_ids = batch.get("input_ids", None)
            if input_ids is None:
                return

            import re
            table_data = []
            num_to_log = min(10, input_ids.shape[0])
            tokenizer = self.processor.tokenizer

            for i in range(num_to_log):
                raw = tokenizer.decode(input_ids[i], skip_special_tokens=False)
                samples_in_seq = raw.split("<|im_start|>user")

                for j, sample_text in enumerate(samples_in_seq):
                    if not sample_text.strip():
                        continue

                    clean = sample_text
                    n_video_pads = clean.count("<|video_pad|>")
                    n_timestamps = len(re.findall(r"<[\d.]+ seconds>", clean))

                    clean = re.sub(
                        r"(<[\d.]+ seconds>)?<\|vision_start\|>(<\|video_pad\|>)+<\|vision_end\|>",
                        "", clean,
                    )

                    prompt = ""
                    response = ""
                    if "<|im_end|>" in clean:
                        parts = clean.split("<|im_end|>")
                        prompt = parts[0].strip()
                        for part in parts[1:]:
                            if "assistant" in part:
                                resp_text = part.split("assistant", 1)[-1]
                                response = resp_text.strip()
                                break

                    response = response.replace("<|im_end|>", "").strip()
                    if not prompt and not response:
                        continue

                    video_info = f"{n_timestamps} frames ({n_video_pads} visual tokens)"
                    table_data.append([f"{i}.{j}", video_info, prompt, response, input_ids[i].shape[0]])

            table = wandb.Table(
                columns=["sample", "video_info", "prompt", "ground_truth", "packed_seq_tokens"],
                data=table_data,
            )
            wandb.log({"training_data/samples": table}, step=0)
            logger.info(f"Logged {len(table_data)} training data samples to wandb")
        except Exception as e:
            logger.warning(f"Failed to log training data samples: {e}")
            import traceback
            traceback.print_exc()
