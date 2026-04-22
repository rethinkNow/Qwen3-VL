"""Wandb training callback: logs config + sample data, and saves processor on every checkpoint."""
from __future__ import annotations

import json
import logging
import os
import re

import wandb
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class WandbTrainingCallback(TrainerCallback):
    """Three responsibilities:

    1. on_train_begin: log full training config + a batch of training samples to wandb tables.
    2. on_save: save the processor alongside the model so external eval (e.g. sequoia
       checkpoint watcher) can load each checkpoint without missing preprocessor_config.json.
    3. on_save: log the checkpoint step to wandb so the wandb timeline marks it.
    """

    def __init__(self, processor, data_args=None):
        self.processor = processor
        self.data_args = data_args

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return
        if wandb.run is None:
            return
        self._log_training_config(args)
        self._log_training_data_samples(kwargs.get("train_dataloader"))

    def on_save(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            self.processor.save_pretrained(checkpoint_dir)
        except Exception as e:
            logger.warning(f"Failed to save processor to {checkpoint_dir}: {e}")

        if wandb.run is not None:
            wandb.log({"checkpoint/step": state.global_step}, step=state.global_step)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def _log_training_config(self, args):
        try:
            config = {
                "model": getattr(args, "model_name_or_path", ""),
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_epsilon": args.adam_epsilon,
                "max_grad_norm": args.max_grad_norm,
                "optim": args.optim,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "gradient_checkpointing": args.gradient_checkpointing,
                "model_max_length": getattr(args, "model_max_length", None),
                "video_max_frames": getattr(self.data_args, "video_max_frames", None) if self.data_args else None,
                "video_min_frames": getattr(self.data_args, "video_min_frames", None) if self.data_args else None,
                "video_fps": getattr(self.data_args, "video_fps", None) if self.data_args else None,
                "max_pixels": getattr(self.data_args, "max_pixels", None) if self.data_args else None,
                "video_max_pixels": getattr(self.data_args, "video_max_pixels", None) if self.data_args else None,
                "save_strategy": str(args.save_strategy),
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "output_dir": args.output_dir,
                "run_name": args.run_name,
            }

            if args.deepspeed:
                ds_path = args.deepspeed
                if os.path.exists(ds_path):
                    with open(ds_path) as f:
                        config["deepspeed"] = json.load(f)
                else:
                    config["deepspeed"] = ds_path

            wandb.config.update({"training": config}, allow_val_change=True)

            descriptions = {
                "model": "Base model name or path",
                "learning_rate": "Peak learning rate",
                "lr_scheduler_type": "LR schedule (cosine, linear, etc)",
                "warmup_ratio": "Fraction of steps for LR warmup",
                "warmup_steps": "Number of warmup steps (overrides ratio if > 0)",
                "weight_decay": "L2 regularization weight",
                "adam_beta1": "Adam optimizer beta1",
                "adam_beta2": "Adam optimizer beta2",
                "adam_epsilon": "Adam optimizer epsilon",
                "max_grad_norm": "Gradient clipping max norm",
                "optim": "Optimizer type",
                "per_device_train_batch_size": "Batch size per GPU",
                "gradient_accumulation_steps": "Gradient accumulation steps",
                "num_train_epochs": "Number of training epochs",
                "max_steps": "Max training steps (-1 = use epochs)",
                "bf16": "BFloat16 training",
                "fp16": "Float16 training",
                "gradient_checkpointing": "Recompute activations to save memory",
                "model_max_length": "Max sequence length for packing",
                "video_max_frames": "Max frames per video",
                "video_min_frames": "Min frames per video",
                "video_fps": "Video sampling rate (frames per second)",
                "max_pixels": "Max pixels per image frame",
                "video_max_pixels": "Per-frame H*W pixel cap for video (e.g. 200704 = 448x448)",
                "save_strategy": "When to save checkpoints",
                "save_steps": "Save checkpoint every N steps",
                "save_total_limit": "Max checkpoints to keep (None = all)",
                "output_dir": "Checkpoint output directory",
                "run_name": "Wandb run name",
            }
            table_data = []
            for key, value in config.items():
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                desc = descriptions.get(key, "")
                table_data.append([key, str(value), desc])
            table = wandb.Table(columns=["parameter", "value", "description"], data=table_data)
            wandb.log({"training_config/table": table}, step=0)
            logger.info("Training config logged to wandb")
        except Exception as e:
            logger.warning(f"Failed to log training config: {e}")

    def _log_training_data_samples(self, train_dataloader):
        try:
            if train_dataloader is None:
                return

            raw_dataset = getattr(train_dataloader, "dataset", None)
            raw_samples = []
            if raw_dataset and hasattr(raw_dataset, "list_data_dict"):
                raw_samples = raw_dataset.list_data_dict[:20]

            batch_iter = iter(train_dataloader)
            batch = next(batch_iter)
            input_ids = batch.get("input_ids", None)
            if input_ids is None:
                return

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
                    n_raw_frames = n_timestamps * 2
                    tokens_per_group = n_video_pads // n_timestamps if n_timestamps > 0 else 0

                    clean = re.sub(
                        r"(<[\d.]+ seconds>)?<\|vision_start\|>(<\|video_pad\|>)+<\|vision_end\|>",
                        "", clean,
                    )

                    prompt, response = "", ""
                    if "<|im_end|>" in clean:
                        parts = clean.split("<|im_end|>")
                        prompt = parts[0].strip()
                        for part in parts[1:]:
                            if "assistant" in part:
                                response = part.split("assistant", 1)[-1].strip()
                                break

                    response = response.replace("<|im_end|>", "").strip()
                    if not prompt and not response:
                        continue

                    video_file = ""
                    sample_idx = i * 4 + j
                    if sample_idx < len(raw_samples):
                        s = raw_samples[sample_idx]
                        video_file = s.get("video", s.get("image", ""))
                        if isinstance(video_file, str):
                            video_file = os.path.basename(video_file)

                    video_info = f"{n_raw_frames} frames, {n_timestamps} groups, {tokens_per_group} tokens/group, {n_video_pads} visual tokens"
                    table_data.append([f"{i}.{j}", video_file, video_info, prompt, response, input_ids[i].shape[0]])

            table = wandb.Table(
                columns=["sample", "video_file", "video_info", "prompt", "ground_truth", "packed_seq_tokens"],
                data=table_data,
            )
            wandb.log({"training_data/samples": table}, step=0)
            logger.info(f"Logged {len(table_data)} training data samples to wandb")
        except Exception as e:
            logger.warning(f"Failed to log training data samples: {e}")
