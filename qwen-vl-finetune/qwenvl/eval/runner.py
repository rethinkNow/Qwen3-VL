from __future__ import annotations

import logging
import os
from typing import Any

import gc

import torch
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info
from transformers.video_processing_utils import VideoMetadata

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")


class EvalVideoDataset(Dataset):
    """Dataset for parallel video decoding + tokenization in eval.

    Each worker does the full pipeline: video decode → process_vision_info →
    apply_chat_template → processor tokenization. Returns ready-to-generate
    inputs (input_ids, pixel_values, etc). Main thread only moves to GPU + generates.
    """

    def __init__(self, samples, benchmark, processor, fps=2.0, max_frames=2048):
        self.samples = samples
        self.benchmark = benchmark
        self.processor = processor
        self.fps = fps
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_info = self.benchmark.format_prompt(sample)
        video_path = prompt_info["video"]
        prompt_text = prompt_info["prompt"]

        if not os.path.exists(video_path):
            return {"idx": idx, "sample": sample, "video_path": video_path, "success": False}

        ext = os.path.splitext(video_path)[1].lower()
        is_image = ext in IMAGE_EXTENSIONS

        try:
            if is_image:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": f"file://{video_path}"},
                    {"type": "text", "text": prompt_text},
                ]}]
            else:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": f"file://{video_path}", "fps": self.fps, "max_frames": self.max_frames},
                    {"type": "text", "text": prompt_text},
                ]}]

            # Full pipeline in worker: decode → tokenize → ready for GPU
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True,
            )

            video_tensors = []
            video_metadata_list = []
            if video_inputs:
                for vi in video_inputs:
                    if isinstance(vi, tuple):
                        video_tensors.append(vi[0])
                        meta_dict = vi[1]
                        video_metadata_list.append(VideoMetadata(
                            total_num_frames=meta_dict.get("total_num_frames", 0),
                            fps=meta_dict.get("fps"),
                            frames_indices=meta_dict.get("frames_indices"),
                            video_backend=meta_dict.get("video_backend"),
                        ))
                    else:
                        video_tensors.append(vi)

            # Tokenize in worker (no padding — single sample)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_tensors if video_tensors else None,
                padding=False,
                return_tensors="pt",
                do_sample_frames=False,
                video_metadata=video_metadata_list if video_metadata_list else None,
            )

            return {
                "idx": idx,
                "sample": sample,
                "video_path": video_path,
                "inputs": inputs,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"Worker failed to process {video_path}: {e}")
            return {"idx": idx, "sample": sample, "video_path": video_path, "success": False}


def _eval_collate_fn(batch):
    """Pass through — main thread handles padding + GPU transfer."""
    return batch


def _build_vllm_messages(video_path: str, prompt_text: str, is_image: bool) -> list[dict]:
    """Build OpenAI-style chat messages for vLLM."""
    if is_image:
        content = [
            {"type": "image_url", "image_url": {"url": f"file://{video_path}"}},
            {"type": "text", "text": prompt_text},
        ]
    else:
        content = [
            {"type": "video_url", "video_url": {"url": f"file://{video_path}"}},
            {"type": "text", "text": prompt_text},
        ]
    return [{"role": "user", "content": content}]


def run_inference_vllm(
    llm,
    sampling_params,
    samples: list[dict],
    benchmark,
    verbose: bool = False,
    processor=None,
) -> list[dict]:
    """Run inference using vLLM backend.

    NOTE: vLLM has a known timestamp bug for Qwen3-VL (issue #30847) where
    timestamps for long videos may be compressed. Short videos work correctly.
    Use --backend transformers for accurate temporal grounding evaluation.

    When verbose=True, processes one sample at a time for interleaved logging.
    When verbose=False, batches all samples for maximum throughput.

    Returns:
        list of {sample: dict, prediction: str}
    """
    predictions = []

    if verbose:
        for sample in samples:
            prompt_info = benchmark.format_prompt(sample)
            video_path = prompt_info["video"]
            prompt_text = prompt_info["prompt"]

            if not os.path.exists(video_path):
                logger.warning(f"File not found: {video_path}")
                print(f"\n  [SKIP] File not found: {video_path}", flush=True)
                predictions.append({"sample": sample, "prediction": ""})
                continue

            ext = os.path.splitext(video_path)[1].lower()
            is_image = ext in IMAGE_EXTENSIONS

            print(f"\n  [INPUT] File: {os.path.basename(video_path)}", flush=True)
            print(f"  [INPUT] Prompt: {prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}", flush=True)
            print(f"  [INPUT] Type: {'image' if is_image else 'video'}", flush=True)

            messages = _build_vllm_messages(video_path, prompt_text, is_image)
            outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text.strip()

            print(f"  [OUTPUT] {output_text[:300]}{'...' if len(output_text) > 300 else ''}", flush=True)
            predictions.append({"sample": sample, "prediction": output_text})
    else:
        all_messages = []
        valid_indices = []

        for i, sample in enumerate(samples):
            prompt_info = benchmark.format_prompt(sample)
            video_path = prompt_info["video"]
            prompt_text = prompt_info["prompt"]

            if not os.path.exists(video_path):
                logger.warning(f"File not found: {video_path}")
                continue

            ext = os.path.splitext(video_path)[1].lower()
            is_image = ext in IMAGE_EXTENSIONS

            messages = _build_vllm_messages(video_path, prompt_text, is_image)
            all_messages.append(messages)
            valid_indices.append(i)

        if all_messages:
            logger.info(f"vLLM batch generating {len(all_messages)} samples...")
            outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)
        else:
            outputs = []

        output_idx = 0
        for i, sample in enumerate(samples):
            if i in valid_indices:
                text = outputs[output_idx].outputs[0].text.strip()
                output_idx += 1
            else:
                text = ""
            predictions.append({"sample": sample, "prediction": text})

    return predictions


def _prepare_sample(sample, benchmark, processor):
    """Pre-process a single sample: decode video/image, build tokens with timestamps."""
    prompt_info = benchmark.format_prompt(sample)
    video_path = prompt_info["video"]
    prompt_text = prompt_info["prompt"]

    if not os.path.exists(video_path):
        return None

    try:
        return _prepare_sample_inner(video_path, prompt_text, processor)
    except Exception as e:
        logger.warning(f"Failed to process {video_path}: {e}")
        return None


def _prepare_sample_inner(video_path, prompt_text, processor):
    """Inner processing that may raise on video decoding failures."""

    ext = os.path.splitext(video_path)[1].lower()
    is_image = ext in IMAGE_EXTENSIONS

    if is_image:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{video_path}"},
            {"type": "text", "text": prompt_text},
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": 2.0, "max_frames": 2048},
            {"type": "text", "text": prompt_text},
        ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True,
    )

    video_tensors = []
    video_metadata_list = []
    if video_inputs:
        for vi in video_inputs:
            if isinstance(vi, tuple):
                video_tensors.append(vi[0])
                meta_dict = vi[1]
                video_metadata_list.append(VideoMetadata(
                    total_num_frames=meta_dict.get("total_num_frames", 0),
                    fps=meta_dict.get("fps"),
                    frames_indices=meta_dict.get("frames_indices"),
                    video_backend=meta_dict.get("video_backend"),
                ))
            else:
                video_tensors.append(vi)

    return {
        "text": text,
        "image_inputs": image_inputs,
        "video_tensors": video_tensors,
        "video_metadata_list": video_metadata_list,
        "is_image": is_image,
        "video_path": video_path,
        "prompt_text": prompt_text,
    }


@torch.no_grad()
def run_inference(
    model,
    processor,
    samples: list[dict],
    benchmark,
    max_new_tokens: int = None,
    batch_size: int = 1,
    num_workers: int = 4,
    fps: float = 2.0,
    max_frames: int = 2048,
    verbose: bool = False,
) -> list[dict]:
    """Run model inference with parallel video decoding workers.

    Uses a DataLoader with num_workers for video decoding (CPU-bound),
    overlapping with model inference (GPU-bound).

    Args:
        model: Qwen3-VL model (on GPU, eval mode)
        processor: AutoProcessor
        samples: list of benchmark samples
        benchmark: Benchmark instance (for format_prompt)
        max_new_tokens: max generation length
        batch_size: number of samples per batch
        num_workers: number of video decoding workers
        verbose: if True, log per-sample input/output

    Returns:
        list of {sample: dict, prediction: str}
    """
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    # Use per-benchmark generation config if not explicitly overridden
    if max_new_tokens is None:
        max_new_tokens = getattr(benchmark, "max_new_tokens", 512)
    rep_penalty = getattr(benchmark, "repetition_penalty", 1.0)

    dataset = EvalVideoDataset(samples, benchmark, processor, fps=fps, max_frames=max_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_eval_collate_fn,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
    )

    for batch in dataloader:
        # Verbose: log inputs
        if verbose:
            for item in batch:
                if item["success"]:
                    print(f"\n  [INPUT #{item['idx']+1}] File: {os.path.basename(item['video_path'])}", flush=True)
                else:
                    print(f"\n  [SKIP #{item['idx']+1}] File not found or decode failed", flush=True)

        # Filter successful items (already tokenized by workers)
        valid_items = [item for item in batch if item["success"]]

        if not valid_items:
            for item in batch:
                predictions.append({"sample": item["sample"], "prediction": ""})
            continue

        # Pad pre-tokenized inputs from workers into a batch
        processor.tokenizer.padding_side = "left"
        if len(valid_items) == 1:
            # Single sample — no padding needed
            inputs = valid_items[0]["inputs"]
        else:
            # Pad multiple pre-tokenized samples to same length
            input_ids_list = [item["inputs"]["input_ids"].squeeze(0) for item in valid_items]
            padded = processor.tokenizer.pad(
                {"input_ids": input_ids_list},
                padding=True,
                return_tensors="pt",
            )
            # Merge pixel values from each sample
            inputs = padded
            pv_key = "pixel_values_videos" if "pixel_values_videos" in valid_items[0]["inputs"] else "pixel_values"
            if pv_key in valid_items[0]["inputs"]:
                inputs[pv_key] = torch.cat([item["inputs"][pv_key] for item in valid_items], dim=0)
            if "image_grid_thw" in valid_items[0]["inputs"]:
                inputs["image_grid_thw"] = torch.cat([item["inputs"]["image_grid_thw"] for item in valid_items], dim=0)
            if "video_grid_thw" in valid_items[0]["inputs"]:
                inputs["video_grid_thw"] = torch.cat([item["inputs"]["video_grid_thw"] for item in valid_items], dim=0)

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if verbose:
            print(f"  [BATCH] {len(valid_items)} samples, padded to {inputs['input_ids'].shape[1]} tokens", flush=True)

        # Generate
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if rep_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = rep_penalty
        generated_ids = model.generate(**inputs, **gen_kwargs)

        # Decode outputs
        input_len = inputs["input_ids"].shape[1]
        output_texts = []
        for b_idx in range(len(valid_items)):
            output_ids = generated_ids[b_idx, input_len:]
            text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
            output_texts.append(text)

        processor.tokenizer.padding_side = "right"

        # Map outputs back to samples
        valid_idx = 0
        for item in batch:
            if item["success"]:
                output_text = output_texts[valid_idx].strip()
                valid_idx += 1
                if verbose:
                    print(f"  [OUTPUT #{item['idx']+1}] {output_text[:300]}{'...' if len(output_text) > 300 else ''}", flush=True)
            else:
                output_text = ""
            predictions.append({"sample": item["sample"], "prediction": output_text})

        # Cleanup
        del inputs, generated_ids, output_texts
        gc.collect()
        torch.cuda.empty_cache()

        if len(predictions) % (batch_size * 5) == 0:
            logger.info(f"Eval inference: {len(predictions)}/{len(samples)} samples")

    return predictions
