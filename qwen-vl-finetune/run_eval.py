import sys, time, json, os, argparse, math
sys.path.insert(0, ".")

# === chotaVLM training-time overrides ===
# These MUST match training to get eval ≡ training behavior.
# Sources of truth in the training repo:
#   - qwenvl/data/data_processor.py:_patch_qwen3vl_smart_resize  (per-frame video cap)
#   - scripts/sft_qwen3_2b.sh: --video_max_pixels, --model_max_length
PER_FRAME_MAX_PIXELS = 200704   # 448 * 448; per-frame H*W cap on top of Qwen3-VL's total budget
MODEL_MAX_LENGTH = 131072       # tokenizer truncation length used during training

from transformers.models.qwen3_vl import video_processing_qwen3_vl as _vp_mod

_orig_smart_resize = _vp_mod.smart_resize

def _smart_resize_capped(num_frames, height, width, temporal_factor=2, factor=32, min_pixels=None, max_pixels=None):
    h, w = _orig_smart_resize(
        num_frames, height, width,
        temporal_factor=temporal_factor, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    if h * w > PER_FRAME_MAX_PIXELS:
        beta = math.sqrt(h * w / PER_FRAME_MAX_PIXELS)
        h = max(factor, math.floor(h / beta / factor) * factor)
        w = max(factor, math.floor(w / beta / factor) * factor)
    return h, w

_vp_mod.smart_resize = _smart_resize_capped


def _print_eval_settings(processor, model_path):
    """One-time dump of all settings that determine eval behavior. Useful for catching drift."""
    print("=" * 70, flush=True)
    print("CHOTAVLM EVAL SETTINGS", flush=True)
    print("=" * 70, flush=True)
    print(f"Model checkpoint: {model_path}", flush=True)
    print(f"--- Runtime overrides (must match training) ---", flush=True)
    print(f"  PER_FRAME_MAX_PIXELS:   {PER_FRAME_MAX_PIXELS}  (= {int(math.sqrt(PER_FRAME_MAX_PIXELS))}x{int(math.sqrt(PER_FRAME_MAX_PIXELS))})", flush=True)
    print(f"  MODEL_MAX_LENGTH:       {MODEL_MAX_LENGTH}", flush=True)
    print(f"  smart_resize patched:   {_vp_mod.smart_resize.__name__ == '_smart_resize_capped'}", flush=True)
    if processor is not None:
        vp = processor.video_processor
        ip = processor.image_processor
        tok = processor.tokenizer
        print(f"--- Video processor (from video_preprocessor_config.json) ---", flush=True)
        print(f"  fps:                   {vp.fps}", flush=True)
        print(f"  max_frames:            {vp.max_frames}", flush=True)
        print(f"  min_frames:            {vp.min_frames}", flush=True)
        print(f"  patch_size:            {vp.patch_size}", flush=True)
        print(f"  merge_size:            {vp.merge_size}", flush=True)
        print(f"  temporal_patch_size:   {vp.temporal_patch_size}", flush=True)
        print(f"  size:                  {vp.size}  (Qwen3-VL total T*H*W budget; per-frame cap added by patch above)", flush=True)
        print(f"--- Image processor (from preprocessor_config.json) ---", flush=True)
        print(f"  max_pixels:            {getattr(ip, 'max_pixels', 'n/a')}", flush=True)
        print(f"  min_pixels:            {getattr(ip, 'min_pixels', 'n/a')}", flush=True)
        print(f"  size:                  {ip.size}", flush=True)
        print(f"--- Tokenizer ---", flush=True)
        print(f"  model_max_length:      {tok.model_max_length}  (overridden to {MODEL_MAX_LENGTH} below)", flush=True)
    # quick smart_resize sanity test
    try:
        h_native, w_native = 1080, 1920
        h_out, w_out = _vp_mod.smart_resize(60, h_native, w_native, temporal_factor=2, factor=32,
                                            min_pixels=4096, max_pixels=25165824)
        print(f"--- smart_resize sanity test ---", flush=True)
        print(f"  60-frame {h_native}x{w_native} -> {h_out}x{w_out} = {h_out*w_out} pixels/frame "
              f"({'PATCH ACTIVE' if h_out*w_out <= PER_FRAME_MAX_PIXELS else 'PATCH NOT ACTIVE'})", flush=True)
    except Exception as e:
        print(f"  smart_resize test failed: {e}", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Model name or path to checkpoint")
    parser.add_argument("--samples", type=int, default=10, help="Samples per benchmark (0 = all)")
    parser.add_argument("--benchmarks", type=str, default="all", help="Comma-separated benchmark names or 'all'")
    parser.add_argument("--verbose", action="store_true", help="Print input/output for each sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for subset sampling")
    parser.add_argument("--output", type=str, default="baseline_eval_verbose.md", help="Output markdown path")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge scoring")
    parser.add_argument("--backend", type=str, default="transformers", choices=["vllm", "transformers"], help="Inference backend")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--jsonl", type=str, default="", help="Path to JSONL file for custom_jsonl benchmark")
    args = parser.parse_args()

    # Set JSONL path as env var for custom_jsonl benchmark
    if args.jsonl:
        os.environ["CUSTOM_JSONL_PATH"] = args.jsonl

    from qwenvl.eval.benchmarks import BENCHMARK_REGISTRY
    from qwenvl.eval.judge import GPT4oJudge

    import qwenvl.eval.benchmarks.dream1k
    import qwenvl.eval.benchmarks.carebench
    import qwenvl.eval.benchmarks.et_bench
    import qwenvl.eval.benchmarks.activitynet
    import qwenvl.eval.benchmarks.next_gqa
    import qwenvl.eval.benchmarks.qwen_builtin
    import qwenvl.eval.benchmarks.custom_jsonl

    ALL_BENCHMARKS = ["dream1k","carebench","et_bench_dvc","et_bench_tvg","activitynet","next_gqa","videomme","mmmu","realworldqa","mathvision"]
    # custom_jsonl is NOT in ALL_BENCHMARKS — only runs when explicitly requested with --jsonl

    if args.benchmarks == "all":
        benchmarks = ALL_BENCHMARKS
    elif "custom_jsonl" in args.benchmarks and not args.jsonl:
        print("ERROR: --jsonl is required when using custom_jsonl benchmark", flush=True)
        sys.exit(1)
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    # --- Load model ---
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from qwenvl.eval.runner import run_inference_vllm
        print(f"Loading model with vLLM: {args.model}", flush=True)
        print("WARNING: vLLM has a known timestamp bug for Qwen3-VL long videos (issue #30847).", flush=True)
        print("For accurate temporal grounding, use --backend transformers.", flush=True)
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=65536,
            enforce_eager=True,
            mm_processor_kwargs={"fps": 2, "min_frames": 4, "max_frames": 2048},
            media_io_kwargs={"video": {"num_frames": 600}},
            limit_mm_per_prompt={"video": 1, "image": 1},
            allowed_local_media_path="/",
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=512)
        model = None
        processor = None
        print("NOTE: smart_resize patch only affects transformers backend; vLLM uses its own video decoder.", flush=True)
        print(f"  PER_FRAME_MAX_PIXELS={PER_FRAME_MAX_PIXELS} (eval ≠ training behavior under vLLM)", flush=True)
    else:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from qwenvl.eval.runner import run_inference
        print(f"Loading model with transformers: {args.model}", flush=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            cache_dir=os.environ.get("HF_HOME", None),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        model.eval()
        processor_path = args.model
        if os.path.isdir(args.model) and not os.path.exists(os.path.join(args.model, "preprocessor_config.json")):
            raise FileNotFoundError(
                f"preprocessor_config.json not found in {args.model}. "
                f"Checkpoint is incomplete — ensure processor was saved with the model."
            )
        processor = AutoProcessor.from_pretrained(
            processor_path,
            cache_dir=os.environ.get("HF_HOME", None),
        )
        # Match training-time tokenizer truncation length (saved JSON has Qwen3-VL default).
        processor.tokenizer.model_max_length = MODEL_MAX_LENGTH
        _print_eval_settings(processor, args.model)
        llm = None
        sampling_params = None

    judge = None if args.no_judge else GPT4oJudge()

    mode_str = "all samples" if args.samples == 0 else f"{args.samples} samples"
    backend_str = args.backend

    # Open output file for progressive writing
    outf = open(args.output, "w")
    outf.write(f"# Qwen3-VL Evaluation: {os.path.basename(args.model)}\n\n")
    outf.write(f"{mode_str} per benchmark, backend={backend_str}, verbose={'on' if args.verbose else 'off'}\n\n")
    outf.write("---\n\n")
    outf.flush()

    all_metrics = {}
    t_total = time.time()

    for bench_name in benchmarks:
        if bench_name not in BENCHMARK_REGISTRY:
            print(f"  SKIP: {bench_name} not found in registry", flush=True)
            continue

        bench = BENCHMARK_REGISTRY[bench_name]
        if args.samples == 0:
            samples = bench.load_all()
        else:
            samples = bench.load_subset(args.samples, seed=args.seed)

        outf.write(f"\n## {bench_name} ({len(samples)} samples)\n\n")
        outf.flush()

        if not samples:
            print(f"  {bench_name}: NO SAMPLES LOADED", flush=True)
            outf.write("**NO SAMPLES LOADED**\n\n")
            outf.flush()
            all_metrics[bench_name] = {}
            continue

        print(f"Running {bench_name}...", flush=True)

        # --- Run inference with selected backend ---
        if args.backend == "vllm":
            predictions = run_inference_vllm(llm, sampling_params, samples, bench, verbose=args.verbose, processor=processor)
        else:
            predictions = run_inference(model, processor, samples, bench, batch_size=args.batch_size, verbose=args.verbose)

        for i, p in enumerate(predictions):
            s = p["sample"]
            prompt_info = bench.format_prompt(s)

            if args.verbose:
                print(f"\n  [{bench_name} #{i+1}] File: {os.path.basename(s.get('video', ''))}", flush=True)
                print(f"  [PROMPT] {prompt_info['prompt'][:300]}", flush=True)
                print(f"  [OUTPUT] {p['prediction'][:400]}", flush=True)
                if 'events' in s:
                    print(f"  [GT] Events: {s['events'][:3]}", flush=True)
                if 'gt_span' in s:
                    print(f"  [GT] Span: {s['gt_span']}", flush=True)
                if 'reference' in s:
                    print(f"  [GT] Ref: {s['reference'][:200]}", flush=True)
                if 'answer' in s:
                    print(f"  [GT] Answer: {s['answer']}", flush=True)

            outf.write(f"### Sample {i+1}\n\n")
            outf.write(f"**File:** `{os.path.basename(s.get('video', ''))}`\n\n")
            outf.write(f"**Prompt:**\n```\n{prompt_info['prompt'][:500]}\n```\n\n")
            outf.write(f"**Model Output:**\n```\n{p['prediction'][:800]}\n```\n\n")

            if "reference" in s:
                outf.write(f"**GT Reference:**\n```\n{s['reference'][:500]}\n```\n\n")
            if "events" in s:
                events_str = "\n".join(f"  {e}" for e in s["events"][:5])
                outf.write(f"**GT Events:**\n```\n{events_str}\n```\n\n")
            if "gt_span" in s:
                outf.write(f"**GT Span:** `{s['gt_span']}`\n\n")
            if "answer" in s:
                outf.write(f"**GT Answer:** `{s['answer']}`\n\n")
            outf.write("---\n\n")
            outf.flush()

        j = judge if (bench.requires_judge and judge is not None) else None
        scores = bench.score(predictions, judge=j)
        all_metrics[bench_name] = scores

        score_parts = [f"{k}={v}" for k, v in scores.items()]
        outf.write(f"**Scores:** {', '.join(score_parts)}\n\n")
        outf.write("---\n\n")
        outf.flush()
        print(f"  {bench_name}: {scores}", flush=True)

    elapsed = time.time() - t_total

    outf.write(f"\n## Summary ({int(elapsed)}s total)\n\n")
    outf.write("| Benchmark | Metrics |\n")
    outf.write("|---|---|\n")
    for bench_name, scores in all_metrics.items():
        score_parts = [f"{k}={v}" for k, v in scores.items()]
        outf.write(f"| {bench_name} | {', '.join(score_parts)} |\n")
    outf.close()

    print(f"\nWritten to {args.output}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
