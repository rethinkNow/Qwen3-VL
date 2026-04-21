import sys, time, json, os, argparse
sys.path.insert(0, ".")


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
