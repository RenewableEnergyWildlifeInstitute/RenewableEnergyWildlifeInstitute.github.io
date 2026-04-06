#!/usr/bin/env python3
"""
Benchmark inference throughput and cost metrics for generate_prediction settings.

This script is designed for cross-instance benchmarking (for example, different
HPC/AWS configurations). It keeps the generation setup aligned with
generate_prediction.py and writes repeat-level results to CSV so you can append
them to your modeling table.

Example:
    python .github/scripts/benchmark_generate_prediction.py \
      --instance-name g4dn.xlarge \
      --instance-hourly-usd 0.526 \
      --doc-key AB12CD34 \
      --batch-sizes 1,4 \
      --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import generate_prediction as gp


def parse_batch_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        n = int(value)
        if n <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        sizes.append(n)
    if not sizes:
        raise ValueError("At least one batch size is required.")
    return sorted(set(sizes))


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def run_batch_once(pipe, prompts: list[str], batch_size: int) -> tuple[list[str], float]:
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(max_new_tokens=gp.MAX_NEW_TOKENS, do_sample=False)
    t0 = time.perf_counter()
    result = pipe(
        prompts,
        generation_config=gen_cfg,
        return_full_text=False,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - t0

    # HF pipeline returns list[list[dict]] when given a list of prompts,
    # or list[dict] when given a single string. Normalise to list[str].
    generated = [
        (row[0] if isinstance(row, list) else row)["generated_text"]
        for row in result
    ]
    return generated, elapsed


def ensure_csv_header(path: Path, fieldnames: list[str]):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark generate_prediction inference throughput and cost."
    )
    parser.add_argument("--instance-name", required=True, help="Label for this hardware config")
    parser.add_argument(
        "--instance-hourly-usd",
        type=float,
        required=True,
        help="Hourly instance cost in USD (same region/pricing mode across runs)",
    )
    parser.add_argument(
        "--doc-key",
        default=None,
        help="Specific parent_key to benchmark; omit to sample one randomly",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face model ID override (also sets HF_MODEL_ID for this run)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4",
        help="Comma-separated batch sizes to test (for example: 1,4)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of measured repeats per batch size")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per batch size before measured repeats",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when --doc-key is omitted",
    )
    parser.add_argument(
        "--output-csv",
        default=str(gp.REPO_ROOT / "benchmark" / "benchmark_runs.csv"),
        help="Path to append repeat-level benchmark rows",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(gp.REPO_ROOT / "benchmark" / "benchmark_summary.csv"),
        help="Path to write per-run summary grouped by batch size",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")
    if args.model_id:
        os.environ["HF_MODEL_ID"] = args.model_id

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    random.seed(args.seed)

    print("Loading benchmark document...")
    doc = gp.load_document(args.doc_key)
    print(f"Using parent_key: {doc['parent_key']}")

    # Keep inference settings aligned with generate_prediction.py.
    base_prompt = f"Text: {doc['fulltext']}\\nTags:"

    print("Loading model/tokenizer...")
    t_model = time.perf_counter()
    pipe, tokenizer = gp.load_model()
    model_load_s = time.perf_counter() - t_model
    prompt = gp.truncate_prompt(base_prompt, tokenizer, pipe)

    input_tokens_per_doc = token_len(tokenizer, prompt)
    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    device = "cuda" if getattr(pipe.model, "device", None) and "cuda" in str(pipe.model.device) else "auto"

    rows: list[dict] = []
    run_ts = datetime.now(timezone.utc).isoformat()

    print("Starting benchmark runs...")
    for batch_size in batch_sizes:
        prompts = [prompt] * batch_size

        for i in range(args.warmup_runs):
            _, warmup_elapsed = run_batch_once(pipe, prompts, batch_size)
            print(f"Warmup batch={batch_size} run={i + 1}/{args.warmup_runs} took {warmup_elapsed:.3f}s")

        for repeat_idx in range(1, args.repeats + 1):
            outputs, wall_time_s = run_batch_once(pipe, prompts, batch_size)
            output_tokens_total = sum(token_len(tokenizer, text) for text in outputs)
            input_tokens_total = input_tokens_per_doc * batch_size
            total_tokens = input_tokens_total + output_tokens_total

            if wall_time_s <= 0:
                raise RuntimeError("Measured wall_time_s must be positive.")

            tokens_per_s = total_tokens / wall_time_s
            tokens_per_hr = tokens_per_s * 3600.0
            docs_per_hr = (batch_size / wall_time_s) * 3600.0
            cost_per_1m_tokens_usd = (args.instance_hourly_usd / tokens_per_hr) * 1_000_000
            cost_per_doc_usd = args.instance_hourly_usd / docs_per_hr

            row = {
                "run_timestamp_utc": run_ts,
                "instance_name": args.instance_name,
                "instance_hourly_usd": args.instance_hourly_usd,
                "model_id": model_id,
                "device": device,
                "parent_key": doc["parent_key"],
                "batch_size": batch_size,
                "repeat_index": repeat_idx,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "max_new_tokens": gp.MAX_NEW_TOKENS,
                "input_tokens_per_doc": input_tokens_per_doc,
                "input_tokens_total": input_tokens_total,
                "output_tokens_total": output_tokens_total,
                "total_tokens": total_tokens,
                "wall_time_s": wall_time_s,
                "tokens_per_s": tokens_per_s,
                "tokens_per_hr": tokens_per_hr,
                "docs_per_hr": docs_per_hr,
                "cost_per_1m_tokens_usd": cost_per_1m_tokens_usd,
                "cost_per_doc_usd": cost_per_doc_usd,
                "model_load_s": model_load_s,
            }
            rows.append(row)
            print(
                f"batch={batch_size} repeat={repeat_idx}/{args.repeats} "
                f"time={wall_time_s:.3f}s tokens/s={tokens_per_s:.1f} $/1M={cost_per_1m_tokens_usd:.4f}"
            )

    if not rows:
        raise RuntimeError("No benchmark rows generated.")

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)

    fieldnames = list(rows[0].keys())
    ensure_csv_header(output_csv, fieldnames)

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("batch_size", as_index=False)
        .agg(
            repeats=("repeat_index", "count"),
            wall_time_s_mean=("wall_time_s", "mean"),
            wall_time_s_std=("wall_time_s", "std"),
            tokens_per_s_mean=("tokens_per_s", "mean"),
            tokens_per_s_std=("tokens_per_s", "std"),
            docs_per_hr_mean=("docs_per_hr", "mean"),
            cost_per_1m_tokens_usd_mean=("cost_per_1m_tokens_usd", "mean"),
            cost_per_doc_usd_mean=("cost_per_doc_usd", "mean"),
        )
        .sort_values("batch_size")
    )
    summary.insert(0, "run_timestamp_utc", run_ts)
    summary.insert(1, "instance_name", args.instance_name)
    summary.insert(2, "parent_key", doc["parent_key"])
    summary.insert(3, "model_id", model_id)
    summary.insert(4, "max_new_tokens", gp.MAX_NEW_TOKENS)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, mode="a", index=False, header=not summary_csv.exists())

    print("\nBenchmark complete.")
    print(f"Repeat rows appended: {output_csv}")
    print(f"Summary rows appended: {summary_csv}")
    print("\nSummary for this run:")
    print(summary.to_string(index=False, justify="left"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
