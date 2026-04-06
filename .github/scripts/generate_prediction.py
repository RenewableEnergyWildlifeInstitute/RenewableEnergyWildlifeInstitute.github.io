#!/usr/bin/env python3
"""
Generate a single LLM tag prediction and append it to evaluator/data/predictions.json.

Designed to run inside a GitHub Actions workflow.  Reuses the same pipeline logic
as single_doc_pipeline_test.py (document loading, LLM inference, output parsing,
tag filtering) but writes the result back to the predictions file that powers
the evaluator web app.

Usage (from repo root):
    python .github/scripts/generate_prediction.py              # random document
    python .github/scripts/generate_prediction.py --doc-key XYZ  # specific document
"""

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 0.  PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .github/scripts -> repo root
PREDICTIONS_PATH = REPO_ROOT / "evaluator" / "data" / "predictions.json"
PARQUET_PATH = (
    REPO_ROOT / "UVA-CAPSTONE-SP-26" / "data" / "full_text"
    / "fulltext_processing_dataset.parquet"
)
TAG_DICT_PATH = (
    REPO_ROOT / "UVA-CAPSTONE-SP-26" / "REWI-Materials"
    / "metadata_Dictionary_v2.xlsx"
)

MAX_NEW_TOKENS = 120

# ---------------------------------------------------------------------------
# 1.  DOCUMENT LOADING
# ---------------------------------------------------------------------------

def parse_parent_tags(raw_tags) -> list[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        return [str(t).strip() for t in raw_tags if str(t).strip()]
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(";") if t.strip()]
    return []


def pick_fulltext(row: pd.Series) -> str:
    for col in ("zotero_fulltext", "pymupdf4llm_md", "docling_text"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_document(doc_key: str | None = None) -> dict:
    """Load a document from the parquet dataset (random or by key)."""
    if not PARQUET_PATH.exists():
        sys.exit(f"ERROR: Parquet not found: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)

    if doc_key:
        matches = df[df["parent_key"] == doc_key]
        if matches.empty:
            sys.exit(f"ERROR: No document with parent_key={doc_key}")
        row = matches.iloc[0]
    else:
        text_series = (
            df["zotero_fulltext"].fillna("").astype(str).str.strip()
            + df["pymupdf4llm_md"].fillna("").astype(str).str.strip()
            + df["docling_text"].fillna("").astype(str).str.strip()
        )
        eligible = df.index[text_series != ""].tolist()
        if not eligible:
            sys.exit("ERROR: No rows with usable full text in parquet.")
        row = df.loc[random.choice(eligible)]

    fulltext = pick_fulltext(row)
    if not fulltext:
        sys.exit("ERROR: Selected row has no usable full text.")

    return {
        "parent_key": str(row.get("parent_key", "")),
        "title": str(row.get("parent_title", "")),
        "abstract": str(row.get("parent_abstract", "")),
        "ground_truth_tags": parse_parent_tags(row.get("parent_tags")),
        "fulltext": fulltext,
    }


# ---------------------------------------------------------------------------
# 2.  TAG DICTIONARY
# ---------------------------------------------------------------------------

def load_allowed_tags() -> tuple[set[str], dict[str, str]]:
    if not TAG_DICT_PATH.exists():
        sys.exit(f"ERROR: Tag dictionary not found: {TAG_DICT_PATH}")
    tag_dict = pd.read_excel(TAG_DICT_PATH)
    allowed = set(tag_dict["Tag"].dropna().str.strip())
    lookup = {t.lower(): t for t in allowed}
    print(f"Loaded {len(allowed)} allowed tags.")
    return allowed, lookup


# ---------------------------------------------------------------------------
# 3.  LLM INFERENCE
# ---------------------------------------------------------------------------

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use bfloat16 on all devices to halve memory usage (~2.5GB vs ~5GB for 1B model).
    # bfloat16 is natively supported on modern CPUs (AVX-512 BF16) and all CUDA GPUs.
    model_dtype = torch.bfloat16

    print(f"Loading model: {model_id} on {device} …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=model_dtype, token=hf_token
        )
    except OSError as exc:
        msg = str(exc)
        if "gated repo" in msg.lower() or "401" in msg:
            hint = (
                "ERROR: Hugging Face model access failed (gated repo).\\n"
                f"Model: {model_id}\\n"
                "You need BOTH:\\n"
                "1) Access approval for this model on huggingface.co\\n"
                "2) HF_TOKEN set in your environment with read access\\n\\n"
                "Quick alternatives:\\n"
                "- Export credentials: export HF_TOKEN=hf_xxx\\n"
                "- Or switch to a public model: export HF_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            raise SystemExit(hint) from exc
        raise
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=model_dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    print("Model loaded.")
    return pipe, tokenizer


def create_prompt(doc_text: str, tokenizer, allowed_tags: set[str]) -> str:
    messages = [
        {"role": "system", "content": "You are an expert renewable energy document tagger. The following is a document that you must tag:"},
        {"role": "system", "content": doc_text},
        {"role": "system", "content": "Respond with only tags from the following list: " + str(allowed_tags)},
        {"role": "system", "content": "Do not respond with any other text."},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def truncate_prompt(prompt: str, tokenizer, pipe) -> str:
    model_max = getattr(pipe.model.config, "max_position_embeddings", None)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    limits = [l for l in [model_max, tok_limit] if isinstance(l, int) and 0 < l < 1_000_000]
    if not limits:
        return prompt
    ctx = min(limits)
    max_input = max(256, ctx - MAX_NEW_TOKENS - 16)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_input:
        return prompt
    trimmed = ids[-max_input:]
    print(f"  Truncated prompt tokens {len(ids)} -> {len(trimmed)} (context {ctx})")
    return tokenizer.decode(trimmed, skip_special_tokens=False)


def run_inference(pipe, prompt: str) -> str:
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    result = pipe(prompt, generation_config=gen_cfg, return_full_text=False)
    return result[0]["generated_text"]


# ---------------------------------------------------------------------------
# 4.  OUTPUT PARSING  (mirrors single_doc_pipeline_test.py)
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", text).strip()


def _extract_tag_names_regex(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r'"tag_name"\s*:\s*"([^"]+)"', text) if m.strip()]


def extract_json_tags(text: str) -> list[str]:
    text = _strip_markdown_fences(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _extract_tag_names_regex(text)
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return _extract_tag_names_regex(text)
    if isinstance(data, dict):
        for key in ("identified_tags", "tags"):
            if key in data and isinstance(data[key], list):
                tags = []
                for e in data[key]:
                    if isinstance(e, dict):
                        t = e.get("tag_name") or e.get("tag") or e.get("name", "")
                        if t:
                            tags.append(str(t).strip())
                    elif isinstance(e, str):
                        tags.append(e.strip())
                return tags
    return _extract_tag_names_regex(text)


def extract_python_list_tags(text: str) -> list[str]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group())
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (ValueError, SyntaxError):
        pass
    return []


def extract_bullet_tags(text: str) -> list[str]:
    pattern = r"^\s*(?:[\*\-\u2022]|\d+[\.)\]])\s+(.+)$"
    return [m.strip().rstrip(",").strip() for m in re.findall(pattern, text, re.MULTILINE) if m.strip()]


def extract_comma_tags(text: str) -> list[str]:
    tags = []
    for c in text.split(","):
        clean = re.sub(r"^[\d\.\s]+", "", c.strip()).strip()
        if clean:
            tags.append(clean)
    return tags


def parse_llm_output(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = raw.strip()
    if "{" in text:
        tags = extract_json_tags(text)
        if tags:
            return tags
    if "[" in text:
        tags = extract_python_list_tags(text)
        if tags:
            return tags
    bullets = extract_bullet_tags(text)
    if bullets:
        return bullets
    return extract_comma_tags(text)


def filter_to_allowed(candidates: list[str], lookup: dict[str, str]) -> list[str]:
    valid, seen = [], set()
    for c in candidates:
        k = c.strip().lower()
        if k in lookup and k not in seen:
            valid.append(lookup[k])
            seen.add(k)
    return valid


# ---------------------------------------------------------------------------
# 5.  PREDICTIONS FILE I/O
# ---------------------------------------------------------------------------

def load_predictions() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_predictions(predictions: list[dict]):
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(predictions)} predictions to {PREDICTIONS_PATH}")


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a single LLM tag prediction.")
    parser.add_argument("--doc-key", default=None, help="Specific parent_key (omit for random)")
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Load document
    doc = load_document(args.doc_key)
    print(f"Document: {doc['parent_key']} — {doc['title'][:80]}")

    # Load tags
    allowed_tags, tag_lookup = load_allowed_tags()

    # Load model & run inference
    t_model = time.perf_counter()
    pipe, tokenizer = load_model()
    print(f"Model load: {time.perf_counter() - t_model:.1f}s")

    prompt = f"Text: {doc['fulltext']}\nTags:" # No need for chat_template when using non-instruction tuned model
#    prompt = create_prompt(doc["fulltext"], tokenizer, allowed_tags)
    prompt = truncate_prompt(prompt, tokenizer, pipe)

    print("Running inference …")
    t_infer = time.perf_counter()
    raw_response = run_inference(pipe, prompt)
    print(f"Inference: {time.perf_counter() - t_infer:.1f}s")
    print(f"Raw response: {raw_response[:500]}")

    # Parse & filter
    candidates = parse_llm_output(raw_response)
    predicted_tags = filter_to_allowed(candidates, tag_lookup)

    gt_filtered = [t for t in doc["ground_truth_tags"] if t.strip().lower() in tag_lookup]

    print(f"Ground truth:     {gt_filtered}")
    print(f"LLM tags (raw):   {candidates}")
    print(f"LLM tags (valid): {predicted_tags}")

    # Build prediction entry (matches evaluator/data/predictions.json schema)
    entry = {
        "key": doc["parent_key"],
        "title": doc["title"],
        "abstract": doc["abstract"],
        "predicted_tags": predicted_tags,
        "full_text": "",
        "ground_truth_tags": gt_filtered,
    }

    # Append to predictions.json (skip if key already exists)
    predictions = load_predictions()
    existing_keys = {p["key"] for p in predictions}
    if entry["key"] in existing_keys:
        print(f"Key {entry['key']} already in predictions — updating in place.")
        predictions = [p if p["key"] != entry["key"] else entry for p in predictions]
    else:
        predictions.append(entry)
        print(f"Appended new prediction for key {entry['key']}.")

    save_predictions(predictions)

    print(f"\nTotal time: {time.perf_counter() - t_start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
