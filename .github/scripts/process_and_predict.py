#!/usr/bin/env python3
"""
Unified Zotero Processing + LLM Tag Prediction Pipeline
========================================================

Given a list of parent keys, this pipeline:
        1. Checks per-parent full_texts files for cached full text
  2. For uncached keys: fetches from Zotero API, extracts PDF text, cleans it
  3. Sends each document to the LLM for tag prediction
    4. Saves results to predictions.json with tag-level prediction history

Designed to run inside a GitHub Actions workflow.

Usage (from repo root):
    python .github/scripts/process_and_predict.py --parent-keys KEY1 KEY2 KEY3
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_and_predict")

# Keep pipeline logs readable by suppressing verbose dependency request logs.
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = REPO_ROOT / "evaluator" / "data" / "predictions.json"
FULL_TEXTS_DIR = REPO_ROOT / "evaluator" / "data" / "full_texts"
SNAPSHOT_PATH = REPO_ROOT / "evaluator" / "data" / "zotero_snapshot.json"
TAG_DICT_PATH = REPO_ROOT / "evaluator" / "data" / "tag_dictionary.json"

MAX_NEW_TOKENS = 60
FULLTEXT_MIN_LEN = 20
MAX_PAGES = 15
MPS_MAX_INPUT_TOKENS = 4096
DEFAULT_MODEL_ID = "jme-datasci/rewi-tagger"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1
PROMPT_SUFFIX = "\nTags: "

# ---------------------------------------------------------------------------
# 1. TAG DICTIONARY
# ---------------------------------------------------------------------------

def load_allowed_tags() -> tuple[set, dict]:
    """Load allowed tags from the JSON tag dictionary."""
    if not TAG_DICT_PATH.exists():
        sys.exit(f"ERROR: Tag dictionary not found: {TAG_DICT_PATH}")
    with open(TAG_DICT_PATH, "r", encoding="utf-8") as f:
        tag_dict = json.load(f)
    allowed = set()
    for tags in tag_dict.values():
        allowed.update(tags)
    lookup = {t.lower(): t for t in allowed}
    log.info(f"Loaded {len(allowed)} allowed tags from {TAG_DICT_PATH.name}")
    return allowed, lookup

# ---------------------------------------------------------------------------
# 2. PROCESSED DATASET (CACHE) I/O
# ---------------------------------------------------------------------------

def load_records_for_parent(parent_key: str) -> list[dict]:
    """Load cached full-text records for a single parent key."""
    path = FULL_TEXTS_DIR / f"{parent_key}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_records_for_parent(parent_key: str, records: list[dict]):
    """Write full-text records for a single parent key to its own JSON file."""
    FULL_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    path = FULL_TEXTS_DIR / f"{parent_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(records)} record(s) to {path.name}")

# ---------------------------------------------------------------------------
# 3. ZOTERO SNAPSHOT HELPERS
# ---------------------------------------------------------------------------

def load_snapshot() -> dict:
    """Load the Zotero snapshot and build a lookup by parent key."""
    if not SNAPSHOT_PATH.exists():
        sys.exit(f"ERROR: Zotero snapshot not found: {SNAPSHOT_PATH}")
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    return snapshot


def build_parent_lookup(snapshot: dict) -> dict:
    """Build {parent_key: item} from the snapshot items list."""
    lookup = {}
    for item in snapshot.get("items", []):
        d = item.get("data", {})
        if "parentItem" not in d and d.get("itemType") not in ("attachment", "note"):
            lookup[item["key"]] = item
    return lookup

# ---------------------------------------------------------------------------
# 4. TEXT CLEANING (ported from data_processing_pipeline.py)
# ---------------------------------------------------------------------------

_RE_PICTURE_TEXT = re.compile(
    r"\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*", re.DOTALL
)
_RE_IMAGE_PLACEHOLDER = re.compile(r"\*\*==>.*?omitted <==\*\*", re.DOTALL)
_RE_BR_TAGS = re.compile(r"<br\s*/?>")
_RE_UNICODE_GARBAGE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F]+")
_RE_DOT_LEADERS = re.compile(r"^.*\.{10,}.*$", re.MULTILINE)
_RE_HORIZ_RULES = re.compile(r"^-{5,}$", re.MULTILINE)
_RE_EMPTY_BULLETS = re.compile(r"^\s*-\s*$", re.MULTILINE)
_RE_BLANK_PAGE = re.compile(
    r"^\s*this\s+page\s+(?:is\s+|has\s+been\s+)?(?:intentionally|intentio\s*nally|purposely|deliberately)\s+left\s+blank\.?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_TRAILING_PAGE_NUM = re.compile(r"[\s,\-–—]*(?:p\.?\s*)?\d{1,4}\s*$", re.IGNORECASE)
_RE_LEADING_PAGE_NUM = re.compile(r"^\s*(?:page\s*)?\d{1,4}[\s,\-–—]*", re.IGNORECASE)
_RE_COLLAPSE_SPACE = re.compile(r"\s+")


def _normalize_for_header_comparison(line):
    s = line.strip()
    s = _RE_TRAILING_PAGE_NUM.sub("", s)
    s = _RE_LEADING_PAGE_NUM.sub("", s)
    s = _RE_COLLAPSE_SPACE.sub(" ", s).strip().lower()
    return s


def _remove_repeated_headers_footers(text, min_occurrences=3):
    lines = text.split("\n")
    normalized = []
    for line in lines:
        stripped = line.strip()
        norm = _normalize_for_header_comparison(stripped) if len(stripped) >= 5 else ""
        normalized.append(norm)

    norm_counts = defaultdict(int)
    for norm in normalized:
        if norm:
            norm_counts[norm] += 1

    repeated = set()
    for norm, count in norm_counts.items():
        threshold = 5 if len(norm) < 10 else min_occurrences
        if count >= threshold:
            repeated.add(norm)

    if not repeated:
        return text

    return "\n".join(
        line for line, norm in zip(lines, normalized) if norm not in repeated
    )


def clean_extracted_text(text: str) -> str:
    """Full cleaning pipeline — order matters."""
    if not text:
        return ""
    for fn in (
        lambda t: _RE_PICTURE_TEXT.sub("", t),
        lambda t: _RE_IMAGE_PLACEHOLDER.sub("", t),
        lambda t: _RE_BR_TAGS.sub("", t),
        lambda t: unicodedata.normalize("NFKC", t),
        lambda t: _RE_UNICODE_GARBAGE.sub("", t.replace("\ufffd", "")),
        lambda t: _RE_DOT_LEADERS.sub("", t),
        lambda t: _RE_HORIZ_RULES.sub("", t),
        lambda t: _RE_EMPTY_BULLETS.sub("", t),
        lambda t: _RE_BLANK_PAGE.sub("", t),
        _remove_repeated_headers_footers,
        lambda t: "\n".join(line.rstrip() for line in _RE_MULTI_NEWLINE.sub("\n\n", t).splitlines()),
    ):
        text = fn(text)
    return text.strip()

# ---------------------------------------------------------------------------
# 5. PDF TEXT EXTRACTION (via Zotero API)
# ---------------------------------------------------------------------------

def extract_pdf_text(child_key: str, zot_client) -> tuple[str, str]:
    """Extract text from a PDF child attachment via Zotero API.

    Returns (raw_text, method).
    """
    # Try Zotero fulltext index first
    try:
        ft = zot_client.fulltext_item(child_key)
        raw = (ft or {}).get("content", "").strip()
        if raw and len(raw) > FULLTEXT_MIN_LEN:
            return raw, "zotero_fulltext"
    except Exception as e:
        log.debug(f"  Fulltext index miss for {child_key}: {e}")

    # Fall back to downloading PDF and extracting locally
    try:
        import pymupdf
        import pymupdf4llm

        pymupdf.TOOLS.mupdf_display_errors(False)

        pdf_bytes = zot_client.file(child_key)
        if not pdf_bytes:
            return "", "download_empty"

        doc = None
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            pages_extracted = min(MAX_PAGES, total_pages) if MAX_PAGES else total_pages
            pages = list(range(pages_extracted)) if MAX_PAGES else None
            md_text = pymupdf4llm.to_markdown(
                doc, pages=pages, page_chunks=False, use_ocr=False,
                footer=False, header=False, ignore_code=True,
                ignore_graphics=True, ignore_images=True,
            )
            raw = md_text.strip() if md_text else ""
            if raw:
                return raw, "pymupdf"
        except Exception as e:
            log.warning(f"  pymupdf failed for {child_key}: {e}")
            return "", "pymupdf_failed"
        finally:
            if doc is not None:
                doc.close()
    except ImportError:
        log.warning("  pymupdf/pymupdf4llm not installed — skipping local extraction")
    except Exception as e:
        log.warning(f"  File download failed for {child_key}: {e}")
        return "", "download_failed"

    return "", "no_text_extracted"


def process_parent_key(parent_key: str, zot_client, parent_item: dict) -> list[dict]:
    """Process a single parent key: fetch children, extract text, return records."""
    VALID_LINK_MODES = {"imported_file"}
    records = []

    kids = zot_client.children(parent_key)
    valid_children = [
        c for c in kids
        if c["data"].get("contentType") == "application/pdf"
        and c["data"].get("linkMode") in VALID_LINK_MODES
    ]

    if not valid_children:
        log.warning(f"  No qualifying PDF children for {parent_key}")
        return records

    parent_data = parent_item.get("data", {})
    title = (parent_data.get("title") or parent_data.get("filename") or "").strip()
    abstract = (parent_data.get("abstractNote") or "").strip()

    for child in valid_children:
        child_key = child["key"]
        log.info(f"  Extracting text for child {child_key} of parent {parent_key}")

        raw_text, method = extract_pdf_text(child_key, zot_client)
        full_text = clean_extracted_text(raw_text) if raw_text else ""

        record = {
            "parent_key": parent_key,
            "child_key": child_key,
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
        }
        records.append(record)

        if full_text:
            log.info(f"    OK ({method}) — {len(full_text)} chars")
        else:
            log.warning(f"    No text extracted ({method})")

    return records

# ---------------------------------------------------------------------------
# 6. LLM INFERENCE (kept from generate_prediction.py)
# ---------------------------------------------------------------------------

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_id = os.getenv("HF_MODEL_ID", DEFAULT_MODEL_ID)
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if torch.cuda.is_available():
        device = "cuda"
        model_dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU path (Metal Performance Shaders).
        device = "mps"
        model_dtype = torch.float16
    else:
        device = "cpu"
        model_dtype = torch.float32

    log.info(f"Loading model: {model_id} on {device} (dtype={model_dtype}) …")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model_kwargs = {
            "dtype": model_dtype,
            "token": hf_token,
        }
        if device == "mps":
            # Avoid SDPA on Apple Silicon for long prompts; it can try to allocate
            # enormous temporary attention buffers and fail despite modest model size.
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        if device in ("cuda", "mps"):
            model = model.to(device)
    except OSError as exc:
        msg = str(exc)
        if "gated repo" in msg.lower() or "401" in msg:
            hint = (
                "ERROR: Hugging Face model access failed (gated repo).\n"
                f"Model: {model_id}\n"
                "Set HF_TOKEN in your environment with read access,\n"
                "or switch to a public model: export HF_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            raise SystemExit(hint) from exc
        raise
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if getattr(model, "generation_config", None) is not None:
        # These defaults can trigger warnings when do_sample=False.
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=model_dtype,
        trust_remote_code=True,
        device=device,
    )
    log.info("Model loaded.")
    return pipe, tokenizer, model_id


def build_prompt(document_text: str) -> str:
    """Build the model prompt for tag prediction."""
    return f"{document_text}{PROMPT_SUFFIX}"


def truncate_prompt(document_text: str, tokenizer, pipe) -> str:
    model_max = getattr(pipe.model.config, "max_position_embeddings", None)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    limits = [l for l in [model_max, tok_limit] if isinstance(l, int) and 0 < l < 1_000_000]
    if not limits:
        return build_prompt(document_text)
    ctx = min(limits)
    model_device = getattr(getattr(pipe, "model", None), "device", None)
    device_type = getattr(model_device, "type", "cpu")
    if device_type == "mps":
        ctx = min(ctx, MPS_MAX_INPUT_TOKENS)

    # Reserve context for generation tokens and a small safety margin.
    max_input = max(256, ctx - MAX_NEW_TOKENS - 16)

    suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]
    doc_ids = tokenizer(document_text, add_special_tokens=False)["input_ids"]

    reserved = len(suffix_ids)
    max_doc_tokens = max(1, max_input - reserved)

    if len(doc_ids) <= max_doc_tokens:
        prompt_tokens = reserved + len(doc_ids)
        log.info(f"  Prompt tokens: {prompt_tokens} (context cap {ctx})")
        return build_prompt(document_text)

    trimmed_doc_ids = doc_ids[:max_doc_tokens]
    trimmed_doc = tokenizer.decode(trimmed_doc_ids, skip_special_tokens=False)
    log.info(
        f"  Truncated document tokens {len(doc_ids)} -> {len(trimmed_doc_ids)} "
        f"(context {ctx}, reserved {reserved})"
    )
    return build_prompt(trimmed_doc)


def run_inference(pipe, prompt: str) -> str:
    from transformers import GenerationConfig

    temperature = float(os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    top_p = float(os.getenv("LLM_TOP_P", str(DEFAULT_TOP_P)))
    repetition_penalty = float(
        os.getenv("LLM_REPETITION_PENALTY", str(DEFAULT_REPETITION_PENALTY))
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    result = pipe(prompt, generation_config=gen_cfg, return_full_text=False)
    return result[0]["generated_text"]

# ---------------------------------------------------------------------------
# 7. OUTPUT PARSING (mirrors generate_prediction.py)
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
        data = json.loads(text[start : end + 1])
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
    candidates = [m.strip().rstrip(",").strip() for m in re.findall(pattern, text, re.MULTILINE) if m.strip()]
    # Ignore common non-tag artifacts when the model drifts into scraped/forum text.
    noise_markers = ("posted by", "reply", "comments", "http://", "https://")
    return [c for c in candidates if not any(n in c.lower() for n in noise_markers)]


def extract_comma_tags(text: str) -> list[str]:
    tags = []
    for c in text.split(","):
        clean = re.sub(r"^[\d\.\s]+", "", c.strip()).strip()
        if clean:
            tags.append(clean)
    return tags


def strip_generation_noise(text: str) -> str:
    """Trim common trailing noise (e.g., forum/comment text) from model output."""
    if not text:
        return ""
    # Cut everything after known noisy section headers/phrases.
    markers = [
        r"\n\s*##+\s*comments?\b",
        r"\n\s*comments?\b",
        r"\n\s*[\u2022\-*]\s*posted by\b",
        r"\n\s*posted by\b",
    ]
    cut = len(text)
    for pat in markers:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cut = min(cut, m.start())
    return text[:cut].strip()


def parse_llm_output(raw: str) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    text = strip_generation_noise(raw.strip())
    if not text:
        return []

    # Prefer the first non-empty line when it looks like a comma-separated tag list.
    first_non_empty = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if "," in first_non_empty:
        tags = extract_comma_tags(first_non_empty)
        if tags:
            return tags

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


def has_nonempty_predicted_tags(record: dict) -> bool:
    tags = record.get("predicted_tags", [])
    if not isinstance(tags, list) or not tags:
        return False

    for tag_obj in tags:
        if isinstance(tag_obj, str) and tag_obj.strip():
            return True
        if isinstance(tag_obj, dict) and str(tag_obj.get("tag", "")).strip():
            return True
    return False

# ---------------------------------------------------------------------------
# 8. PREDICTIONS I/O
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
    log.info(f"Saved {len(predictions)} predictions to {PREDICTIONS_PATH.name}")

# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process Zotero items and generate LLM tag predictions."
    )
    parser.add_argument(
        "--parent-keys", nargs="+", required=True,
        help="One or more Zotero parent keys to process and predict tags for.",
    )
    parser.add_argument(
        "--skip-predict", action="store_true",
        help="Only process text (skip LLM prediction step).",
    )
    args = parser.parse_args()

    parent_keys = list(dict.fromkeys(args.parent_keys))  # deduplicate, preserve order
    log.info(f"Pipeline started with {len(parent_keys)} parent key(s): {parent_keys}")

    t_start = time.perf_counter()

    # --- Load snapshot for metadata ---
    snapshot = load_snapshot()
    parent_lookup = build_parent_lookup(snapshot)
    log.info(f"Snapshot: {len(parent_lookup)} parents in library")

    # --- Determine which keys need processing ---
    keys_to_process = []
    docs_for_prediction = []

    for pk in parent_keys:
        cached = load_records_for_parent(pk)
        has_text = any(r.get("full_text", "").strip() for r in cached)

        if has_text:
            log.info(f"  {pk}: found in per-document cache — skipping processing")
            docs_for_prediction.extend(cached)
        else:
            if pk not in parent_lookup:
                log.warning(f"  {pk}: not found in Zotero snapshot — skipping")
                continue
            keys_to_process.append(pk)

    # --- Process uncached keys via Zotero API ---
    if keys_to_process:
        zotero_api_key = os.getenv("ZOTERO_API_KEY")
        zotero_group_id = os.getenv("ZOTERO_GROUP_ID")
        if not zotero_api_key or not zotero_group_id:
            sys.exit(
                "ERROR: ZOTERO_API_KEY and ZOTERO_GROUP_ID must be set "
                "to process uncached keys."
            )

        from pyzotero import Zotero

        zot = Zotero(
            library_id=zotero_group_id,
            library_type="group",
            api_key=zotero_api_key,
        )

        log.info(f"Processing {len(keys_to_process)} uncached parent key(s) via Zotero API…")

        for pk in keys_to_process:
            parent_item = parent_lookup[pk]
            new_records = process_parent_key(pk, zot, parent_item)

            save_records_for_parent(pk, new_records)
            docs_for_prediction.extend(new_records)
    else:
        log.info("All keys found in cache — no Zotero API calls needed")

    # --- Filter to docs with actual text ---
    docs_with_text = [d for d in docs_for_prediction if d.get("full_text", "").strip()]
    log.info(f"{len(docs_with_text)} document(s) with text ready for prediction")

    if not docs_with_text:
        log.warning("No documents with text to predict — exiting")
        return

    if args.skip_predict:
        log.info("--skip-predict set — skipping LLM prediction step")
        return

    # --- Load tags and model ---
    allowed_tags, tag_lookup = load_allowed_tags()

    t_model = time.perf_counter()
    pipe, tokenizer, model_id = load_model()
    log.info(f"Model load: {time.perf_counter() - t_model:.1f}s")

    # --- Run predictions ---
    predictions = load_predictions()
    before_cleanup = len(predictions)
    predictions = [p for p in predictions if has_nonempty_predicted_tags(p)]
    removed_empty = before_cleanup - len(predictions)
    if removed_empty:
        log.info(f"Removed {removed_empty} existing record(s) with zero predicted tags")

    existing_child_keys = {p.get("child_key") for p in predictions}
    predicted_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for doc in docs_with_text:
        child_key = doc.get("child_key", doc.get("parent_key", ""))
        log.info(f"Predicting tags for {child_key} — {doc['title'][:60]}")

        prompt = truncate_prompt(doc["full_text"], tokenizer, pipe)

        t_infer = time.perf_counter()
        raw_response = run_inference(pipe, prompt)
        log.info(f"  Inference: {time.perf_counter() - t_infer:.1f}s")
        log.info(f"  Raw response: {raw_response[:300]}")

        candidates = parse_llm_output(raw_response)
        predicted_tags = filter_to_allowed(candidates, tag_lookup)

        log.info(f"  LLM tags (raw):   {candidates}")
        log.info(f"  LLM tags (valid): {predicted_tags}")

        if not predicted_tags:
            log.info(f"  No valid tags predicted for {child_key}; skipping write/update")
            continue

        # Build predicted_tags with prediction history structure
        predicted_tags_with_history = [
            {
                "tag": tag,
                "predictions": [
                    {
                        "predicted_date": predicted_date,
                        "model_id": model_id
                    }
                ]
            }
            for tag in predicted_tags
        ]

        entry = {
            "parent_key": doc["parent_key"],
            "child_key": child_key,
            "title": doc["title"],
            "abstract": doc["abstract"],
            "ground_truth_tags": [],
            "predicted_tags": predicted_tags_with_history,
        }

        # Update or append with merge logic
        if child_key in existing_child_keys:
            log.info(f"  Updating existing prediction for {child_key}")
            # Find existing record
            existing_record = next((p for p in predictions if p.get("child_key") == child_key), None)
            if existing_record:
                # Merge predicted_tags with prediction history
                existing_tags_dict = {}
                for tag_obj in existing_record.get("predicted_tags", []):
                    # Backward compatibility: support legacy string tags if any remain.
                    if isinstance(tag_obj, str):
                        existing_tags_dict[tag_obj] = {
                            "tag": tag_obj,
                            "predictions": []
                        }
                    elif isinstance(tag_obj, dict) and tag_obj.get("tag"):
                        existing_tags_dict[tag_obj["tag"]] = tag_obj

                history_updates = 0
                
                for new_tag_obj in predicted_tags_with_history:
                    tag_name = new_tag_obj["tag"]
                    if tag_name in existing_tags_dict:
                        # Tag already exists: append this run if not already present.
                        tag_history = existing_tags_dict[tag_name].setdefault("predictions", [])
                        already_recorded = any(
                            h.get("predicted_date") == predicted_date and h.get("model_id") == model_id
                            for h in tag_history
                        )
                        if not already_recorded:
                            log.info(f"    Tag '{tag_name}' already predicted; appending {model_id} on {predicted_date}")
                            tag_history.append({
                                "predicted_date": predicted_date,
                                "model_id": model_id
                            })
                            history_updates += 1
                    else:
                        # New tag: add with single prediction
                        log.info(f"    New tag '{tag_name}' added")
                        existing_tags_dict[tag_name] = new_tag_obj
                        history_updates += 1
                
                # Only bump top-level metadata when this run is recorded in tag history.
                if history_updates > 0:
                    existing_record["predicted_tags"] = list(existing_tags_dict.values())
                    log.info(f"  Recorded {history_updates} tag-history update(s)")
                else:
                    log.info("  No new tag-history updates; leaving record metadata unchanged")
        else:
            predictions.append(entry)
            existing_child_keys.add(child_key)
            log.info(f"  Appended new prediction for {child_key}")

    save_predictions(predictions)

    log.info(f"\nTotal time: {time.perf_counter() - t_start:.1f}s")
    log.info("Done.")


if __name__ == "__main__":
    main()
