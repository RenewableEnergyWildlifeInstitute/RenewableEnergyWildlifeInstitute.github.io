#!/usr/bin/env python3
"""Build and publish the REWI community explorer snapshot.

This script reads the Zotero snapshot, runs document-community detection on
lemmatized titles + abstracts, then writes:
1) evaluator/rewi_community_explorer.html (published explorer)
2) evaluator/data/community_explorer_snapshot.json (admin metadata)
3) evaluator/data/community_explorer_assignments.csv (parent-level assignments)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_community
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = REPO_ROOT / "evaluator" / "data" / "zotero_snapshot.json"
OUTPUT_HTML_PATH = REPO_ROOT / "evaluator" / "rewi_community_explorer.html"
OUTPUT_META_PATH = REPO_ROOT / "evaluator" / "data" / "community_explorer_snapshot.json"
OUTPUT_ASSIGNMENTS_CSV_PATH = REPO_ROOT / "evaluator" / "data" / "community_explorer_assignments.csv"

BANNED_TERMS = {"et", "al", "pdf"}
GENERIC_STOP_TERMS = {
    "full", "text", "snapshot", "attachment", "html", "file", "document", "version",
    "sciencedirect", "pubmed", "sage", "report", "proceedings", "proceeding",
}

UNIGRAM_MAX_COMMUNITIES = 50
BIGRAM_MAX_COMMUNITIES = 50
MIN_DOCS_FOR_COMMUNITIES = 10

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def ensure_nltk_resources() -> None:
    resources = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def load_parent_items() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot not found: {SNAPSHOT_PATH}")

    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    items = snapshot.get("items", [])

    rows: list[dict] = []
    for item in items:
        data = item.get("data", {})
        if data.get("parentItem"):
            continue
        if data.get("itemType") in {"attachment", "note"}:
            continue
        rows.append(
            {
                "key": str(item.get("key", "")).strip(),
                "title": str(data.get("title") or "").strip(),
                "abstract": str(data.get("abstractNote") or "").strip(),
                "authors_raw": " ".join(
                    [
                        str(c.get("lastName") or c.get("name") or "").strip()
                        for c in (data.get("creators") or [])
                    ]
                ).strip(),
            }
        )

    df = pd.DataFrame(rows)
    df = df[(df["title"].str.len() > 0) | (df["abstract"].str.len() > 0)].copy()
    return df


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    if not text:
        return ""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    tokens = [t for t in tokens if not t.isdigit() and len(t) > 2]
    if not tokens:
        return ""
    try:
        pos_tags = nltk.pos_tag(tokens)
        lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    except Exception:
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmas)


def build_author_stop_terms(df: pd.DataFrame) -> set[str]:
    author_terms: set[str] = set()
    for value in df["authors_raw"].dropna().astype(str):
        for token in re.findall(r"\b[a-z]{3,}\b", value.lower()):
            author_terms.add(token)
    return author_terms


def run_document_topic_communities(
    text_series: pd.Series,
    matrix: pd.DataFrame,
    topic_stop_terms: set[str],
    min_similarity: float,
    k_neighbors: int,
) -> tuple[pd.DataFrame, dict[int, int]]:
    if matrix.empty:
        return pd.DataFrame(), {}

    n_docs = matrix.shape[0]
    k_neighbors = min(k_neighbors, max(5, n_docs - 1))

    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
    nn_model.fit(matrix.values)
    distances, indices = nn_model.kneighbors(matrix.values)

    doc_graph = nx.Graph()
    doc_graph.add_nodes_from(range(n_docs))

    for doc_i in range(n_docs):
        for nbr_idx, dist in zip(indices[doc_i][1:], distances[doc_i][1:]):
            similarity = 1.0 - float(dist)
            if similarity < min_similarity:
                continue
            if doc_graph.has_edge(doc_i, nbr_idx):
                if similarity > doc_graph[doc_i][nbr_idx]["weight"]:
                    doc_graph[doc_i][nbr_idx]["weight"] = similarity
            else:
                doc_graph.add_edge(doc_i, nbr_idx, weight=similarity)

    isolates = list(nx.isolates(doc_graph))
    if isolates:
        doc_graph.remove_nodes_from(isolates)

    if doc_graph.number_of_nodes() == 0:
        return pd.DataFrame(), {}

    try:
        communities = list(nx_community.louvain_communities(doc_graph, weight="weight", resolution=1.0, seed=42))
    except Exception:
        communities = list(nx_community.greedy_modularity_communities(doc_graph, weight="weight"))

    communities = sorted(communities, key=len, reverse=True)

    term_names = matrix.columns.to_numpy()
    rows = []
    assignments: dict[int, int] = {}

    for community_id, nodes in enumerate(communities):
        node_list = sorted(nodes)
        for node in node_list:
            assignments[node] = community_id

        centroid = matrix.iloc[node_list].mean(axis=0).to_numpy()
        ranked_idx = np.argsort(centroid)[::-1]

        top_terms = []
        for i in ranked_idx:
            term = term_names[i]
            if centroid[i] <= 0:
                break
            if any(tok in topic_stop_terms for tok in term.split()):
                continue
            top_terms.append(term)
            if len(top_terms) == 6:
                break

        sample_titles = text_series.iloc[node_list].astype(str).head(3).tolist()
        rows.append(
            {
                "community": community_id,
                "documents": len(node_list),
                "top_terms": ", ".join(top_terms),
                "sample_documents": " | ".join(sample_titles),
            }
        )

    summary = pd.DataFrame(rows).sort_values("documents", ascending=False).reset_index(drop=True)
    return summary, assignments


def collect_community_titles(assignments: dict[int, int], doc_labels: pd.Series, max_titles: int = 50) -> dict[int, list[str]]:
    bucket: dict[int, list[str]] = defaultdict(list)
    for idx, cid in assignments.items():
        title = str(doc_labels.iloc[idx]) if idx < len(doc_labels) else str(idx)
        bucket[cid].append(title)
    return {int(k): sorted(set(v))[:max_titles] for k, v in bucket.items()}


def build_community_json(summary: pd.DataFrame, title_map: dict[int, list[str]], mode: str) -> list[dict]:
    if summary is None or summary.empty:
        return []
    records = []
    for _, row in summary.iterrows():
        cid = int(row["community"])
        terms = [t.strip() for t in str(row.get("top_terms", "")).split(",") if t.strip()]
        records.append(
            {
                "id": cid,
                "mode": mode,
                "docs": int(row["documents"]),
                "terms": terms,
                "titles": title_map.get(cid, []),
            }
        )
    records.sort(key=lambda r: r["docs"], reverse=True)
    return records


def render_html(data_json: str, snapshot_label: str, total_docs: int) -> str:
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\"/>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
<title>REWI Community Explorer</title>
<style>
  :root{{--bg:#f4f6fa;--card:#fff;--border:#d1d9e6;--primary:#2563eb;--tag-bg:#e0eaff;--tag-fg:#1e3a8a;--title-fg:#1e293b;--meta-fg:#64748b;--expand-bg:#f8fafc;}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:\"Segoe UI\",system-ui,sans-serif;background:var(--bg);color:var(--title-fg);}}
  header{{background:#1e293b;color:#f1f5f9;padding:1.1rem 2rem;position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}}
  header h1{{font-size:1.2rem;font-weight:700;}}
  .meta{{font-size:.82rem;color:#cbd5e1;}}
  .controls{{display:flex;gap:.5rem;margin-left:auto;}}
  .mode-btn{{padding:.4rem 1rem;border:2px solid #94a3b8;border-radius:6px;background:transparent;color:#f1f5f9;cursor:pointer;font-size:.9rem;}}
  .mode-btn.active{{border-color:#60a5fa;background:#1d4ed8;color:#fff;}}
  main{{max-width:1400px;margin:1.2rem auto;padding:0 1.25rem 2rem;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem;}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);}}
  .card-header{{display:flex;align-items:center;gap:.75rem;padding:.85rem 1rem;cursor:pointer;user-select:none;}}
  .swatch{{width:14px;height:38px;border-radius:4px;flex-shrink:0;}}
  .card-meta{{flex:1;min-width:0;}}
  .card-title{{font-size:.82rem;font-weight:700;color:var(--meta-fg);text-transform:uppercase;letter-spacing:.05em;}}
  .card-count{{font-size:1.35rem;font-weight:800;color:var(--title-fg);line-height:1.1;}}
  .tag-list{{display:flex;flex-wrap:wrap;gap:.35rem;padding:0 1rem .85rem;}}
  .tag{{background:var(--tag-bg);color:var(--tag-fg);font-size:.78rem;font-weight:600;padding:.25rem .55rem;border-radius:20px;white-space:nowrap;}}
  .toggle-row{{display:flex;align-items:center;justify-content:space-between;padding:.5rem 1rem;border-top:1px solid var(--border);cursor:pointer;}}
  .toggle-label{{font-size:.8rem;color:var(--primary);font-weight:600;}}
  .title-list{{display:none;padding:.25rem 1rem .85rem;background:var(--expand-bg);border-top:1px solid var(--border);}}
  .title-list.open{{display:block;}}
  .title-list ol{{padding-left:1.2rem;}}
  .title-list li{{font-size:.82rem;color:var(--title-fg);margin:.28rem 0;line-height:1.4;}}
</style>
</head>
<body>
<header>
  <h1>REWI Community Explorer</h1>
  <div class=\"meta\">Monthly snapshot: {snapshot_label} | Documents analyzed: {total_docs}</div>
  <div class=\"controls\">
    <button class=\"mode-btn active\" onclick=\"setMode('unigram')\" id=\"btn-unigram\">Unigram</button>
    <button class=\"mode-btn\" onclick=\"setMode('bigram')\" id=\"btn-bigram\">Bigram</button>
  </div>
</header>
<main><div class=\"grid\" id=\"grid\"></div></main>
<script>
const DATA = {data_json};
const COLORS = {json.dumps(PALETTE)};
let mode = "unigram";
function setMode(m){{
  mode = m;
  document.getElementById("btn-unigram").classList.toggle("active", m === "unigram");
  document.getElementById("btn-bigram").classList.toggle("active", m === "bigram");
  render();
}}
function escHtml(s){{
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\"/g, "&quot;");
}}
function toggleTitles(el){{
  const card = el.closest(".card");
  const panel = card.querySelector(".title-list");
  panel.classList.toggle("open");
}}
function render(){{
  const communities = DATA[mode] || [];
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  communities.forEach((c, ci) => {{
    const color = COLORS[ci % COLORS.length];
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header" onclick="toggleTitles(this)">
        <div class="swatch" style="background:${{color}}"></div>
        <div class="card-meta">
          <div class="card-title">Community ${{c.id}}</div>
          <div class="card-count">${{c.docs.toLocaleString()}} docs</div>
        </div>
      </div>
      <div class="tag-list">${{(c.terms || []).map(t => `<span class="tag">${{escHtml(t)}}</span>`).join("")}}</div>
      <div class="toggle-row" onclick="toggleTitles(this)"><span class="toggle-label">View titles (${{(c.titles || []).length}} shown)</span></div>
      <div class="title-list"><ol>${{(c.titles || []).map(t => `<li>${{escHtml(t)}}</li>`).join("")}}</ol></div>`;
    grid.appendChild(card);
  }});
}}
render();
</script>
</body>
</html>
"""


def main() -> None:
    ensure_nltk_resources()
    df = load_parent_items()
    if df.empty:
        raise RuntimeError("No parent records with title/abstract were found in zotero_snapshot.json")

    lemmatizer = WordNetLemmatizer()
    all_parent_df = df.copy()

    metadata_title_pattern = r"^(?:\s*(?:pdf|full\s+text|snapshot|sciencedirect\s+snapshot)\s*)+$"

    df = df.copy()
    df["text_raw"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_clean"] = df["text_raw"].map(clean_text)
    df["text_lemma"] = df["text_clean"].map(lambda x: lemmatize_text(x, lemmatizer))

    df = df[df["text_lemma"].str.len() > 0].copy()
    metadata_mask = df["title"].str.lower().str.match(metadata_title_pattern, na=False)
    df = df[~metadata_mask].copy()
    # Keep a contiguous index so downstream boolean masks align reliably.
    df = df.reset_index(drop=True)

    skip_reason = None
    if len(df) < MIN_DOCS_FOR_COMMUNITIES:
        skip_reason = (
            "Not enough records for stable community detection after filtering "
            f"(found {len(df)}, need at least {MIN_DOCS_FOR_COMMUNITIES})"
        )

    author_stop_terms = build_author_stop_terms(df)
    topic_stop_terms = BANNED_TERMS | GENERIC_STOP_TERMS | author_stop_terms

    unigram_summary = pd.DataFrame()
    unigram_assignments: dict[int, int] = {}
    df_u = df.copy()

    bigram_summary = pd.DataFrame()
    bigram_assignments: dict[int, int] = {}
    df_b = df.copy()

    if skip_reason is None:
        try:
            unigram_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 1),
                max_features=5000,
                min_df=2,
                token_pattern=r"\b[a-z]{3,}\b",
            )
            unigram_matrix_raw = unigram_vectorizer.fit_transform(df["text_lemma"])
            unigram_matrix = pd.DataFrame(unigram_matrix_raw.toarray(), columns=unigram_vectorizer.get_feature_names_out())

            unigram_nonzero_mask = unigram_matrix.sum(axis=1) > 0
            df_u = df.loc[unigram_nonzero_mask.to_numpy()].reset_index(drop=True)
            unigram_matrix = unigram_matrix.loc[unigram_nonzero_mask].reset_index(drop=True)

            unigram_summary, unigram_assignments = run_document_topic_communities(
                text_series=df_u["title"].fillna(df_u["key"]),
                matrix=unigram_matrix,
                topic_stop_terms=topic_stop_terms,
                min_similarity=0.20,
                k_neighbors=15,
            )
        except ValueError as exc:
            skip_reason = f"Unable to build unigram features: {exc}"

    if skip_reason is None:
        try:
            bigram_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(2, 2),
                max_features=5000,
                min_df=3,
                token_pattern=r"\b[a-z]{3,}\b",
            )
            bigram_matrix_raw = bigram_vectorizer.fit_transform(df["text_lemma"])
            bigram_matrix = pd.DataFrame(bigram_matrix_raw.toarray(), columns=bigram_vectorizer.get_feature_names_out())

            bigram_nonzero_mask = bigram_matrix.sum(axis=1) > 0
            df_b = df.loc[bigram_nonzero_mask.to_numpy()].reset_index(drop=True)
            bigram_matrix = bigram_matrix.loc[bigram_nonzero_mask].reset_index(drop=True)

            bigram_summary, bigram_assignments = run_document_topic_communities(
                text_series=df_b["title"].fillna(df_b["key"]),
                matrix=bigram_matrix,
                topic_stop_terms=topic_stop_terms,
                min_similarity=0.16,
                k_neighbors=12,
            )
        except ValueError as exc:
            skip_reason = f"Unable to build bigram features: {exc}"

    unigram_title_map = collect_community_titles(unigram_assignments, df_u["title"].fillna(df_u["key"])) if unigram_assignments else {}
    bigram_title_map = collect_community_titles(bigram_assignments, df_b["title"].fillna(df_b["key"])) if bigram_assignments else {}

    unigram_key_assignments: dict[str, int] = {}
    for row_idx, community_id in unigram_assignments.items():
        if row_idx < len(df_u):
            key = str(df_u.iloc[row_idx]["key"])
            unigram_key_assignments[key] = int(community_id)

    bigram_key_assignments: dict[str, int] = {}
    for row_idx, community_id in bigram_assignments.items():
        if row_idx < len(df_b):
            key = str(df_b.iloc[row_idx]["key"])
            bigram_key_assignments[key] = int(community_id)

    assignments_df = all_parent_df[["key", "title"]].copy()
    assignments_df["unigram_community"] = assignments_df["key"].map(unigram_key_assignments).astype("Int64")
    assignments_df["bigram_community"] = assignments_df["key"].map(bigram_key_assignments).astype("Int64")
    assignments_df = assignments_df.rename(columns={"key": "parent_key"})
    OUTPUT_ASSIGNMENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(OUTPUT_ASSIGNMENTS_CSV_PATH, index=False, encoding="utf-8")

    unigram_data = build_community_json(unigram_summary, unigram_title_map, "unigram")[:UNIGRAM_MAX_COMMUNITIES]
    bigram_data = build_community_json(bigram_summary, bigram_title_map, "bigram")[:BIGRAM_MAX_COMMUNITIES]

    now = datetime.now(timezone.utc)
    snapshot_label = now.strftime("%Y-%m")
    all_data = {"unigram": unigram_data, "bigram": bigram_data}

    OUTPUT_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML_PATH.write_text(
        render_html(
            data_json=json.dumps(all_data, ensure_ascii=False),
            snapshot_label=snapshot_label,
            total_docs=len(df),
        ),
        encoding="utf-8",
    )

    meta = {
        "generated_at": now.isoformat(),
        "snapshot_month": snapshot_label,
        "source": "evaluator/data/zotero_snapshot.json",
        "mode": "titles_abstracts_lemmatized",
        "document_count": int(len(df)),
        "filtered_metadata_like_titles": int(metadata_mask.sum()),
        "unigram_communities": int(len(unigram_data)),
        "bigram_communities": int(len(bigram_data)),
        "community_build_status": "skipped" if skip_reason else "ok",
        "community_build_note": skip_reason,
        "published_html": "evaluator/rewi_community_explorer.html",
        "assignments_csv": "evaluator/data/community_explorer_assignments.csv",
    }
    OUTPUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote explorer HTML: {OUTPUT_HTML_PATH}")
    print(f"Wrote metadata JSON: {OUTPUT_META_PATH}")
    print(f"Wrote assignments CSV: {OUTPUT_ASSIGNMENTS_CSV_PATH}")
    if skip_reason:
        print(f"Community detection skipped: {skip_reason}")


if __name__ == "__main__":
    main()
