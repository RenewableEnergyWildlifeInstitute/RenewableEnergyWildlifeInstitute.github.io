# Renewable Energy Wildlife Institute Tagging Pipeline

This repository contains the end-to-end REWI literature tagging system, including automated ingestion, text processing, model inference, and expert review tooling.

It includes:

- A static evaluator application for human review and export
- Automated Zotero snapshot refresh and model inference workflows
- Data artifacts used by the tagging pipeline
- Dashboard assets and rendered pages
- Project documentation and paper deliverables

## Repository Layout

```text
.
├── .github/
│   ├── workflows/            # GitHub Actions workflows
│   └── scripts/              # Pipeline and validation scripts
├── evaluator/                # Static review app (GitHub Pages)
│   ├── admin.html
│   ├── review.html
│   ├── js/
│   └── data/
├── docs/                     # Published static pages
├── rmarkdown/                # Source Rmd pages for docs/
├── data/                     # Geospatial and supporting datasets
└── REWI_Paper/               # Final report artifacts
```

## Core Capabilities

1. Keep a current Zotero snapshot in-repo using scheduled or manual workflow runs.
2. Run the text processing and model tagging pipeline on:
   - Free GitHub-hosted CPU runners for smaller jobs
   - Self-hosted runners (local Mac or cloud GPU) for faster large-batch jobs
3. Assign reviewers in the evaluator app, collect corrections, and export validated outputs.
4. Maintain full auditability using JSON artifacts committed to git.

## Quick Start (Operations)

### 1) Configure Repository Secrets

In GitHub repository settings, add the following secrets:

- `ZOTERO_GROUP_ID`
- `ZOTERO_API_KEY`
- `HF_TOKEN`
- `HF_MODEL_ID` (optional override; workflow/script default is `jme-datasci/rewi-tagger`)

### 2) Refresh Zotero Snapshot

Run workflow: **Refresh Zotero Snapshot**

- File: `.github/workflows/zotero-snapshot.yml`
- Trigger: daily schedule and manual dispatch
- Output: `evaluator/data/zotero_snapshot.json`

### 3) Run Processing + Prediction

Run workflow: **Run Pipeline — Process & Predict Tags**

- File: `.github/workflows/run-model.yml`
- Required input: `parent_keys` (space-separated Zotero parent keys)
- Optional input: `runner_type` (`github` or `self-hosted`)
- Outputs:
  - `evaluator/data/full_texts/<PARENT_KEY>.json`
  - `evaluator/data/predictions.json`

### 4) Review and Export

Open evaluator pages:

- Admin: `evaluator/admin.html`
- Reviewer: `evaluator/review.html?reviewer=<email>`

The evaluator app setup and reviewer workflow are documented in `evaluator/README.md`.

## Automation Details

### Snapshot workflow

- File: `.github/workflows/zotero-snapshot.yml`
- Uses the Zotero API to generate and commit `evaluator/data/zotero_snapshot.json`
- Validates cached processed records in `evaluator/data/full_texts/` against current snapshot keys

### Process-and-predict workflow

- File: `.github/workflows/run-model.yml`
- Runs `.github/scripts/process_and_predict.py` for selected parent keys
- Uses `jme-datasci/rewi-tagger` by default unless `HF_MODEL_ID` is overridden
- Reuses per-parent cache files in `evaluator/data/full_texts/`
- Writes/updates `evaluator/data/predictions.json`

## Runtime Modes

### GitHub-hosted CPU

- No infrastructure setup
- Best for small or occasional runs
- Longer processing time

### Self-hosted Runner (Local or Cloud)

- Best for larger batches and faster turnaround
- Can run on Apple Silicon local machines (M1/M2) or AWS GPU instances
- The team validated local M1 execution at approximately 15 seconds per document for full single-document pipeline runs

## Data Artifacts

- `evaluator/data/zotero_snapshot.json`: latest trimmed Zotero library snapshot
- `evaluator/data/full_texts/`: per-parent processed full text cache
- `evaluator/data/predictions.json`: model predictions and prediction history
- `evaluator/data/assignments.json`: reviewer assignments
- `evaluator/data/reviews/*.json`: reviewer outputs
- `evaluator/data/tag_dictionary.json`: allowed tags and categories

## Security Notes

- Do not store GitHub PATs in repository files.
- `evaluator/data/config.json` should only contain `owner` and `repo`.
- PATs used by evaluator admins are stored in browser local storage on the admin device.
- Use fine-grained PATs scoped to this repository with minimal permissions.

## Additional Documentation

- [Workflow Operations Guide](.github/README.md): workflow operations and troubleshooting
- [Evaluator App Guide](evaluator/README.md): evaluator UI setup and reviewer/admin usage
- [Predictions Schema](evaluator/data/PREDICTIONS_SCHEMA.md): structure and history model for `predictions.json`

## Live Application

- [REWI Evaluator GitHub Page](https://renewableenergywildlifeinstitute.github.io/evaluator/)
