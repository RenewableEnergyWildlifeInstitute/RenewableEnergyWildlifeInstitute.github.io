# GitHub Actions Operations Guide

This document explains the automation in `.github/workflows/` for REWI maintainers.

## Workflows

## 1) Refresh Zotero Snapshot

- File: `workflows/zotero-snapshot.yml`
- Triggers:
  - Scheduled daily (`0 9 * * *`)
  - Manual (`workflow_dispatch`)
- Purpose:
  - Pull latest items from Zotero Group API
  - Trim fields for repository use
  - Write `evaluator/data/zotero_snapshot.json`
  - Validate processed dataset consistency in `evaluator/data/full_texts/`
  - Commit and push changes if any

Required secrets:

- `ZOTERO_GROUP_ID`
- `ZOTERO_API_KEY` (required for private libraries)

## 2) Run Pipeline — Process & Predict Tags

- File: `workflows/run-model.yml`
- Trigger: manual (`workflow_dispatch`)
- Inputs:
  - `parent_keys` (required, space-separated)
  - `runner_type` (`github` or `self-hosted`, default `github`)
- Purpose:
  - Run `.github/scripts/process_and_predict.py`
  - Process full text for specified parent keys
  - Run model inference and tag filtering
  - Update and commit prediction artifacts

Required secrets:

- `HF_TOKEN`
- `ZOTERO_API_KEY`
- `ZOTERO_GROUP_ID`

Optional secret:

- `HF_MODEL_ID` (workflow/script default: `jme-datasci/rewi-tagger`)

## Script Dependencies

Installed from `.github/scripts/requirements.txt`:

- pandas
- pyarrow
- transformers
- torch
- accelerate
- python-dotenv
- pyzotero
- pymupdf
- pymupdf4llm

## Artifact Paths

- Snapshot output: `evaluator/data/zotero_snapshot.json`
- Processed full text cache: `evaluator/data/full_texts/*.json`
- Predictions: `evaluator/data/predictions.json`

## Operational Modes

### GitHub-hosted mode

- Use `runner_type=github`
- Lowest operational overhead
- CPU inference path

### Self-hosted mode

- Use `runner_type=self-hosted`
- Requires online self-hosted runner
- Supports local Mac or cloud GPU runner strategies

## Common Troubleshooting

- Missing Zotero secret values: snapshot and pipeline runs will fail early.
- Hugging Face auth/model access errors: verify `HF_TOKEN` and model permissions.
- No changes committed: expected when outputs do not differ from repository state.
- Self-hosted runner unavailable: rerun with `runner_type=github`.
