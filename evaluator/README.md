# REWI Tag Evaluation Tool

A static GitHub Pages tool for REWI domain experts to manually review and evaluate metadata tags predicted by our LLM pipeline. All data is persisted as JSON files in the GitHub repo via the GitHub REST API.

## Architecture

```
evaluator/
├── index.html              # Entry point — routes to admin or reviewer view
├── admin.html              # Admin dashboard (setup, monitor, assign, export)
├── review.html             # Reviewer interface (evaluate tags per document)
├── js/
│   ├── app.js              # Core logic, config loading, CSV parsing
│   ├── github.js           # GitHub API wrapper (read/write files)
│   ├── tags.js             # Tag dictionary (71 tags, 10 categories)
│   └── export.js           # Export/aggregation utilities
├── css/
│   └── styles.css          # Custom styles
├── data/                   # Data directory (read/written via GitHub API)
│   ├── config.json         # App config (repo owner, repo name; no PAT)
│   ├── predictions.json    # Uploaded predictions data
│   ├── assignments.json    # Reviewer-to-document assignments
│   └── reviews/            # Per-reviewer evaluation files
└── README.md
```

## Deployment to GitHub Pages

1. Push the `evaluator/` directory to your repo's `main` branch (or whichever branch you deploy from).

2. In GitHub repo settings, go to **Pages** and set:
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or your branch)
   - **Folder**: `/evaluator` (if at repo root, use `/`)

3. The tool will be available at:
   ```
   https://<owner>.github.io/<repo>/evaluator/
   ```

## First-Time Setup

### 1. Create a GitHub Personal Access Token (PAT)

1. Go to [GitHub Settings > Developer settings > Personal access tokens > Fine-grained tokens](https://github.com/settings/tokens?type=beta).
2. Click **Generate new token**.
3. Configure:
   - **Token name**: `rewi-evaluator` (or similar)
   - **Expiration**: Set an appropriate duration (90 days recommended, rotate when expired)
   - **Repository access**: Select **Only select repositories** and choose this repo
   - **Permissions**: Under **Repository permissions**, set **Contents** to **Read and write**. No other permissions are needed.
4. Copy the token — you'll need it for setup.

### 2. Configure the Admin Dashboard

1. Navigate to the admin page: `https://<owner>.github.io/<repo>/evaluator/admin.html`
2. Enter the default passphrase: `rewi-admin`
3. In the **Setup** tab:
   - Enter the repository **Owner** (e.g., `RenewableEnergyWildlifeInstitute`)
   - Enter the repository **Name** (e.g., `RenewableEnergyWildlifeInstitute.github.io`)
   - Paste the **Personal Access Token**
   - Click **Test Connection** to verify
   - Click **Save Configuration**
4. Change the admin passphrase to something secure.

### 3. Predictions Data

Predictions are created and maintained by the model pipeline and stored in:

- `evaluator/data/predictions.json`

By default, pipeline inference uses `jme-datasci/rewi-tagger` unless `HF_MODEL_ID` is overridden in repository secrets.

The admin dashboard **does not** support manual prediction uploads.

### 4. Assign Reviewers

1. Go to the **Assignments** tab.
2. Enter a reviewer's email address.
3. Select documents to assign using the checkboxes.
4. Click **Save Assignment**.
5. Copy the generated reviewer link and share it with the reviewer.

**Reviewer link format:**
```
https://<owner>.github.io/<repo>/evaluator/review.html?reviewer=email@example.com
```

## For Reviewers

1. Open the link provided by your admin.
2. You'll see your assigned documents with a progress indicator.
3. Click a document to start reviewing.
4. For each predicted tag, mark it as correct or incorrect.
5. Add any missing tags the model should have predicted.
6. Click **Save Progress** to save without completing, or **Submit Review** when done.
7. Your progress persists across devices — the data is stored in the GitHub repo.

## Exporting Results

1. Go to the admin dashboard **Export** tab.
2. Click **Export as JSON** or **Export as CSV**.
3. The export includes:
   - Overall precision, recall, and F1 score
   - Per-category and per-tag breakdowns
   - All individual review records

**Metrics computed from evaluations:**
- **Precision** = correct predictions / total predictions
- **Recall** = correct predictions / (correct predictions + missing tags)
- **F1** = harmonic mean of precision and recall

## Zotero Dashboard Snapshot Mode

The admin Zotero tab supports a snapshot-first loading flow to avoid browser storage quota errors with large libraries.

### Why this mode exists

- Browsers enforce a small quota for `localStorage`.
- A full Zotero item payload can exceed that quota.
- Snapshot mode keeps large data in the repository (`evaluator/data/zotero_snapshot.json`) instead of browser storage.

### Setup

1. Add repository secrets:
   - `ZOTERO_GROUP_ID` (required)
   - `ZOTERO_API_KEY` (required for private libraries; optional for public libraries)
2. Run the GitHub Actions workflow **Refresh Zotero Snapshot**:
   - Manual run: **Actions** -> **Refresh Zotero Snapshot** -> **Run workflow**
   - Scheduled run: daily via cron
3. Open the admin Zotero tab and click **Load Snapshot**.

### Notes on secrets

- GitHub Pages is static and cannot read repository secrets at browser runtime.
- Secrets are available only inside GitHub Actions jobs.
- The workflow uses those secrets server-side and commits a snapshot JSON that the page can safely read.

## Security Considerations

- **PAT Storage**: The PAT is stored only in browser `localStorage` on the device where setup is performed. It is **not** written to `data/config.json`.
- **Committed config**: `data/config.json` contains only `owner` and `repo`.
- **Scope the PAT narrowly**: Use a fine-grained token with **only Contents read/write access** to this single repository.
- **Rotate regularly**: Set a 90-day expiration and create a new token when it expires. Re-enter it in admin setup on each device that needs write access.
- **Admin passphrase**: The admin passphrase is stored in the admin's browser `localStorage`. Change it from the default immediately after setup.
- **No server-side code**: The entire tool runs client-side. The PAT is used directly from the browser to call the GitHub API.

## Data Format

Predictions schema details are documented in:

- [evaluator/data/PREDICTIONS_SCHEMA.md](data/PREDICTIONS_SCHEMA.md)

Each reviewer's evaluations are stored in `data/reviews/{sanitized_email}.json`:

```json
{
  "reviewer_email": "reviewer@example.com",
  "reviews": [
    {
      "key": "ZOTERO_KEY",
      "title": "Document Title",
      "review_timestamp": "2025-03-01T14:30:00Z",
      "status": "completed",
      "predicted_tags": ["tag1", "tag2", "tag3"],
      "evaluations": {
        "tag1": "correct",
        "tag2": "correct",
        "tag3": "incorrect"
      },
      "missing_tags": ["tag4", "tag5"]
    }
  ]
}
```

## Tag Dictionary

71 tags across 10 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Technology Type | 8 | Land-based Wind, Offshore Wind, PV Solar |
| Data Collection | 4 | New Field Data Collected, No Field Data Used |
| FWS Region | 8 | Northeast, Southeast, Midwest |
| Continent | 7 | North America, Europe, Asia |
| Taxa Group | 11 | Birds, Bats, Vegetation, Fish |
| Special Interest Taxa | 10 | Migratory Tree Bats, Eagles, Pollinators |
| Solar Interaction Summary | 4 | Population-Level Interactions |
| Wind Interaction Summary | 6 | Collisions, Habitat Impacts, Siting |
| General Purpose | 4 | Review Paper, REWI Product or Coauthor |
| Hot Topics | 9 | Curtailment, Fatality, Biodiversity |

## Troubleshooting

- **"Cannot load app configuration"**: The admin hasn't completed setup yet, or the config.json is empty.
- **"Rate limit exceeded"**: Authenticated requests allow 5,000 requests/hour; unauthenticated requests are much lower. Configure a PAT in admin setup for write operations and higher limits.
- **"Write conflict"**: Another reviewer or the admin modified the same file simultaneously. Try saving again — the app fetches the latest version before writing.
- **Reviewer sees no documents**: The admin hasn't assigned documents to their email, or used a different email address.
