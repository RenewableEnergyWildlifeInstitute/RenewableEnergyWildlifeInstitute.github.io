# REWI Tag Evaluation Tool

A static GitHub Pages tool for REWI domain experts to manually review and evaluate metadata tags predicted by our LLM pipeline. All data is persisted as JSON files in the GitHub repo via the GitHub REST API.

## Architecture

```
evaluator/
├── index.html              # Entry point — routes to admin or reviewer view
├── admin.html              # Admin dashboard (setup, upload, assign, export)
├── review.html             # Reviewer interface (evaluate tags per document)
├── js/
│   ├── app.js              # Core logic, config loading, CSV parsing
│   ├── github.js           # GitHub API wrapper (read/write files)
│   ├── tags.js             # Tag dictionary (71 tags, 10 categories)
│   └── export.js           # Export/aggregation utilities
├── css/
│   └── styles.css          # Custom styles
├── data/                   # Data directory (read/written via GitHub API)
│   ├── config.json         # App config (repo owner, repo name, PAT)
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
   - Enter the repository **Name** (e.g., `UVA-Capstone-Su-25`)
   - Paste the **Personal Access Token**
   - Click **Test Connection** to verify
   - Click **Save Configuration**
   - The PAT is stored only in your browser's local storage and is **not** written to the repository.
4. Change the admin passphrase to something secure.

### 3. Upload Predictions Data

1. Go to the **Predictions** tab in the admin dashboard.
2. Upload a CSV or JSON file. Required columns:
   - `key` — unique document identifier (Zotero key)
   - `title` — document title
   - `predicted_tags` — list of predicted tags (JSON array, semicolon-separated, or comma-separated)
3. Optional columns:
   - `abstract` — document abstract
   - `full_text` — full text of the document
   - `ground_truth_tags` — existing ground-truth tags
4. Click **Upload & Save to Repo**.

**Example CSV:**
```csv
key,title,abstract,predicted_tags,ground_truth_tags
ABC123,"Wind Farm Impact Study","This study examines...","['Land-based Wind', 'Birds', 'Collisions']","['Land-based Wind', 'Birds']"
DEF456,"Solar Panel Effects","An analysis of...","PV Solar; Vegetation; Pollinators","PV Solar; Vegetation"
```

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

## Data Format

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
