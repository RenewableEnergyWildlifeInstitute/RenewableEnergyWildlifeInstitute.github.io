# Predictions Schema Documentation

## Overview
The `predictions.json` file stores model predictions for each document, including full prediction history with dates and model IDs. Each prediction record represents the accumulated prediction history for a single child document.

## Record Structure

```json
{
  "parent_key": "unique_parent_identifier",
  "child_key": "unique_child_identifier",
  "title": "Document Title",
  "abstract": "Abstract text...",
  "ground_truth_tags": [],
  "predicted_tags": [
    {
      "tag": "tag_name_1",
      "predictions": [
        {
          "predicted_date": "2024-01-15",
          "model_id": "model_v1"
        },
        {
          "predicted_date": "2024-01-16",
          "model_id": "model_v2"
        }
      ]
    },
    {
      "tag": "tag_name_2",
      "predictions": [
        {
          "predicted_date": "2024-01-15",
          "model_id": "model_v1"
        }
      ]
    }
  ]
}
```

## Field Descriptions

### Top-level Fields
- **parent_key** (string): Unique identifier for the parent document (from Zotero or source system)
- **child_key** (string): Unique identifier for the child document/item
- **title** (string): Document title for display in the reviewer interface
- **abstract** (string): Document abstract (optional)
- **ground_truth_tags** (array): Array of verified/ground truth tag strings (populated by reviewers)
- **predicted_tags** (array): Array of tag objects with prediction history (see below)

### Predicted Tags Objects
Each tag object represents a single tag and its complete prediction history:

- **tag** (string): The tag name (e.g., "DataVisualization")
- **predictions** (array): Array of prediction records for this tag
  - **predicted_date** (string): ISO date YYYY-MM-DD when this tag was predicted
  - **model_id** (string): The model version that predicted this tag

## Deduplication & History Behavior

When the pipeline runs and encounters a document (identified by child_key):

1. **First run**: Creates a new prediction record with all predicted tags, each with a single prediction entry (date + model_id)

2. **Subsequent runs**:
   - **New tags**: Added to the record with a single prediction entry
   - **Existing tags**: The new prediction (date + model_id) is APPENDED to the predictions array for that tag

### Example: Tag History Across Runs

**Run 1 (2024-01-15, model_v1):**
- Predicts tags: ["DataVisualization", "Classification"]

**Run 2 (2024-01-16, model_v2):**
- Predicts tags: ["DataVisualization", "NeuralNetworks"]

**Result in predictions.json:**
```json
{
  "child_key": "item_123",
  "predicted_tags": [
    {
      "tag": "DataVisualization",
      "predictions": [
        {"predicted_date": "2024-01-15", "model_id": "model_v1"},
        {"predicted_date": "2024-01-16", "model_id": "model_v2"}
      ]
    },
    {
      "tag": "Classification",
      "predictions": [
        {"predicted_date": "2024-01-15", "model_id": "model_v1"}
      ]
    },
    {
      "tag": "NeuralNetworks",
      "predictions": [
        {"predicted_date": "2024-01-16", "model_id": "model_v2"}
      ]
    }
  ]
}
```

## Benefits of This Structure

1. **Full Auditability**: Track which model predicted which tag and when
2. **Consistency Detection**: Identify tags consistently predicted across model versions (quality signal)
3. **Disagreement Analysis**: Find tags where model versions disagree (potential improvement areas)
4. **Single Source of Truth**: No data loss; all predictions preserved in one record per document
5. **Scalable**: Works with any number of model versions without schema changes

## Frontend Handling

The reviewer interface uses the `extractTagNames()` utility (defined in `js/app.js`) to extract just the tag names from the predictions array when needed:

```javascript
// Gets ["DataVisualization", "Classification", "NeuralNetworks"] from the structure above
const tagNames = extractTagNames(predictedTags);
```

This maintains backward compatibility and keeps the UI logic clean.
