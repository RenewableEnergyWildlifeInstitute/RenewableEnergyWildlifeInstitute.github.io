/**
 * Export and aggregation utilities for the admin view.
 */

/** Load all reviewer files from data/reviews/ */
async function loadAllReviews() {
  const files = await githubAPI.listFiles('evaluator/data/reviews');
  const allReviews = [];

  for (const file of files) {
    if (!file.name.endsWith('.json')) continue;
    try {
      const data = await githubAPI.readFile(file.path);
      if (data && data.reviews) {
        allReviews.push(data);
      }
    } catch (err) {
      console.warn(`Failed to load ${file.name}:`, err);
    }
  }

  return allReviews;
}

/**
 * Compute the validated tags for a single review record.
 * Validated = ground truth tags ∪ correctly-predicted tags ∪ reviewer-added missing tags.
 */
function computeValidatedTags(record, groundTruthTags) {
  const validated = new Set(groundTruthTags || []);
  for (const [tag, verdict] of Object.entries(record.evaluations || {})) {
    if (verdict === 'correct') validated.add(tag);
  }
  for (const tag of (record.missing_tags || [])) {
    validated.add(tag);
  }
  return [...validated].sort();
}

/** Aggregate all reviews into a flat array for export.
 *  predictionsMap: optional { key -> prediction object } for ground_truth_tags lookup. */
function aggregateReviews(allReviewerData, predictionsMap = {}) {
  const records = [];

  for (const reviewerData of allReviewerData) {
    for (const review of reviewerData.reviews) {
      const groundTruthTags = (predictionsMap[review.key] || {}).ground_truth_tags || [];
      records.push({
        reviewer_email: reviewerData.reviewer_email,
        key: review.key,
        title: review.title,
        status: review.status,
        review_timestamp: review.review_timestamp,
        predicted_tags: review.predicted_tags,
        evaluations: review.evaluations,
        missing_tags: review.missing_tags || [],
        validated_tags: computeValidatedTags(review, groundTruthTags)
      });
    }
  }

  return records;
}

/**
 * Build a per-document summary of validated tags from completed reviews.
 * Unions validated_tags across all reviewers for the same document.
 */
function buildDocumentSummary(records) {
  const docMap = {};
  for (const r of records) {
    if (r.status !== 'completed') continue;
    if (!docMap[r.key]) {
      docMap[r.key] = { key: r.key, title: r.title, validatedSet: new Set() };
    }
    for (const tag of (r.validated_tags || [])) {
      docMap[r.key].validatedSet.add(tag);
    }
  }
  return Object.values(docMap).map(d => ({
    key: d.key,
    title: d.title,
    validated_tags: [...d.validatedSet].sort()
  }));
}

/** Compute metrics from aggregated reviews */
function computeMetrics(records) {
  let totalCorrect = 0;
  let totalPredicted = 0;
  let totalMissing = 0;

  const perTag = {};
  const perCategory = {};

  for (const record of records) {
    if (record.status !== 'completed') continue;

    const evals = record.evaluations || {};
    const missing = record.missing_tags || [];

    for (const [tag, verdict] of Object.entries(evals)) {
      totalPredicted++;
      if (verdict === 'correct') totalCorrect++;

      // Per-tag
      if (!perTag[tag]) perTag[tag] = { correct: 0, incorrect: 0, missing: 0 };
      if (verdict === 'correct') perTag[tag].correct++;
      else perTag[tag].incorrect++;

      // Per-category
      const cat = TAG_TO_CATEGORY[tag] || 'Unknown';
      if (!perCategory[cat]) perCategory[cat] = { correct: 0, incorrect: 0, missing: 0 };
      if (verdict === 'correct') perCategory[cat].correct++;
      else perCategory[cat].incorrect++;
    }

    for (const tag of missing) {
      totalMissing++;
      if (!perTag[tag]) perTag[tag] = { correct: 0, incorrect: 0, missing: 0 };
      perTag[tag].missing++;

      const cat = TAG_TO_CATEGORY[tag] || 'Unknown';
      if (!perCategory[cat]) perCategory[cat] = { correct: 0, incorrect: 0, missing: 0 };
      perCategory[cat].missing++;
    }
  }

  const precision = totalPredicted > 0 ? totalCorrect / totalPredicted : 0;
  const recall = (totalCorrect + totalMissing) > 0 ? totalCorrect / (totalCorrect + totalMissing) : 0;
  const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

  return {
    overall: { precision, recall, f1, totalCorrect, totalPredicted, totalMissing },
    perTag,
    perCategory
  };
}

/** Export aggregated reviews as JSON and trigger download */
function exportJSON(allReviewerData, predictionsMap = {}) {
  const records = aggregateReviews(allReviewerData, predictionsMap);
  const metrics = computeMetrics(records);

  const output = {
    exported_at: new Date().toISOString(),
    overall_metrics: metrics.overall,
    per_category_metrics: metrics.perCategory,
    per_tag_metrics: metrics.perTag,
    reviews: records
  };

  downloadFile(JSON.stringify(output, null, 2), 'rewi_evaluations_export.json', 'application/json');
}

/** Export aggregated reviews as CSV and trigger download */
function exportCSV(allReviewerData, predictionsMap = {}) {
  const records = aggregateReviews(allReviewerData, predictionsMap);

  const rows = [['reviewer_email', 'key', 'title', 'status', 'review_timestamp', 'predicted_tags', 'evaluations_correct', 'evaluations_incorrect', 'missing_tags', 'validated_tags']];

  for (const r of records) {
    const evals = r.evaluations || {};
    const correct = Object.entries(evals).filter(([, v]) => v === 'correct').map(([k]) => k);
    const incorrect = Object.entries(evals).filter(([, v]) => v === 'incorrect').map(([k]) => k);

    rows.push([
      r.reviewer_email,
      r.key,
      `"${(r.title || '').replace(/"/g, '""')}"`,
      r.status,
      r.review_timestamp,
      `"${extractTagNames(r.predicted_tags).join('; ')}"`,
      `"${correct.join('; ')}"`,
      `"${incorrect.join('; ')}"`,
      `"${(r.missing_tags || []).join('; ')}"`,
      `"${(r.validated_tags || []).join('; ')}"`
    ]);
  }

  const csv = rows.map(r => r.join(',')).join('\n');
  downloadFile(csv, 'rewi_evaluations_export.csv', 'text/csv');
}

/** Trigger a file download in the browser */
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
