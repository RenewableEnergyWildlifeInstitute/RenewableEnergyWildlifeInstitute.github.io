/**
 * Core application logic — shared utilities for admin and reviewer views.
 */

/** Load GitHub API config from data/config.json or localStorage fallback */
async function loadConfig() {
  // Try localStorage first for speed
  const cached = localStorage.getItem('rewi_config');
  if (cached) {
    try {
      const config = JSON.parse(cached);
      if (config.owner && config.repo && config.token) {
        githubAPI.configure(config.owner, config.repo, config.token);
        return config;
      }
    } catch { /* ignore bad cache */ }
  }
  return null;
}

/** Save config to both localStorage and the repo */
async function saveConfig(owner, repo, token) {
  const config = { owner, repo, token };
  localStorage.setItem('rewi_config', JSON.stringify(config));
  githubAPI.configure(owner, repo, token);

  // Write config to repo (without the token for safety — token stays in localStorage only)
  // Actually, per spec, we store a write-enabled token in config.json for reviewers
  await githubAPI.writeFile('evaluator/data/config.json', {
    owner,
    repo,
    token
  }, 'Update app configuration');

  return config;
}

/**
 * Initialize the app for a reviewer.
 * Loads config from the repo's config.json (no admin setup needed on their end).
 */
async function initForReviewer() {
  // First, try to determine repo info from the current page URL
  // GitHub Pages URL format: https://<owner>.github.io/<repo>/
  const hostname = window.location.hostname;
  const pathname = window.location.pathname;

  let owner, repo;

  // Check localStorage first
  const cached = localStorage.getItem('rewi_config');
  if (cached) {
    try {
      const config = JSON.parse(cached);
      if (config.owner && config.repo && config.token) {
        githubAPI.configure(config.owner, config.repo, config.token);
        return config;
      }
    } catch { /* ignore */ }
  }

  // Try to infer from GitHub Pages URL
  if (hostname.endsWith('.github.io')) {
    owner = hostname.replace('.github.io', '');
    // Repo name is the first path segment
    const segments = pathname.split('/').filter(Boolean);
    repo = segments[0] || '';
  }

  if (!owner || !repo) {
    throw new Error('Cannot determine repo from URL. Please contact your admin.');
  }

  let config = null;

  // Strategy 1: fetch config.json via the GitHub Pages URL.
  // This works for both public AND private repos because GitHub Pages always
  // serves the static files publicly, regardless of repo visibility.
  try {
    const pagesUrl = `https://${owner}.github.io/${repo}/evaluator/data/config.json`;
    const pagesRes = await fetch(pagesUrl, { cache: 'no-cache' });
    if (pagesRes.ok) {
      config = await pagesRes.json();
    }
  } catch { /* ignore, fall through to API */ }

  // Strategy 2: fall back to the GitHub REST API (works for public repos).
  if (!config) {
    const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/evaluator/data/config.json`;
    const apiRes = await fetch(apiUrl, {
      headers: { 'Accept': 'application/vnd.github.v3+json' }
    });

    if (!apiRes.ok) {
      throw new Error('Cannot load app configuration. Please contact your admin to set up the tool.');
    }

    const data = await apiRes.json();
    const content = atob(data.content.replace(/\n/g, ''));
    const bytes = new Uint8Array(content.length);
    for (let i = 0; i < content.length; i++) {
      bytes[i] = content.charCodeAt(i);
    }
    config = JSON.parse(new TextDecoder('utf-8').decode(bytes));
  }

  // Cache it locally for subsequent page loads
  localStorage.setItem('rewi_config', JSON.stringify(config));
  githubAPI.configure(config.owner, config.repo, config.token);
  return config;
}

/** Load predictions from the repo */
async function loadPredictions() {
  const cached = localStorage.getItem('rewi_predictions');
  let predictions = null;

  try {
    predictions = await githubAPI.readFile('evaluator/data/predictions.json');
    if (predictions) {
      localStorage.setItem('rewi_predictions', JSON.stringify(predictions));
    }
  } catch (err) {
    console.warn('Failed to load predictions from GitHub, using cache:', err);
    if (cached) predictions = JSON.parse(cached);
  }

  return Array.isArray(predictions) ? predictions.map(normalizePredictionRecord) : [];
}

/** Load assignments from the repo */
async function loadAssignments() {
  const cached = localStorage.getItem('rewi_assignments');
  let assignments = null;

  try {
    assignments = await githubAPI.readFile('evaluator/data/assignments.json');
    if (assignments) {
      localStorage.setItem('rewi_assignments', JSON.stringify(assignments));
    }
  } catch (err) {
    console.warn('Failed to load assignments from GitHub, using cache:', err);
    if (cached) assignments = JSON.parse(cached);
  }

  return assignments || {};
}

/** Load a reviewer's evaluations */
async function loadReviewerData(email) {
  const safeEmail = sanitizeEmail(email);
  const cacheKey = `rewi_reviews_${safeEmail}`;
  const cached = localStorage.getItem(cacheKey);
  let reviews = null;

  try {
    reviews = await githubAPI.readFile(`evaluator/data/reviews/${safeEmail}.json`);
    if (reviews) {
      localStorage.setItem(cacheKey, JSON.stringify(reviews));
    }
  } catch (err) {
    console.warn('Failed to load reviews from GitHub, using cache:', err);
    if (cached) reviews = JSON.parse(cached);
  }

  const reviewerData = reviews || { reviewer_email: email, reviews: [] };
  reviewerData.reviews = Array.isArray(reviewerData.reviews)
    ? reviewerData.reviews.map(normalizeReviewRecord)
    : [];
  return reviewerData;
}

/** Save a reviewer's evaluations to GitHub */
async function saveReviewerData(email, data) {
  const safeEmail = sanitizeEmail(email);
  const cacheKey = `rewi_reviews_${safeEmail}`;

  // Save to localStorage immediately
  localStorage.setItem(cacheKey, JSON.stringify(data));

  // Write to GitHub
  await githubAPI.writeFile(
    `evaluator/data/reviews/${safeEmail}.json`,
    data,
    `Update reviews for ${email}`
  );
}

/** Convert email to a safe filename */
function sanitizeEmail(email) {
  return email.toLowerCase().replace(/[^a-z0-9]/g, '_');
}

/** Parse a CSV string into an array of objects */
function parseCSV(csvText) {
  const lines = csvText.split('\n').filter(line => line.trim());
  if (lines.length < 2) throw new Error('CSV must have a header row and at least one data row.');

  // Parse header
  const headers = parseCSVLine(lines[0]);

  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length === 0) continue;

    const record = {};
    headers.forEach((header, idx) => {
      let val = values[idx] || '';
      // Try to parse arrays (e.g., "['tag1', 'tag2']" or "tag1; tag2")
      if (header === 'predicted_tags' || header === 'ground_truth_tags') {
        val = parseTagList(val);
      }
      record[header.trim()] = val;
    });
    records.push(record);
  }

  return records;
}

/** Parse a single CSV line, handling quoted fields */
function parseCSVLine(line) {
  const fields = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === ',' && !inQuotes) {
      fields.push(current.trim());
      current = '';
    } else {
      current += ch;
    }
  }
  fields.push(current.trim());
  return fields;
}

/** Parse a tag list from various formats */
function parseTagList(val) {
  if (Array.isArray(val)) return val;
  if (!val || val === '[]' || val === '') return [];

  // Try JSON array
  try {
    const parsed = JSON.parse(val.replace(/'/g, '"'));
    if (Array.isArray(parsed)) return parsed.map(t => t.trim());
  } catch { /* not JSON */ }

  // Try semicolon-separated
  if (val.includes(';')) return val.split(';').map(t => t.trim()).filter(Boolean);

  // Try comma-separated (but be careful with tags that contain commas)
  if (val.includes(',')) return val.split(',').map(t => t.trim()).filter(Boolean);

  // Single tag
  return [val.trim()];
}

function normalizePredictionRecord(record) {
  const parentKey = String(record?.parent_key || record?.key || '').trim();
  const childKey = String(record?.child_key || '').trim();
  const documentKey = String(record?.document_key || childKey || parentKey).trim();

  return {
    ...record,
    key: parentKey,
    parent_key: parentKey,
    child_key: childKey,
    document_key: documentKey,
    title: String(record?.title || record?.child_title || '').trim(),
    child_title: String(record?.child_title || '').trim(),
    abstract: typeof record?.abstract === 'string' ? record.abstract : '',
    full_text: typeof record?.full_text === 'string' ? record.full_text : '',
    predicted_tags: parseTagList(record?.predicted_tags),
    ground_truth_tags: parseTagList(record?.ground_truth_tags)
  };
}

function normalizeReviewRecord(record) {
  const parentKey = String(record?.parent_key || '').trim();
  const childKey = String(record?.child_key || '').trim();
  const documentKey = String(record?.document_key || childKey || record?.key || parentKey).trim();

  return {
    ...record,
    key: documentKey,
    parent_key: parentKey,
    child_key: childKey,
    document_key: documentKey
  };
}

function getPredictionDocumentKey(prediction) {
  return String(prediction?.document_key || prediction?.child_key || prediction?.key || prediction?.parent_key || '').trim();
}

function getPredictionParentKey(prediction) {
  return String(prediction?.parent_key || prediction?.key || '').trim();
}

function getReviewDocumentKey(review) {
  return String(review?.document_key || review?.child_key || review?.key || review?.parent_key || '').trim();
}

function predictionMatchesDocumentKey(prediction, documentKey) {
  const normalizedKey = String(documentKey || '').trim();
  if (!normalizedKey) return false;
  return getPredictionDocumentKey(prediction) === normalizedKey
    || getPredictionParentKey(prediction) === normalizedKey;
}

/** Group tags by category */
function groupTagsByCategory(tags) {
  const groups = {};
  for (const tag of tags) {
    const cat = TAG_TO_CATEGORY[tag] || 'Unknown';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(tag);
  }
  return groups;
}

/** Show a toast notification */
function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

/** Show/hide a loading spinner on a button */
function setButtonLoading(btn, loading) {
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.textContent = 'Loading...';
    btn.disabled = true;
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
    btn.disabled = false;
  }
}
