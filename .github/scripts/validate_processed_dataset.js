#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
      continue;
    }
    args[key] = next;
    i += 1;
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function isNonEmptyString(v) {
  return typeof v === 'string' && v.trim() !== '';
}

function mergeValues(current, incoming, fieldName) {
  if (current == null || current === '') return incoming;
  if (incoming == null || incoming === '') return current;

  if (Array.isArray(current) && Array.isArray(incoming)) {
    const seen = new Set(current.map((x) => JSON.stringify(x)));
    const merged = current.slice();
    for (const item of incoming) {
      const key = JSON.stringify(item);
      if (!seen.has(key)) {
        seen.add(key);
        merged.push(item);
      }
    }
    return merged;
  }

  if (
    typeof current === 'object' &&
    typeof incoming === 'object' &&
    !Array.isArray(current) &&
    !Array.isArray(incoming)
  ) {
    const merged = { ...current };
    for (const [k, v] of Object.entries(incoming)) {
      if (merged[k] == null || merged[k] === '') merged[k] = v;
    }
    return merged;
  }

  if (typeof current === 'string' && typeof incoming === 'string') {
    if (current === incoming) return current;
    if (fieldName === 'full_text' || fieldName === 'abstract') {
      return incoming.length > current.length ? incoming : current;
    }
    return current.length >= incoming.length ? current : incoming;
  }

  return current;
}

function mergeDuplicateChildRows(rows) {
  const grouped = new Map();
  const output = [];
  let duplicateRowsMerged = 0;

  for (const row of rows) {
    const childKey = isNonEmptyString(row.child_key) ? row.child_key.trim() : '';
    if (!childKey) {
      output.push(row);
      continue;
    }

    if (!grouped.has(childKey)) {
      const copy = { ...row, child_key: childKey };
      grouped.set(childKey, copy);
      output.push(copy);
      continue;
    }

    duplicateRowsMerged += 1;
    const existing = grouped.get(childKey);
    for (const [field, value] of Object.entries(row)) {
      if (field === 'child_key') continue;
      existing[field] = mergeValues(existing[field], value, field);
    }
  }

  return { mergedRows: output, duplicateRowsMerged };
}

function validateAgainstSnapshot(processedRows, snapshotKeys) {
  const missingParent = [];
  const missingChild = [];

  for (const row of processedRows) {
    const parent = isNonEmptyString(row.parent_key) ? row.parent_key.trim() : '';
    const child = isNonEmptyString(row.child_key) ? row.child_key.trim() : '';

    if (parent && !snapshotKeys.has(parent)) {
      missingParent.push(parent);
    }
    if (child && !snapshotKeys.has(child)) {
      missingChild.push(child);
    }
  }

  const uniqueMissingParent = [...new Set(missingParent)].sort();
  const uniqueMissingChild = [...new Set(missingChild)].sort();

  return {
    uniqueMissingParent,
    uniqueMissingChild
  };
}

function main() {
  const args = parseArgs(process.argv);
  const snapshotPath = args.snapshot || 'evaluator/data/zotero_snapshot.json';
  const processedPath = args.processed || 'evaluator/data/default_final_output.json';
  const shouldWrite = Boolean(args.write);

  const snapshot = readJson(snapshotPath);
  const processed = readJson(processedPath);

  if (!snapshot || !Array.isArray(snapshot.items)) {
    throw new Error(`Snapshot file must contain an 'items' array: ${snapshotPath}`);
  }
  if (!Array.isArray(processed)) {
    throw new Error(`Processed file must be a JSON array: ${processedPath}`);
  }

  const snapshotKeys = new Set(
    snapshot.items
      .map((item) => (item && item.key ? String(item.key).trim() : ''))
      .filter(Boolean)
  );

  const { mergedRows, duplicateRowsMerged } = mergeDuplicateChildRows(processed);
  if (shouldWrite && duplicateRowsMerged > 0) {
    writeJson(processedPath, mergedRows);
  }

  const { uniqueMissingParent, uniqueMissingChild } = validateAgainstSnapshot(mergedRows, snapshotKeys);

  console.log(
    JSON.stringify(
      {
        processed_rows_before: processed.length,
        processed_rows_after: mergedRows.length,
        duplicate_rows_merged: duplicateRowsMerged,
        missing_parent_keys: uniqueMissingParent.length,
        missing_child_keys: uniqueMissingChild.length
      },
      null,
      2
    )
  );

  if (uniqueMissingParent.length > 0 || uniqueMissingChild.length > 0) {
    if (uniqueMissingParent.length > 0) {
      console.error('Missing parent keys in Zotero snapshot (sample):', uniqueMissingParent.slice(0, 20).join(', '));
    }
    if (uniqueMissingChild.length > 0) {
      console.error('Missing child keys in Zotero snapshot (sample):', uniqueMissingChild.slice(0, 20).join(', '));
    }
    process.exit(1);
  }

  console.log('Validation passed.');
}

main();
