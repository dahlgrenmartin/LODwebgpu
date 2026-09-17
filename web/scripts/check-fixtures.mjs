#!/usr/bin/env node
/**
 * Verify every model file the manifest references is actually present.
 *
 * vite copies public/ without complaint when the fixtures are absent, so the
 * build stays green and the published site 404s on every weight. This runs
 * before the build in CI, and can be run by hand:
 *
 *   node scripts/check-fixtures.mjs public/models
 *   node scripts/check-fixtures.mjs dist/models
 */
import fs from 'node:fs';
import path from 'node:path';

const dir = process.argv[2] ?? 'public/models';
const manifestPath = path.join(dir, 'manifest.json');

if (!fs.existsSync(manifestPath)) {
  console.error(`::error::${manifestPath} is missing - no fixtures in this tree`);
  process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

// `weights` is a list once the blob is sharded to stay under 100 MB, and a bare
// string in older manifests. Treating the array as a string silently builds one
// bogus comma-joined path and reports every shard as missing.
const weights = Array.isArray(manifest.weights)
  ? manifest.weights
  : [manifest.weights];

const required = [
  ...manifest.resolutions.map((r) => r.graph),
  ...weights,
  manifest.encoder,
  manifest.encoderWeights,
  'lod_graph.json',
  'lod_graph.weights.bin',
].filter(Boolean);

const missing = required.filter((f) => !fs.existsSync(path.join(dir, f)));
if (missing.length) {
  console.error(`::error::model fixtures missing from ${dir}: ${missing.join(', ')}`);
  process.exit(1);
}

const bytes = required.reduce(
  (n, f) => n + fs.statSync(path.join(dir, f)).size, 0);
const oversized = required
  .map((f) => [f, fs.statSync(path.join(dir, f)).size])
  .filter(([, n]) => n > 100 * 1024 * 1024);

if (oversized.length) {
  for (const [f, n] of oversized) {
    console.error(`::error::${f} is ${(n / 1048576).toFixed(1)} MB, over GitHub's 100 MB limit`);
  }
  process.exit(1);
}

console.log(
  `${required.length} fixtures present in ${dir}, ` +
  `${(bytes / 1048576).toFixed(0)} MB, largest under 100 MB`);
