// Run alongside test_research_routing.mjs with the same YAML parser.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { parse } from 'yaml';

for (const [workflow, job, reports] of [
  ['ci', 'eval-sanity', ['reports/eval-sanity-report.json', 'reports/eval-sanity-report.md']],
  ['ci', 'eval-full', ['eval-full-report.md']],
  ['fcoref-cli', 'pipeline', ['${{ steps.replay.outputs.receipt }}']],
]) {
  test(`${job} shares only its reviewed QA reports`, () => {
    const jobs = parse(readFileSync(`.github/workflows/${workflow}.yml`, 'utf8')).jobs;
    const paths = jobs[job].steps
      .filter(step => /^actions\/(?:cache(?:\/(?:save|restore))?|upload-artifact)@/.test(step.uses ?? ''))
      .flatMap(step => step.with.path.trim().split(/\s*\n\s*/));
    assert.deepEqual(paths.sort(), reports);
  });
}
