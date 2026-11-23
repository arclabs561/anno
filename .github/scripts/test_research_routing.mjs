// From the repository root, install test-only dependencies in the ignored Cargo target directory:
// npm install --prefix target/routing --no-save --no-package-lock --ignore-scripts @actions/expressions@0.3.61 yaml@2.8.4
// cp .github/scripts/test_research_routing.mjs target/routing/ && node target/routing/test_research_routing.mjs
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { data, Evaluator, Lexer, Parser } from '@actions/expressions';
import { parse } from 'yaml';

const workflow = parse(readFileSync('.github/workflows/research.yml', 'utf8'));
const job = workflow.jobs.experiment;
const selector = job.steps.find(step => step.id === 'route');
const source = {
  repository_id: '1104910992',
  event_name: 'push',
  event: { deleted: false },
  ref_type: 'branch',
  ref: 'refs/heads/research/offsets',
};
const gate = new Parser(new Lexer(job.if).lex().tokens, ['github'], []).parse();

const cases = [
  ['research push', {}, 'research'],
  ['nested mutation push', { ref: 'refs/heads/mutation/spans/union' }, 'mutation'],
  ['selected research branch', { event_name: 'workflow_dispatch' }, 'research'],
  ['selected mutation branch', { event_name: 'workflow_dispatch', ref: 'refs/heads/mutation/spans' }, 'mutation'],
  ['deleted branch', { event: { deleted: true } }],
  ['fork push', { repository_id: '42' }],
  ['fork dispatch', { repository_id: '42', event_name: 'workflow_dispatch' }],
  ['fork PR to main', { event_name: 'pull_request', ref: 'refs/pull/3/merge', head_ref: 'mutation/spans', base_ref: 'main' }],
  ['same-repository PR', { event_name: 'pull_request', ref: 'refs/pull/9/merge', head_ref: 'research/m', base_ref: 'main' }],
  ['PR targeting a research branch', { event_name: 'pull_request_target', base_ref: 'research/offsets' }],
  ['selected main branch with unrelated head', { event_name: 'workflow_dispatch', ref: 'refs/heads/main', head_ref: 'research/offsets' }],
  ['selected tag', { event_name: 'workflow_dispatch', ref_type: 'tag', ref: 'refs/tags/research/v1' }],
  ['unrelated prefix', { ref: 'refs/heads/mutation-fix' }],
  ['embedded prefix', { ref: 'refs/heads/fix/research/offsets' }],
  ['legacy eval branch', { ref: 'refs/heads/eval-large' }],
  ['case variation on dispatch', { event_name: 'workflow_dispatch', ref: 'refs/heads/Mutation/spans' }],
];

for (const [description, fields, expected] of cases) {
  test(description, () => {
    const github = { ...source, ...fields };
    const context = JSON.parse(JSON.stringify({ github }), data.reviver);
    if (!new Evaluator(gate, context).evaluate().value) {
      assert.equal(expected, undefined);
      return;
    }
    const directory = mkdtempSync(join(tmpdir(), 'anno-route-'));
    try {
      const output = join(directory, 'result');
      const result = spawnSync('bash', ['-eu', '-c', selector.run], {
        env: { GITHUB_REF: github.ref, GITHUB_OUTPUT: output },
        encoding: 'utf8',
      });
      assert.equal(result.status === 0, expected !== undefined, result.stderr);
      const route = existsSync(output) ? readFileSync(output, 'utf8').trim() : '';
      assert.equal(route, expected ? 'lane=' + expected : '');
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
}

test('runner, permissions and concurrent cost stay bounded', () => {
  assert.deepEqual(workflow.on, { push: { branches: ['research/**', 'mutation/**'] }, workflow_dispatch: null });
  assert.deepEqual(job.concurrency, { group: 'anno-optional-research', 'cancel-in-progress': false });
  assert.deepEqual(workflow.permissions, { contents: 'read' });
  assert.deepEqual(Object.keys(workflow.jobs), ['experiment']);
  assert.equal(job['runs-on'], 'ubuntu-24.04');
  assert.ok(job['timeout-minutes'] > 0 && job['timeout-minutes'] <= 30);
  assert.equal(job.permissions, undefined);
  assert.equal(job.environment, undefined);
  assert.equal(job.strategy, undefined);
  const actions = job.steps.filter(step => step.uses);
  assert.equal(actions.length, 1);
  assert.match(actions[0].uses, /^actions\/checkout@[0-9a-f]{40}$/);
  assert.equal(actions[0].with['persist-credentials'], false);
  for (const step of job.steps.filter(s => s.run && s !== selector)) {
    const lane = step.name.includes('coreference') ? 'research' : 'mutation';
    assert.equal(step.if, "steps.route.outputs.lane == '" + lane + "'");
  }
});
