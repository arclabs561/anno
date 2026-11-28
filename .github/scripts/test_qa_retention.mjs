// Run alongside test_research_routing.mjs with the same YAML parser.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { parse } from 'yaml';

for (const [workflow, job, reports] of [
  ['ci', 'eval-sanity', ['reports/eval-sanity-report.json', 'reports/eval-sanity-report.md']],
  ['ci', 'eval-full', ['eval-full-report.md']],
  ['fcoref-cli', 'pipeline', [
    ...['config.json', 'encoder.onnx', 'scorer_weights.safetensors', 'tokenizer.json']
      .map(file => '${{ env.FCOREF_MODEL_PATH }}/' + file),
    '${{ steps.replay.outputs.receipt }}',
  ]],
]) {
  test(`${job} shares only its reviewed QA files`, () => {
    const jobs = parse(readFileSync(`.github/workflows/${workflow}.yml`, 'utf8')).jobs;
    const paths = jobs[job].steps
      .filter(step => /^actions\/(?:cache(?:\/(?:save|restore))?|upload-artifact)@/.test(step.uses ?? ''))
      .flatMap(step => step.with.path.trim().split(/\s*\n\s*/));
    assert.deepEqual(paths.sort(), reports);
  });
}

const ci = parse(readFileSync('.github/workflows/ci.yml', 'utf8'));
const figures = parse(readFileSync('.github/workflows/dataset-figures.yml', 'utf8'));
const defaultRef = 'refs/heads/trunk';

async function eligible(condition, { event = 'push', ref = defaultRef, role = '', bucket = '', inputs = {} } = {}) {
  const { data, Evaluator, Lexer, Parser } = await import('@actions/expressions');
  const github = { event_name: event, ref, event: { repository: { default_branch: 'trunk' } } };
  const vars = { ANNO_AWS_ROLE_ARN: role, ANNO_S3_BUCKET: bucket };
  const context = JSON.parse(JSON.stringify({ github, vars, inputs }), data.reviver);
  const expression = new Parser(new Lexer(condition).lex().tokens, ['github', 'vars', 'inputs'], []).parse();
  return Boolean(new Evaluator(expression, context).evaluate().value);
}

test('generic and AWS evaluation jobs run once with the appropriate permissions', async () => {
  for (const [name, event] of [['test-eval', 'push'], ['matrix-test', 'schedule'], ['eval-sanity', 'push'], ['eval-full', 'workflow_dispatch']]) {
    const normal = ci.jobs[name];
    const trusted = ci.jobs[name + '-trusted'];
    assert.equal(normal.permissions['id-token'], undefined);
    assert.equal(trusted.permissions['id-token'], 'write');
    const setup = normal.steps.find(s => s.uses === './.github/actions/dataset-cache');
    for (const [overrides, expected] of [
      [{}, false], [{role: 'approved'}, false], [{bucket: 'example-data'}, false],
      [{role: 'approved', bucket: 'example-data'}, true],
      [{role: 'approved', bucket: 'example-data', ref: 'refs/heads/eval-testing'}, false],
      [{role: 'approved', bucket: 'example-data', event: 'pull_request', ref: 'refs/pull/42/merge'}, false],
    ]) {
      const invocation = {event, ...overrides};
      assert.equal(await eligible(trusted.if, invocation), expected, name);
      assert.equal(await eligible(setup.if, invocation), expected, name);
      assert.equal(await eligible(normal.if, invocation), !expected && (name !== 'eval-full' || invocation.event === 'workflow_dispatch'), name);
    }
  }
});

test('figure generation is credential-free; each live audit has an independent switch', async () => {
  assert.equal(figures.jobs.reproduce.permissions?.['id-token'], undefined);
  for (const [job, option, other] of [['s3', 'with_cache_audit', 'with_publisher_audit'], ['publisher-files', 'with_publisher_audit', 'with_cache_audit']]) {
    const audit = figures.jobs[job];
    for (const [event, ref, inputs, expected] of [
      ['workflow_dispatch', defaultRef, {}, false],
      ['workflow_dispatch', defaultRef, {[other]: true}, false],
      ['workflow_dispatch', defaultRef, {[option]: true, bucket: 'explicit-dataset-bucket'}, true],
      ['pull_request', defaultRef, {[option]: true}, false],
      ['workflow_dispatch', 'refs/heads/work', {[option]: true}, false],
    ]) {
      assert.equal(await eligible(audit.if, {event, ref, inputs}), expected);
    }
    const called = parse(readFileSync(audit.uses.slice(2), 'utf8'));
    const jobDefinition = Object.values(called.jobs)[0];
    assert.deepEqual(Object.keys(called.on).sort(), ['workflow_call', 'workflow_dispatch']);
    assert.equal(jobDefinition.env.BUCKET, '${{ inputs.bucket || vars.ANNO_S3_BUCKET }}');
    const check = jobDefinition.steps.findIndex(s => s.run?.includes('check_dataset_cache.py'));
    const assume = jobDefinition.steps.findIndex(s => s.uses?.startsWith('aws-actions/'));
    assert.ok(check >= 0 && check < assume);
  }
});

test('shared model and policy checkpoints have no raw input-cache upload', () => {
  const checkpoint = ci.jobs['matrix-test'].steps.find(s => s.name === 'Publish trusted matrix checkpoint');
  assert.deepEqual(checkpoint.with.path.trim().split(/\s*\n\s*/).map(p => p.split('/').pop()).sort(),
    ['eval-history.db', 'eval-results.jsonl', 'linucb_global_state.json', 'muxer_history.*.json']);
  const model = parse(readFileSync('.github/workflows/native-checkpoint-replay.yml', 'utf8'));
  const runs = model.jobs.replay.steps.filter(s => s.run?.includes('--receipt '));
  assert.equal(runs.length, 2);
  assert.ok(runs.every(s => s.run.includes('--aggregate --receipt')));
  for (const job of Object.values(ci.jobs)) {
    for (const step of job.steps ?? []) {
      if (/^actions\/cache(?:\/(?:save|restore))?@/.test(step.uses ?? '')) {
        assert.ok(step.with.path.trim().split(/\s*\n\s*/).every(path => path.startsWith('~/.cargo/bin/')));
      }
    }
  }
});


test('destination validation rejects missing roles, invalid bucket names and wildcard resource policies', () => {
  for (const [bucket, role, ok] of [
    ['my-test-cache', 'approved', true], ['', 'approved', false], ['my-test-cache', '', false],
    ['bucket/*', 'approved', false], ['bad..name', 'approved', false], ['127.0.0.1', 'approved', false],
    ['bucket\nprivate-marker', 'approved', false],
  ]) {
    const checked = spawnSync('python3', ['.github/scripts/check_dataset_cache.py'], {
      env: {...process.env, BUCKET: bucket, CACHE_ROLE: role}, encoding: 'utf8',
    });
    assert.equal(checked.status === 0, ok);
    assert.ok(!checked.stderr.includes('private-marker'));
  }
});
