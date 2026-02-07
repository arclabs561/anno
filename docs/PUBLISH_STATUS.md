# Publish status

This repo is **not** doing crates.io publishing as its primary release mechanism right now.

We treat Git tags + GitHub Releases as the release log.

## What is publishable?

- **Potentially publishable (later)**:
  - `anno` (the facade crate)
- **Not intended to be published**:
  - `anno-core`, `anno-metrics`, `anno-eval`, `anno-cli` (workspace crates used for internal structure/tooling)

## Why publishing is paused

Today the `anno` facade depends on workspace-only crates via path dependencies
(notably `anno-lib` and `anno-core`).
Crates.io publishing would therefore require either:

- publishing the internal crates too (which expands the public semver surface), or
- collapsing the split back into a single publishable crate.

We’re explicitly choosing neither for now.

## CI checks

The CI workflow runs a best-effort “publish validation” job (currently expected to fail) and
uploads a short markdown report as `publish-validation-report`.

## Local checks

If you want to create a release tag, follow `docs/TAGS.md`.

