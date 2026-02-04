# Publish status

This repo is **not** doing crates.io publishing as its primary checkpoint mechanism right now.

We treat Git tags + GitHub Releases as the “checkpoint log”.

## What is publishable?

- **Potentially publishable (later)**:
  - `anno` (the facade crate)
- **Not intended to be published**:
  - `anno-core`, `anno-eval`, `anno-cli` (workspace crates used for internal structure/tooling)

## Why publishing is paused

Today `anno` depends on `anno-core` as a workspace crate (`path = "../anno-core"`).
Crates.io publishing would therefore require either:

- publishing `anno-core` too (which expands the public semver surface), or
- collapsing `anno-core` back into `anno`.

We’re explicitly choosing neither for now.

## CI checks

The CI workflow runs a best-effort “publish validation” job (currently expected to fail) and
uploads a short markdown report as `publish-validation-report`.

## Local checks

If you want a checkpoint, follow `docs/CHECKPOINTS.md`.

