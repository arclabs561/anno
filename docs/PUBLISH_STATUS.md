# Publish status

This repo publishes a single crate: `anno`.

## What is publishable?

- **Published / intended to be published**:
  - `anno` (the public facade crate)
- **Not intended to be published**:
  - `anno-core`, `anno-eval`, `anno-cli` (workspace crates used for internal structure/tooling)

## CI checks

The CI workflow runs `cargo publish --dry-run` for the publishable crates and uploads a short markdown report as an artifact named `publish-validation-report`.

## Local checks

From the repo root:

```bash
just validate-publish
```

If you get an unexpected failure, start by running:

```bash
cargo publish --dry-run -p anno
```

