# Publish status

## Current packages

The workspace currently assigns version `0.12.0` to its three crates.io
packages. `anno-py` is a separate PyPI wheel and has `publish = false` for
crates.io.

| Crate | Package name | Publish | Notes |
|-------|-------------|---------|-------|
| `crates/anno` | `anno` | yes | Main library; owns the public extraction API. |
| `crates/anno-eval` | `anno-eval` | yes | Evaluation harnesses and datasets; depends on `anno`. |
| `crates/anno-cli` | `anno-cli` | yes | Command-line interface; depends on `anno` and `anno-eval`. |

The publish order is `anno` → `anno-eval` → `anno-cli`; the version constraints
between these packages must be updated together for a release.

## Publish command

The publish workflow (`.github/workflows/publish.yml`) fires on `v*` tag push and on workflow_dispatch with `confirm=publish`. Authentication uses crates.io trusted publishing (OIDC), no API tokens.

```bash
# Dispatch after pushing the release commit and verifying its CI:
gh workflow run publish.yml --ref main -f confirm=publish
```

Verify all three registry versions before creating and pushing the release tag
at the published commit. Check for existing local and remote tags first; never
reuse a tag from unrelated history. Push only the intended tag. The tag event
reruns the idempotent publish workflow.

The workflow publishes bottom-up by dependency order. Each step uses
`publish-crate.sh`, which treats an already-uploaded version as success so a
rerun can continue after a partial failure.

## Trusted-publisher configuration (crates.io)

Each published crate needs a trusted publisher entry on crates.io with:

- Repository: `arclabs561/anno`
- Workflow: `publish.yml`
- Environment: `crates-io`

If `cargo publish` returns `403 Forbidden: provided access token is not valid for crate <name>`, the entry for that crate is missing or has a mismatched workflow/environment. Fix on https://crates.io/crates/<name>/settings.
