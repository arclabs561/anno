# Publish status

## Current packages

The three crates.io packages share one release version. `anno-py` is a separate PyPI wheel and has `publish = false` for
crates.io.

| Crate | Package name | Publish | Notes |
|-------|-------------|---------|-------|
| `crates/anno` | `anno` | yes | Main library; owns the public extraction API. |
| `crates/anno-eval` | `anno-eval` | yes | Evaluation harnesses and datasets; depends on `anno`. |
| `crates/anno-cli` | `anno-cli` | yes | Command-line interface; depends on `anno` and `anno-eval`. |

The publish order is `anno` → `anno-eval` → `anno-cli`; the version constraints
between these packages must be updated together for a release.

## Publish command

The publish workflow (`.github/workflows/publish.yml`) fires on `v*` tag push and on workflow_dispatch from `main` with `confirm=publish`. Authentication uses crates.io trusted publishing (OIDC), no API tokens.

```bash
# Dispatch after pushing the release commit and verifying its CI:
gh workflow run publish.yml --ref main -f confirm=publish
```

Verify all three registry versions before creating and pushing the release tag
at the published commit. Check for existing local and remote tags first; never
reuse a tag from unrelated history. Push only the intended tag. The tag event
reruns the idempotent publish workflow.

A read-only job checks that the three package versions agree and that any release
tag matches that exact version before authentication. The publish job packages
each crate from the locked checkout in dependency order. It confirms the SHA-256
of the candidate archive against the crates.io index before advancing. A retry
can reuse an existing, unyanked version with identical bytes. A mismatch stops
the release and identifies the crate to investigate.

## Trusted-publisher configuration (crates.io)

Each published crate needs a trusted publisher entry on crates.io with:

- Repository: `arclabs561/anno`
- Workflow: `publish.yml`
- Environment: `crates-io`

If `cargo publish` returns `403 Forbidden: provided access token is not valid for crate <name>`, the entry for that crate is missing or has a mismatched workflow/environment. Fix on https://crates.io/crates/<name>/settings.
