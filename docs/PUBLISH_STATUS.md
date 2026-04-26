# Publish status

## Current state (post-2026-04-26 Phase B, unreleased)

Three crates publish to crates.io. Phase B (2026-04-26) folded `anno-core`, `anno-metrics`, and `anno-graph` back into `anno`.

| Crate | Package name | Publish | Notes |
|-------|-------------|---------|-------|
| `crates/anno` | `anno` | yes | The main library. Includes the type foundation (`anno::core`), coref scoring (`anno::metrics`, behind `analysis`), and KG export (`anno::graph`, behind `graph`). |
| `crates/anno-eval` | `anno-eval` | yes | Evaluation harnesses, datasets, muxer sampling. |
| `crates/anno-cli` | `anno-cli` | yes | Full CLI binary. `cargo install anno-cli`. |

The legacy `anno-core 0.8.0`, `anno-metrics 0.8.0`, and `anno-graph 0.8.0` remain frozen on crates.io. Users on those crates keep working until they upgrade; the next `anno` release exposes the equivalent surface at `anno::core::*`, `anno::metrics::*`, and `anno::graph::*`.

## Publish command

The publish workflow (`.github/workflows/publish.yml`) fires on `v*` tag push and on workflow_dispatch with `confirm=publish`. Authentication uses crates.io trusted publishing (OIDC), no API tokens.

```bash
# Tag-triggered (preferred):
git tag vX.Y.Z && git push --follow-tags

# Manual dispatch (idempotent, safe to re-run after a partial failure):
GITHUB_TOKEN= gh workflow run publish.yml -f confirm=publish
```

The publish chain runs bottom-up by dep order: `anno -> anno-eval -> anno-cli`. Each step uses `publish-crate.sh`, which treats "crate version X is already uploaded" as success so re-runs after partial failures pick up where they left off.

## Trusted-publisher configuration (crates.io)

Each published crate needs a trusted publisher entry on crates.io with:

- Repository: `arclabs561/anno`
- Workflow: `publish.yml`
- Environment: `crates-io`

If `cargo publish` returns `403 Forbidden: provided access token is not valid for crate <name>`, the entry for that crate is missing or has a mismatched workflow/environment. Fix on https://crates.io/crates/<name>/settings.

## Known issue

The 0.7.0 publish (2026-04-25) succeeded for anno-core and anno-metrics but failed mid-chain on anno-graph due to a trusted-publisher misalignment for 5 of 7 entries. Rather than re-running 0.7.0, the release rolled forward to 0.8.0 directly. anno-core 0.7.0 and anno-metrics 0.7.0 remain on crates.io as orphan releases (no functional impact; users on `anno = "0.6"` keep working, and `anno = "0.8"` resolves cleanly to anno-core 0.8.0 + anno-metrics 0.8.0).
