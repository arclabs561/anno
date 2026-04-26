# Publish status

## Current state (2026-04-26)

Six crates publish to crates.io as a chain. The 0.8.0 release is the first under the post-facade layout.

| Crate | Package name | Publish | Notes |
|-------|-------------|---------|-------|
| `crates/anno` | `anno` | yes | The main library. Renamed from `anno-lib` in 0.8.0. |
| `crates/anno-core` | `anno-core` | yes | Stable type foundation. |
| `crates/anno-metrics` | `anno-metrics` | yes | Coreference scoring + cluster encoders. |
| `crates/anno-graph` | `anno-graph` | yes | Graph/KG export, optional. |
| `crates/anno-eval` | `anno-eval` | yes | Evaluation harnesses, datasets, muxer sampling. |
| `crates/anno-cli` | `anno-cli` | yes | Full CLI binary. `cargo install anno-cli`. |

## Trajectory (Phase B target)

Phase A (this 0.8.0) collapsed the `anno` facade. Phase B aims for 3 published crates:

| Future state | Plan |
|--------------|------|
| `anno` | Absorbs anno-core, anno-metrics, anno-graph as internal modules. Publishes alone. |
| `anno-eval` | Stays separate; distinct audience and heavier dep graph. |
| `anno-cli` | Stays separate; binary. |

Rationale: keep the published surface to crates that have distinct release cadence, distinct downstream users, or distinct native-toolchain requirements. The current `*-core`/`*-metrics`/`*-graph` split has none of those properties at present.

## Publish command

The publish workflow (`.github/workflows/publish.yml`) fires on `v*` tag push and on workflow_dispatch with `confirm=publish`. Authentication uses crates.io trusted publishing (OIDC), no API tokens.

```bash
# Tag-triggered (preferred):
git tag vX.Y.Z && git push --follow-tags

# Manual dispatch (idempotent, safe to re-run after a partial failure):
GITHUB_TOKEN= gh workflow run publish.yml -f confirm=publish
```

The publish chain runs bottom-up by dep order: `anno-core → anno-metrics → anno-graph → anno → anno-eval → anno-cli`. Each step uses `publish-crate.sh`, which treats "crate version X is already uploaded" as success so re-runs after partial failures pick up where they left off.

## Trusted-publisher configuration (crates.io)

Each published crate needs a trusted publisher entry on crates.io with:

- Repository: `arclabs561/anno`
- Workflow: `publish.yml`
- Environment: `crates-io`

If `cargo publish` returns `403 Forbidden: provided access token is not valid for crate <name>`, the entry for that crate is missing or has a mismatched workflow/environment. Fix on https://crates.io/crates/<name>/settings.

## Known issue

The 0.7.0 publish (2026-04-25) succeeded for anno-core and anno-metrics but failed mid-chain on anno-graph due to a trusted-publisher misalignment for 5 of 7 entries. Rather than re-running 0.7.0, the release rolled forward to 0.8.0 directly. anno-core 0.7.0 and anno-metrics 0.7.0 remain on crates.io as orphan releases (no functional impact; users on `anno = "0.6"` keep working, and `anno = "0.8"` resolves cleanly to anno-core 0.8.0 + anno-metrics 0.8.0).
