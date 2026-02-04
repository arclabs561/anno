## Checkpoints (tags + GitHub Releases)

We use tags/releases as the primary “checkpoint log” while the internal crate split is still fluid.

### Create a checkpoint

From repo root:

```bash
# Make sure you’re on main and clean.
git status --porcelain

# Run the highest-signal local checks (CI should be green too).
cargo fmt --all
cargo test --workspace --all-features
cargo deny check
cargo audit

# Tag + push.
git tag "v<version>"
git push origin "v<version>"
```

Then create a GitHub Release for that tag (notes can be short: “checkpoint: <one-liner>”).

### Notes

- This keeps checkpoints **immutable** and easy to cite without committing to a crates.io semver surface.
- When we later decide to publish to crates.io, we can either publish `anno-core` too or collapse it into `anno`.

