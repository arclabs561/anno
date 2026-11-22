# Building anno

Cargo's normal freshness and incremental artifacts are the fastest path for a
short edit-and-test loop:

```sh
just t parser_test_name
```

For a CI-shaped check, disable incremental compilation. This makes Rust library
compilations eligible for reuse by `sccache` when another clean target directory
or repository has already built the same crate with the same compiler, target,
features, and flags:

```sh
just check-cached
```

`check-cached` sets `CARGO_INCREMENTAL=0` only for that invocation. It does not
configure a compiler cache itself; configure `RUSTC_WRAPPER` or Cargo's
`build.rustc-wrapper` on the machine that runs the build.

To inspect cache counters, run:

```sh
RUSTC_WRAPPER=/path/to/your-sccache-wrapper just cache-stats
```

Using the same wrapper matters when it supplies a remote cache configuration.
Without `RUSTC_WRAPPER`, `just cache-stats` reports the local `sccache` server's
view, which can omit remote tiers configured only by that wrapper. Cache hit
rates are meaningful only across comparable compiler version, target, feature,
and `RUSTFLAGS` settings.

## Optional research runs

A maintainer can push a branch named `research/<topic>` to run the existing
coreference benchmark in Criterion quick mode, or `mutation/<topic>` to check
the discontinuous-span property against up to six mutations. The same branch
can be selected in **Actions → Optional research → Run workflow** once the
workflow is on the default branch. Use the job log for timings and mutation
results. For local mutation checks, install cargo-mutants 27.1.0 and run
`just mutants-fast`.

Both tasks share one hosted Ubuntu runner slot with a 30-minute limit per run.
The latest pending request replaces an older pending request. Branch routing
uses the selected ref in the upstream repository. A contributor's pull request
to main runs the usual CI checks regardless of its source branch name.
