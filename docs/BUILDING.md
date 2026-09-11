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
