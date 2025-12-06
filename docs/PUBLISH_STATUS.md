# Publish Status

## Published

- `anno` v0.2.0

## Ready

- `anno-core` v0.2.0

## Issues

### anno-coalesce
Missing version requirement for `anno-core` dependency.

Fix: `coalesce/Cargo.toml`: `anno-core = { version = "0.2.0", path = "../anno-core" }`

### anno-strata
Missing version requirement for `anno-core` dependency.

Fix: `strata/Cargo.toml`: `anno-core = { version = "0.2.0", path = "../anno-core" }`

### anno
Missing version requirements for `anno-coalesce` and `anno-strata`.

Fix: `anno/Cargo.toml`:
- `anno-coalesce = { version = "0.2.0", path = "../coalesce" }`
- `anno-strata = { version = "0.2.0", path = "../strata", optional = true }`

## Publish Order

1. `anno-core`
2. `anno-coalesce`
3. `anno-strata`
4. `anno`

## Not Published

- `anno-cli` (publish = false)
