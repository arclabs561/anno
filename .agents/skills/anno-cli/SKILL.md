---
name: anno-cli
description: Use the anno command-line interface for extraction, redaction, coreference, corpus queries, batch processing, model management, evaluation, and other CLI operations.
---

# Anno CLI

Start with the documentation for the Anno revision you are running. In a checkout, open `docs/README.md`. With an installed binary, check `anno --version` and use the corresponding release docs when available. The [project documentation index](https://github.com/arclabs561/anno/blob/main/docs/README.md) links the current guides.

- For installation and worked CLI examples, read `docs/QUICKSTART.md`.
- For model selection, optional features and runtime requirements, read `docs/BACKENDS.md`.
- For interpreting results, saving annotations or highlighting spans in the source, read `docs/CONTRACT.md`.
- For scoring a result or reproducing a regression, read `docs/QA.md`. The bundled `anno-qa` skill covers multi-step investigations.

For any command, including operations without a worked example, use `anno --help` to see what the installed build exposes. Follow the relevant command path with `--help` at each level to resolve its inputs, output formats and options. Feature-enabled builds may expose additional commands.
