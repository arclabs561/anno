# Native checkpoint replay

The Linux CPU job reads immutable revisions and SHA-256 hashes from [models.json](models.json). It loads BERT-large, ModernBERT, NuNER Zero-span INT8 and GLiNER2.5 FP32 through `anno extract`. The cases check the runtime identity, entity type, occurrence IDs, and half-open character and UTF-8 byte offsets into the source. The GLiNER2.5 cases also compare probabilities with the upstream Python output (absolute tolerance 0.001), including repeated names, Unicode, missing terminal punctuation and an empty prediction.

To reproduce the [workflow](../../../.github/workflows/native-checkpoint-replay.yml) with Python 3.11+ on Linux x86_64, build the CLI with `gliner2-fastino` and link it to the ONNX Runtime 1.24.2 CPU library from the specified wheel. Set `LD_LIBRARY_PATH` to its directory. Then run:

```sh
python3 scripts/qa/checkpoint_replay/run.py --binary target/debug/anno \
  --models /path/to/model-cache --fetch --receipt checkpoint-receipt.json
```

Omit `--fetch` for an offline replay. The verifier gives the loader only the verified files and runs one inference process at a time on up to two allowed CPUs. The hosted workflow additionally limits the replay service to 6 GiB with swap disabled. The receipt records binary and artifact hashes, ONNX Runtime identity, the cases that completed and cold elapsed time including model load. Its memory field is the largest individual child process peak RSS. The cases exercise integration and have no corpus-level accuracy denominator.
