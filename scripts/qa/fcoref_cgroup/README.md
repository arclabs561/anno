# FCoref replay under Linux container resource limits

The existing real-weight PR workflow runs this benchmark and uploads the `fcoref-cgroup` JSON artifact. Run it locally on an x86_64 Linux Docker host with delegated cgroup v2 CPU, cpuset, memory and swap controllers. The runner verifies the daemon capabilities, then starts a fresh network-disabled container for each measurement. The entrypoint checks its cgroup **before** model load. The runner checks the measured process's controls and thread affinity afterward. A successful receipt requires a one-CPU quota, one allowed logical CPU, a hard 2 GiB memory ceiling and zero swap allowance.

Use Python 3.11+, the FP32 model from the [real-weight workflow](../../../.github/workflows/fcoref-cli.yml), the Linux x86_64 CPU library from ONNX Runtime 1.24.4, and an already-local container image with a compatible glibc, libstdc++, `/bin/sh` and `cat`. The image and benchmark run on the same machine as the Docker daemon so it can bind-mount the inputs. Use a CPU number allowed by the daemon. The image is selected by its immutable local image ID. The replay verifies the hashes of the model and runtime and records the binary hash. It never pulls an image or dispatches a workflow.

```bash
cargo build --locked --release -p anno --no-default-features \
  --features onnx,ort/load-dynamic --example fcoref_resource_profile

python3 scripts/qa/fcoref_cgroup/replay.py \
  --binary target/release/examples/fcoref_resource_profile \
  --model /absolute/path/to/fcoref_onnx \
  --ort /absolute/path/to/libonnxruntime.so.1.24.4 \
  --image my-local-runtime:ubuntu24 --cpu 0 --output fcoref-cgroup.json
```

The default matrix runs 42 containers: three repeats, one and two ONNX encoder threads, and seven token-count/budget pairs. The 511/512/513 cases cross the encoder-segment boundary. The 520/521/522 cases test a fixed document cap of 521. The canonical 521-token input also runs with a cap of 520. Each accepted case checks mention surfaces against its fixture and all three expected clusters at the shifted source offsets. The canonical 521-token case also checks every recorded reference mention and logit. Each rejected case must report the document-budget error for its exact encoded length and budget.

The FCoref CLI workflow schedules the caller-limit replay on the default branch every Wednesday at 05:37 UTC. Select **Replay the 4096-token caller limit** to run it manually. Its six containers run one and two encoder threads on the 521-token reference, a 4,097-token input over the 4,096 cap, and a 4,096-token input at the cap. The equivalent local options are `--caller-budget --repeats 1`. The count includes both special tokens, which the Rust worker checks using the pinned tokenizer. Nonbreaking whitespace before the long input gives source byte and character offsets different values. A successful accepted run must return finite scores on sorted, unique, verbatim word ranges. Each returned cluster must link distinct observed mentions. The long input has no reference cluster partition. A refusal, timeout, or OOM at the cap produces a failed receipt. These calls exercise Anno's pretokenized FCoref API at the largest [document budget accepted by Aurelius's language lane](https://github.com/arlyon/aurelius/blob/1319bcdcee28ee5b668b41a2a4f85dd0722b460f/lanes/src/language.rs#L20). Aurelius's own tokenization, scheduling and other language processing are outside the measured process.

Each container measures model loading, one first call, and five warm calls. Its receipt includes process high-water RSS (`VmHWM`), cgroup peak memory if the kernel exposes it, OOM counters, quotas and affinities. The cgroup peak includes memory charged to the whole container. The host wrapper's wall time also includes Docker startup. An early refusal records the daemon capabilities and returns failure. No resource-limited timing exists for such a refusal. Run the driver tests with `python3 -m unittest discover -s scripts/qa/fcoref_cgroup`.

The output path must be new. The driver publishes a complete JSON snapshot before preflight, before and after each case, and at termination. A cancelled run retains its last snapshot, including all verified rows and the active case when known. A verification failure records that case and any container diagnostics separately. Only `status: completed` denotes a complete matrix. CI uploads the receipt from a directory unique to that attempt.
