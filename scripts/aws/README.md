# scripts/aws

AWS-based hardware validators for anno. Convenience wrappers, not infrastructure.

## What this is

Manual-run scripts that spin up an EC2 instance, build anno on it, run a single
smoke binary, and tear everything down. One job: prove a feature flag actually
exercises the hardware it claims to, on real hardware, before users hit silent
fallback in production.

## What this is not

- **Not CI.** No PR triggers, no scheduled runs. Run it when you change an EP
  or upgrade a driver, not on every commit.
- **Not an eval harness.** Quality and accuracy live in `anno-eval`. These
  scripts validate "does this hardware path execute" -- not "does it execute
  correctly across a benchmark suite."
- **Not a perf tracker.** The CPU-vs-GPU ratio is a sanity check (≥3x means
  CUDA actually attached), not a number to graph over time. That belongs in
  `/perf`.
- **Not durable infrastructure.** No CloudFormation, no Terraform, no Pulumi.
  Every run is fresh spin-up + spin-down. If you find yourself wanting
  long-lived runners, switch to a proper IaC tool -- don't grow this script.

## Per-EP coverage

| EP        | Status      | Hardware                       | Notes |
|-----------|-------------|--------------------------------|-------|
| CUDA      | wired       | g4dn.xlarge (Tesla T4, default)| `./gpu-smoke.sh cuda`. Override `ANNO_INSTANCE_TYPE` to use g5.xlarge / g6.xlarge if you need A10G/L4 headroom. |
| TensorRT  | not wired   | g5.xlarge               | same hardware as CUDA; see anno issue #19 |
| DirectML  | not wired   | g4dn (Windows GPU)      | requires Windows EC2; higher friction |
| ROCm      | not on AWS  | n/a                     | EC2 has no AMD GPU instance types; use Azure MI300x or Hetzner |
| CoreML    | not on AWS  | n/a                     | no Apple Silicon in EC2; validate locally on macOS dev box (already shipped, commit f891a31) |

When ROCm or DirectML matter enough to test, those go in their own directory
(`scripts/azure/`, `scripts/hetzner/`, etc.) -- not bolted onto this script.

## Usage

```bash
# Required: working AWS profile.
export AWS_PROFILE=your-profile

# Run the CUDA smoke. Defaults to us-east-1, g4dn.xlarge.
./scripts/aws/gpu-smoke.sh cuda

# Override region or instance.
AWS_REGION=us-west-2 ANNO_INSTANCE_TYPE=g5.xlarge ./scripts/aws/gpu-smoke.sh cuda
```

Cost: g4dn.xlarge on-demand is ~$0.53/hr; a smoke takes ~5 minutes including
AMI boot, workspace sync, and build. Expect ~$0.05-0.10 per run.

## Safety properties

The script is built around three guarantees:

1. **Always teardown.** A single trap on `EXIT INT TERM` deletes the instance,
   key pair, and security group regardless of how the script ends (success,
   build failure, ctrl-C, signal). The trap is idempotent.
2. **Watchdog timeout.** If the script runs longer than `ANNO_MAX_MINUTES`
   (default 25), it kills itself, which fires the trap. No runaway instances.
3. **No persistent secrets.** Each run mints a temporary EC2 key pair and SG,
   both deleted at end. No long-lived keys in the repo.

## What gets created and torn down

Per run, the script creates and then deletes:

- 1 EC2 key pair (named `anno-smoke-<ep>-<ts>`)
- 1 security group (SSH from your current public IP only)
- 1 EC2 instance with a GP3 EBS root volume, `DeleteOnTermination=true`

If the script is interrupted in a way that bypasses the trap (kernel panic,
laptop closed before EXIT fires, etc.), check for orphans:

```bash
aws ec2 describe-instances --filters 'Name=tag:Name,Values=anno-smoke-*' \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]' --output table
aws ec2 describe-security-groups --filters 'Name=group-name,Values=anno-smoke-*' \
  --query 'SecurityGroups[].[GroupId,GroupName]' --output table
aws ec2 describe-key-pairs --filters 'Name=key-name,Values=anno-smoke-*' \
  --query 'KeyPairs[].[KeyName]' --output table
```

## Adding a new EP

1. Confirm the EP is AWS-compatible. If it's not, add a row to the matrix
   above and a `case` arm in `gpu-smoke.sh` that prints the right alternative
   ("not on AWS -- see X").
2. Write `crates/anno/examples/onnx_<ep>_smoke.rs`. Mirror `onnx_cuda_smoke.rs`
   structure: build two sessions (with and without the EP), N inferences each,
   assert speedup threshold.
3. Add a `case` arm in `gpu-smoke.sh` mapping EP → `INSTANCE_TYPE`,
   `AMI_FILTER`, `CARGO_FEATURES`, `SMOKE_BIN`, `SSH_USER`.
4. Run it once to confirm. Update this README's matrix.

Resist combining EPs into one binary. One file per EP, separate failure modes,
separate evolution.
