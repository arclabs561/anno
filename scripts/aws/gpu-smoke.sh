#!/usr/bin/env bash
# Validate that an ONNX execution provider actually accelerates inference on
# real GPU hardware. Single-job convenience wrapper: launch EC2, build, run
# the matching `examples/onnx_<ep>_smoke.rs`, terminate. Always teardown.
#
# Non-goals (do NOT bloat this script):
# - CI integration (this is manual-run; self-hosted runners are a separate decision).
# - Multi-instance, multi-region, or fleet orchestration (use Pulumi/Terraform if you need that).
# - End-to-end benchmarking or perf tracking (anno-eval / /perf).
# - Persistent infra (every run is fresh spin-up + spin-down).
#
# Per-EP dispatch (only `cuda` is wired today; structure honors future EPs):
#   cuda      AWS g5.xlarge   CUDA 12.x via Deep Learning AMI
#   tensorrt  AWS g5.xlarge   not implemented (issue #19)
#   directml  AWS g4dn Win    not implemented (issue #19, Windows EC2)
#   rocm      not on AWS      no AMD GPUs in EC2; use Azure/Hetzner
#   coreml    not on AWS      validate locally on Apple Silicon
#
# Required env:
#   AWS_PROFILE         AWS CLI profile (uses 'default' if unset)
#   AWS_REGION          AWS region (default us-east-1)
# Optional env:
#   ANNO_INSTANCE_TYPE  override default instance for the chosen EP
#   ANNO_MAX_MINUTES    hard timeout in minutes for the whole run (default 25)

set -euo pipefail

EP="${1:-cuda}"
AWS_REGION="${AWS_REGION:-us-east-1}"
MAX_MINUTES="${ANNO_MAX_MINUTES:-25}"

# AWS CLI base args. Array form so multi-token expansions stay split correctly.
AWS_ARGS=(--region "$AWS_REGION")
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_ARGS=(--profile "$AWS_PROFILE" "${AWS_ARGS[@]}")
fi
aws_cli() { aws "${AWS_ARGS[@]}" "$@"; }

case "$EP" in
  cuda)
    # g4dn.xlarge (Tesla T4) is the default: cheapest CUDA-12-capable AWS GPU,
    # easily clears the 3x-speedup smoke threshold, ~$0.53/hr on-demand. Override
    # via ANNO_INSTANCE_TYPE for production-shaped workloads (g5.xlarge A10G,
    # g6.xlarge L4) where CUDA throughput headroom matters.
    INSTANCE_TYPE="${ANNO_INSTANCE_TYPE:-g4dn.xlarge}"
    AMI_FILTER='Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*'
    AMI_OWNER='amazon'
    CARGO_FEATURES='onnx,onnx-cuda'
    SMOKE_BIN='onnx_cuda_smoke'
    SSH_USER='ubuntu'
    ;;
  tensorrt|directml)
    echo "ep=$EP: not implemented yet (anno issue #19). Wired EPs: cuda. Exiting." >&2
    exit 2
    ;;
  rocm)
    echo "ep=rocm: AWS does not offer AMD GPU instances. Use Azure (MI300x) or Hetzner dedicated. Exiting." >&2
    exit 3
    ;;
  coreml)
    echo "ep=coreml: not an AWS path. Run locally on Apple Silicon: cargo run --example ..." >&2
    echo "(no Apple Silicon in EC2; CoreML EP shipped in commit f891a31.)" >&2
    exit 3
    ;;
  *)
    echo "Unknown EP: $EP. Valid: cuda, tensorrt, directml, rocm, coreml" >&2
    exit 64
    ;;
esac

echo "[smoke] ep=$EP region=$AWS_REGION instance=$INSTANCE_TYPE features=$CARGO_FEATURES"

# Cleanup state. Populated incrementally as resources are created. The single
# trap fires on EXIT (success, error, signal) and tears each created resource
# down in reverse order. Idempotent: each branch checks if its var is set.
INSTANCE_ID=""
KEY_NAME=""
KEY_PATH=""
SG_ID=""
WATCHDOG_PID=""
# shellcheck disable=SC2329  # invoked via trap, not directly
cleanup() {
  local rc=$?
  set +e
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill -9 "$WATCHDOG_PID" 2>/dev/null || true
  fi
  if [[ -n "$INSTANCE_ID" ]]; then
    echo "[smoke] terminating instance $INSTANCE_ID"
    aws_cli ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws_cli ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" || true
  fi
  if [[ -n "$SG_ID" ]]; then
    echo "[smoke] deleting security group $SG_ID"
    aws_cli ec2 delete-security-group --group-id "$SG_ID" >/dev/null || true
  fi
  if [[ -n "$KEY_NAME" ]]; then
    echo "[smoke] deleting key pair $KEY_NAME"
    aws_cli ec2 delete-key-pair --key-name "$KEY_NAME" >/dev/null || true
  fi
  if [[ -n "$KEY_PATH" && -f "$KEY_PATH" ]]; then
    rm -f "$KEY_PATH"
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# Watchdog: kill the script (which triggers the trap) if it runs too long.
( sleep $((MAX_MINUTES * 60)) && echo "[smoke] timeout after ${MAX_MINUTES}m, killing" >&2 && kill -TERM $$ ) &
WATCHDOG_PID=$!

# 1. Find latest matching AMI.
echo "[smoke] resolving AMI..."
AMI_ID=$(aws_cli ec2 describe-images \
  --owners "$AMI_OWNER" \
  --filters "Name=name,Values=$AMI_FILTER" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)
if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  echo "[smoke] no AMI matched: $AMI_FILTER" >&2
  exit 4
fi
echo "[smoke] AMI: $AMI_ID"

# 2. Caller's public IP, for SG ingress allow-list.
MY_IP=$(curl -sS https://checkip.amazonaws.com | tr -d '\n')
if [[ -z "$MY_IP" ]]; then
  echo "[smoke] could not resolve caller IP" >&2
  exit 5
fi
echo "[smoke] caller IP: $MY_IP"

# 3. Temp key pair.
TS=$(date +%s)
KEY_NAME="anno-smoke-$EP-$TS"
KEY_PATH="$(mktemp -t "${KEY_NAME}.XXXXXX")"
aws_cli ec2 create-key-pair --key-name "$KEY_NAME" \
  --query 'KeyMaterial' --output text > "$KEY_PATH"
chmod 600 "$KEY_PATH"
echo "[smoke] key pair: $KEY_NAME"

# 4. Discover VPC + subnet. AWS accounts without a default VPC must explicitly
# place the SG and instance into one. Honor env overrides; otherwise auto-pick
# from a single VPC, error if ambiguous.
if [[ -z "${ANNO_VPC_ID:-}" ]]; then
  VPC_COUNT=$(aws_cli ec2 describe-vpcs --query 'length(Vpcs[])' --output text)
  if [[ "$VPC_COUNT" == "0" ]]; then
    echo "[smoke] no VPCs in $AWS_REGION. Create one or set ANNO_VPC_ID." >&2
    exit 7
  elif [[ "$VPC_COUNT" == "1" ]]; then
    ANNO_VPC_ID=$(aws_cli ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text)
  else
    # Prefer default VPC if any; else require explicit choice.
    ANNO_VPC_ID=$(aws_cli ec2 describe-vpcs --filters 'Name=is-default,Values=true' \
      --query 'Vpcs[0].VpcId' --output text)
    if [[ -z "$ANNO_VPC_ID" || "$ANNO_VPC_ID" == "None" ]]; then
      echo "[smoke] $VPC_COUNT VPCs in $AWS_REGION and no default. Set ANNO_VPC_ID." >&2
      exit 7
    fi
  fi
fi
echo "[smoke] VPC: $ANNO_VPC_ID"

if [[ -z "${ANNO_SUBNET_ID:-}" ]]; then
  # Prefer a subnet that auto-assigns public IPs (we need to SSH in from outside).
  ANNO_SUBNET_ID=$(aws_cli ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$ANNO_VPC_ID" "Name=map-public-ip-on-launch,Values=true" \
    --query 'Subnets[0].SubnetId' --output text)
  if [[ -z "$ANNO_SUBNET_ID" || "$ANNO_SUBNET_ID" == "None" ]]; then
    echo "[smoke] no public-IP-auto-assign subnet in $ANNO_VPC_ID. Set ANNO_SUBNET_ID or fix the VPC." >&2
    exit 7
  fi
fi
echo "[smoke] subnet: $ANNO_SUBNET_ID"

# Layer-2 dead-man's switch: instance terminates itself after 30 min if the
# launching client somehow loses its grip and the in-process trap never fires
# (laptop sleep, kill -9, network partition, etc). The +30 value gives ~15 min
# of slack over the script's watchdog so a legitimate longer build does not
# get kneecapped. InstanceInitiatedShutdownBehavior=terminate is the load-
# bearing pairing flag: without it, `shutdown -h` only stops the instance
# (still costs EBS storage; lingers in stopped state). With it, AWS fully
# terminates and reaps the volume.
USER_DATA=$(printf '#!/bin/bash\nshutdown -h +30\n')

# 5. Temp security group with SSH from caller only.
SG_NAME="anno-smoke-$EP-$TS"
SG_ID=$(aws_cli ec2 create-security-group --group-name "$SG_NAME" \
  --description "Temporary anno smoke SG (auto-deleted)" \
  --vpc-id "$ANNO_VPC_ID" \
  --query 'GroupId' --output text)
aws_cli ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null
echo "[smoke] security group: $SG_ID"

# 6. Launch.
INSTANCE_ID=$(aws_cli ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$ANNO_SUBNET_ID" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "$USER_DATA" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=80,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$SG_NAME},{Key=Project,Value=anno},{Key=Realm,Value=hardware-test},{Key=ManagedBy,Value=anno-smoke-script},{Key=auto-terminate,Value=true},{Key=CreatedBy,Value=anno},{Key=anno:allow_gpu,Value=true}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "[smoke] instance: $INSTANCE_ID"

aws_cli ec2 wait instance-running --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws_cli ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "[smoke] public IP: $PUBLIC_IP"

# 6. Wait for SSH (Deep Learning AMI takes ~60-90s after instance-running).
echo "[smoke] waiting for SSH..."
SSH_OPTS=(
  -i "$KEY_PATH"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=10
  -o LogLevel=ERROR
)
for i in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" 'echo ok' >/dev/null 2>&1; then
    echo "[smoke] SSH up after ${i}0s"
    break
  fi
  if [[ "$i" -eq 30 ]]; then
    echo "[smoke] SSH never came up" >&2
    exit 6
  fi
  sleep 10
done

# 7. Sanity-check NVIDIA driver and CUDA.
ssh "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" 'nvidia-smi -L && (nvcc --version 2>/dev/null | grep release || echo "[remote] nvcc not on PATH; runtime libs will be dlopened")'

# 8. Sync workspace. `--filter=':- .gitignore'` makes rsync honor every
# gitignore in the tree (one less list to maintain). The hardcoded
# excludes cover only `.git/` itself (never gitignored) and a couple of
# patterns that are tracked but uselessly large for a remote build
# (testdata fixtures, generated reports). `--info=progress2` prints a
# single live progress line so the run does not look frozen during
# upload.
REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "[smoke] rsyncing workspace from $REPO_ROOT..."
# rsync's -e wants a single string; build it from the array.
SSH_E="ssh $(printf '%q ' "${SSH_OPTS[@]}")"
t_rsync_start=$(date +%s)
rsync -az --info=progress2 --delete \
  --filter=':- .gitignore' \
  --exclude '.git/' \
  --exclude 'testdata/' \
  --exclude 'evals/' \
  --exclude 'reports/' \
  --exclude 'archive/' \
  --exclude 'hack/' \
  -e "$SSH_E" \
  "$REPO_ROOT/" "$SSH_USER@$PUBLIC_IP:~/anno/"
echo "[smoke] rsync complete in $(( $(date +%s) - t_rsync_start ))s"

# 9. Run smoke. Build is on the box (rust toolchain ships with the DL AMI;
# if not, the script will fail clearly here -- that's a deliberate signal
# rather than a silent fallback). `-tt` forces a pseudo-TTY so cargo emits
# line-buffered progress instead of full-buffering when piped. Stage echos
# tell you which phase is active without needing to ssh in separately.
echo "[smoke] building + running on remote..."
SMOKE_RC=0
# shellcheck disable=SC2029  # SMOKE_BIN/CARGO_FEATURES intentionally expand on the client
ssh -tt "${SSH_OPTS[@]}" "$SSH_USER@$PUBLIC_IP" "
  set -e
  echo '[remote] toolchain check'
  if ! command -v cargo >/dev/null; then
    echo '[remote] installing rustup...'
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable -q
    source \$HOME/.cargo/env
  fi
  cd ~/anno
  echo '[remote] cargo build + smoke run'
  export CARGO_TERM_COLOR=always
  cargo run --release --example $SMOKE_BIN --features '$CARGO_FEATURES'
  echo '[remote] smoke run done'
" || SMOKE_RC=$?

if [[ "$SMOKE_RC" -eq 0 ]]; then
  echo "[smoke] PASS (ep=$EP)"
else
  echo "[smoke] FAIL exit=$SMOKE_RC (ep=$EP)" >&2
fi

# Trap fires here, terminates instance + cleans up key/SG.
exit "$SMOKE_RC"
