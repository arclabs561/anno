#!/usr/bin/env bash
# Sweep stale anno smoke resources. Manual recovery + belt-and-suspenders
# for cases the in-process trap and instance self-shutdown both missed.
#
# Usage:
#   ./scripts/aws/cleanup-stale.sh           # default: terminate anything > 60 min old
#   ./scripts/aws/cleanup-stale.sh 5         # custom threshold (minutes)
#   ./scripts/aws/cleanup-stale.sh 0 dry     # dry-run: list, don't terminate
#
# Filters by tag Project=anno + ManagedBy=anno-smoke-script (set by
# gpu-smoke.sh). Will not touch unrelated resources.

set -euo pipefail

THRESHOLD_MIN="${1:-60}"
DRY_RUN="${2:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

AWS_ARGS=(--region "$AWS_REGION")
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_ARGS=(--profile "$AWS_PROFILE" "${AWS_ARGS[@]}")
fi
aws_cli() { aws "${AWS_ARGS[@]}" "$@"; }

# BSD date (macOS) and GNU date (Linux) have incompatible flags for
# relative time. Try BSD first, fall back to GNU.
cutoff=$(date -u -v-"${THRESHOLD_MIN}"M +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
       || date -u -d "${THRESHOLD_MIN} minutes ago" +%Y-%m-%dT%H:%M:%SZ)
echo "[cleanup] threshold: ${THRESHOLD_MIN} min (cutoff $cutoff)${DRY_RUN:+ [DRY-RUN]}"

# Stale instances tagged by us, older than cutoff. Includes stopping/stopped
# in case shutdown -h fired but terminate didn't (paranoid).
INSTANCES=$(aws_cli ec2 describe-instances \
  --filters \
    'Name=tag:Project,Values=anno' \
    'Name=tag:ManagedBy,Values=anno-smoke-script' \
    'Name=instance-state-name,Values=running,pending,stopping,stopped' \
  --query "Reservations[].Instances[?LaunchTime<='$cutoff'].InstanceId" \
  --output text)

if [[ -n "$INSTANCES" ]]; then
  count=$(echo "$INSTANCES" | wc -w | tr -d ' ')
  echo "[cleanup] stale instances ($count): $INSTANCES"
  if [[ "$DRY_RUN" != "dry" ]]; then
    # shellcheck disable=SC2086  # intentional word-split on instance id list
    aws_cli ec2 terminate-instances --instance-ids $INSTANCES >/dev/null
    echo "[cleanup] terminated $count instance(s)"
  fi
else
  echo "[cleanup] no stale instances"
fi

# Orphan security groups: anno-smoke-* with no attached network interfaces.
ORPHAN_SGS=$(aws_cli ec2 describe-security-groups \
  --filters 'Name=group-name,Values=anno-smoke-*' \
  --query 'SecurityGroups[].GroupId' --output text)
for sg in $ORPHAN_SGS; do
  in_use=$(aws_cli ec2 describe-network-interfaces \
    --filters "Name=group-id,Values=$sg" \
    --query 'length(NetworkInterfaces[])' --output text)
  if [[ "$in_use" == "0" ]]; then
    echo "[cleanup] orphan SG: $sg"
    if [[ "$DRY_RUN" != "dry" ]]; then
      aws_cli ec2 delete-security-group --group-id "$sg" 2>/dev/null || true
    fi
  fi
done

# Orphan key pairs: anno-smoke-*. Key pairs cost nothing but clutter the
# console and the trap's first-line cleanup may have missed them.
ORPHAN_KPS=$(aws_cli ec2 describe-key-pairs \
  --filters 'Name=key-name,Values=anno-smoke-*' \
  --query 'KeyPairs[].KeyName' --output text)
for kp in $ORPHAN_KPS; do
  echo "[cleanup] orphan key pair: $kp"
  if [[ "$DRY_RUN" != "dry" ]]; then
    aws_cli ec2 delete-key-pair --key-name "$kp" 2>/dev/null || true
  fi
done

echo "[cleanup] done"
