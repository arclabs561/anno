#!/bin/sh
# Run as the entrypoint inside a fresh, private cgroup namespace.
set -eu
read -r quota period < /sys/fs/cgroup/cpu.max
[ "$quota" = "$period" ] && [ "$period" -gt 0 ] ||
    { echo "expected one CPU of CFS quota" >&2; exit 78; }
[ "$(cat /sys/fs/cgroup/cpuset.cpus.effective)" = "$EXPECTED_CPU" ] ||
    { echo "unexpected effective cpuset" >&2; exit 78; }
[ "$(cat /sys/fs/cgroup/memory.max)" = "2147483648" ] ||
    { echo "expected a hard 2 GiB cgroup memory limit" >&2; exit 78; }
[ "$(cat /sys/fs/cgroup/memory.swap.max)" = "0" ] ||
    { echo "expected swap to be disabled" >&2; exit 78; }
exec /measure /weights "$@"
