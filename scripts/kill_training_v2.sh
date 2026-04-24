#!/bin/bash
set -euo pipefail

# Force-kill all processes related to train_dgx_v2.py for the current user.
# Uses SIGKILL (-9) as requested.

SELF_PID=$$
PARENT_PID=$PPID
USER_ID=$(id -u)

PATTERNS=(
  "accelerate launch.*scripts/train_dgx_v2.py"
  "scripts/train_dgx_v2.py"
  "python.*train_dgx_v2.py"
)

pids=""
for pattern in "${PATTERNS[@]}"; do
  matches=$(pgrep -u "$USER_ID" -f "$pattern" || true)
  if [[ -n "$matches" ]]; then
    pids+=$'\n'"$matches"
  fi
done

# Unique + strip empty lines.
pids=$(echo "$pids" | tr ' ' '\n' | sed '/^$/d' | sort -u)

# Do not kill this script or its parent shell.
pids=$(echo "$pids" | grep -v -E "^(${SELF_PID}|${PARENT_PID})$" || true)

if [[ -z "$pids" ]]; then
  echo "No running train_dgx_v2 processes found for user $(id -un)."
  exit 0
fi

echo "Killing these PIDs with SIGKILL (-9):"
echo "$pids"

echo "$pids" | xargs -r kill -9

echo "Done."
