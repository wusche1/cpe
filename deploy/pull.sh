#!/usr/bin/env bash
# Manual rescue for a run whose sync-back watcher died: wait for the remote job to
# reach a terminal state, rsync the results folder back, then tear the cluster
# down. Without this the 30-min autostop reclaims the machine with the results
# still on it.
#
#   ./pull.sh <cluster_name> [<cluster_name> ...]
#
# Status is parsed by keyword, never by column: SUBMITTED/STARTED contain spaces.
set -u
EXP=experiments/pwlock_generalisation/evaluation
LOCAL_ROOT=/workspace/cpe_worktree/pw-transfer/$EXP/results

for CLUSTER in "$@"; do
(
  while true; do
    q=$(timeout 120 uv run sky queue "$CLUSTER" 2>&1)
    if grep -qE "SUCCEEDED|FAILED|CANCELLED" <<<"$q"; then
      echo "[$CLUSTER] terminal: $(grep -oE 'SUCCEEDED|FAILED_SETUP|FAILED|CANCELLED' <<<"$q" | head -1)"
      break
    fi
    if grep -qE "not found|does not exist" <<<"$q"; then
      echo "[$CLUSTER] cluster gone before rescue"; exit 1
    fi
    sleep 60
  done

  mkdir -p "$LOCAL_ROOT/$CLUSTER"
  if timeout 900 rsync -az "$CLUSTER:sky_workdir/$EXP/results/$CLUSTER/" \
        "$LOCAL_ROOT/$CLUSTER/"; then
    echo "[$CLUSTER] synced -> $LOCAL_ROOT/$CLUSTER"
    ls "$LOCAL_ROOT/$CLUSTER" | tr '\n' ' '; echo
    timeout 600 uv run sky down -y "$CLUSTER" >/dev/null 2>&1 \
      && echo "[$CLUSTER] torn down"
  else
    echo "[$CLUSTER] RSYNC FAILED — leaving cluster up for manual recovery"
  fi
) &
done
wait
