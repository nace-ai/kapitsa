#!/bin/bash
# usage: locked.sh <command...>: runs the command under the shared GPU lock (lockf -k -t 540),
# retrying while the lock stays busy (exit 75). The command itself must finish in < ~9 minutes.
while true; do
  lockf -k -t 540 /tmp/kapitsa-gpu.lock "$@"
  rc=$?
  [ $rc -ne 75 ] && exit $rc
  echo "[locked.sh] lock busy, retrying" >&2
done
