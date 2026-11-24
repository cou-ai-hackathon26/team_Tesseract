#!/usr/bin/env bash
# Free a TCP port before starting a dev server.
#
# A stale uvicorn left over from a previous run is the most common way
# `npm run dev` fails here: the API dies with "Address already in use" while the
# frontend starts normally and then looks broken. Clearing the port makes the
# command idempotent instead of something you have to debug.
set -euo pipefail

port="${1:?usage: free-port.sh <port>}"

command -v lsof >/dev/null 2>&1 || exit 0

pids="$(lsof -ti:"$port" 2>/dev/null || true)"
[ -z "$pids" ] && exit 0

echo "port $port in use by $(echo "$pids" | tr '\n' ' ')- stopping it"
# shellcheck disable=SC2086
kill $pids 2>/dev/null || true

for _ in $(seq 1 20); do
  [ -z "$(lsof -ti:"$port" 2>/dev/null || true)" ] && exit 0
  sleep 0.25
done

echo "port $port still busy, forcing"
# shellcheck disable=SC2086
kill -9 $pids 2>/dev/null || true
sleep 0.5
exit 0
