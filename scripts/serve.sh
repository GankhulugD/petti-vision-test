#!/bin/sh
# One process group: Next.js on $PORT (public) and the model on PETTI_PORT (internal).
set -eu

export PETTI_PORT="${PETTI_PORT:-8765}"
export PORT="${PORT:-7860}"
export HOSTNAME="${HOSTNAME:-0.0.0.0}"
export PETTI_DEVICE="${PETTI_DEVICE:-cpu}"
export PETTI_HALF="${PETTI_HALF:-0}"
export HF_HOME="${HF_HOME:-/app/model/.cache}"

cd /app/model
python server.py &
MODEL_PID=$!

cd /app/web
node server.js &
WEB_PID=$!

trap 'kill "$MODEL_PID" "$WEB_PID" 2>/dev/null || true' INT TERM
wait "$WEB_PID"
