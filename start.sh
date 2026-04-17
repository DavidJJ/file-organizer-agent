#!/usr/bin/env bash
# Starts the stack, auto-detecting whether to run the bundled Ollama service.
#
# If Ollama is already reachable on the host (port 11434), it is used as-is
# and the Docker Ollama service is skipped.  Otherwise the Ollama service
# (and model puller) are started as part of the stack.
#
# Usage:
#   ./start.sh              # auto-detect (default)
#   ./start.sh --ollama     # force Docker Ollama even if host Ollama is running
#   ./start.sh --no-ollama  # skip Docker Ollama unconditionally

set -euo pipefail

OLLAMA_HOST_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

detect_ollama() {
  curl -sf "${OLLAMA_HOST_URL}/api/tags" > /dev/null 2>&1
}

# --- Parse flags ---
FORCE_OLLAMA=""
for arg in "$@"; do
  case "$arg" in
    --ollama)    FORCE_OLLAMA="yes" ;;
    --no-ollama) FORCE_OLLAMA="no"  ;;
  esac
done

# --- Decide whether to start the Docker Ollama service ---
if [ "$FORCE_OLLAMA" = "yes" ]; then
  USE_DOCKER_OLLAMA=true
  echo "ℹ️  --ollama flag set: starting bundled Ollama service."
elif [ "$FORCE_OLLAMA" = "no" ]; then
  USE_DOCKER_OLLAMA=false
  echo "ℹ️  --no-ollama flag set: skipping bundled Ollama service."
elif detect_ollama; then
  USE_DOCKER_OLLAMA=false
  echo "✅ Ollama detected at ${OLLAMA_HOST_URL} — skipping bundled Ollama service."
else
  USE_DOCKER_OLLAMA=true
  echo "⚠️  Ollama not detected at ${OLLAMA_HOST_URL} — starting bundled Ollama service."
fi

# --- Build the docker compose command ---
COMPOSE_ARGS=("docker" "compose" "up" "--detach" "--remove-orphans")

if [ "$USE_DOCKER_OLLAMA" = true ]; then
  COMPOSE_ARGS+=("--profile" "ollama")
fi

# Pass through any extra arguments (e.g. --build, specific service names)
for arg in "$@"; do
  case "$arg" in
    --ollama|--no-ollama) ;;   # already consumed above
    *) COMPOSE_ARGS+=("$arg") ;;
  esac
done

echo "▶ ${COMPOSE_ARGS[*]}"
exec "${COMPOSE_ARGS[@]}"
