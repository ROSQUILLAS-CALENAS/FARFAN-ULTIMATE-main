#!/usr/bin/env bash
set -euo pipefail

# One-click health check wrapper
# - Activates .venv or venv if present
# - Falls back to system Python
# - Runs project_health_check.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
cd "$ROOT_DIR"

PYBIN=""

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate || true
  PYBIN="$(command -v python || command -v python3)"
elif [ -d venv ]; then
  # shellcheck disable=SC1091
  . venv/bin/activate || true
  PYBIN="$(command -v python || command -v python3)"
else
  PYBIN="$(command -v python3 || command -v python)"
fi

if [ -z "$PYBIN" ]; then
  echo "[ERROR] Python not found. Please install Python 3." >&2
  exit 1
fi

chmod +x project_health_check.py || true

# Pass-through arguments (e.g., --full, --json)
exec "$PYBIN" ./project_health_check.py "$@"
