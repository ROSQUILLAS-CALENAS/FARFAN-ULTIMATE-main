#!/usr/bin/env bash
set -euo pipefail

# Portable logger
log(){ printf '%s [%s] %s\n' "$(date '+%F %T')" "$1" "$2"; }
i(){ log INFO "$*"; }
e(){ log ERROR "$*" >&2; }
ok(){ log OK "$*"; }

# Canonical repo root
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"

# Load config if provided
CONFIG_FILE="${1:-config.env}"
[ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE" || true

REPO="${REPO:-farfan-app}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
PORT="${PORT:-8000}"
AUTO_INSTALL="${AUTO_INSTALL:-false}"

has_docker(){ command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }

run_with_docker(){
  if [ ! -f "$DOCKERFILE" ]; then
    e "No existe $DOCKERFILE en $ROOT_DIR"; return 1
  fi
  if [ ! -x "$ROOT_DIR/run_docker.sh" ]; then
    chmod +x "$ROOT_DIR/run_docker.sh" || true
  fi
  i "Docker disponible: ejecutando run_docker.sh"
  "$ROOT_DIR/run_docker.sh" "$CONFIG_FILE"
}

create_venv_and_install(){
  command -v python3 >/dev/null 2>&1 || { e "python3 no encontrado. Instala Python 3."; return 1; }
  if [ ! -d .venv ]; then
    i "Creando entorno virtual (.venv)"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  . .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  if [ -f requirements-minimal.txt ]; then
    i "Instalando requirements-minimal.txt"
    python -m pip install -r requirements-minimal.txt
  elif [ -f requirements_minimal.txt ]; then
    i "Instalando requirements_minimal.txt"
    python -m pip install -r requirements_minimal.txt
  elif [ -f requirements.txt ]; then
    i "Instalando requirements.txt"
    python -m pip install -r requirements.txt
  else
    i "No hay archivo de requisitos; continuando sin instalación"
  fi
}

run_locally(){
  i "Ejecución local (sin Docker) en $ROOT_DIR"
  if [ "${AUTO_INSTALL}" = "true" ]; then
    create_venv_and_install || return 1
    # shellcheck disable=SC1091
    . .venv/bin/activate
  else
    if [ -d .venv ]; then
      # shellcheck disable=SC1091
      . .venv/bin/activate
    else
      i "AUTO_INSTALL=false y no hay .venv; se usará Python del sistema (pueden faltar dependencias)."
    fi
  fi
  command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || { e "Python no disponible"; return 1; }
  PYBIN=$(command -v python || command -v python3)
  i "Lanzando: $PYBIN main.py -v (puerto esperado: $PORT)"
  "$PYBIN" main.py -v
}

if has_docker; then
  run_with_docker || { e "Fallo con Docker, intentando ejecución local"; run_locally; }
else
  e "Docker no disponible o daemon no accesible; ejecutando localmente"
  run_locally
fi
