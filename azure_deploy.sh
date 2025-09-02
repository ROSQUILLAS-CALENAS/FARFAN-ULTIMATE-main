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

# Defaults (same as deploy.sh), but we force-confirm below
RG="${RG:-farfan-rg}"; LOC="${LOC:-eastus}"; REPO="${REPO:-farfan-app}"
ACI_NAME="${ACI_NAME:-farfan-aci}"; DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CPU="${CPU:-2}"; MEMORY="${MEMORY:-4}"; PORT="${PORT:-8000}"
WAIT_READY="${WAIT_READY:-true}"; WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"
CLEAN_REPO="${CLEAN_REPO:-false}"

# Prerequisite quick checks
command -v az >/dev/null 2>&1 || { e "Azure CLI 'az' no encontrado. Instala: https://aka.ms/azcli"; exit 1; }
az account show >/dev/null 2>&1 || { e "No has iniciado sesión en Azure. Ejecuta: az login"; exit 1; }
[ -f "$DOCKERFILE" ] || { e "No existe $DOCKERFILE en $ROOT_DIR"; exit 1; }

# Force confirmed execution for Azure deployment
export CONFIRM="true"

i "Despliegue confirmado a Azure: RG=$RG LOC=$LOC ACI=$ACI_NAME REPO=$REPO PORT=$PORT CPU=$CPU MEM=${MEMORY}Gi WAIT_READY=$WAIT_READY CLEAN_REPO=$CLEAN_REPO"

i "Invocando deploy.sh con configuración: $CONFIG_FILE"
"$ROOT_DIR/deploy.sh" "$CONFIG_FILE"