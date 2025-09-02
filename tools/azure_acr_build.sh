#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/azure_acr_build.sh -r <registry-name> [-g <resource-group>] [-t <tag>] [-f <dockerfile>] [--platform <os/arch>] [--no-push]

Options:
  -r, --registry          Azure Container Registry name (required or set ACR_NAME env)
  -g, --resource-group    Azure Resource Group of the ACR (optional)
  -t, --tag, --image      Image tag (default: farfan-app:latest)
  -f, --file              Dockerfile path (default: Dockerfile)
      --platform          Build platform, e.g., linux/amd64 (optional)
      --no-push           Build without pushing to the registry
  -h, --help              Show this help and exit

Examples:
  # Use env ACR_NAME and default tag farfan-app:latest
  ACR_NAME=MyRegistry bash tools/azure_acr_build.sh

  # Explicit registry, RG, and tag
  bash tools/azure_acr_build.sh -r MyRegistry -g farfan-rg -t farfan-app:latest

  # Specify platform (if needed)
  bash tools/azure_acr_build.sh -r MyRegistry --platform linux/amd64
EOF
}

registry="${ACR_NAME:-}"
resource_group=""
tag="farfan-app:latest"
dockerfile="Dockerfile"
platform=""
no_push=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--registry)
      registry="$2"; shift 2;;
    -g|--resource-group)
      resource_group="$2"; shift 2;;
    -t|--tag|--image|--image-tag)
      tag="$2"; shift 2;;
    -f|--file)
      dockerfile="$2"; shift 2;;
    --platform)
      platform="$2"; shift 2;;
    --no-push)
      no_push="--no-push"; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 1;;
  esac
done

if ! command -v az >/dev/null 2>&1; then
  echo "Error: Azure CLI 'az' is not installed or not on PATH." >&2
  echo "Install: https://learn.microsoft.com/cli/azure/install-azure-cli" >&2
  exit 1
fi

if [[ -z "${registry}" ]]; then
  echo "Error: registry not specified. Provide -r/--registry or set ACR_NAME environment variable." >&2
  usage
  exit 2
fi

# Ensure we run from repo root (script resides in tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${dockerfile}" ]]; then
  echo "Warning: Dockerfile '${dockerfile}' not found in ${REPO_ROOT}" >&2
fi

# Optional: check Azure login state
if ! az account show >/dev/null 2>&1; then
  echo "You are not logged into Azure CLI. Run: az login" >&2
fi

cmd=(az acr build --registry "${registry}" --image "${tag}" --file "${dockerfile}")

if [[ -n "${resource_group}" ]]; then
  cmd+=(--resource-group "${resource_group}")
fi
if [[ -n "${platform}" ]]; then
  cmd+=(--platform "${platform}")
fi
if [[ -n "${no_push}" ]]; then
  cmd+=(${no_push})
fi
cmd+=(".")

echo "Repository root: ${REPO_ROOT}"
echo "Executing: ${cmd[*]}"
"${cmd[@]}"
