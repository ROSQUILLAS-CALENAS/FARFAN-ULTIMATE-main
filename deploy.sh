#!/usr/bin/env bash
set -euo pipefail

log(){ printf '%s [%s] %s\n' "$(date '+%F %T')" "$1" "$2"; }
i(){ log INFO "$*"; }
e(){ log ERROR "$*" >&2; }
ok(){ log OK "$*"; }

CONFIG_FILE="${1:-config.env}"; [ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE" || true
RG="${RG:-farfan-rg}"; LOC="${LOC:-eastus}"; REPO="${REPO:-farfan-app}"
ACI_NAME="${ACI_NAME:-farfan-aci}"; DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CPU="${CPU:-2}"; MEMORY="${MEMORY:-4}"; PORT="${PORT:-8000}"

# Safety and readiness config
CONFIRM="${CONFIRM:-false}"
WAIT_READY="${WAIT_READY:-true}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"

# If not confirmed, perform a dry-run and exit without making changes
if [ "$CONFIRM" != "true" ]; then
  log INFO "DRY-RUN: No resources will be created. Review planned deployment:"
  printf ' Resource Group: %s (location: %s)\n' "$RG" "$LOC"
  printf ' ACR Repository: %s (repo: %s)\n' "<existing or new>" "$REPO"
  printf ' ACI Name: %s  CPU: %s  Memory: %sGi  Port: %s\n' "$ACI_NAME" "$CPU" "$MEMORY" "$PORT"
  printf ' Dockerfile: %s  Image tags: %s:latest and a timestamp\n' "$DOCKERFILE" "$REPO"
  printf ' Wait for readiness: %s (timeout: %ss)\n' "$WAIT_READY" "$WAIT_TIMEOUT"
  printf '\nTo proceed, set CONFIRM=true in %s or export CONFIRM=true and rerun: bash deploy.sh %s\n' "$CONFIG_FILE" "$CONFIG_FILE"
  exit 0
fi

# Prerrequisitos
for c in az; do command -v "$c" >/dev/null || { e "falta $c"; exit 1; }; done
az account show >/dev/null 2>&1 || { e "haz 'az login'"; exit 1; }
[ -f "$DOCKERFILE" ] || { e "no existe $DOCKERFILE"; exit 1; }

# Providers
for NS in Microsoft.ContainerRegistry Microsoft.ContainerInstance; do
  st=$(az provider show -n "$NS" --query registrationState -o tsv 2>/dev/null || echo NotRegistered)
  if [ "$st" != "Registered" ]; then
    i "registrando $NS"; az provider register -n "$NS" >/dev/null
    while [ "$(az provider show -n "$NS" --query registrationState -o tsv)" != "Registered" ]; do sleep 3; done
  fi
done
ok "providers OK"

# RG + ACR
az group create -n "$RG" -l "$LOC" >/dev/null
ACR_NAME="$(az acr list -g "$RG" --query '[0].name' -o tsv)"
if [ -z "${ACR_NAME:-}" ]; then
  ACR_NAME="farfanacr$RANDOM"
  az acr create -g "$RG" -n "$ACR_NAME" --sku Basic -l "$LOC" >/dev/null
fi
ok "RG=$RG ACR=$ACR_NAME"

# Build & push
TS=$(date +%Y%m%d%H%M%S)
az acr build -g "$RG" -r "$ACR_NAME" -t "$REPO:$TS" -t "$REPO:latest" -f "$DOCKERFILE" .
IMAGE="$ACR_NAME.azurecr.io/$REPO:$TS"
ok "imagen $IMAGE"

# Optional cleanup of older ACR images for this repo (preserve current TS and latest)
CLEAN_REPO="${CLEAN_REPO:-false}"
if [ "$CLEAN_REPO" = "true" ]; then
  KEEP_DIGESTS=""
  for T in "$TS" "latest"; do
    D=$(az acr repository show --name "$ACR_NAME" --image "$REPO:$T" --query digest -o tsv 2>/dev/null || true)
    [ -n "$D" ] && KEEP_DIGESTS="$KEEP_DIGESTS $D"
  done
  ALL_DIGESTS=$(az acr repository show-manifests --name "$ACR_NAME" --repository "$REPO" --query "[].digest" -o tsv 2>/dev/null || true)
  for D in $ALL_DIGESTS; do
    SKIP=false
    for K in $KEEP_DIGESTS; do
      if [ "$D" = "$K" ]; then SKIP=true; break; fi
    done
    if [ "$SKIP" = false ]; then
      i "Eliminando manifest antiguo: $D"
      az acr repository delete --name "$ACR_NAME" --image "$REPO@$D" --yes >/dev/null || true
    fi
  done
  ok "Limpieza ACR completada (CLEAN_REPO=true)"
fi

# Credenciales ACR
az acr update -n "$ACR_NAME" --admin-enabled true >/dev/null
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query 'passwords[0].value' -o tsv)

# Despliegue ACI (reemplazo)
az container delete -g "$RG" -n "$ACI_NAME" --yes --no-wait 2>/dev/null || true
DNS="farfan-$RANDOM"
az container create -g "$RG" -n "$ACI_NAME" \
  --image "$IMAGE" \
  --registry-login-server "$ACR_NAME.azurecr.io" \
  --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
  --cpu "$CPU" --memory "$MEMORY" --ports "$PORT" \
  --dns-name-label "$DNS" -l "$LOC" --restart-policy Always \
  --os-type Linux \
  --command-line "python canonical_web_server.py --port $PORT --no-analysis" >/dev/null
ok "ACI creado"

# Espera opcional hasta que el contenedor esté listo
if [ "$WAIT_READY" = "true" ]; then
  i "Esperando a que el contenedor esté listo (timeout ${WAIT_TIMEOUT}s)..."
  start_ts=$(date +%s)
  while true; do
    st=$(az container show -g "$RG" -n "$ACI_NAME" --query 'instanceView.state' -o tsv 2>/dev/null || echo "")
    if [ "$st" = "Running" ]; then ok "Estado ACI: $st"; break; fi
    if [ "$st" = "Terminated" ] || [ "$st" = "Succeeded" ] || [ "$st" = "Failed" ]; then e "Estado ACI: $st"; break; fi
    now_ts=$(date +%s)
    if [ $((now_ts - start_ts)) -ge "$WAIT_TIMEOUT" ]; then e "Timeout esperando estado Running"; break; fi
    sleep 5
  done
fi

# Estado + logs
FQDN=$(az container show -g "$RG" -n "$ACI_NAME" --query 'ipAddress.fqdn' -o tsv)
STATE=$(az container show -g "$RG" -n "$ACI_NAME" --query 'instanceView.state' -o tsv)
echo "STATE=$STATE URL=http://$FQDN:$PORT"
az container show -g "$RG" -n "$ACI_NAME" --query "{state:instanceView.state,fqdn:ipAddress.fqdn,image:containers[0].image}" -o table || true
# Azure CLI compatibility: some versions don't support --tail
if az container logs -h 2>&1 | grep -q -- "--tail"; then
  az container logs -g "$RG" -n "$ACI_NAME" --tail 80 || true
else
  az container logs -g "$RG" -n "$ACI_NAME" 2>&1 | tail -n 80 || true
fi
