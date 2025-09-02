#!/usr/bin/env bash
set -euo pipefail

# ---- logging helpers ----
log(){ printf '%s [%s] %s\n' "$(date '+%F %T')" "$1" "$2"; }
i(){ log INFO "$*"; }
e(){ log ERROR "$*" >&2; }
ok(){ log OK "$*"; }

# ---- canonical repo root ----
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
cd "$ROOT_DIR"

# ---- load config (if present) ----
CONFIG_FILE="${1:-config.env}"
[ -f "$CONFIG_FILE" ] && . "$CONFIG_FILE" || true

# ---- defaults ----
REPO="${REPO:-farfan-app}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
PORT="${PORT:-8000}"
CPU="${CPU:-2}"
MEMORY="${MEMORY:-4}"  # in GB
CONTAINER_NAME="${CONTAINER_NAME:-$REPO}"
READ_ONLY_FS="${READ_ONLY_FS:-false}" # set to "true" to enforce read-only root fs

# ---- pre-checks (integrity & prerequisites) ----
command -v docker >/dev/null 2>&1 || { e "Docker CLI no encontrado. Instala Docker Desktop/Engine"; exit 1; }
# daemon check
if ! docker info >/dev/null 2>&1; then
  e "Docker daemon no accesible. Asegúrate de que Docker está en ejecución."; exit 1;
fi

# validate Dockerfile with canonical path
DOCKERFILE_PATH="$ROOT_DIR/$DOCKERFILE"
[ -f "$DOCKERFILE_PATH" ] || { e "No existe Dockerfile en: $DOCKERFILE_PATH"; exit 1; }

TS="$(date +%Y%m%d%H%M%S)"
IMAGE_TS_TAG="$REPO:$TS"

# ---- build image (tag latest + timestamp) ----
i "Construyendo imagen Docker: $REPO (tags: latest, $TS)"
docker build --pull --file "$DOCKERFILE_PATH" -t "$REPO:latest" -t "$IMAGE_TS_TAG" "$ROOT_DIR"
ok "Imagen construida: $REPO:latest y $IMAGE_TS_TAG"

# ---- stop/remove existing container (scoped by name) ----
i "Reemplazando contenedor si existe: $CONTAINER_NAME"
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null || true
  ok "Contenedor previo eliminado: $CONTAINER_NAME"
fi

# ---- remove previous images for this repo (safe) ----
i "Limpiando imágenes antiguas del repo $REPO (conservando: latest y $TS)"
# list images for this repository and keep only IDs not matching the two keepers
older_ids=$(docker images "$REPO" --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
  | awk -v keep1="$REPO:latest" -v keep2="$IMAGE_TS_TAG" '$1!=keep1 && $1!=keep2 {print $2}' \
  | sort -u)
if [ -n "${older_ids:-}" ]; then
  for img in $older_ids; do
    docker rmi "$img" >/dev/null || true
  done
  ok "Imágenes antiguas removidas (si existían)"
else
  ok "No se encontraron imágenes antiguas para limpiar"
fi
# prune dangling images for this repo specifically
dangling_ids=$(docker images --filter "dangling=true" --filter "reference=$REPO" -q)
if [ -n "${dangling_ids:-}" ]; then
  docker rmi $dangling_ids >/dev/null || true
fi

# ---- run container with hardened flags ----
MEM_LIMIT="${MEMORY}g"
RUN_OPTS=(
  --name "$CONTAINER_NAME"
  -p "$PORT:$PORT"
  --restart unless-stopped
  --cpus "$CPU"
  --memory "$MEM_LIMIT"
  --security-opt no-new-privileges
  --cap-drop ALL
)

if [ "$READ_ONLY_FS" = "true" ]; then
  RUN_OPTS+=(--read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m --tmpfs /var/tmp:rw,noexec,nosuid,size=64m)
fi

i "Levantando contenedor: $CONTAINER_NAME (puerto $PORT)"
docker run -d "${RUN_OPTS[@]}" "$REPO:latest"
ok "Contenedor en ejecución: $CONTAINER_NAME"
echo "URL: http://localhost:$PORT"

i "Mostrando logs (Ctrl+C para salir)"
docker logs -n 80 -f "$CONTAINER_NAME" || true
