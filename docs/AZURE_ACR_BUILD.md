# Azure ACR Build Guide for FARFAN-ULTIMATE

This guide helps you build and push the Docker image to Azure Container Registry (ACR) without common pitfalls.

## Prerequisites
- Azure CLI installed and logged in: `az login`
- An existing Azure Container Registry (ACR)
- This repository cloned locally

## Correct project path
Your project root is:

```
/Users/recovered/PycharmProjects/FARFAN-ULTIMATE
```

Always run build commands from this directory.

## Quick start (recommended)
Use the helper script which validates inputs and runs `az acr build` with the right context:

```
cd /Users/recovered/PycharmProjects/FARFAN-ULTIMATE
# Either export ACR_NAME or pass --registry/-r
export ACR_NAME=MyRegistry
bash tools/azure_acr_build.sh -g farfan-rg -t farfan-app:latest
```

Options:
- `-r/--registry` or `ACR_NAME` env: ACR name (required)
- `-g/--resource-group`: Resource group of your ACR (optional)
- `-t/--tag`: Image tag (default: `farfan-app:latest`)
- `-f/--file`: Dockerfile path (default: `Dockerfile`)
- `--platform`: e.g., `linux/amd64` (optional)
- `--no-push`: Build without pushing (optional)

## Raw Azure CLI (manual)
You can also invoke the Azure CLI directly:

```
cd /Users/recovered/PycharmProjects/FARFAN-ULTIMATE
# Ensure ACR_NAME is set, or pass --registry explicitly
export ACR_NAME=MyRegistry
az acr build \
  --registry "$ACR_NAME" \
  --resource-group farfan-rg \
  --image farfan-app:latest \
  --file Dockerfile \
  .
```

## Common errors and fixes
- Error: `cd: no such file or directory: /Users/recovered/FARFAN-ULTIMATE`
  - Fix: Use the correct path: `/Users/recovered/PycharmProjects/FARFAN-ULTIMATE`
- Error: `argument --registry/-r: expected one argument`
  - Fix: Set the environment variable or pass the argument, e.g.: `export ACR_NAME=MyRegistry` or `--registry MyRegistry`
- Error: `az: command not found`
  - Fix: Install Azure CLI: https://learn.microsoft.com/cli/azure/install-azure-cli
- Warning: Large build context
  - Note: The repo includes a `.dockerignore` that excludes `venv/` and `.venv/` to avoid uploading local virtual environments.

## Verify your ACR
List registries in your resource group:

```
az acr list --resource-group farfan-rg --output table
```

## About the Dockerfile
A Dockerfile is provided at the repo root targeting Python 3.12 slim. It installs `requirements.txt`, copies the project, and runs `main.py -v` by default. You can override the command by passing `--build-arg` and/or setting a different `CMD` during runtime as needed.

## Notes
- The helper script auto-detects the repo root and ensures the build context is `.` at the correct location.
- If your local shell cannot execute the script directly, use `bash tools/azure_acr_build.sh ...`.
