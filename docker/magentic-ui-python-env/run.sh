#!/bin/bash

# Docker network name - must match the ParaView container network
NETWORK_NAME=my-network
CONTAINER_NAME=python-client

# Create network if it doesn't exist (in case ParaView container isn't running yet)
docker network create ${NETWORK_NAME} 2>/dev/null || true

# Get the absolute path to the workspace root (2 levels up from this script)
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Starting Python environment container on network: ${NETWORK_NAME}"
echo "Container name: ${CONTAINER_NAME}"
echo "Workspace mounted at: /workspace"
echo "Connect to ParaView using hostname: paraview-server"
echo ""
echo "Running interactive bash shell..."

docker run --rm -it \
  --name ${CONTAINER_NAME} \
  --network ${NETWORK_NAME} \
  --platform linux/arm64 \
  -v "${WORKSPACE_ROOT}:/workspace" \
  -w /workspace \
  ghcr.io/microsoft/magentic-ui-python-env:latest \
  bash
