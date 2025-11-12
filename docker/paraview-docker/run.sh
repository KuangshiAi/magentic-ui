#!/bin/bash

# Docker network name for ParaView and Python environment to communicate
NETWORK_NAME=my-network
CONTAINER_NAME=paraview-server

# Create network if it doesn't exist
docker network create ${NETWORK_NAME} 2>/dev/null || true

echo "Starting ParaView container on network: ${NETWORK_NAME}"
echo "Container name: ${CONTAINER_NAME}"
echo "Access ParaView GUI at: http://localhost:6080"
echo "pvserver listening on port: 11111"

docker run --platform=linux/amd64 -it --rm \
  --name ${CONTAINER_NAME} \
  --network ${NETWORK_NAME} \
  -p 6080:6080 -p 5900:5900 -p 11111:11111 \
  -v ../data:/home/MCPagent/data \
  paraview-novnc:latest