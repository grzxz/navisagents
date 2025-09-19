#!/bin/bash

set -e

# Stop existing container if built and running
docker stop navis-agents 2>/dev/null || true
docker rm navis-agents 2>/dev/null || true

# Build and run
docker build -t navis-agents .
docker run -it --rm --name navis-agents -p 8001:8001 -v "$(pwd)/data:/app/data" navis-agents