#!/bin/bash

set -e

echo "Pulling latest code from Git..."
git pull

echo "Building Docker image..."
docker build -t larry:nnwipqc .

echo "Running Docker container..."
docker run -it --rm --gpus "device=3" -v ./results:/app/results larry:nnwipqc