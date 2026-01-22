#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Universal container build script that works with Apptainer or Docker

set -e

# Check which container runtime is available
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNTIME="apptainer"
    echo "[INFO] Using Apptainer"
elif command -v docker &> /dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "[INFO] Using Docker"
else
    echo "[ERROR] Neither Apptainer nor Docker is available"
    echo "[ERROR] Please install either Apptainer or Docker to continue"
    exit 1
fi

# Build based on detected runtime
if [ "$CONTAINER_RUNTIME" = "apptainer" ]; then
    echo "[INFO] Building with Apptainer..."
    
    # Use shared cache directory on same filesystem as runner's workspace
    # This prevents filling up the home filesystem and allows image reuse across jobs
    APPTAINER_CACHE_DIR="${RUNNER_WORKSPACE:-$(dirname "$(pwd)")}/apptainer_cache"
    mkdir -p "$APPTAINER_CACHE_DIR"
    
    # Set Apptainer temp directory to avoid filling up /tmp
    # Use the same filesystem as the cache directory
    export APPTAINER_TMPDIR="$APPTAINER_CACHE_DIR/tmp"
    mkdir -p "$APPTAINER_TMPDIR"
    
    # Build Apptainer image from definition file (only if it doesn't exist)
    if [ ! -f "$APPTAINER_CACHE_DIR/iris-dev.sif" ]; then
        echo "[INFO] Building new Apptainer image at $APPTAINER_CACHE_DIR/iris-dev.sif..."
        echo "[INFO] Using temp directory: $APPTAINER_TMPDIR"
        apptainer build "$APPTAINER_CACHE_DIR/iris-dev.sif" apptainer/iris.def
    else
        echo "[INFO] Using existing Apptainer image at $APPTAINER_CACHE_DIR/iris-dev.sif"
    fi
    
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    echo "[INFO] Checking Docker images..."
    # Use GitHub variable if set, otherwise default to iris-dev
    IMAGE_NAME=${DOCKER_IMAGE_NAME:-"iris-dev"}
    
    # Check if the image exists
    if docker image inspect "$IMAGE_NAME" &> /dev/null; then
        echo "[INFO] Using existing Docker image: $IMAGE_NAME"
    else
        echo "[WARNING] Docker image $IMAGE_NAME not found"
        echo "[INFO] Please build it using: ./build_triton_image.sh"
        echo "[INFO] Or pull it if available from registry"
    fi
fi

echo "[INFO] Container build completed successfully with $CONTAINER_RUNTIME"
