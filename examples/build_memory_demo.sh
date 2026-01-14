#!/bin/bash

# Build script for memory_copy_demo.cu
# Automatically detects GPU architecture and builds appropriately

echo "========================================="
echo "Building Memory Copy Demo"
echo "========================================="

# Detect GPU compute capability
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

echo "Detected GPU Compute Capability: SM $GPU_ARCH"

# Determine which architecture to compile for
if [ "$GPU_ARCH" -ge "120" ]; then
    ARCH_FLAG="sm_120"
    echo "Compiling for SM 120 (GeForce Blackwell) - TMA only"
elif [ "$GPU_ARCH" -ge "100" ]; then
    ARCH_FLAG="sm_100a"
    echo "Compiling for SM 100 (Datacenter Blackwell) - All features"
elif [ "$GPU_ARCH" -ge "90" ]; then
    ARCH_FLAG="sm_90a"
    echo "Compiling for SM 90 (Hopper) - TMA + Multicast"
elif [ "$GPU_ARCH" -ge "80" ]; then
    ARCH_FLAG="sm_80"
    echo "Compiling for SM 80 (Ampere) - CPASYNC only"
else
    echo "ERROR: GPU SM $GPU_ARCH not supported (requires SM 80+)"
    exit 1
fi

# Build command
echo ""
echo "Building with: nvcc -arch=$ARCH_FLAG"

nvcc -arch=$ARCH_FLAG \
     -I../include \
     -std=c++17 \
     --expt-relaxed-constexpr \
     -O3 \
     memory_copy_demo.cu \
     -o memory_copy_demo

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build successful!"
    echo "========================================="
    echo "Run with: ./memory_copy_demo"
    echo ""
else
    echo ""
    echo "========================================="
    echo "Build failed!"
    echo "========================================="
    exit 1
fi
