# Memory Copy Feature Demonstration

This demo shows three CUDA memory copy mechanisms in a single file:

## Features Demonstrated

1. **CPASYNC** (SM 80-89) - Asynchronous copy with software pipelining
   - Thread-level async copies from GMEM → SMEM
   - Uses `cp.async` instruction
   - Multiple threads issue copies, then wait via barriers

2. **TMA** (SM 90+) - Tensor Memory Accelerator
   - Hardware-accelerated bulk tensor transfer
   - Single thread initiates, all threads wait on barrier
   - More efficient than CPASYNC for large transfers

3. **TMA Multicast** (SM 90+, datacenter only)
   - Broadcasts TMA data to multiple CTAs in a cluster
   - Single memory load shared across cluster
   - NOT available on GeForce RTX 50 series (SM 120)

## Quick Start

### Automatic Build (Recommended)

```bash
cd /home/jerry/Documents/cutlass/examples
./build_memory_demo.sh
./memory_copy_demo
```

The build script automatically detects your GPU and compiles accordingly.

### Manual Build

For **RTX 5090** (SM 120):
```bash
nvcc -arch=sm_120 -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
```

For **Hopper/H100** (SM 90):
```bash
nvcc -arch=sm_90a -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
```

For **Datacenter Blackwell** (SM 100):
```bash
nvcc -arch=sm_100a -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
```

For **Ampere/Ada** (SM 80):
```bash
nvcc -arch=sm_80 -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
```

## Expected Output on RTX 5090

```
========================================
Memory Copy Feature Demonstration
========================================
GPU: NVIDIA GeForce RTX 5090
Compute Capability: SM 120
========================================

=== CPASYNC Example (SM 80+) ===
CPASYNC requires SM 80-89 (Ampere/Ada). Skipping...

=== TMA Example (SM 90+) ===
TMA Result: PASSED
Single-thread initiated bulk transfer of 256x256 elements

=== TMA Multicast Example (SM 90+ Datacenter) ===
TMA Multicast NOT supported on SM 120 (GeForce Blackwell)
This feature requires datacenter GPUs (H100, B100/B200)

========================================
Feature Support Summary:
========================================
SM 80-89 (Ampere/Ada):  CPASYNC ✓
SM 90    (Hopper):      TMA ✓, TMA Multicast ✓
SM 100   (Datacenter):  TMA ✓, TMA Multicast ✓
SM 120   (GeForce):     TMA ✓ (no multicast)
========================================
```

## What Works on Your GPU

| GPU | CPASYNC | TMA | TMA Multicast |
|-----|---------|-----|---------------|
| RTX 3090/4090 (SM 80/89) | ✅ | ❌ | ❌ |
| H100 (SM 90) | ❌ | ✅ | ✅ |
| B100/B200 (SM 100) | ❌ | ✅ | ✅ |
| **RTX 5090 (SM 120)** | ❌ | ✅ | ❌ |

## Key Concepts

### CPASYNC
- Multi-threaded async copies
- Software pipelining with `cp.async.commit_group` and `cp.async.wait_group`
- Good for irregular access patterns

### TMA
- Single-threaded bulk transfer
- Hardware manages the entire copy
- Requires barrier synchronization
- More efficient than CPASYNC for regular tensor patterns

### TMA Multicast
- One TMA load broadcasts to multiple CTAs
- Reduces memory bandwidth
- Requires cluster launch (`__cluster_dims__`)
- Hardware feature removed from GeForce GPUs

## Files Created

- `memory_copy_demo.cu` - Main demonstration code
- `build_memory_demo.sh` - Automatic build script
- `MEMORY_COPY_DEMO_README.md` - This file

## Troubleshooting

**Build fails:**
- Make sure CUTLASS include directory exists: `ls ../include/cute/`
- Check CUDA version: `nvcc --version` (requires CUDA 11.8+ for SM 90+)

**Runtime fails:**
- Check GPU: `nvidia-smi`
- Verify compute capability matches compiled architecture

**TMA Multicast doesn't work:**
- This is expected on RTX 5090 (no multicast hardware)
- Requires datacenter GPU (H100, B100, B200)
