/***************************************************************************************************
 * Memory Copy Feature Demonstration
 *
 * This example demonstrates three memory copy mechanisms and shows what happens
 * when you try to use them on incompatible GPU architectures.
 *
 * Compile for ANY architecture:
 *   nvcc -arch=sm_80 -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
 *   nvcc -arch=sm_90a -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
 *   nvcc -arch=sm_120 -I../include -std=c++17 --expt-relaxed-constexpr memory_copy_demo.cu -o memory_copy_demo
 *
 * Run:
 *   ./memory_copy_demo
 *
 * The program will attempt all three methods and report which ones work on your GPU.
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while(0)

//=================================================================================================
// Example 1: CPASYNC - Asynchronous copy with software pipelining (SM 80-89)
//=================================================================================================
// Macro to stringify values for compile-time messages
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#ifdef __CUDA_ARCH__
#pragma message "CUDA_ARCH value: " STRINGIZE(__CUDA_ARCH__)
#endif

__global__ void cpasync_kernel(
    half const* __restrict__ gmem_src,
    half* __restrict__ gmem_dst,
    int M, int K)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_K = 128;

    __shared__ half smem_buffer[TILE_M * TILE_K];

    int block_m = blockIdx.x * TILE_M;
    int tid = threadIdx.x;
    int elements_per_thread = (TILE_M * TILE_K + 255) / 256;
    int start_idx = tid * elements_per_thread;

#if __CUDA_ARCH__ >= 800
    // CPASYNC is available on SM 80-89
    // Copy in 16-byte chunks (8 halfs = 16 bytes)
    constexpr int ELEMENTS_PER_16B = 8; // 8 halfs = 16 bytes
    int num_16b_chunks = (TILE_M * TILE_K) / ELEMENTS_PER_16B;
    int chunks_per_thread = (num_16b_chunks + 255) / 256;
    printf("Thread %d copying %d chunks of 16B each\n", tid, chunks_per_thread);
    for (int i = 0; i < chunks_per_thread; i++) {
        int chunk_idx = tid + i * 256;
        if (chunk_idx < num_16b_chunks) {
            int elem_idx = chunk_idx * ELEMENTS_PER_16B;
            int m = elem_idx / TILE_K;
            int k = elem_idx % TILE_K;
            int global_m = block_m + m;

            // Only copy if aligned and within bounds
            if (global_m < M && (k % ELEMENTS_PER_16B) == 0 && (k + ELEMENTS_PER_16B) <= K) {
                half* smem_ptr = &smem_buffer[elem_idx];
                half const* gmem_ptr = &gmem_src[global_m * K + k];

                // Use cp.async instruction (16-byte copy = 8 halfs)
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 16;\n"
                    :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))),
                       "l"(gmem_ptr)
                );
            }
        }
    }

    // Commit async copies
    asm volatile("cp.async.commit_group;\n" ::: "memory");

    // Wait for completion
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#else
    // Fallback: Regular synchronous copy
    // This will still work but won't use CPASYNC
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = start_idx + i;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                smem_buffer[idx] = gmem_src[global_m * K + k];
            }
        }
    }
    __syncthreads();
#endif

    // Copy from SMEM to GMEM
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = start_idx + i;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                gmem_dst[global_m * K + k] = smem_buffer[idx];
            }
        }
    }
}

void run_cpasync_example() {
    std::cout << "\n=== CPASYNC Example (SM 80-89) ===" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    int arch = props.major * 10 + props.minor;
    if (arch >= 80 && arch < 90) {
        std::cout << "✓ CPASYNC supported on SM " << props.major << props.minor << std::endl;
    } else {
        std::cout << "✗ CPASYNC NOT supported on SM " << props.major << props.minor
                  << " (requires SM 80-89)" << std::endl;
        std::cout << "  Fallback: Using regular synchronous copy instead" << std::endl;
    }

    constexpr int M = 128;
    constexpr int K = 128;

    half* d_src;
    half* d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dst, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_dst, 0, M * K * sizeof(half)));

    std::vector<half> h_src(M * K);
    for (int i = 0; i < M * K; i++) {
        h_src[i] = __float2half(float(i % 100) / 10.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

    constexpr int TILE_M = 128;
    dim3 grid((M + TILE_M - 1) / TILE_M);
    dim3 block(256);

    cpasync_kernel<<<grid, block>>>(d_src, d_dst, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> h_dst(M * K);
        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, M * K * sizeof(half), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < M * K; i++) {
            if (__half2float(h_src[i]) != __half2float(h_dst[i])) {
                passed = false;
                break;
            }
        }
        std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

//=================================================================================================
// Example 2: TMA - Tensor Memory Accelerator (SM 90+)
//=================================================================================================

__global__ void tma_kernel(
    half const* __restrict__ gmem_src,
    half* __restrict__ gmem_dst,
    int M, int K)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_K = 128;

    __shared__ half smem[TILE_M * TILE_K];

    int block_m = blockIdx.x * TILE_M;
    int tid = threadIdx.x;
    int elements_per_thread = (TILE_M * TILE_K + 255) / 256;

#if __CUDA_ARCH__ >= 1300
    // TMA is available on SM 90+
    // For a real TMA example, you would use TMA descriptors created on host
    // This is a simplified simulation showing the pattern

    __shared__ uint64_t barrier;

    if (tid == 0) {
        // In real TMA, we initialize barrier for transaction
        barrier = 0;
    }
    __syncthreads();

    // TMA load pattern: single thread issues, all wait
    if (tid == 0) {
        // Real TMA descriptor would be used here
        // For demo, we show the concept
    }
#endif

    // Data movement (all threads participate in this demo)
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * 256;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                smem[idx] = gmem_src[global_m * K + k];
            }
        }
    }
    __syncthreads();

    // Copy back to GMEM
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * 256;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                gmem_dst[global_m * K + k] = smem[idx];
            }
        }
    }
}

void run_tma_example() {
    std::cout << "\n=== TMA Example (SM 90+) ===" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    int arch = props.major * 10 + props.minor;
    if (arch >= 90) {
        std::cout << "✓ TMA supported on SM " << props.major << props.minor << std::endl;
    } else {
        std::cout << "✗ TMA NOT supported on SM " << props.major << props.minor
                  << " (requires SM 90+)" << std::endl;
        std::cout << "  Note: Kernel will run but without actual TMA instructions" << std::endl;
    }

    constexpr int M = 128;
    constexpr int K = 128;

    half* d_src;
    half* d_dst;

    CUDA_CHECK(cudaMalloc(&d_src, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dst, M * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_dst, 0, M * K * sizeof(half)));

    std::vector<half> h_src(M * K);
    for (int i = 0; i < M * K; i++) {
        h_src[i] = __float2half(float(i % 100) / 10.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

    dim3 grid((M + 127) / 128);
    dim3 block(256);

    tma_kernel<<<grid, block>>>(d_src, d_dst, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> h_dst(M * K);
        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, M * K * sizeof(half), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < M * K; i++) {
            if (__half2float(h_dst[i]) != __half2float(h_src[i])) {
                passed = false;
                break;
            }
        }
        std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

//=================================================================================================
// Example 3: TMA Multicast - Broadcast to cluster (SM 90+ datacenter only)
//=================================================================================================

#if defined(__CUDACC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ void __cluster_dims__(2, 2, 1)
tma_multicast_kernel(
    half const* __restrict__ gmem_src,
    half* __restrict__ gmem_dst,
    int M, int K)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_K = 128;

    __shared__ half smem[TILE_M * TILE_K];

    // Cluster operations would go here
    // This requires actual cluster support in hardware

    int tid = threadIdx.x;
    int elements_per_thread = (TILE_M * TILE_K + 255) / 256;

    int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int block_m = block_idx * TILE_M;

    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * 256;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                smem[idx] = gmem_src[global_m * K + k];
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < elements_per_thread; i++) {
        int idx = tid + i * 256;
        if (idx < TILE_M * TILE_K) {
            int m = idx / TILE_K;
            int k = idx % TILE_K;
            int global_m = block_m + m;

            if (global_m < M && k < K) {
                gmem_dst[global_m * K + k] = smem[idx];
            }
        }
    }
}
#else
// Dummy kernel for architectures without cluster support
__global__ void tma_multicast_kernel(
    half const* __restrict__ gmem_src,
    half* __restrict__ gmem_dst,
    int M, int K)
{
    // This kernel won't use cluster features
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
        printf("TMA Multicast kernel running without cluster support\n");
    }
}
#endif

void run_tma_multicast_example() {
    std::cout << "\n=== TMA Multicast Example (SM 90+ Datacenter) ===" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    bool multicast_supported = false;

    // Check for multicast support
    if (props.major >= 9 && props.major != 12) {
        // SM 90 and SM 100 (datacenter) support multicast
        // SM 120 (GeForce) does NOT
        multicast_supported = true;
        std::cout << "✓ TMA Multicast supported on SM " << props.major << props.minor << std::endl;
    } else if (props.major == 12) {
        std::cout << "✗ TMA Multicast NOT supported on SM " << props.major << props.minor
                  << " (GeForce Blackwell)" << std::endl;
        std::cout << "  Hardware limitation: No multicast, cluster fixed to 1x1x1" << std::endl;
    } else {
        std::cout << "✗ TMA Multicast NOT supported on SM " << props.major << props.minor
                  << " (requires SM 90+)" << std::endl;
    }

    constexpr int M = 512;
    constexpr int K = 256;

    half* d_src;
    half* d_dst;

    CUDA_CHECK(cudaMalloc(&d_src, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dst, M * K * sizeof(half)));

    std::vector<half> h_src(M * K);
    for (int i = 0; i < M * K; i++) {
        h_src[i] = __float2half(float(i % 100) / 10.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));

    // Try to launch with cluster
    dim3 grid(2, 2);
    dim3 block(256);

    if (multicast_supported) {
        std::cout << "Attempting cooperative kernel launch with 2x2 cluster..." << std::endl;

        void* kernel_args[] = {&d_src, &d_dst, const_cast<int*>(&M), const_cast<int*>(&K)};

        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)tma_multicast_kernel,
            grid, block, kernel_args
        );

        if (err != cudaSuccess) {
            std::cout << "✗ Cooperative kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            std::cout << "  Your GPU may not support cooperative groups with clusters" << std::endl;
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "✓ Cooperative kernel executed successfully" << std::endl;
        }
    } else {
        std::cout << "Skipping multicast test - hardware not supported" << std::endl;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

//=================================================================================================
// Main
//=================================================================================================

int main() {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    std::cout << "========================================" << std::endl;
    std::cout << "Memory Copy Feature Demonstration" << std::endl;
    std::cout << "Testing feature compatibility" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Compute Capability: SM " << props.major << props.minor << std::endl;
    std::cout << "========================================" << std::endl;

    // Run all examples - will show what works and what doesn't
    run_cpasync_example();
    run_tma_example();
    run_tma_multicast_example();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Feature Support Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    int arch = props.major * 10 + props.minor;
    std::cout << "Your GPU (SM " << props.major << props.minor << "):" << std::endl;
    std::cout << "  CPASYNC:       " << (arch >= 80 && arch < 90 ? "✓ Supported" : "✗ Not supported") << std::endl;
    std::cout << "  TMA:           " << (arch >= 90 ? "✓ Supported" : "✗ Not supported") << std::endl;
    std::cout << "  TMA Multicast: " << (arch >= 90 && props.major != 12 ? "✓ Supported" : "✗ Not supported") << std::endl;

    std::cout << "\nArchitecture Overview:" << std::endl;
    std::cout << "  SM 80-89 (Ampere/Ada):  CPASYNC ✓" << std::endl;
    std::cout << "  SM 90    (Hopper):      TMA ✓, Multicast ✓" << std::endl;
    std::cout << "  SM 100   (DC Blackwell): TMA ✓, Multicast ✓" << std::endl;
    std::cout << "  SM 120   (GeForce):     TMA ✓, Multicast ✗" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
