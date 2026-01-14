/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


/*! \file
    \brief Pure FP8 Grouped GEMM example using CUTLASS 3x APIs for the NVIDIA Blackwell SM120 architecture.

    This example demonstrates an implementation of Grouped GEMM using a TMA + Blackwell SM120 TensorOp-based warp-specialized kernel
    for pure FP8 (float_e4m3_t) precision with tensorwide scaling (applied in epilogue) and BF16 output.

    This example uses the same approach as cuBLAS FP8 GEMM:
    - Pure FP8 data (float_e4m3_t) without MX block-scaled wrappers
    - Tensorwide scaling via alpha/beta in the epilogue
    - No per-block scale factors (unlike 79g's previous groupwise scaling)

    This provides a fair comparison against 79f (MXFP8 x MXFP4) to measure the overhead of block scaling.

    For this example all scheduling work is performed on the device, utilizing the device-side modification of TMA descriptors
    to move between groups/problem_count (represented by groups).
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/#encoding-a-tensor-map-on-device

    To run this example:

      $ ./examples/79_blackwell_geforce_gemm/79g_blackwell_geforce_fp8_grouped_gemm --m=2048 --n=2048 --k=2048 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes.
      Skipping any of the problem dimensions randomizes it across the different groups.
      Same applies for alpha and beta values that are randomized across the different groups.

    To run this example for a set of problems using the benchmark option:

      $ ./examples/79_blackwell_geforce_gemm/79g_blackwell_geforce_fp8_grouped_gemm --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <float.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "helper.h"
using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using ElementInput = cutlass::float_e4m3_t;                                // Element type for Input matrix operands (Pure FP8)

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
// A matrix configuration (Pure FP8 - no MX wrapper, tensorwide scaling)
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand (Pure FP8)
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment (128 bits / 8 bits = 16)

// B matrix configuration (Pure FP8 - no MX wrapper, tensorwide scaling)
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand (Pure FP8)
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Alignment (128 bits / 8 bits = 16)

// C/D matrix configuration (BF16 output)
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operands (BF16 output)
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operands
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands

constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Alignment of D matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for internal computation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // TensorOp (pure FP8, no block scaling)

// Kernel Perf config
// Cluster Shape fixed to 1x1x1 for GeForce
using ThreadBlockShape    = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_1,_1>;

// D = alpha * acc + beta * C
// Tensorwide scaling is done via alpha in the epilogue (like cuBLAS)

// Collective Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag *, AlignmentC,
    ElementD, LayoutDTag *, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Cooperative kernel schedule - Plain layouts (no tuple, no scale factors)
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                            // Auto schedule for best performance
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Pingpong kernel schedule
using CollectiveMainloopPingpong = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag *, AlignmentA,
    ElementB, LayoutBTag *, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong                  // Pingpong schedule for grouped GEMM
>::CollectiveOp;

using GemmKernelPingpong = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopPingpong,
    CollectiveEpilogue
>;

using GemmPingpong = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelPingpong>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// Host-side allocations
std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

using HostTensorA = cutlass::HostTensor<ElementA, cutlass::layout::PackedVectorLayout>;
using HostTensorB = cutlass::HostTensor<ElementB, cutlass::layout::PackedVectorLayout>;
using HostTensorC = cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout>;
using HostTensorD = cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>;
std::vector<HostTensorA> block_A;
std::vector<HostTensorB> block_B;
std::vector<HostTensorC> block_C;
std::vector<HostTensorD> block_D;
std::vector<HostTensorD> block_ref_D;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<const ElementA *> ptr_A;
cutlass::DeviceAllocation<const ElementB *> ptr_B;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<ElementD *> ptr_D;
cutlass::DeviceAllocation<ElementD *> ptr_ref_D;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;

// Note, this is an array of pointers to alpha and beta scaling values per group
cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
// Command line options parsing
struct Options {

  bool help = false;
  bool verification = true;
  bool use_pdl = false;

  float alpha = std::numeric_limits<float>::max();
  float beta  = std::numeric_limits<float>::max();
  float norm_constant = 1.0;
  int iterations = 10;
  int m = 1024, n = 2048, k = 512, groups = 10;
  RasterOrderOptions raster_order = RasterOrderOptions::AlongN;
  int max_sm_count = INT_MAX;
  std::string benchmark_path;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<ElementInput>::value;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    if (cmd.check_cmd_line_flag("no_verif")) {
      verification = false;
    }
    if (cmd.check_cmd_line_flag("use_pdl")) {
      use_pdl = true;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, std::numeric_limits<float>::max());
    cmd.get_cmd_line_argument("beta",  beta,  std::numeric_limits<float>::max());
    cmd.get_cmd_line_argument("norm_constant",  norm_constant,  float(1.0));
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);
    cmd.get_cmd_line_argument("max_sm_count", max_sm_count, INT_MAX);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster_order = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster_order = RasterOrderOptions::AlongM;
    }
  }

  void randomize_problems(cutlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_host.reserve(groups);

    for (int i = groups; i > 0; i--) {
      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;
      if (m < 1) {
        m = alignment * ((rand() % 64) + 1);
      }
      if (n < 1) {
        n = alignment * ((rand() % 64) + 1);
      }
      if (k < 1) {
        k = alignment * ((rand() % 64) + 1);
      }
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      cutlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      cutlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
      }
    }
    groups = static_cast<int>(problem_sizes_host.size());

    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "79g_blackwell_geforce_fp8_grouped_gemm\n\n"
      << "  Blackwell Pure FP8 Grouped GEMM with Tensorwide Scaling and BF16 output.\n"
      << "  Uses the same approach as cuBLAS FP8 GEMM (no per-block scale factors).\n\n"
      << "Options:\n\n"
      << "  --help                                                       If specified, displays this usage statement\n\n"
      << "  --m=<int>                                                    Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                                                    Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                                                    Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>                                               Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>                                                Epilogue scalar alpha (includes tensorwide scale)\n"
      << "  --beta=<f32>                                                 Epilogue scalar beta\n"
      << "  --norm_constant=<f32>                                        Epilogue scalar normalization constant for the output matrix\n\n"
      << "  --raster=<char>                                              CTA Rasterization direction (N for along N, M for along M)\n\n"
      << "  --iterations=<int>                                           Number of profiling iterations to perform\n\n"
      << "  --benchmark=<str>                                            Executes a benchmark problem size\n"
      << "  --max_sm_count=<int>                                         Run kernels using only these number of SMs\n"
      << "  --no_verif                                                   Do not run (host-side) verification kernels\n"
      << "  --use_pdl                                                    Launch kernel with PDL (Programmatic Dependent Launch) enabled\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "79g_blackwell_geforce_fp8_grouped_gemm" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data
void allocate(const Options &options) {
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto stride_A_val = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B_val = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C_val = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D_val = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_A = make_layout(make_shape(M, K, 1), stride_A_val);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B_val);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C_val);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D_val);

    stride_A_host.push_back(stride_A_val);
    stride_B_host.push_back(stride_B_val);
    stride_C_host.push_back(stride_C_val);
    stride_D_host.push_back(stride_D_val);

    block_A.push_back(HostTensorA(cutlass::make_Coord(size(layout_A))));
    block_B.push_back(HostTensorB(cutlass::make_Coord(size(layout_B))));
    block_C.push_back(HostTensorC(cutlass::make_Coord(size(layout_C))));
    block_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
    block_ref_D.push_back(HostTensorD(cutlass::make_Coord(size(layout_D))));
  }
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  uint64_t seed = 2020;
  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementD *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {

    initialize_block(block_A.at(i).host_view(), seed + 2021);
    initialize_block(block_B.at(i).host_view(), seed + 2022);
    initialize_block(block_C.at(i).host_view(), seed + 2023);

    block_A.at(i).sync_device();
    block_B.at(i).sync_device();
    block_C.at(i).sync_device();

    ptr_A_host.at(i) = block_A.at(i).device_data();
    ptr_B_host.at(i) = block_B.at(i).device_data();
    ptr_C_host.at(i) = block_C.at(i).device_data();
    ptr_D_host.at(i) = block_D.at(i).device_data();

    alpha_host.push_back((options.alpha == std::numeric_limits<float>::max()) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == std::numeric_limits<float>::max()) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename GemmType>
typename GemmType::Arguments args_from_options(Options &options, bool host_problem_shapes_available = true)
{
  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id), options.max_sm_count);

  typename GemmType::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;

  // If alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
  // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
  if (options.alpha != std::numeric_limits<float>::max()){
    // Single alpha for all groups
    fusion_args.alpha = options.alpha;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.alpha = 0;
    fusion_args.alpha_ptr_array = alpha_device.get();
    // Only one alpha per each group
    fusion_args.dAlpha = {_0{}, _0{}, 1};
  }
  if (options.beta != std::numeric_limits<float>::max()) {
    // Single beta for all groups
    fusion_args.beta = options.beta;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {_0{}, _0{}, 0};
  }
  else {
    fusion_args.beta = 0;
    fusion_args.beta_ptr_array = beta_device.get();
    // Only one beta per each group
    fusion_args.dBeta = {_0{}, _0{}, 1};
  }

  typename GemmType::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = options.raster_order;

  if (host_problem_shapes_available) {
    arguments = typename GemmType::Arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info, scheduler
    };
  }
  else {
    arguments = typename GemmType::Arguments {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), nullptr},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info, scheduler
    };
  }

  return arguments;
}

bool verify(const Options &options) {
  using namespace cute;
  bool passed = true;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    auto stride_A_val = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B_val = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C_val = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D_val = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_A = make_layout(make_shape(M, K, 1), stride_A_val);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B_val);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C_val);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D_val);

    // Create the arguments for host reference implementation
    Tensor tensor_A = make_tensor(make_iterator(block_A.at(i).host_data()), layout_A);
    Tensor tensor_B = make_tensor(make_iterator(block_B.at(i).host_data()), layout_B);
    auto tensor_C = cute::make_tensor(make_iterator(block_C.at(i).host_data()), layout_C);
    auto tensor_ref_D = cute::make_tensor(make_iterator(block_ref_D.at(i).host_data()), layout_D);

    // Standard mainloop params (no scale factors)
    cutlass::reference::host::GettMainloopParams<ElementAccumulator,
        decltype(tensor_A),
        decltype(tensor_B)
      >
    mainloop_params{tensor_A, tensor_B};

    // Standard epilogue params
    cutlass::reference::host::GettEpilogueParams<
        ElementAccumulator,                   // ElementScalar
        ElementAccumulator,                   // ElementAccumulator
        ElementAccumulator,                   // ElementCompute
        decltype(tensor_C),                   // TensorC
        decltype(tensor_ref_D)                // TensorD
      > epilogue_params;

    epilogue_params.C = tensor_C;
    epilogue_params.D = tensor_ref_D;
    epilogue_params.alpha = alpha_host.at(i);
    epilogue_params.beta = beta_host.at(i);

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    // Comparison
    block_D.at(i).sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    passed &= cutlass::reference::host::TensorEquals(block_ref_D.at(i).host_view(), block_D.at(i).host_view());
    // Check that the tensors have non-zero norms
    passed &= (cutlass::reference::host::TensorNorm(block_ref_D.at(i).host_view()) > 0);
    passed &= (cutlass::reference::host::TensorNorm(block_D.at(i).host_view()) > 0);
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename GemmType>
int run(Options &options, bool host_problem_shapes_available = true)
{
  std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
  for (int32_t i = 0; i < options.groups; ++i) {
    std::cout << "    " << options.problem_sizes_host.at(i);
    std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
  }
  std::cout << "  Groups      : " << options.groups  << std::endl;

  // Instantiate CUTLASS kernel depending on templates
  GemmType gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<GemmType>(options, host_problem_shapes_available);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmType::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));

  cudaDeviceSynchronize();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.verification) {
    std::cout << "  Host-side verification is now running - may be very slow for large cases." << std::endl;
    result.passed = verify(options);
    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
    if (!result.passed) {
      exit(-1);
    }
  }
  else {
    std::cout << "  Verification is turned off for this run." << std::endl;
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ options.use_pdl));
    }
    timer.stop();

    // Compute average setup and runtime and GFLOPs.
    float elapsed_ms       = timer.elapsed_millis();
    result.avg_runtime_ms  = double(elapsed_ms) / double(options.iterations);
    result.gflops          = options.gflops(result.avg_runtime_ms / 1000.0, options.problem_sizes_host);

    std::cout << "  Avg runtime : " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  TFLOPS      : " << result.gflops / 1000.0 << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit for SM120 support,
  // or CUDA 12.9 or higher for SM121 support.
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 ||
       ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
       )
     ) {
    std::cerr << "This example requires CUDA 12.8 or newer for SM120 support.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#elif defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 9)) {
    std::cerr << "This example requires CUDA 12.9 or newer for SM121 support.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
#endif

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Blackwell Architecture (compute capability 120 or 121).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  allocate(options);
  initialize(options);

  //
  // Evaluate CUTLASS kernels
  //

  std::cout << "Running kernel with Cooperative kernel schedule:" << std::endl;
  run<Gemm>(options, false /*host_problem_shapes_available*/);
  std::cout << "Running kernel with Pingpong kernel schedule:" << std::endl;
  run<GemmPingpong>(options, false /*host_problem_shapes_available*/);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
