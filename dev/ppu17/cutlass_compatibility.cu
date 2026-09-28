// Compile-only contracts for the actual FA epilogue and no-cluster pipeline.
// No CUDA entrypoint in this file is executed by the local checker.
#include "flash_fwd_launch_template.h"

using Arrival = cutlass::detail::FlashAttentionWarpgroupArrival;
using Pipeline = cutlass::PipelineTmaAsyncNoCluster<2>;
using BasePipeline = cutlass::PipelineTmaAsync<2>;

#if CUTLASS_VERSION == 360
static_assert(cute::is_same_v<Pipeline, BasePipeline>, "3.6 uses its original matched pipeline");
#else
static_assert(!cute::is_same_v<Pipeline, BasePipeline>, "4.3 must retain warpgroup, not per-thread arrivals");
#endif

// Independent enumeration: one loader warpgroup, then 1..7 complete consumer
// warpgroups. Count every thread which will actually signal the barrier.
constexpr bool arrivals_match() {
  for (unsigned consumers = 128; consumers <= 896; consumers += 128) {
    unsigned signals = 0;
    for (unsigned thread = 128; thread < 128 + consumers; ++thread) {
      signals += Arrival::signals(thread);
    }
    if (signals != consumers / 128 || signals != Arrival::count(consumers)) return false;
  }
  return true;
}
static_assert(arrivals_match(), "barrier expected count differs from actual consumer arrivals");

template <class Element, int Dim>
constexpr bool output_store_matches() {
  using Epilogue = flash::CollectiveEpilogueFwd<
      cute::Shape<cute::_128, cute::Int<Dim>, cute::_64>,
      cute::Shape<cute::_1, cute::_1, cute::_1>, Element, cutlass::arch::Sm90,
      256, false, false, false>;
  return cute::is_same_v<typename Epilogue::CopyOpR2S, cute::SM90_U32x4_STSM_N>;
}
static_assert(output_store_matches<cutlass::half_t, 64>());
static_assert(output_store_matches<cutlass::half_t, 128>());
static_assert(output_store_matches<cutlass::half_t, 256>());
static_assert(output_store_matches<cutlass::bfloat16_t, 64>());
static_assert(output_store_matches<cutlass::bfloat16_t, 128>());
static_assert(output_store_matches<cutlass::bfloat16_t, 256>());

// Real constructor/release code with two consumer warpgroups (D128/D256).
// PTX must initialize the empty barrier with 2, not 256. D64 uses three
// consumer warpgroups, also covered by arrivals_match() and the full TU gate.
extern "C" __global__ void fa_pipeline_warpgroup_receipt() {
  extern __shared__ __align__(128) char storage_bytes[];
  auto& storage = *reinterpret_cast<Pipeline::SharedStorage*>(storage_bytes);
  Pipeline::Params params;
  params.transaction_bytes = 128;
  params.role = Pipeline::ThreadCategory::Consumer;
  params.num_consumers = 256;
  Pipeline pipeline(storage, params, cute::make_shape(cute::_1{}, cute::_1{}, cute::_1{}));
  __syncthreads();
  if (threadIdx.x >= 128 && threadIdx.x < 384) {
    pipeline.consumer_release(typename Pipeline::PipelineState{});
  }
}
