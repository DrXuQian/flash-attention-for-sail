/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include<cutlass/pipeline/sm90_pipeline.hpp>
#include "ppu17_cutlass_compat.h"

namespace cutlass {

using namespace cute;

namespace detail {
// Keep the producer's expected count and the consumers' release predicate
// together. Host contracts enumerate the exact same policy over CTA threads.
struct FlashAttentionWarpgroupArrival {
  CUTLASS_HOST_DEVICE static constexpr uint32_t count(uint32_t consumers) {
    return consumers / NumThreadsPerWarpGroup;
  }
  CUTLASS_HOST_DEVICE static constexpr bool signals(uint32_t thread) {
    return thread % NumThreadsPerWarpGroup == 0;
  }
};
} // namespace detail

#if defined(FLASHATTN_PPU17) && CUTLASS_VERSION == 360
// This PPU CUTLASS3.6 tree already signals once per consumer warpgroup for
// a 1x1x1 cluster. Its three-argument constructor and matching barrier counts
// predate the newer API targeted by the FA workaround below.
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
using PipelineTmaAsyncNoCluster = Base;
#else
////////////////////////////////////////////////////////////////////////////////////////////////////

// As of Cutlass v3.6.0, if size(ClusterShape) == 1, PipelineTmaAsync has all threads
// signaling the barrier during consumer_release. This causes a perf regression in FA3
// forward pass (especially hdim 128 causal). We instead reimplement the version of
// PipelineTmaAsync before v3.6.0 where only 1 out of 128 threads signals the barrier.
//
// PPU CUTLASS4.3 has the same per-thread cluster1 policy. Use this matched
// init/release pair there too: the count AND the signalling predicate must
// remain one arrival per consumer warpgroup.
// Assumption: params.num_consumers % NumThreadsPerWarpGroup == 0
template <int Stages_, class Base=cutlass::PipelineTmaAsync<Stages_>>
class PipelineTmaAsyncNoCluster: public Base {
public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = typename Base::PipelineState;

  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;

  static
  CUTLASS_DEVICE
  void
  init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    if (is_initializing_warp) {
      // Barrier FULL and EMPTY init
      constexpr int producer_arv_cnt = 1;
      uint32_t const num_consumer_warpgroups_per_cluster = detail::FlashAttentionWarpgroupArrival::count(params.num_consumers);
      uint32_t const multicast_consumer_arrival_count = num_consumer_warpgroups_per_cluster;

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template<class ClusterShape, class InitBarriers, class InitMasks>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {}, InitMasks = {})
      : Base(storage, params, make_shape(_1{}, _1{}, _1{}) /*cluster_shape*/, cute::false_type{} /*init_barriers*/, cute::false_type{} /*init_masks*/)
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) {

#if defined(FLASHATTN_PPU17)
    static_assert(CUTE_STATIC_V(size(ClusterShape{})) == 1, "PPU1.7 admits cluster1 only");
    CUTLASS_ASSERT(params.num_consumers > 0 && params.num_consumers % NumThreadsPerWarpGroup == 0);
    CUTLASS_ASSERT(params.num_producers == 1 && params.initializing_warp == 0);
#endif
    int warp_idx = canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> || cute::is_same_v<InitBarriers, cute::false_type>);
    static_assert(cute::is_same_v<InitMasks, cute::true_type> || cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, params);
    }

  }

  // Constructor
  template<class ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, cute::true_type{}, cute::true_type{}) { }

  template<class ClusterShape, class InitBarriers>
  CUTLASS_DEVICE
  PipelineTmaAsyncNoCluster(SharedStorage& storage, Params params, ClusterShape cluster_shape, InitBarriers = {})
      : PipelineTmaAsyncNoCluster(storage, params, cluster_shape, InitBarriers{}, cute::true_type{}) { }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

private:
  EmptyBarrier* const empty_barrier_ptr_ = nullptr;

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    empty_barrier_ptr_[stage].arrive(0 /*dst_blockid_*/, uint32_t(detail::FlashAttentionWarpgroupArrival::signals(threadIdx.x)) & (!skip) /*is_signaling_thread*/);
  }

};


////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
} // end namespace cutlass
