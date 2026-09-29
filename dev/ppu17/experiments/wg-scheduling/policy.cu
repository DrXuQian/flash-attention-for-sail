// Actual shipping collective policy/types, executed only on CPU.
#include <cuda_runtime.h>
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"
#include "tile_size.h"
#include <iostream>

using namespace cute;
template<class Element = cutlass::half_t, bool Causal = false, int M = 128, int N = 128>
using Main = flash::CollectiveMainloopFwdSm90<2, Shape<_1, _1, _1>,
    Shape<Int<M>, Int<N>, _128>, 128, Element, float, cutlass::arch::Sm90,
    Causal, false, false, false, false, false, false, true, true, false, false, false>;
using Epi = flash::CollectiveEpilogueFwd<Shape<_128, _128, _128>, Shape<_1, _1, _1>,
    cutlass::half_t, cutlass::arch::Sm90, Main<>::NumMmaThreads,
    false, false, false, false>;
using Kernel = flash::FlashAttnFwdSm90<Main<>, Epi, flash::StaticPersistentTileScheduler<false>>;

constexpr auto selected = tile_size_fwd_sm90(128, 128, false, false, 2);
static_assert(std::get<0>(selected) == 128 && std::get<1>(selected) == 128);
static_assert(Main<>::IndependentWarpGroups == flash::kPpu17IndependentWG);
static_assert(Main<>::UseSchedulerBarrier == !flash::kPpu17IndependentWG);
static_assert(Main<>::requires_kv_wait(0), "WG0 must wait for K/V readiness");
static_assert(Main<>::requires_kv_wait(1) == flash::kPpu17IndependentWG,
              "WG1 must wait for K/V when pingpong is disabled");
static_assert(Main<>::NumMmaThreads == 256 && Main<>::NumMmaThreadsQK == 256);
static_assert(Main<>::NumProducerThreads == 32 && Main<>::NumMmaWarpGroups == 2);
static_assert(Main<>::NumMmaThreadsQK + Main<>::NumProducerThreads == 288);
static_assert(Kernel::MaxThreadsPerBlock == 384 && Kernel::SharedStorageSize == 166912);
static_assert(Kernel::LoadRegisterRequirement == 24 && Kernel::MmaRegisterRequirement == 240);
static_assert(Main<>::TmaTransactionBytesQ == 32768 && Main<>::TmaTransactionBytesK == 32768
              && Main<>::TmaTransactionBytesV == 32768);
static_assert(!Main<cutlass::half_t, true>::IndependentWarpGroups);
static_assert(Main<cutlass::half_t, true>::UseSchedulerBarrier);
static_assert(!Main<cutlass::bfloat16_t>::IndependentWarpGroups);
static_assert(Main<cutlass::bfloat16_t>::UseSchedulerBarrier);
static_assert(!Main<cutlass::half_t, false, 64>::IndependentWarpGroups);
static_assert(!Main<cutlass::half_t, false, 128, 176>::IndependentWarpGroups);
static_assert(Main<cutlass::half_t, false, 128, 176>::UseSchedulerBarrier);

int main() {
    using Arrival = cutlass::detail::FlashAttentionWarpgroupArrival;
    int signals = 0;
    for (int thread = 128; thread < int(Kernel::MaxThreadsPerBlock); ++thread)
        signals += Arrival::signals(thread);
    if (signals != 2 || signals != int(Arrival::count(Main<>::NumMmaThreads))) return 1;
    std::cout << "[WG policy] PASS independent=" << Main<>::IndependentWarpGroups
              << " math_threads=256 producer_threads=32 CTA_threads=384 QueryEmpty=288"
              << " KV_release_arrivals=" << signals << " shared_bytes=" << Kernel::SharedStorageSize
              << " KV_wait_WG0=" << Main<>::requires_kv_wait(0)
              << " KV_wait_WG1=" << Main<>::requires_kv_wait(1)
              << " device=NOT_RUN\n";
}
