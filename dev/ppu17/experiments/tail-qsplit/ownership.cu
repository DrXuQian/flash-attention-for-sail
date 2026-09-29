// Actual collective accumulator layouts and actual scheduler divmods, executed
// only on CPU. No launch, driver call or parallel reimplementation of GEMM.
#include <cuda_runtime.h>
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"
#include "tile_size.h"
#include "qsplit_plan.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cute;
using Plan = flash::Ppu17TailQSplitPlan;
template<int M>
using Main = flash::CollectiveMainloopFwdSm90<2, Shape<_1, _1, _1>,
    Shape<Int<M>, _128, _128>, 128, cutlass::half_t, float, cutlass::arch::Sm90,
    false, false, false, false, false, false, false, true, true, false, false, false>;
template<int M>
using Epi = flash::CollectiveEpilogueFwd<Shape<Int<M>, _128, _128>, Shape<_1, _1, _1>,
    cutlass::half_t, cutlass::arch::Sm90, Main<M>::NumMmaThreads, false, false, false, false>;
template<int M>
using Kernel = flash::FlashAttnFwdSm90<Main<M>, Epi<M>,
    flash::StaticPersistentTileScheduler<false>>;
constexpr auto selected = tile_size_fwd_sm90(128, 128, false, false, 2);
static_assert(std::get<0>(selected) == 64 && std::get<1>(selected) == 128 &&
              std::get<2>(selected) && std::get<3>(selected));
static_assert(Kernel<64>::MaxThreadsPerBlock == 256);
static_assert(Kernel<128>::MaxThreadsPerBlock == 384);

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

template<class Mainloop>
void visit(int tile, flash::StaticPersistentTileScheduler<false>::Params const& p,
           std::vector<unsigned char>& counts, const std::string& plant, bool half) {
    constexpr int M = Mainloop::kBlockM;
    using Mma = typename Mainloop::TiledMmaPV;
    const auto coord = flash::StaticPersistentTileScheduler<false>::WorkTileInfo{tile}.get_block_coord(p);
    int block = get<0>(coord), head = get<1>(coord), batch = get<2>(coord);
    require(batch == 0 && get<3>(coord) == 0, "unexpected batch/K split");
    if (half && plant == "head") --head;
    if (half && plant == "origin") block ^= 1;
    require(head * 1024 + block * M == tile * M, "wrong head or row origin");
    auto identity = make_identity_tensor(Shape<Int<M>, _128>{});
    for (int tid = 0; tid < Mainloop::NumMmaThreads; ++tid) {
        auto xy = Mma{}.get_slice(tid).partition_C(identity);
        static_assert(size(xy) == 64);
        for (int row = 0; row < 2; ++row) for (int col = 0; col < 32; ++col) {
            // Iterate the real fragment linearly; the separate lane/warp
            // anchor below knows only the public SM90 accumulator mapping.
            const int index = (col / 2) * 4 + row * 2 + col % 2;
            const auto mn = xy(index);
            const int m = get<0>(mn), n = get<1>(mn);
            const int want_m = 64 * (tid / 128) + 16 * ((tid % 128) / 32)
                             + (tid % 32) / 4 + 8 * row;
            const int want_n = 8 * (col / 2) + 2 * (tid % 4) + col % 2;
            require(m == want_m && n == want_n, "actual accumulator layout differs from lane/warp anchor");
            ++counts.at((head * 1024 + block * M + m) * 128 + n);
        }
    }
}

int main(int argc, char** argv) {
    try {
        const std::string plant = argc > 1 ? argv[1] : "none";
        const std::string mode = argc > 2 ? argv[2] : "uniform";
        require(mode == "uniform" || mode == "hybrid", "unknown ownership mode");
        require(plant == "none" || plant == "duplicate" || plant == "omit" ||
                plant == "head" || plant == "origin" || plant == "denominator", "unknown plant");
        using S = flash::StaticPersistentTileScheduler<false>;
        auto full = S::to_underlying_arguments({8, 56, 1, 1, 1, 1024, 1024, 128, 128, 2});
        auto half = S::to_underlying_arguments({16, 56, 1, 1, 1, 1024, 1024, 128, 128, 2});
        require(full.total_blocks == 448 && half.total_blocks == 896, "scheduler denominator");
        full.total_blocks = mode == "hybrid" ? Plan::Prefix : 0;
        std::vector<unsigned char> counts(56 * 1024 * 128);
        int full_tasks = 0, half_tasks = 0;
        for (int worker = 0; worker < Plan::Grid; ++worker) {
            for (int tile = worker; tile < full.total_blocks; tile += Plan::Grid) {
                visit<Main<128>>(tile, full, counts, plant, false);
                ++full_tasks;
            }
            if (mode == "uniform") {
                for (int tile = worker; tile < half.total_blocks; tile += Plan::Grid) {
                    if (plant == "omit" && tile == 895) continue;
                    visit<Main<64>>(plant == "duplicate" && tile == 895 ? 894 : tile,
                                    half, counts, plant, true);
                    ++half_tasks;
                }
            } else if (worker < Plan::TailHalves) {
                if (plant == "omit" && worker == 15) continue;
                const int tile = Plan::HalfStart + (plant == "duplicate" && worker == 15 ? 14 : worker);
                visit<Main<64>>(tile, half, counts, plant, true);
                ++half_tasks;
            }
        }
        require(mode == "uniform" ? (full_tasks == 0 && half_tasks == 896) :
                                    (full_tasks == 440 && half_tasks == 16), "task coverage denominator");
        std::size_t checked = 0;
        for (auto count : counts) { require(count == 1, "output not owned exactly once"); ++checked; }
        if (plant == "denominator") --checked;
        require(checked == 7340032, "output denominator must be 7340032");
        require(Plan::admits(1, 1024, 1024, 56, 56, 128, 128, 40), "valid geometry refused");
        for (int bad : {20, 39, 41, 72})
            require(!Plan::admits(1, 1024, 1024, 56, 56, 128, 128, bad), "wrong SM count admitted");
        require(!Plan::admits(1, 1024, 1024, 55, 56, 128, 128, 40), "wrong geometry admitted");
        std::cout << "[tail ownership] PASS mode=" << mode << " cells=" << checked << " rows=57344"
                  << " prefix=" << full_tasks << " halves=" << half_tasks
                  << " full_threads=" << Kernel<128>::MaxThreadsPerBlock
                  << " half_threads=" << Kernel<64>::MaxThreadsPerBlock
                  << " full_smem=" << Kernel<128>::SharedStorageSize
                  << " half_smem=" << Kernel<64>::SharedStorageSize
                  << " old_pipeline_offset=" << offsetof(Kernel<128>::SharedStorage, pipelines)
                  << " numerical_kernel=NOT_RUN\n";
    } catch (const std::exception& error) {
        std::cerr << "[tail ownership] FAIL: " << error.what() << '\n';
        return 1;
    }
}
