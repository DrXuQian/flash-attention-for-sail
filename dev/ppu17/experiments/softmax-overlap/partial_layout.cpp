// Host-only enumeration of the actual collective's accumulator and token map.
// Device functions appear only in unevaluated type deduction. No CUDA calls.
#include <cuda_runtime.h>
#include "utils.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace cute;
using Mainloop = flash::CollectiveMainloopFwdSm90<
    2, Shape<_1, _1, _1>, Shape<_128, Int<176>, _128>, 128,
    cutlass::half_t, float, cutlass::arch::Sm90,
    false, false, false, false, false, false, false,
    true, true, false, false, false>;
using Mma = Mainloop::TiledMmaQK;
using Fragment = decltype(partition_fragment_C(Mma{}, Shape<_128, Int<176>>{}));
using RowLayout = decltype(flash::convert_layout_acc_rowcol(typename Fragment::layout_type{}));
static_assert(Mainloop::NumMmaThreads == 256);
static_assert(size<0>(RowLayout{}) == 2 && size<1>(RowLayout{}) == 44);
static_assert(size(typename Fragment::layout_type{}) == 88);
#if FLASHATTN_PPU17_SOFTMAX_OVERLAP == 2
static_assert(Mainloop::PartialSoftmaxOverlap && Mainloop::KeepSoftmaxOverlap);
#elif FLASHATTN_PPU17_SOFTMAX_OVERLAP >= 3
static_assert(Mainloop::RowwiseSoftmaxOverlap && Mainloop::KeepSoftmaxOverlap);
#endif

void require(bool yes, char const* message) {
    if (!yes) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        std::string plant = argc > 1 ? argv[1] : "none";
        require(plant == "none" || plant == "row" || plant == "token" || plant == "coverage",
                "unknown plant");
        auto identity = make_identity_tensor(Shape<_128, Int<176>>{});
        std::array<int, 128 * 176> visits{};
        constexpr int Words = Mainloop::RowwiseReadDependency ? 512 : 256;
        std::array<int, Words> token_owners{};
        int row0_count = 0, row1_count = 0;
        for (int tid = 0; tid < 256; ++tid) {
            auto coordinates = Mma{}.get_slice(tid).partition_C(identity);
            std::array<int, 88> fragment_visits{};
            const int token = plant == "token" ? tid % 128 : tid;
            ++token_owners.at(token);
            if constexpr (Mainloop::RowwiseReadDependency) ++token_owners.at(256 + token);
            for (int row = 0; row < 2; ++row) {
                for (int col = 0; col < 44; ++col) {
                    if (plant == "coverage" && tid == 255 && row == 1 && col == 43) continue;
                    const int index = RowLayout{}(plant == "row" ? 1 - row : row, col);
                    ++fragment_visits.at(index);
                    const auto mn = coordinates(index);
                    const int m = get<0>(mn), n = get<1>(mn);
                    // Independent SM90 accumulator lane/warp coordinates.
                    const int want_m = 64 * (tid / 128) + 16 * ((tid % 128) / 32)
                                     + (tid % 32) / 4 + 8 * row;
                    const int want_n = 8 * (col / 2) + 2 * (tid % 4) + col % 2;
                    require(m == want_m && n == want_n, "actual row layout differs from logical coordinate anchor");
                    ++visits.at(m * 176 + n);
                    (row == 0 ? row0_count : row1_count)++;
                }
            }
            for (int count : fragment_visits) require(count == 1, "fragment coverage is not exact-once");
        }
        for (int count : visits) require(count == 1, "full QK coverage is not exact-once");
        for (int count : token_owners) require(count == 1, "private token has multiple/missing owners");
        require(row0_count == 11264 && row1_count == 11264, "row denominator changed");
        std::cout << "[FA17 partial layout] PASS: actual collective 256 threads x 2 rows x 44 scores; "
                     "22528 QK cells exact-once; row0/row1 disjoint; " << Words << " private word owners\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "[FA17 partial layout] FAIL: " << error.what() << '\n';
        return 1;
    }
}
