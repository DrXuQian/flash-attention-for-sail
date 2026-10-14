// Host test of the actual device iterator. No independent device-math model.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "softmax_row_schedule.h"

using FLASH_NAMESPACE::detail::for_each_softmax_column;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <int Rows, int Columns, int Begin>
bool check_trace(bool drop_last = false, bool swap_columns = false) {
    std::array<std::vector<int>, Rows> traces;
    int visits = 0;
    for_each_softmax_column<Rows, Columns, Begin>([&](int row, int col) {
        if (drop_last && row == Rows-1 && col == Columns-1) return;
        if (swap_columns && col < Begin+2 && col < Columns) col = 2*Begin+1-col;
        traces[row].push_back(col);
        ++visits;
    });
    if (visits != Rows*(Columns-Begin)) return false;
    for (int row = 0; row < Rows; ++row) {
        if (int(traces[row].size()) != Columns-Begin) return false;
        for (int col = Begin; col < Columns; ++col)
            if (traces[row][col-Begin] != col) return false;
    }
    return true;
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

template <bool ZeroInit>
void check_ordered_math(uint32_t& rng, bool masked) {
    // Four independently scaled rows, sixteen columns: the exact 128/64/W4
    // per-thread score shape. CPU exp2 is only a host order test, not a claim
    // about PPU exp2 approximation. Device raw-bit comparison is still required.
    auto random = [&]() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return float(int(rng & 65535)-32768)/2048.f;
    };
    std::array<std::array<float, 16>, 4> source, control, candidate;
    std::array<float, 4> max, sum_control, sum_candidate;
    for (int row = 0; row < 4; ++row) {
        max[row] = masked ? -INFINITY : std::abs(random());
        sum_candidate[row] = sum_control[row] = random();
        for (int col = 0; col < 16; ++col)
            source[row][col] = masked ? -INFINITY : random();
    }
    const float scale = 0.127517432f;
    auto evaluate = [&](int row, int col) {
        float bias = max[row] == -INFINITY ? 0.f : max[row]*scale;
        return std::exp2(std::fma(source[row][col], scale, -bias));
    };
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 16; ++col)
            control[row][col] = evaluate(row, col);
    for_each_softmax_column<4, 16>([&](int row, int col) {
        candidate[row][col] = evaluate(row, col);
    });
    for (int row = 0; row < 4; ++row) {
        sum_control[row] = ZeroInit ? control[row][0] : sum_control[row]+control[row][0];
        sum_candidate[row] = ZeroInit ? candidate[row][0] : sum_candidate[row]+candidate[row][0];
        for (int col = 1; col < 16; ++col) sum_control[row] += control[row][col];
    }
    for_each_softmax_column<4, 16, 1>([&](int row, int col) {
        sum_candidate[row] += candidate[row][col];
    });
    for (int row = 0; row < 4; ++row) {
        require(bits(sum_control[row]) == bits(sum_candidate[row]), "ordered sum changed");
        for (int col = 0; col < 16; ++col)
            require(bits(control[row][col]) == bits(candidate[row][col]), "element changed");
    }
}

int main() {
    require(check_trace<4,16,0>() && check_trace<4,16,1>(), "exact geometry trace");
    require(check_trace<1,1,0>() && check_trace<1,1,1>(), "degenerate trace");
    require(check_trace<8,32,0>() && check_trace<8,32,1>(), "extended trace");
    require(!check_trace<4,16,0>(true), "missing element negative escaped");
    require(!check_trace<4,16,1>(false,true), "within-row order negative escaped");
    uint32_t rng = 0x13579bdf;
    for (int sample = 0; sample < 4096; ++sample) {
        check_ordered_math<true>(rng, sample%31 == 0);
        check_ordered_math<false>(rng, sample%31 == 0);
    }
    std::cout << "row traversal + per-row FP32 order: 8192 fixtures PASS; "
                 "missing-element/order-swap negatives EXPECTED_RED; device math NOT_RUN\n";
}
