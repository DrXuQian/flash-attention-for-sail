/******************************************************************************
 * Copyright (c) 2026, FlashAttention contributors.
 ******************************************************************************/
#pragma once

#include <cute/config.hpp>
#include "namespace_config.h"

namespace FLASH_NAMESPACE {
namespace detail {

// A scheduling permutation, not a new reduction: project this traversal onto
// any row and its column order is still BeginColumn, ..., Columns - 1.
// Host tests execute this same iterator, including the nonzero-initial sum.
template <int Rows, int Columns, int BeginColumn = 0, typename Function>
CUTE_HOST_DEVICE void for_each_softmax_column(Function&& function) {
    static_assert(Rows > 0 && Columns > 0);
    static_assert(BeginColumn >= 0 && BeginColumn <= Columns);
    CUTE_UNROLL
    for (int column = BeginColumn; column < Columns; ++column) {
        CUTE_UNROLL
        for (int row = 0; row < Rows; ++row) {
            function(row, column);
        }
    }
}

}  // namespace detail
}  // namespace FLASH_NAMESPACE
