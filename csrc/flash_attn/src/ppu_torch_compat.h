/******************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstddef>

#if defined(FLASHATTN_PPU_DEVICE_COMPILE)

#include <cstdint>
#include <tuple>

namespace at {

// Keep this layout identical to ATen/cuda/detail/PhiloxCudaStateRaw.cuh.  PPU
// device translation units only need the by-value kernel parameter; the host
// extension still uses PyTorch's authoritative definition.
struct PhiloxCudaState {
    union Payload {
        uint64_t val;
        int64_t* ptr;
    };

    Payload seed_{};
    Payload offset_{};
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};

static_assert(sizeof(PhiloxCudaState) == 24,
              "PPU Philox kernel argument must match PyTorch's host ABI");
static_assert(alignof(PhiloxCudaState) == 8,
              "PPU Philox kernel argument alignment must match PyTorch's host ABI");
static_assert(offsetof(PhiloxCudaState, seed_) == 0);
static_assert(offsetof(PhiloxCudaState, offset_) == 8);
static_assert(offsetof(PhiloxCudaState, offset_intragraph_) == 16);
static_assert(offsetof(PhiloxCudaState, captured_) == 20);

namespace cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(PhiloxCudaState arg) {
    if (arg.captured_) {
        return std::make_tuple(
            static_cast<uint64_t>(*arg.seed_.ptr),
            static_cast<uint64_t>(*arg.offset_.ptr + arg.offset_intragraph_));
    }
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
}

}  // namespace cuda::philox
}  // namespace at

#else

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "philox_unpack.cuh"

static_assert(sizeof(at::PhiloxCudaState) == 24,
              "PyTorch changed the Philox kernel argument ABI");
static_assert(alignof(at::PhiloxCudaState) == 8,
              "PyTorch changed the Philox kernel argument alignment");
static_assert(offsetof(at::PhiloxCudaState, seed_) == 0);
static_assert(offsetof(at::PhiloxCudaState, offset_) == 8);
static_assert(offsetof(at::PhiloxCudaState, offset_intragraph_) == 16);
static_assert(offsetof(at::PhiloxCudaState, captured_) == 20);

#endif
