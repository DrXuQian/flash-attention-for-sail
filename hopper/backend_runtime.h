#pragma once

// USE_PPU is the legacy PPU1.0/1.5 AIU algorithm switch, not a synonym for
// every PPU runtime. PPU1.7 uses the CUDA-compatible SM90 API and layouts.
#if defined(FLASHATTN_PPU17)
#if defined(USE_PPU) || (defined(USE_AIU) && USE_AIU)
#error "PPU1.7 SM90 must not select the legacy USE_PPU/USE_AIU implementation"
#endif
#if !defined(ACOMPUTE_VERSION) || ACOMPUTE_VERSION != 10700
#error "PPU1.7 requires ACOMPUTE_VERSION=10700"
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL))
#error "PPU1.7 forward requires real SM90a device compilation, not a legacy target"
#endif
#include <cuda_runtime.h>
#else
#include <hggc.h>
#include <hggc_runtime_api.h>
#endif

namespace flash::runtime {
#if defined(FLASHATTN_PPU17)
using Stream = cudaStream_t;
using Error = cudaError_t;
constexpr Error success = cudaSuccess;
inline Error get_device(int* device) { return cudaGetDevice(device); }
inline Error last_error() { return cudaGetLastError(); }
inline const char* error_string(Error error) { return cudaGetErrorString(error); }
template <class Kernel>
inline Error set_dynamic_smem(Kernel kernel, int bytes) {
    return cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}
#else
using Stream = hggcStream_t;
using Error = hggcError_t;
constexpr Error success = hggcSuccess;
inline Error get_device(int* device) { return hggcGetDevice(device); }
inline Error last_error() { return hggcGetLastError(); }
inline const char* error_string(Error error) { return hggcGetErrorString(error); }
template <class Kernel>
inline Error set_dynamic_smem(Kernel kernel, int bytes) {
    return hggcFuncSetAttribute(reinterpret_cast<const void*>(kernel), hggcFuncAttributeMaxDynamicSharedMemorySize, bytes);
}
#endif
}  // namespace flash::runtime

// The generated legacy units still name hggcStream_t; SM90 units name
// cudaStream_t. Both match this alias in their respective builds, with no
// cross-runtime stream reinterpretation in the PPU1.7 path.
using FlashStream = flash::runtime::Stream;
