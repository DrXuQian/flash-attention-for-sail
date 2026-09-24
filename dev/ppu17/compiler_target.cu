// Compile-only architecture receipt. This is intentionally compiled with the
// same flags as the generated forward units, not with a forced __CUDA_ARCH__.
#include <cuda_runtime.h>
#if !defined(ACOMPUTE_VERSION) || ACOMPUTE_VERSION != 10700
#error "PPU1.7 requires ACOMPUTE_VERSION=10700"
#endif
#if !defined(FLASHATTN_PPU17_SOURCE_CHECK) && !defined(__HGGC__)
#error "native PPU1.7 build requires the PPU compiler; CUDA-only checks are not native evidence"
#endif
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ != 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error "compiler did not select SM90a; legacy PPU1.0/1.5 fallback is forbidden"
#endif
#endif
__global__ void flash_ppu17_target_receipt(int* result) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) *result = __CUDA_ARCH__;
#endif
}
