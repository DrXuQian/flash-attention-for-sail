#pragma once

#include <cutlass/version.h>

// Architecture and dependency API are separate axes. Both admitted backends
// implement SM90, but differ in the epilogue selector and TMA barrier policy.
#if defined(FLASHATTN_PPU17)
#if CUTLASS_VERSION != 360 && CUTLASS_VERSION != 430
#error "PPU1.7 forward admits PPU CUTLASS 3.6.0 or 4.3.0 only"
#endif
#if defined(FLASHATTN_PPU17_EXPECTED_CUTLASS_VERSION) && \
    CUTLASS_VERSION != FLASHATTN_PPU17_EXPECTED_CUTLASS_VERSION
#error "PPU1.7 CUTLASS include/version mismatch: compiler headers differ from selected backend"
#endif
#if CUTLASS_VERSION == 360
#define FLASHATTN_PPU17_BACKEND_ID "cutlass36-sm90-forward-v1"
#else
#define FLASHATTN_PPU17_BACKEND_ID "cutlass43-sm90-forward-v1"
#endif
#endif
