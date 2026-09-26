#pragma once

// Opt-in physical-Hopper experiment. No effect on any shipping default.
// Host metadata allocation and device scheduler selection MUST use the same
// predicate: a SingleTileScheduler never consumes a tile-count semaphore.
#if (defined(FLASHATTN_PPU17_CAUSAL_N) || defined(FLASHATTN_PPU17_CAUSAL_SINGLE)) && \
    !defined(FLASHATTN_PPU17_CAUSAL_EXPERIMENT)
#error "causal tuning knobs require the explicitly scoped experiment build"
#endif
#if defined(FLASHATTN_PPU17_CAUSAL_EXPERIMENT) && !defined(FLASHATTN_PPU17)
#error "causal tuning requires the PPU1.7 Hopper source branch"
#endif

#ifndef FLASHATTN_PPU17_CAUSAL_N
#define FLASHATTN_PPU17_CAUSAL_N 80
#endif
#ifndef FLASHATTN_PPU17_CAUSAL_SINGLE
#define FLASHATTN_PPU17_CAUSAL_SINGLE 0
#endif

namespace flash::ppu17_causal {
static_assert(FLASHATTN_PPU17_CAUSAL_N == 64 || FLASHATTN_PPU17_CAUSAL_N == 80,
              "registered causal N inventory is exactly {64,80}");
static_assert(FLASHATTN_PPU17_CAUSAL_SINGLE == 0 || FLASHATTN_PPU17_CAUSAL_SINGLE == 1,
              "registered scheduler inventory is LPT or single-tile");

constexpr bool in_scope(int d, int dv, int bytes, bool causal, bool local) {
#if defined(FLASHATTN_PPU17_CAUSAL_EXPERIMENT)
    return d == 256 && dv == 256 && bytes == 2 && causal && !local;
#else
    return false;
#endif
}

constexpr bool single_tile(int arch, int d, int dv, int bytes,
                           bool causal, bool local, bool varlen, bool split, bool packed) {
    return FLASHATTN_PPU17_CAUSAL_SINGLE && arch == 90 &&
        in_scope(d, dv, bytes, causal, local) && !varlen && !split && !packed;
}
}  // namespace flash::ppu17_causal
