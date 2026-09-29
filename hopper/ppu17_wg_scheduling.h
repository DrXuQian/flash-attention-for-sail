#pragma once

// Opt-in experiment only. Keep independent-WG scheduling separate from
// changes to geometry, exponent arithmetic, or operand-overlap lifetimes.
#if defined(FLASHATTN_PPU17_INDEPENDENT_WG)
#if !defined(FLASHATTN_PPU17) || (FLASHATTN_PPU17_INDEPENDENT_WG != 0 && FLASHATTN_PPU17_INDEPENDENT_WG != 1)
#error "FLASHATTN_PPU17_INDEPENDENT_WG requires explicit PPU1.7 and value 0 or 1"
#endif
#if FLASHATTN_PPU17_INDEPENDENT_WG
#if !defined(FLASHATTN_PPU17_KV_TILE128) || !FLASHATTN_PPU17_KV_TILE128
#error "independent-WG experiment requires explicit KV128"
#endif
#if (defined(FLASHATTN_PPU17_SOFTMAX_OVERLAP) && FLASHATTN_PPU17_SOFTMAX_OVERLAP) || defined(FLASHATTN_PPU17_Q_TAIL_MODE)
#error "independent-WG experiment must not be composed with exp or Q-tail experiments"
#endif
#endif
#endif

namespace flash {
#if defined(FLASHATTN_PPU17_INDEPENDENT_WG) && FLASHATTN_PPU17_INDEPENDENT_WG
inline constexpr bool kPpu17IndependentWG = true;
#else
inline constexpr bool kPpu17IndependentWG = false;
#endif
}  // namespace flash
