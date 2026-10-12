#ifdef FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
// Shared by workspace query and execution; shape is after the single-query GQA swap.
inline int holmes_select_splits(int b, int h, int s_q, int s_k, int d,
                                bool swapped, int num_sm, int requested = 0,
                                int max_splits = 128) {
    TORCH_CHECK(requested >= 0 && requested <= 128, "Holmes FA2 splits must be in [0, 128]");
    TORCH_CHECK(max_splits >= 1 && max_splits <= 128, "Holmes FA2 split capacity must be in [1, 128]");
    if (requested > 0) return std::min(requested, max_splits);
    if (s_q == 0 || s_k == 0) return 1;
    TORCH_CHECK(num_sm > 0, "Holmes FA2 requires a positive SM count");
#ifdef USE_PPU
    // Split kernel dispatch specializes on the rounded head dimension, e.g.
    // logical D=104 selects Headdim=128 and therefore kBlockN=64.
    const int kernel_head_dim = ((d + (d <= 128 ? 31 : 63)) / (d <= 128 ? 32 : 64))
                                * (d <= 128 ? 32 : 64);
    const int block_m = swapped ? 16 : 64;
    const bool even_mn = s_k % 128 == 0 && s_q % block_m == 0;
    const int block_n = swapped ? 64 : ((kernel_head_dim >= 128 || even_mn) ? 64 : 256);
#else
    const int block_m = 64;
    const int block_n = d <= 64 ? 256 : (d <= 128 ? 128 : 64);
#endif
    const size_t m_blocks = (size_t(s_q) + block_m - 1) / block_m;
    const size_t n_blocks = (size_t(s_k) + block_n - 1) / block_n;
    const size_t work = holmes_checked_mul(holmes_checked_mul(size_t(b), size_t(h)), m_blocks);
    // The heuristic immediately returns one split above this threshold. Clamp
    // before its int arithmetic so large valid shapes cannot overflow.
    if (work >= 0.8f * (double(num_sm) * 2)) return 1;
    TORCH_CHECK(num_sm <= std::numeric_limits<int>::max() / 2,
                "Holmes FA2 SM count exceeds the supported range");
    // Restrict candidates before scoring so capacity limits retain v2.8.2's
    // eligibility and efficiency rules instead of decrementing its chosen split.
    return num_splits_heuristic(int(work), num_sm * 2, int(n_blocks), max_splits);
}
#endif // FLASH_ATTN_CPP_INFERENCE_FOR_HOLMES
