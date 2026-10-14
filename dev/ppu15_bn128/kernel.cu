// Isolated geometry experiment; all device math is the shipping FA2 body.
#include <cmath>
#include <cstdint>
#include <new>
#include "flash_fwd_launch_template.h"

#ifndef FA15_CONFIG
#error "FA15_CONFIG must select one of the three registered geometries"
#endif
#if defined(__HGGC_ARCH__)
static_assert(__HGGC_ARCH__ == 150, "this experiment is PPU1.5 only");
#endif

namespace {
constexpr int kHeadDim = 128;
constexpr int kHeads = 56;
constexpr int kBlockM = FA15_CONFIG == 2 ? 64 : 128;
constexpr int kBlockN = FA15_CONFIG == 0 ? 64 : 128;
constexpr int kWarps = FA15_CONFIG == 1 ? 8 : 4;
static_assert(FA15_CONFIG >= 0 && FA15_CONFIG <= 2);

template <class T>
using Traits = Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kWarps,
                                       false, false, T>;

template <class T>
auto kernel() {
    // Exactly the shipping specialization for S=8398: non-even MN, even D,
    // no dropout/causal/local/alibi/softcap/returned probability matrix.
    return &FLASH_NAMESPACE::flash_fwd_kernel<Traits<T>, false, false, false,
                                             false, false, true, false, false>;
}

struct Call {
    FLASH_NAMESPACE::Flash_fwd_params params{};
    bool bf16 = false;
};

template <class T>
int resources(int* out) {
    auto fn = kernel<T>();
    auto rc = hggcFuncSetAttribute(fn, hggcFuncAttributeMaxDynamicSharedMemorySize,
                                  Traits<T>::kSmemSize);
    if (rc != hggcSuccess) return int(rc);
    hggcFuncAttributes attr{};
    rc = hggcFuncGetAttributes(&attr, fn);
    if (rc != hggcSuccess) return int(rc);
    int active = 0;
    rc = hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, fn, Traits<T>::kNThreads, Traits<T>::kSmemSize);
    if (rc != hggcSuccess) return int(rc);
    out[0] = kBlockM; out[1] = kBlockN; out[2] = kWarps;
    out[3] = Traits<T>::kSmemSize; out[4] = attr.numRegs;
    out[5] = int(attr.localSizeBytes); out[6] = active;
    return 0;
}

template <class T>
int launch(Call const& call, hggcStream_t stream) {
    auto fn = kernel<T>();
    dim3 grid((call.params.seqlen_q + kBlockM - 1) / kBlockM, 1, kHeads);
    fn<<<grid, Traits<T>::kNThreads, Traits<T>::kSmemSize, stream>>>(call.params);
    return int(hggcGetLastError());
}
}  // namespace

extern "C" int fa15_resources(int dtype, int* out) {
    if (!out || (dtype != 0 && dtype != 1)) return -1;
    return dtype ? resources<cutlass::bfloat16_t>(out)
                 : resources<cutlass::half_t>(out);
}

extern "C" int fa15_runtime_versions(int* out) {
    if (!out) return -1;
    auto rc = hggcRuntimeGetVersion(out);
    if (rc != hggcSuccess) return int(rc);
    return int(hggcDriverGetVersion(out + 1));
}

extern "C" const char* fa15_error_string(int rc) {
    return rc < 0 ? "invalid experiment argument" : hggcGetErrorString(hggcError_t(rc));
}

extern "C" void* fa15_create(int dtype, int seq, void* q, void* k, void* v,
                             void* o, void* lse) {
    // C ABI deliberately admits just B1/H56/D128 contiguous BSHD, dense FWD.
    if ((dtype != 0 && dtype != 1) || seq <= 0 || seq > 16384 ||
        !q || !k || !v || !o || !lse) return nullptr;
    for (void* p : {q, k, v, o})
        if (reinterpret_cast<uintptr_t>(p) % 16) return nullptr;
    auto* call = new (std::nothrow) Call;
    if (!call) return nullptr;
    auto& p = call->params;
    p.q_ptr = q; p.k_ptr = k; p.v_ptr = v; p.o_ptr = o;
    p.softmax_lse_ptr = lse;
    p.b = 1; p.h = kHeads; p.h_k = kHeads; p.h_h_k_ratio = 1;
    p.seqlen_q = seq; p.seqlen_k = seq; p.d = kHeadDim;
    p.seqlen_q_rounded = (seq + 127) / 128 * 128;
    p.seqlen_k_rounded = (seq + 127) / 128 * 128;
    p.d_rounded = kHeadDim;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride =
        p.o_batch_stride = int64_t(seq) * kHeads * kHeadDim;
    p.q_row_stride = p.k_row_stride = p.v_row_stride = p.o_row_stride =
        kHeads * kHeadDim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = kHeadDim;
    p.scale_softmax = 1.f / std::sqrt(float(kHeadDim));
    p.scale_softmax_log2 = p.scale_softmax * M_LOG2E;
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.window_size_left = -1; p.window_size_right = -1;
    p.is_seqlens_k_cumulative = true; p.num_splits = 1;
    p.is_bf16 = call->bf16 = dtype == 1;
    return call;
}

extern "C" int fa15_launch(void* handle, void* stream) {
    if (!handle) return -1;
    auto const& call = *static_cast<Call*>(handle);
    return call.bf16 ? launch<cutlass::bfloat16_t>(call, hggcStream_t(stream))
                     : launch<cutlass::half_t>(call, hggcStream_t(stream));
}

extern "C" void fa15_destroy(void* handle) { delete static_cast<Call*>(handle); }
