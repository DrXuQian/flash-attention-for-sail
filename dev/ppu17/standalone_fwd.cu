// Torch-free host application; device code comes from the shipping
// hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu translation unit.
// An explicitly selected body experiment is printed in softmax_overlap.
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include "flash.h"
#include "tile_size.h"
#include "ppu17_cutlass_compat.h"
#include "ppu17_wg_scheduling.h"
#include "standalone_reference.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef FA17_BUILD_SOURCE_SHA256
#error "build with tools/build_ppu17_standalone.py to bind source provenance"
#endif

namespace {
using namespace fa17_standalone;
using Half = cutlass::half_t;
#if defined(FLASHATTN_PPU17_SOFTMAX_OVERLAP) && FLASHATTN_PPU17_SOFTMAX_OVERLAP
constexpr char kOverlapVariant[] = "row-sum-token";
#else
constexpr char kOverlapVariant[] = "control";
#endif

void check(cudaError_t status, char const* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
public:
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    void upload(std::vector<T> const& host) { check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D"); }
    void download(std::vector<T>& host) { check(cudaMemcpy(host.data(), ptr, host.size() * sizeof(T), cudaMemcpyDeviceToHost), "D2H"); }
};

template <class T> std::uint64_t fingerprint(std::vector<T> const& values) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    auto bytes = reinterpret_cast<unsigned char const*>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(T); ++i) {
        hash ^= bytes[i]; hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::vector<Half> encode(std::vector<float> const& values) {
    std::vector<Half> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = Half(values[i]);
        if (float(result[i]) != values[i]) throw std::runtime_error("fixture is not FP16-exact");
    }
    return result;
}

Flash_fwd_params make_params(int sm_count) {
    Flash_fwd_params p{};
    p.b = p.b_k = kShape.batch;
    p.h = p.h_k = kShape.heads;
    p.seqlen_q = p.seqlen_k = p.seqlen_q_rounded = p.seqlen_k_rounded = kShape.seqlen;
    p.total_q = p.total_k = kShape.batch * kShape.seqlen;
    p.d = p.dv = p.d_rounded = p.dv_rounded = kShape.dim;
    p.q_row_stride = p.k_row_stride = p.v_row_stride = p.o_row_stride = kShape.heads * kShape.dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = kShape.dim;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride = p.o_batch_stride = kShape.elements() / kShape.batch;
    p.v_dim_stride = 1;
    p.scale_softmax = float(1.0 / std::sqrt(double(kShape.dim)));
    p.window_size_left = p.window_size_right = kShape.seqlen - 1;
    p.p_dropout = p.rp_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.num_splits = 1;
    p.arch = 90;
    p.num_sm = sm_count;
    // Zero-initialized flags: noncausal, FP16, no local mask, no split,
    // no packed GQA, no varlen. Static scheduler needs no device counter.
    return p;
}

void describe() {
    constexpr auto tile = tile_size_fwd_sm90(kShape.dim, kShape.dim, false, false, sizeof(Half));
    const auto p = make_params(20);
    std::cout << "{\"batch\":1,\"seqlen_q\":1024,\"seqlen_k\":1024,\"heads\":56,\"kv_heads\":56,"
              << "\"dim\":128,\"dtype\":\"fp16\",\"causal\":false,\"layout\":\"BSHD\","
              << "\"tile_m\":" << std::get<0>(tile) << ",\"tile_n\":" << std::get<1>(tile)
              << ",\"row_stride_elements\":" << p.q_row_stride << ",\"head_stride_elements\":" << p.q_head_stride
              << ",\"batch_stride_elements\":" << p.q_batch_stride << ",\"num_splits\":" << p.num_splits
              << ",\"logical_flops\":" << UINT64_C(4) * kShape.heads * kShape.seqlen * kShape.seqlen * kShape.dim
              << ",\"softmax_overlap\":\"" << kOverlapVariant << "\""
              << ",\"wg_scheduling\":\"" << (flash::kPpu17IndependentWG ? "independent" : "pingpong") << "\""
              << ",\"backend\":\"" << FLASHATTN_PPU17_BACKEND_ID << "\",\"source_sha256\":\""
              << FA17_BUILD_SOURCE_SHA256 << "\"}\n";
}

int positive_int(char const* value, bool zero_allowed = false) {
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (end == value || *end || result < (zero_allowed ? 0 : 1) || result > 65536)
        throw std::invalid_argument("invalid integer option");
    return int(result);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        bool verify = false, self_test = false, describe_only = false;
        int device = 0, expected_sms = 20, cpu_threads = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "PPU1.7 SM90a simulation input: B1/S1024/H56/Hkv56/D128 FP16 noncausal\n"
                          << "  --verify             full CPU FP64 O/LSE reference AFTER one GPU invocation\n"
                          << "  --expected-sms N     runtime admission (default 20; 0 disables assertion)\n"
                          << "  --device N           runtime device ordinal (default 0)\n"
                          << "  --cpu-threads N      CPU reference threads (default 4)\n"
                          << "  --describe           print fixed shape/Params without touching CUDA\n"
                          << "  --host-self-test     CPU oracle checks/negatives; no CUDA calls\n"
                          << "No warmup, repeat, event timer, GPU initializer or GPU reference.\n";
                return 0;
            } else if (arg == "--verify") verify = true;
            else if (arg == "--host-self-test") self_test = true;
            else if (arg == "--describe") describe_only = true;
            else if ((arg == "--expected-sms" || arg == "--device" || arg == "--cpu-threads") && i + 1 < argc) {
                const int value = positive_int(argv[++i], arg != "--cpu-threads");
                if (arg == "--device") device = value;
                else if (arg == "--expected-sms") expected_sms = value;
                else cpu_threads = value;
            } else throw std::invalid_argument("unknown/incomplete option: " + arg);
        }
#ifdef _OPENMP
        omp_set_num_threads(cpu_threads);
#endif
        if (self_test) {
            host_self_test();
            // Also bind the actual half encoder and actual runtime parameter
            // constructor: neither needs a GPU or a simulated launch.
            const auto x = make_inputs({1, 3, 2, 4});
            (void)encode(x.q); (void)encode(x.k); (void)encode(x.v);
            const auto p = make_params(20);
            if (p.is_causal || p.is_bf16 || p.num_splits != 1 || p.num_sm != 20 ||
                p.q_row_stride != 7168 || p.q_head_stride != 128 || p.q_batch_stride != 7340032 ||
                p.q_row_stride != p.k_row_stride || p.k_row_stride != p.v_row_stride ||
                p.v_row_stride != p.o_row_stride || p.tile_count_semaphore)
                throw std::runtime_error("standalone Params self-test failed");
            std::cout << "[FA17 standalone host] oracle=uniform+nonuniform negatives=last-output+nan+extent Params+FP16=PASS device=NOT_RUN\n";
            return 0;
        }
        if (describe_only) { describe(); return 0; }

        check(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        if (prop.major != 9 || prop.minor != 0) throw std::runtime_error("requires SM90-compatible runtime");
        if (expected_sms && prop.multiProcessorCount != expected_sms)
            throw std::runtime_error("expected " + std::to_string(expected_sms) + " SMs, got " +
                                     std::to_string(prop.multiProcessorCount) + "; assertion does not partition hardware");
        describe();
        std::cout << "[FA17 standalone device] name=" << std::quoted(prop.name) << " ordinal=" << device
                  << " measured_sms=" << prop.multiProcessorCount << " measured_l2_bytes=" << prop.l2CacheSize
                  << " grid=runtime-sms target_invocations=1 warmup=0 timer=MODEL-ONLY\n" << std::flush;
        const auto inputs = make_inputs(kShape);
        auto q = encode(inputs.q), k = encode(inputs.k), v = encode(inputs.v);
        // Poison from CPU so missing writes are caught, without cudaMemset or a fill kernel.
        std::vector<Half> o(kShape.elements(), Half(std::numeric_limits<float>::quiet_NaN()));
        std::vector<float> lse(kShape.rows(), std::numeric_limits<float>::quiet_NaN());
        DeviceBuffer<Half> dq(q.size()), dk(k.size()), dv(v.size()), dout(o.size());
        DeviceBuffer<float> dlse(lse.size());
        dq.upload(q); dk.upload(k); dv.upload(v); dout.upload(o); dlse.upload(lse);
        check(cudaDeviceSynchronize(), "input synchronization");
        auto p = make_params(prop.multiProcessorCount);
        p.q_ptr = dq.ptr; p.k_ptr = dk.ptr; p.v_ptr = dv.ptr; p.o_ptr = dout.ptr;
        p.softmax_lse_ptr = dlse.ptr;
        std::cout << "[FA17 standalone input] seed=" << kSeed << " fixture=dyadic-fp16-exact q_fnv="
                  << std::hex << fingerprint(q) << " k_fnv=" << fingerprint(k) << " v_fnv=" << fingerprint(v)
                  << std::dec << " numerics=" << (verify ? "PENDING" : "NOT_CHECKED") << '\n' << std::flush;

        // Exactly the public dispatch's shipping generated specialization.
        run_mha_fwd_<90, Half, 128, 128, false, false, false, false>(p, nullptr);
        check(cudaGetLastError(), "shipping forward launch");
        check(cudaDeviceSynchronize(), "shipping forward completion");

        dout.download(o); dlse.download(lse);
        std::vector<float> output(o.size());
        for (std::size_t i = 0; i < o.size(); ++i) {
            output[i] = float(o[i]);
            if (!std::isfinite(output[i])) throw std::runtime_error("nonfinite/unwritten O at " + std::to_string(i));
        }
        for (std::size_t i = 0; i < lse.size(); ++i)
            if (!std::isfinite(lse[i])) throw std::runtime_error("nonfinite/unwritten LSE at " + std::to_string(i));
        std::cout << "[FA17 standalone output] o_fnv=" << std::hex << fingerprint(o) << " lse_fnv="
                  << fingerprint(lse) << std::dec << " finite=PASS\n" << std::flush;
        if (verify) {
            std::cout << "[FA17 standalone reference] CPU FP64, full O+LSE; this is not simulation timing\n" << std::flush;
            auto ref = reference(kShape, inputs);
            auto oc = compare(output, ref.o), lc = compare(lse, ref.lse);
            std::cout << "[FA17 standalone verify] O_bad=" << oc.bad << '/' << output.size()
                      << " LSE_bad=" << lc.bad << '/' << lse.size() << " O_max_abs=" << oc.max_abs
                      << " LSE_max_abs=" << lc.max_abs << " atol=" << kAtol << " rtol=" << kRtol << '\n';
            if (oc.bad || lc.bad) throw std::runtime_error("CPU FP64 reference mismatch");
        }
        std::cout << "[FA17 standalone] completed_invocations=1 numerics="
                  << (verify ? "CPU-FP64/PASS" : "NOT_CHECKED") << " performance=READ-SIMULATOR-CYCLES\n";
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "[FA17 standalone] FAIL: " << error.what() << '\n';
        return 1;
    }
}
