#pragma once

// CPU-only fixture/oracle. No CUDA, Torch, CUTLASS layout or kernel arithmetic.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fa17_standalone {

struct Shape {
    int batch, seqlen, heads, dim;
    std::size_t rows() const { return std::size_t(batch) * heads * seqlen; }
    std::size_t elements() const { return rows() * dim; }
    std::size_t bshd(int b, int s, int h, int d) const {
        return ((std::size_t(b) * seqlen + s) * heads + h) * dim + d;
    }
    std::size_t bhs(int b, int h, int s) const {
        return (std::size_t(b) * heads + h) * seqlen + s;
    }
};

inline constexpr Shape kShape{1, 1024, 56, 128};
inline constexpr double kAtol = 2e-3, kRtol = 2e-3;
inline constexpr std::uint64_t kSeed = 170020;

struct Inputs { std::vector<float> q, k, v; };
struct Reference { std::vector<double> o, lse; };
struct Comparison {
    std::size_t bad = 0, nonfinite = 0;
    double max_abs = 0;
};

inline std::uint64_t mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

inline void validate_shape(Shape s) {
    if (s.batch <= 0 || s.seqlen <= 0 || s.heads <= 0 || s.dim <= 0)
        throw std::invalid_argument("positive CPU-reference dimensions required");
}

inline Inputs make_inputs(Shape s) {
    validate_shape(s);
    Inputs x{std::vector<float>(s.elements()), std::vector<float>(s.elements()),
             std::vector<float>(s.elements())};
    int channel = 0;
    for (auto* values : {&x.q, &x.k, &x.v}) {
        const auto salt = UINT64_C(0x9e3779b97f4a7c15) * (++channel);
        for (std::size_t i = 0; i < values->size(); ++i) {
            // Exactly representable FP16 dyadics, with different data per
            // position/head/channel. Not the Python runner's normal fixture.
            (*values)[i] = (int(mix(kSeed + i + salt) & 511) - 256) / 256.0f;
        }
    }
    return x;
}

inline Reference reference(Shape s, Inputs const& x) {
    validate_shape(s);
    if (x.q.size() != s.elements() || x.k.size() != s.elements() || x.v.size() != s.elements())
        throw std::invalid_argument("CPU reference input extent mismatch");
    Reference ref{std::vector<double>(s.elements()), std::vector<double>(s.rows())};
    const double scale = 1.0 / std::sqrt(double(s.dim));
    // Parallel CPU heads only. Each row uses a stable FP64 softmax, no online
    // softmax, GPU reference, or copies of the production fragment mapping.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int bh = 0; bh < s.batch * s.heads; ++bh) {
        const int b = bh / s.heads, h = bh % s.heads;
        std::vector<double> kh(std::size_t(s.seqlen) * s.dim), vh(kh.size());
        for (int j = 0; j < s.seqlen; ++j)
            for (int d = 0; d < s.dim; ++d) {
                kh[std::size_t(j) * s.dim + d] = x.k[s.bshd(b, j, h, d)];
                vh[std::size_t(j) * s.dim + d] = x.v[s.bshd(b, j, h, d)];
            }
        std::vector<double> scores(s.seqlen), out(s.dim), query(s.dim);
        for (int i = 0; i < s.seqlen; ++i) {
            for (int d = 0; d < s.dim; ++d) query[d] = x.q[s.bshd(b, i, h, d)];
            double max_score = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < s.seqlen; ++j) {
                double dot = 0;
                for (int d = 0; d < s.dim; ++d) dot += query[d] * kh[std::size_t(j) * s.dim + d];
                scores[j] = dot * scale;
                max_score = std::max(max_score, scores[j]);
            }
            double sum = 0;
            for (double& score : scores) { score = std::exp(score - max_score); sum += score; }
            std::fill(out.begin(), out.end(), 0);
            for (int j = 0; j < s.seqlen; ++j) {
                const double probability = scores[j] / sum;
                for (int d = 0; d < s.dim; ++d) out[d] += probability * vh[std::size_t(j) * s.dim + d];
            }
            for (int d = 0; d < s.dim; ++d) ref.o[s.bshd(b, i, h, d)] = out[d];
            ref.lse[s.bhs(b, h, i)] = max_score + std::log(sum);
        }
    }
    return ref;
}

inline Comparison compare(std::vector<float> const& got, std::vector<double> const& want) {
    if (got.size() != want.size()) throw std::invalid_argument("comparison extent mismatch");
    Comparison result;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++result.bad; ++result.nonfinite;
            continue;
        }
        const double error = std::abs(double(got[i]) - want[i]);
        result.max_abs = std::max(result.max_abs, error);
        result.bad += error > kAtol + kRtol * std::abs(want[i]);
    }
    return result;
}

inline void host_self_test() {
    auto require = [](bool ok) { if (!ok) throw std::runtime_error("CPU oracle self-test failed"); };
    // Uniform scores, distinct batches/heads/columns. Every output is the
    // arithmetic mean of V, and LSE is log(S). Tests BSHD output vs BHS LSE.
    Shape s{2, 3, 2, 4};
    Inputs x{std::vector<float>(s.elements(), 0), std::vector<float>(s.elements(), 0),
             std::vector<float>(s.elements())};
    for (int b = 0; b < s.batch; ++b)
        for (int h = 0; h < s.heads; ++h)
            for (int i = 0; i < s.seqlen; ++i)
                for (int d = 0; d < s.dim; ++d) x.v[s.bshd(b, i, h, d)] = b * 16 + h * 4 + d + i;
    auto ref = reference(s, x);
    for (int b = 0; b < s.batch; ++b)
        for (int h = 0; h < s.heads; ++h)
            for (int i = 0; i < s.seqlen; ++i) {
                require(std::abs(ref.lse[s.bhs(b, h, i)] - std::log(3.0)) < 1e-12);
                for (int d = 0; d < s.dim; ++d)
                    require(std::abs(ref.o[s.bshd(b, i, h, d)] - (b * 16 + h * 4 + d + 1)) < 1e-12);
            }
    // Independent two-token closed form: sigmoid(+/-2), not uniform scores.
    auto nonuniform = reference({1, 2, 1, 1}, {{1, -1}, {1, -1}, {1, 0}});
    require(std::abs(nonuniform.o[0] - 1 / (1 + std::exp(-2.0))) < 1e-12);
    require(std::abs(nonuniform.o[1] - 1 / (1 + std::exp(2.0))) < 1e-12);
    std::vector<float> got(ref.o.begin(), ref.o.end());
    require(compare(got, ref.o).bad == 0);
    got.back() += 1;
    require(compare(got, ref.o).bad == 1);  // full denominator, including last output
    got.back() = std::numeric_limits<float>::quiet_NaN();
    require(compare(got, ref.o).nonfinite == 1);
    got.pop_back();
    bool rejected = false;
    try { (void)compare(got, ref.o); } catch (std::invalid_argument const&) { rejected = true; }
    require(rejected);
}

}  // namespace fa17_standalone
