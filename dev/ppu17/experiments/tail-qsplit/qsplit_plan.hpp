#pragma once

namespace flash {

// Bounded noncausal experiment, NOT a general scheduler/selector. The whole
// prefix retains the existing static scheduler and K order. Only the final
// head's eight M128 tiles become sixteen disjoint M64 tiles.
struct Ppu17TailQSplitPlan {
    static constexpr int Grid = 40;
    static constexpr int Heads = 56;
    static constexpr int Sequence = 1024;
    static constexpr int Dim = 128;
    static constexpr int FullM = 128;
    static constexpr int HalfM = 64;
    static constexpr int FullBlocksPerHead = Sequence / FullM;
    static constexpr int HalfBlocksPerHead = Sequence / HalfM;
    static constexpr int TotalFull = Heads * FullBlocksPerHead;
    static constexpr int Prefix = TotalFull / Grid * Grid;
    static constexpr int TailHalves = (TotalFull - Prefix) * 2;
    static constexpr int HalfStart = Prefix * 2;
    static constexpr int HalfEnd = TotalFull * 2;
    static_assert(Prefix == 440 && TailHalves == 16);
    static_assert(Prefix % FullBlocksPerHead == 0);
    static_assert(TailHalves <= Grid);

    static constexpr bool admits(int batch, int q, int k, int heads,
                                 int kv_heads, int dim, int dv, int sms) {
        return batch == 1 && q == Sequence && k == Sequence && heads == Heads &&
               kv_heads == Heads && dim == Dim && dv == Dim && sms == Grid;
    }
};

} // namespace flash
