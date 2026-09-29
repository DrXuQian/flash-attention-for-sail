# T02 local phase boundary (not simulated admission)

Final status: REJECTED_BEFORE_SIMULATION (32/32B CUDA spill >12/12B budget).
The corrected implementation and its compile negatives are preserved only in
`../rejected-hybrid.patch`; production kernel/launcher edits were removed.

This is a separate delivery experiment, not an exp change. Parent arithmetic
files `softmax.h`, `mainloop_fwd_sm90_tma_gmma_ws.hpp` and `epilogue_fwd.hpp`
remain unchanged. Full and half use their real, separately instantiated TMA
descriptors and CuTe accumulators. The launch argument initializer is shared,
not a second manually copied set of pointer/stride assumptions.

## Drain is more than a CTA barrier

1. Full producer `load_tail` waits for the O transaction barrier and releases
   of both K/V rings. Full consumers complete GMMA and the last TMA store
   (`epilogue.store` already does `tma_store_wait<0>`).
2. `mma` posts256 QueryEmpty arrivals for the *next* Q, including after its
   last tile. `load_tail` does not consume those. A final32-thread producer
   `NamedBarrier::sync(288, QueryEmpty)` must complete that phase. Omitting
   this is a real stale-arrival bug when the next collective expects160.
3. All384threads join before reusing retired tensor storage. The96unused
   producer threads returned from the inlined full collective, not from the
   wrapper, and therefore participate in this join.
4. Only16CTAs have half work. On those, the second math WG relinquishes its
   registers and exits; the first256threads initialize the half collective.
   Its initialization barrier explicitly counts256, not physical CTA384.
5. The half's150528B total storage ends before the full pipeline starts at
   byte165888. Thus no initialized mbarrier is overwritten/reinitialized;
   the half barriers live in retired full-tensor memory. Full allocation is
   166912B. These numbers come from actual compiled types, not size estimates.
6. Reserved epilogue barrier1 completed before the join. The full warp-scheduler
   barriers are not used by the one-math-WG half collective. QueryEmpty is the
   shared named-barrier ID whose pending phase needs the explicit drain.

## Register handover

Full producer/math/math budgets are24/240/240, or64512registers/CTA. After
the join the retiring WG returns240->24. Producer restores24->168 for
initialization; half init's256-thread join precedes its normal56/256 split.
No thread sources a value from a retired register; the real compiler must
validate liveness/resource usage. The measured32B spill in the first compile
already shows that fitting aggregate budgets is not enough for performance.

All setmaxnreg operations are WG-uniform, with explicit synchronization
between successive operations by a WG. PTX's inc may wait for the CTA pool;
no cross-CTA progress dependency is introduced. The ISA definition is the
[NVIDIA setmaxnreg contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg),
not a native PPU1.7 allocator certificate.

## Output ownership

The host probe calls the real scheduler's divmods/coordinate method and the
real full/half TiledMmaPV accumulator partitions. Each actual fragment
coordinate is checked against a separate SM90 warp/lane mapping. All7340032
output cells are covered exactly once, with57344rows. Duplicate half, omitted
half, wrong head, wrong half origin and missing denominator each fail.
This is an address/ownership proof, NOT execution or numeric proof of the
phase boundary. The single simulated invocation plus full CPU oracle is
still required if the compile-resource gate admits a candidate.
