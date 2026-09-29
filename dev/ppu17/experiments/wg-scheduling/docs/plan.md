# W01: independent math warpgroups, original Q128/KV128

Registered 2026-09-29 07:08:57 UTC, before the kernel edit.
Parent `d19d27a`, branch `ppu17-wg-scheduling`. This is NOT Q64, a cache
experiment, a software-exp change or a no-sync kernel.

## Fixed work and hypothesis

FP16/noncausal B1/Sq1024/Sk1024/H56/Hkv56/D128, BSHD, 40-SM model.
Q128/KV128, two stages, 384 CTA threads, two math warpgroups, 448 tasks.
Keep arithmetic, K order, TMA descriptors, storage, registers-per-role,
task mapping and every data/operand-lifetime barrier unchanged.

Only disable `UseSchedulerBarrier` for the admitted FP16 specialization.
The existing non-pingpong path makes BOTH math WGs wait on K/V readiness;
the experiment must use that path, not merely erase the named barriers.
Keep QueryEmpty, Q/O transaction barriers, pipeline consumer release counts,
GMMA wait1/wait0 and epilogue/store completion.

Hypothesis: Hopper's alternating issue policy may add unnecessary waits on
the PPU model. Counterhypothesis: removing it destroys useful GEMM/softmax
staggering and is slower. Neither fewer barrier sites nor lower stall counts
is performance admission.

## Gates fixed before compilation/simulation

- Real generated shipping FP16/D128 TU, assembled and linked. Default C07
  encoded bodies unchanged 2/2 with the flag disabled; candidate causal body
  unchanged. The only noncausal change is the WG scheduling policy and its
  matching per-WG K/V readiness checks.
- Compile the actual mainloop policy/types: two128-thread math WGs,
  32 active producer threads, QueryEmpty count288, original storage and
  matrix types. Both WG0 and WG1 require K/V waits without pingpong.
- Source math/lifetime sites unchanged; native/PTX matrix denominator32
  m64n128k16 static sites, four GMMA completion-wait sites retained.
  Remove scheduler named-barrier sites, NOT Q/O/epilogue data barriers.
- No new CUDA stack/spill above C07's8/12/12B; reject asynchronous-
  serialization diagnostics. Record register/local-memory-site changes.
  CUDA is a control, not a native PPU allocator certificate.
- One target invocation per simulator process, CPU-only FP64 O/LSE reference
  with original atol/rtol0.002. No warmup, repeat, GPU reference or helper.
  Compare parent O/LSE fingerprints if available; a mismatch requires review.
- Complete cycles must beat the user's C07 report240412 on the SAME model.
  Target229376cycles (80% useful MFU) is aspiration, not a result or movable
  admission threshold. No new native PPU spill traffic. Full report/model/
  binary/CPU verdict required; no attribution from a stall percentage alone.

## Negatives

1. Wrong target/value, absent KV128, or composition with exp/Q64 rejects.
2. Plant the old leader-only K/V wait while pingpong is disabled: the actual
   production policy/type gate must fail for WG1, with an unmutated green.
3. Drop a GMMA wait or matrix instruction from real disassembly: accounting
   must reject. Restoring the pingpong body must fail the candidate-only gate.

Native PPU1.7 SDK/model unavailable locally. Compile/native CUDA control and
CPU gates are local; simulator numerics and performance remain NOT_RUN.
