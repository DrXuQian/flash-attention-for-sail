# PPU1.7 Hopper source integration

    updated-at: 2026-09-29 05:49:31 UTC
    working-on: T01 user-reported402013cycles; do not promote (+67.22% versus reported C07); no kernel changes
    blocked-on: current command/binary/model/report identities and CPU O/LSE result not supplied; native PPU1.7 model unavailable locally
    last-commit: 6490d3e (preceding committed checkpoint; implementation7e41b5d)
    branch: ppu17-tail-qsplit
    workspace: /workspace/flash-attn-ppu17-tail-qsplit-source
    scope: FP16/BF16 fixed forward, D64/128/256, causal/noncausal, GQA
    model-target: current uploaded run 40 SM; historical 20-SM model is not this run

Current experiment: dev/ppu17/experiments/tail-qsplit/docs/plan.md.
User requires tail and exp optimization independently. Exp/softmax untouched.
User now reports402013cycles after the T01 handoff:161601more than C07,
+67.22%; candidate must not be promoted. Attribution to T01 is provisional
until command/report identities arrive. Numerical verdict not supplied.
The initial report of a long run is not proof of deadlock. Source audit finds
bounded task loops (at most23 per CTA), QueryEmpty160=128math+32producer,
and no two-math-WG scheduler barrier in the one-WG specialization. These
checks do not establish native PPU liveness or explain the slowdown.
Q128 has two math WGs; Q64 has one and doubles KV task fills. This is not a
clean tail-only change or evidence against a separate cache experiment.
C07 user reports240412cycles; full report/CPU result/identity not yet uploaded.
Actual `/root/perfstatistics.json` is still the old266011cycle C03 loser.
T01 final ELF24dc2a0e: actual M64 collective,255regs/0stack/0spill on CUDA.
T02 corrected440full+16half ELF1f881cb0:32/32B spill >12/12B budget, rejected
before simulation. Its live wrapper/launcher/phase edits removed; exact replay
patch retained. Both actual transition compile negatives have matching green
controls. Final source leaves exp, mainloop, epilogue and kernel unchanged.
M64 is a scaling control with doubled K/V task fills, not a claimed speedup.
Complete final tier68PASS/4SKIP/0FAIL (3missing Torch;1old raw report missing).
Default encoded bodies2/2 unchanged; opt-in causal unchanged; actual7340032
output cells exact-once for each inventory and5plants per inventory red.
No local device or model execution; reported cycles are user evidence only.
Final replay/results and single-call commands:
dev/ppu17/experiments/tail-qsplit/README.md.

Previous experiment: dev/ppu17/experiments/kv128/docs/plan.md.
C07 changes only opt-in KV176->128 on the original pipeline/softmax. Local
63 PASS /4 SKIP /0 FAIL;3missing-Torch Python oracles,1overwritten baseline
raw report. Real default encoded bodies unchanged2/2, opt-in causal unchanged.
Exact generated FP16/D128 body links; CUDA noncausal168regs,8Bstack,12/12Bspill
unchanged; static body2624->2344, oldP32/O64 early-overwrite negatives red.
Selector1536combinations: exactly1changes; wrong target/value/mixed experiment
and wrong denominator plants reject. Source/backend/compiler/ELF hashes bound
in kv128/results.json. PPU lowering, CPU-oracle kernel verdict and performance
NOT_RUN; no default promotion. Fewer padded matrix operations3.03%, but KV
iterations6->8: full modeled cycles must decide, not static instruction count.

Previous experiment: dev/ppu17/experiments/softmax-overlap/docs/plan.md.
Partial-overlap follow-up: C04/C05 native75/15 and76/14 fail45/45; C06 hits
45/45 but spills12->40B, rejected before simulation. Real layout22528cells
and512private words exact-once,3plants red. Default encoded streams2/2 match;
all five experimental live source/build edits restored. Artifacts and exact
C06 replay patch retained; see docs/partial-verdict.md. Next distinct axis
is KV tile128 using the ORIGINAL softmax/pipeline, not these rejected changes.
Uploaded control: 247267 cycles / 74.21% useful MFU. Uploaded C03: 266011
cycles / 68.98%, slower7.58047%, REJECTED. All90 steady EX2 really precede
wait0; the wait's sync stall drops to0. But PPU lowering spills23 logical
private slots:35323904B reads +18964480B writes, exactly closed from PCs to
memory counters. CUDA's unchanged12B spill metadata did not predict this.
No kernel/default route changed during this diagnosis. Application CPU O/LSE
PASS is not present in perfstatistics; do not invent it. Preserve both ELFs.
Detailed verdict: dev/ppu17/experiments/softmax-overlap/docs/simulation-verdict.md.
Current complete local suite:58 PASS /4 SKIP /0 FAIL. Three skips are missing
Torch; one is the original full baseline report overwritten by this upload.
Real C03 passes the90/90 overlap check but fails the no-new-private-traffic
gate as required. Removing one private read while keeping both instruction
totals consistent still fails the independent memory-byte denominator.

C03 fresh real ELF link PASS. Control default machine words unchanged2/2;
candidate causal body unchanged. CUDA native EX2 window19/90->90/90, matrix
sites16QK+22PV fixed, old P44/O64 words not overwritten before completion.
Registers168, stack8B, spill stores/loads12B remain equal to the parent.
Static body2624->2632; +1KiB private shared payload and one XOR/store per math
thread/steady step. Neither source ordering nor NVIDIA SASS is PPU timing.
Local suite PASS56/SKIP3/FAIL0; missing Torch is the reason for3old Python CPU
oracle skips; independent standalone C++ oracle/link and all10new gates pass.
Results/negative controls: dev/ppu17/experiments/softmax-overlap/results.json.
C01 inert and C02 added-spill variants are rejected, not retained in mainloop.
Build/run command: that experiment's README.md. No default-route promotion.

Encoding follow-up: local fixture reproduces UTF8 byte0x82 error at offset22
in old version-header decoding. User's exact failing input is not yet known.
49 unique local tests pass: comments may be non-UTF8, damaged version/opcode tokens
remain rejected, raw diagnostics preserved, rc17 remains failure and rc0
without an object remains failure. Default build only compiles/links;
--inspect-codegen enables optional PTX/disassembly and records SKIP otherwise.
No kernel/CUDA flags changed. Fresh direct AND inspected ELF builds PASS;
8 standalone host checks pass on each. Direct build has no PTX/disassembly
commands; its encoded native kernel words match the earlier standalone2/2.
Evidence: dev/ppu17/results/standalone-20260928/build-encoding.json.

Standalone: real CUTLASS4.3/CUDA12.8 ELF built without Torch/Python libraries.
Shipping generated FP16/D128 unit unchanged; both encoded SM90a kernel bodies
(including control words) match the admitted release43 object, 2/2.
40 local contracts pass (7 standalone + 26 source + 4 timing + 3 CPU oracle),
including real missing-generated-unit link
failure and oracle last-output/NaN/extent negatives. Default one invocation;
--verify adds full CPU FP64 O/LSE only. No GPU jobs were run.
ELF SHA256=74db0013d95f4f84f5e83a53a86d28bd67853eb0f834a526bbd54132323a022c.
Artifacts: /workspace/flash-attn-ppu17-causal-tune-20260926/standalone-20260928-v1.
No production collective, tiling, scheduler or generated-unit modifications.

CUTLASS4.3 complete: release43 AND release36 pass 6/6 units / 12 live bodies,
assembly and real host/internal link. All 12 CUTLASS3.6 encoded kernel streams
including control words match the old admitted build. 33 local tests PASS;
four real header/include mutations EXPECTED_RED; four existing target/link
plants EXPECTED_RED on each backend. Optional old-SDK negative not rerun.
Real 4.3 Python3.12/Torch2.9 package compiled and linked, imported as
cutlass43-sm90-forward-v1; actual wrong-backend admission rejected. Initial
link needed the installed CUDA compat library search path, not a kernel fix.
No GPU device nodes and no device/simulator invocation. Raw artifacts:
/workspace/flash-attn-ppu17-causal-tune-20260926/cutlass43-20260928.
Committed evidence: dev/ppu17/docs/cutlass43-migration.md and
dev/ppu17/results/cutlass43-20260928/validation.json. The older performance
results below remain specific to their 3.6 binary, not the new backend.

S4096/8192/16384 A/B/B/A complete with immutable binaries and full CPU FP64
O/LSE checks. Pooled port kernel MFU60.37/64.26/62.51%; official59.19/65.25/
62.16%. Every relative speed comparison UNRESOLVED; no stable70% result.
Same generated specialization/resources throughout. Snapshot/trace timestamp
joins show SW Power Cap during attention at default350W; no clocks or power
changed and no denominator adjustment. 36 local tests PASS. No GPU samples
retained from three prelaunch BUSY refusals. GPU empty at05:41:33 UTC.
Record: dev/ppu17/docs/h800-sequence-scaling.md; sample/hash-bound data in
dev/ppu17/results/h800-sequence-scaling-20260926/summary.json.

Completed task: official FA3 A/B on the same H800 and input, no kernel edits.
Order A1/B1-r2/B2/A2, each200 warmups+9x50 calls. Pooled kernel-only medians:
admitted port161.02995us/56.4386%, official164.23429us/55.3374%; ranges
overlap, so relative speed UNRESOLVED. Both fail the unchanged70% threshold.
Full-call event spans177.80608/173.62881us also overlap. All four runs pass
the unchanged CPU FP64 O/LSE oracle and within-arm replay; cross-arm output
AND LSE hashes are also identical on this fixture. H800 only, not PPU1.7.
Official extension SHA256=c0611358efe2a1511843ef2638befce6932b1d5855225c7d7d89ebf8583ef585.
Source a8aa52b1, official CUTLASS dc481792 (4.0.0); build-r2 completed with
unmodified sources. First build attempt and official b1 attempt were refused
by idle admission before compilation/GPU work respectively; B1-r2 began after
the foreign parent exited. Target kernels both168 registers, zero stack/spill,
grid114/block384/shared232448B. Static SASS counts3640/3672 are not a dynamic
verdict. Official API uses one counter-fill kernel per call; port uses H2D.
No clock/partition/power changes. GPU empty after final arm. Raw artifacts:
/workspace/flash-attn-official-h800-ab-20260926 (local and remote).
Summary and hashes: dev/ppu17/docs/h800-official-ab.md.

Previous performance task: H800 only, priority BF16 causal shape unchanged.
Numerator=68,753,031,168 useful causal FLOPs; fixed dense BF16 peak=756.5TFLOPS
(OEM1513 rating includes structural sparsity); 70-percent threshold=129.833us.
Keep uninstrumented complete-call event span separate from warmed CUPTI
kernel-only duration. Preserve the admitted binary; no kernel/config edits.
Contract: dev/ppu17/docs/h800-perf-plan.md. Completed runs baseline-v1/v3:
kernel-only warmed CUPTI medians168.810/163.354us (53.84/55.64% useful MFU);
uninstrumented full-call spans174.563/182.928us (52.06/49.68%). All sample
ranges miss the fixed70% target. CPU FP64 O/LSE + admitted fingerprint PASS
before/after timing. One intervening attempt was blocked by another task,
before any GPU work; it contributed no samples. No clocks/kernel/config
changed. Raw evidence: /workspace/flash-attn-ppu17-perf-20260926.
Interpretation/limits: dev/ppu17/docs/h800-causal-performance.md.

H800 preflight (2026-09-26, physical device, not PPU1.7 simulation):
NVIDIA H800 PCIe, SM90, 114 SM, 50 MiB L2; driver 595.71.05,
CUDA toolkit 12.8.93, Python 3.12.3, Torch 2.8.0+cu128, CXX11 ABI enabled.
Two initial samples showed no GPU process and zero memory/utilization. A later
sample caught another task's `single-launch` GPU process, so no FlashAttention
build, forward validation or timing was started. A gap between that task's
launches is not evidence that the task has finished. Respect idle-only admission
for the entire validation; do not terminate other processes or change MIG,
clocks, drivers, or system packages. H800 can validate the shared Hopper path,
not native PPU1.7-specific behavior or the 20-SM / 32-MiB model's performance.
No credentials are recorded in this checkout. Native PPU1.7 remains unverified.
Local preparation after resume: runner hardware mode records measured L2 and
runtime/input identities, without changing its single forward or fixed
CPU-reference tolerances. 21 host contracts + 3 CPU-reference tests PASS.
The exact backend snapshot for this control remains 023e82d, matching the
source/rebase validation; unrelated newer CUTLASS work is not substituted.
At 03:36:08 UTC both remote task parents had exited and the GPU was empty.
Build-only step started after a fresh guard checked GPU processes and other
Python/compiler parents. Remote build directory:
`/workspace/flash-attn-ppu17-control-e7ee864`. Source/backend archive hashes
match locally and remotely. The matched Python3.12/Torch2.8 extension built,
loaded and passed all 22 declared numerical cases with the unchanged FP64 CPU
oracle/tolerances. All per-case before/after occupancy and parent-task checks
were idle. No compute source was changed. The priority BF16 causal case
B1/S2048/H32/Hkv2/D256 has output max_abs=0.009159422 and LSE PASS.
One fresh-process trace contains exactly one actual SM90 attention kernel,
no GPU reference/init/prepare/combine, and the same output fingerprint as
the untraced process. Grid114/block384, 168 registers/thread, 232448 shared
bytes are H800 launch metadata, NOT PPU1.7 capacities or performance claims.
Evidence: dev/ppu17/results/h800-20260926; explanation:
dev/ppu17/docs/h800-validation.md. Binary SHA256:
a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320.
H800 timing is now recorded separately above; native PPU1.7 and simulator
remain unverified. Do not repurpose the earlier single-call trace as timing.

Rebase parent: f056429 (latest DrXuQian/v2.8.2, including upstream 664597d).
Recovery branch: backup/ppu17-before-rebase-20260925 at a9e497a.
Initial rebased compile caught two HGGC-only declarations in the new shared
QSA header; they now use the existing runtime alias. Object compilation then
caught unguarded __ppu_read_firstlane/__ld_smem in its direct-index consumer.
Those legacy intrinsics are now outside the PPU1.7 parser path. QSA remains
unsupported on PPU1.7 and fails closed, without changing upstream legacy QSA.
Fresh validation: 19 host + 3 CPU tests PASS; 6/6 SM90 generated units / 12
bodies PASS; real host/internal link PASS; 5 real compile/link plants
EXPECTED_RED. Legacy PPU1.0 D128 BF16 compile PASS using upstream's updated
actlize submodule 9771215. Evidence: /workspace/flash-attn-ppu17-rebase-20260925.
That rebase checkpoint did not verify a matched extension load; the H800
control above now closes it. Native PPU1.7 / simulator remain NOT VERIFIED.

Historical, before rebase: local host contracts: 18 passed; CPU reference
tests: 3 passed. All six real
generated units / twelve bodies compiled and assembled with WGMMA + TMA
read/write. Shared-source hashes and real host/internal-link closure PASS.
Five real target/link negatives rejected the intended planted defects. Legacy
PPU1.0 D128 BF16 generated unit compiled on HGCC2.1.1. Simulation counter init
uses CPU zero + H2D rather than a GPU fill kernel. Native PPU1.7,
simulation numerics and performance are NOT VERIFIED.
The independent skill lives at /root/.codex/skills/ppu17-hopper-porting and
passed skill validation. No changes
to CUTLASS tree, admitted PPU1.0/1.5 compute bodies, or original worktree.
