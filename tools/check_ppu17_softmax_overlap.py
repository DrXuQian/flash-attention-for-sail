#!/usr/bin/env python3
"""Inspect the exact S1024 FP16 noncausal overlap window, never execute CUDA.

PTX and NVIDIA SASS are compile evidence only. perfstatistics is the user's
PPU execution evidence. Counts across the three layers are not interchangeable.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def window(rows, wait_pattern, exp_pattern):
    waits = [(i, int(m[1])) for i, row in enumerate(rows)
             if (m := re.search(wait_pattern, row))]
    require([n for _, n in waits] == [0, 1, 0, 0],
            f"expected prologue/steady/drain waits [0,1,0,0], got {waits}")
    start, end, drain = waits[1][0], waits[2][0], waits[3][0]
    inside = [i for i in range(start + 1, end) if re.search(exp_pattern, rows[i])]
    after = [i for i in range(end + 1, drain) if re.search(exp_pattern, rows[i])]
    require(len(inside) + len(after) == 90,
            f"steady softmax denominator changed: {len(inside)}+{len(after)} != 90")
    return {"waits": [i for i, _ in waits], "exp2_before_wait0": len(inside),
            "exp2_after_wait0": len(after)}, inside, after


def inspect_ptx(text):
    entries = re.split(r"(?m)(?=^(?:(?:\.visible|\.weak)\s+)?\.entry\s)", text)
    selected = [s for s in entries if s.lstrip().startswith((".entry", ".visible .entry", ".weak .entry"))
                and "StaticPersistentTileScheduler" in s.splitlines()[0]]
    require(len(selected) == 1, "expected exactly one noncausal StaticPersistent PTX body")
    body = selected[0]
    rows = [s.split("//", 1)[0].strip() for s in body.splitlines()]
    result, _, _ = window(rows, r"wgmma\.wait_group\.sync\.aligned\s+([0-7]);", r"\bex2\.approx")
    counts = collections.Counter(re.findall(r"wgmma\.mma_async\.[^\s]*\.(m64n\d+k16)\.", body))
    require(counts == {"m64n176k16": 16, "m64n128k16": 22}, f"PTX matrix denominator changed: {counts}")
    require(not re.search(r"\b(?:ld|st)\.local", body), "PTX local memory introduced")
    return {"layer": "PTX/source", **result, "matrix_sites": dict(counts)}


def sass_body(text):
    parts = re.split(r"(?m)(?=^[ \t]*Function[ \t]*:[ \t]*)", text)
    selected = [s for s in parts if s.strip() and "StaticPersistentTileScheduler" in s.splitlines()[0]]
    require(len(selected) == 1, "expected exactly one noncausal StaticPersistent SASS body")
    rows = []
    for line in selected[0].splitlines():
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*/\*\s*0x", line)
        if m:
            rows.append((int(m[1], 16), m[2]))
    require(bool(rows), "no encoded SASS instructions")
    return rows


def inspect_sass(text):
    rows = sass_body(text)
    inst = [s for _, s in rows]
    result, _, _ = window(inst, r"WARPGROUP\.DEPBAR\.LE\s+gsb0,\s*0x([0-7])", r"\bMUFU\.EX2\b")
    counts = collections.Counter(re.findall(r"\bHGMMA\.(64x\d+x16)\.", "\n".join(inst)))
    require(counts == {"64x176x16": 16, "64x128x16": 22}, f"SASS matrix denominator changed: {counts}")
    # Existing parent already has local accesses: report them, compare to the
    # parent, and inspect ptxas frame/spill bytes. Never invent a zero baseline.
    local = collections.Counter(re.findall(r"\b(LDL|STL)\b", "\n".join(inst)))
    wait1, wait0 = result["waits"][1:3]
    pv = []
    for index, op in enumerate(inst[:wait1]):
        match = re.search(r"HGMMA\.64x128x16\.F32\s+R(\d+),\s*R(\d+),", op)
        if match:
            pv.append((index, int(match[1]), int(match[2])))
    require(len(pv) == 11, "expected eleven steady PV register operand groups")
    p_words = {reg for _, _, base in pv for reg in range(base, base + 4)}
    o_words = {reg for _, base, _ in pv for reg in range(base, base + 64)}
    require(len(p_words) == 44 and len(o_words) == 64 and not (p_words & o_words),
            "unexpected steady P/O register map")
    for pc, op in rows[pv[0][0] + 1:wait0]:
        # Matrix issue may consume register operands asynchronously. No scalar
        # write to old P or accumulating O may be advanced before completion.
        if "HGMMA." in op:
            continue  # the intended asynchronous O accumulation, not a scalar overwrite
        match = re.match(r"(?:@!?\w+\s+)?([\w.]+)\s+R(\d+)\b", op)
        if match:
            width = 1
            if match[1].startswith(("LDL", "LDS", "LDG")):
                width = 4 if ".128" in match[1] else (2 if ".64" in match[1] else 1)
            touched = set(range(int(match[2]), int(match[2]) + width))
            require(not touched & (p_words | o_words),
                    f"early old-P/O overwrite before wait0 at {hex(pc)}: {op}")
    result["wait_pcs"] = [hex(rows[i][0]) for i in result.pop("waits")]
    return {"layer": "NVIDIA-SASS/compile-control", **result, "matrix_sites": dict(counts),
            "static_instructions": len(rows), "local_sites": dict(local),
            "old_P_words": len(p_words), "live_O_words": len(o_words),
            "early_old_P_or_O_writes": 0}


def inspect_report(report):
    require(len(report["ppu"]) == 1, "expected one PPU report")
    p = report["ppu"][0]
    require(p["scheduler_statistics"]["kernel_num"] == 1, "not a single-kernel simulation")
    require(p["scheduler_statistics"]["block_num"] == 40, "expected this experiment's 40-CTA launch")
    rows = sorted(p["instruction_statistics"]["source_view_data"]["data"], key=lambda r: int(r["pc"], 16))
    require(sum(r["executed"] for r in rows) == p["instruction_statistics"]["executed_instructions"],
            "per-PC instruction denominator does not close")
    require(sum(n for _, n in p["instruction_statistics"]["inst_histogram_data"])
            == p["instruction_statistics"]["executed_instructions"],
            "opcode instruction denominator does not close")
    inst = [r["inst"] for r in rows]
    result, before, after = window(inst, r"s\.wait\s+gmma_commit_grp\(([0-7])\)", r"\bv\.exp2\.f32\b")
    waits = result.pop("waits")
    require([rows[i]["executed"] for i in waits] == [3584, 17920, 17920, 3584],
            "wrong prologue/steady/drain execution counts")
    require(all(rows[i]["executed"] == 17920 for i in before + after),
            "some steady EX2 PCs are not executed by the full cohort")
    counts = collections.Counter()
    for row in rows:
        m = re.search(r"v\.mm[a]?\.g\.f32\.f16\.(m64n\d+k16)", row["inst"])
        if m:
            counts[m[1]] += row["executed"]
    require(counts == {"m64n176k16": 172032, "m64n128k16": 236544},
            f"executed matrix denominator changed: {counts}")
    result["wait_pcs"] = [rows[i]["pc"] for i in waits]
    result["wait_sync_warp_cycles"] = [rows[i]["stall_reasons"]["sync"] for i in waits]
    cycles = p["ppu_overview"]["compute_cycles"]
    return {"layer": "PPU-simulation", **result, "compute_cycles": cycles,
            "matrix_warp_executions": dict(counts),
            "executed_instructions": p["instruction_statistics"]["executed_instructions"],
            "private_traffic": inspect_private_traffic(rows, p["memory_statistics"]),
            "useful_mfu_percent": 100 * 30064771072 / (cycles * 163840),
            "peak_scope": "this 40-SM model only; report-derived 40*4096 FLOP/cycle",
            "numerics": "NOT_IN_PERFSTATISTICS"}


def inspect_private_traffic(rows, memory):
    """Account for this lowering's lane-private scratch, not HBM transactions.

    The S1024 body loads all Q/K/V via TMA. Its C03 lowering additionally uses
    [slot + (vreg + %tid) * 4] @sreg-pair for private temporaries. The encoded
    slot is NOT a byte offset. All such accesses have the full 32-lane math
    cohort; close the read-byte total against the independent memory counter.
    Unknown scalar loads are not silently called zero-spill.
    """
    slots = {}
    for row in rows:
        op = row["inst"]
        match = re.fullmatch(
            r"vmem\.(ld|st)\.b32 vreg\d+, \[(0x[0-9a-f]+) \+ "
            r"\(vreg\d+ \+ %tid\) \* 0x4\] @sreg\[\d+:\d+\]", op)
        if not match:
            require(not op.startswith("vmem.ld."), "unclassified scalar read: " + op)
            continue  # output/LSE stores and TMA have different roles
        access, slot = match.groups()
        entry = slots.setdefault(slot, {"ld_warp_executions": 0, "st_warp_executions": 0})
        entry[access + "_warp_executions"] += row["executed"]
    reads = 32 * 4 * sum(s["ld_warp_executions"] for s in slots.values())
    writes = 32 * 4 * sum(s["st_warp_executions"] for s in slots.values())
    require(reads == memory["vmem_inst_read_bytes"], "private read-byte denominator does not close")
    require(writes <= memory["vmem_inst_write_bytes"], "private writes exceed total scalar stores")
    return {"read_bytes": reads, "write_bytes": writes, "slot_count": len(slots),
            "slots": dict(sorted(slots.items())), "scope": "lane-private traffic, not HBM bytes"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ptx", "sass", "report"))
    p.add_argument("input", type=Path)
    p.add_argument("--require-before", type=int, help="explicit static EX2 count required before wait0")
    p.add_argument("--max-private-bytes", type=int,
                   help="PPU report only: explicit read+write budget from its measured control")
    a = p.parse_args()
    raw = a.input.read_bytes()
    if a.kind == "report":
        result = inspect_report(json.loads(raw))
    else:
        result = (inspect_ptx if a.kind == "ptx" else inspect_sass)(raw.decode("utf-8", "backslashreplace"))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    if a.require_before is not None:
        require(result["exp2_before_wait0"] == a.require_before,
                f"overlap postcondition: expected {a.require_before}, got {result['exp2_before_wait0']}")
    if a.max_private_bytes is not None:
        require(a.kind == "report" and a.max_private_bytes >= 0,
                "private-traffic budget requires a PPU report and a nonnegative limit")
        traffic = result["private_traffic"]
        require(traffic["read_bytes"] + traffic["write_bytes"] <= a.max_private_bytes,
                f"private traffic exceeds control budget: {traffic['read_bytes']} read + "
                f"{traffic['write_bytes']} write > {a.max_private_bytes} bytes")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"[FA17 overlap] FAIL: {error}")
