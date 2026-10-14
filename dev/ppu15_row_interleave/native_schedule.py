"""Trace emitted FMA -> EXP2 row identities, not source-loop intentions.

The scaled row maximum is identified by its defining native instruction,
not only its physical register number (the control reuses one register for
two rows). This is a static scheduling observation, NOT a cycle/stall model.
"""
from collections import Counter
import hashlib
from itertools import groupby
import json
from pathlib import Path
import re
import sys

COMMON = Path(__file__).resolve().parents[1] / "ppu15_bn128"
sys.path.insert(0, str(COMMON))
from native import inspect_native


def instructions(disassembly, symbol):
    marker = f"Disassembly of section .text.kernel.{symbol}:"
    if disassembly.count(marker) != 1:
        raise ValueError("native symbol must appear once")
    body = disassembly.split(marker)[1].split("Disassembly of section", 1)[0]
    # Some instructions have extension bytes: assuming exactly eight bytes
    # silently drops sites (including MMAs) from the denominator.
    pattern = r"^\s*([0-9a-f]+):[ \t]+((?:[0-9a-f]{2}[ \t]+)+)([a-z][a-z0-9_.]+)\b([^\n]*)"
    return [{"pc": m[1], "bytes": bytes.fromhex(m[2]).hex(), "opcode": m[3],
             "args": [a.strip().replace(".reuse", "") for a in m[4].split(",")]}
            for m in re.finditer(pattern, body, re.M)]


def destinations(arg):
    if re.fullmatch(r"[vs]reg\d+", arg):
        return [arg]
    match = re.fullmatch(r"([vs]reg)\[(\d+):(\d+)\]", arg)
    return [f"{match[1]}{i}" for i in range(int(match[2]), int(match[3])+1)] if match else []


def row_trace(native):
    definitions, versions, trace = {}, {}, []
    for index, ins in enumerate(native):
        a, op = ins["args"], ins["opcode"]
        if op == "v.exp2.f32" and len(a) == 2 and a[1] in definitions:
            definition = definitions[a[1]]
            fma = definition["instruction"]
            fargs = fma["args"]
            if (fma["opcode"] == "v.fma.f32.rtte" and len(fargs) == 4
                    and re.fullmatch(r"sreg\d+", fargs[1])
                    and re.fullmatch(r"!vreg\d+", fargs[3])):
                if definition["bias_version"] is None:
                    raise ValueError("scaled maximum definition missing")
                trace.append({"exp_pc": ins["pc"], "fma_pc": fma["pc"],
                              "row_key": fargs[3] + "@" + definition["bias_version"],
                              "input_register": a[1],
                              "fma_to_exp_static_distance": index-definition["index"],
                              "next_fma_overwrites_exp_input": (
                                  index+1 < len(native)
                                  and native[index+1]["opcode"] == "v.fma.f32.rtte"
                                  and native[index+1]["args"][0] == a[1])})
        bias_version = versions.get(a[-1].lstrip("!"))
        for dest in destinations(a[0]):
            versions[dest] = ins["pc"]
            definitions[dest] = {"instruction": ins, "index": index,
                                 "bias_version": bias_version}
    # Exact 128x64 / 128-thread fragment: 4 rows * 16 cols, in each of
    # the first-tile and recurrent-tile softmax bodies. Do not shrink this
    # denominator when parsing fails.
    if len(trace) != 128:
        raise ValueError(f"FMA/EXP coverage {len(trace)}/128, expected two complete 4x16 bodies")
    result = {}
    for name, events in (("first_tile", trace[:64]), ("recurrent_tile", trace[64:])):
        counts = Counter(e["row_key"] for e in events)
        if len(counts) != 4 or set(counts.values()) != {16}:
            raise ValueError(f"independent-row coverage failed: {name}: {counts}")
        runs = [len(list(g)) for _, g in groupby(events, key=lambda e: e["row_key"])]
        result[name] = {"events": events, "row_counts": dict(counts),
                        "max_same_row_exp_run": max(runs), "row_switches": len(runs)-1,
                        "next_fma_overwrites_exp_input": sum(e["next_fma_overwrites_exp_input"] for e in events)}
    return result


def compare(contract, control_resources, control_isa, candidate_resources, candidate_isa):
    configs = contract["configs"]
    if [c["name"] for c in configs] != ["control", "row-interleave"]:
        raise ValueError("expected exact control/candidate denominator")
    controls = inspect_native(configs[0], control_resources, control_isa)
    candidates = inspect_native(configs[1], candidate_resources, candidate_isa)
    report = {"scope": "native static schedules; performance and device numerics NOT_RUN", "dtypes": {}}
    for dtype in ("fp16", "bf16"):
        c, v = controls[dtype], candidates[dtype]
        if "flash_row_interleave" not in v["symbol"] or "flash_row_interleave" in c["symbol"]:
            raise ValueError("control/candidate device symbols are not isolated")
        if v["vector_registers"] > contract["admission"]["maximum_vector_registers"]:
            raise ValueError("candidate exceeds preregistered register budget")
        for field in ("static_shared_bytes", "dynamic_shared_bytes"):
            if c[field] != v[field]:
                raise ValueError(f"shared footprint changed: {field}")
        # These instruction families must not change with scheduling.
        families = ("v.mm.", "v.mma.", "tsm.ld.", "v.exp2.", "v.fma.f32.",
                    "v.mul.f32", "v.add.f32", "v.pcnvt.")
        protected = lambda item: {k: n for k, n in item["static_opcodes"].items() if k.startswith(families)}
        if protected(c) != protected(v):
            raise ValueError("math/load/conversion static inventory changed")
        cn = instructions(control_isa, c["symbol"])
        vn = instructions(candidate_isa, v["symbol"])
        ct, vt = row_trace(cn), row_trace(vn)
        for phase in ct:
            if (vt[phase]["max_same_row_exp_run"] >= ct[phase]["max_same_row_exp_run"]
                    or vt[phase]["row_switches"] <= ct[phase]["row_switches"]):
                raise ValueError(f"row interleave not exercised in native {phase}")
        encoded_hash = lambda native: hashlib.sha256(b"".join(bytes.fromhex(i["bytes"]) for i in native)).hexdigest()
        report["dtypes"][dtype] = {
            "control": {**c, "native_bytes_sha256": encoded_hash(cn), "schedule": ct},
            "row-interleave": {**v, "native_bytes_sha256": encoded_hash(vn), "schedule": vt},
            "opcode_delta": {k: v["static_opcodes"].get(k, 0)-c["static_opcodes"].get(k, 0)
                             for k in sorted(set(c["static_opcodes"]) | set(v["static_opcodes"]))
                             if v["static_opcodes"].get(k, 0) != c["static_opcodes"].get(k, 0)}}
    report["status"] = "PASS"
    return report


def check_build(build, contract_path):
    build, contract_path = Path(build), Path(contract_path)
    contract = json.loads(contract_path.read_text())
    read = lambda name, kind: (build / f"{name}.{kind}.txt").read_text()
    report = compare(contract, read("control", "resources"), read("control", "isa"),
                     read("row-interleave", "resources"), read("row-interleave", "isa"))
    (build / "row-schedule.json").write_text(json.dumps(report, indent=2) + "\n")
    for dtype, pair in report["dtypes"].items():
        c, v = pair["control"], pair["row-interleave"]
        print(f"[FA15 row schedule] {dtype} recurrent max_same_row_exp_run="
              f"{c['schedule']['recurrent_tile']['max_same_row_exp_run']}->"
              f"{v['schedule']['recurrent_tile']['max_same_row_exp_run']} "
              f"registers={c['vector_registers']}->{v['vector_registers']} "
              f"stack=0 native-interleave=EXERCISED/PASS", flush=True)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    check_build(args.build, Path(__file__).with_name("contract.json"))
