#!/usr/bin/env python3
"""Pool the registered two-pass cells; missing cells/errors cannot turn green."""
import argparse
import bisect
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from bench_h800_causal_tuning import check_candidate_receipt, inventory, sha
from bench_ppu17_hopper_control import CALLS, SAMPLES, WARMUP, summarize

LABELS = ("N80-LPT", "N64-LPT", "N80-single", "N64-single", "flashinfer", "cudnn")
PHASES = ("full_call_event_span", "instrumented_kernel_only")


def compare(control, subject):
    if subject["max_us"] < control["min_us"]:
        return "FASTER"
    if subject["min_us"] > control["max_us"]:
        return "SLOWER"
    return "UNRESOLVED"


def power_join(path, record, trace):
    monitor = record["power_monitor"]
    power_path = path.parent / "power.csv"
    if sha(power_path) != monitor["sha256"]:
        raise ValueError("power log hash drift")
    tz = datetime.fromisoformat(monitor["host_timezone"]).tzinfo
    kernels = sorted((e for e in trace["traceEvents"] if e.get("cat") == "kernel"), key=lambda e:e["ts"])
    base = trace["baseTimeNanoseconds"] / 1e9
    starts = [base+e["ts"]/1e6 for e in kernels]
    ends = [base+(e["ts"]+e["dur"])/1e6 for e in kernels]
    hits = []
    for raw in csv.DictReader(power_path.read_text().splitlines()):
        row = {k.strip():v.strip() for k,v in raw.items()}
        t = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
        i = bisect.bisect_right(starts,t)-1
        if i >= 0 and t <= ends[i]:
            hits.append({"utc":datetime.fromtimestamp(t,timezone.utc).isoformat(),
                         "sm_mhz":float(row["clocks.current.sm [MHz]"].split()[0]),
                         "power_w":float(row["power.draw [W]"].split()[0]),
                         "sw_power_cap":row["clocks_event_reasons.sw_power_cap"] == "Active"})
    return {"inside_actual_attention_samples":len(hits), "cap_active_samples":sum(x["sw_power_cap"] for x in hits),
            "hits":hits, "scope":"passive point samples, not cycle-weighted clocks or fraction of time throttled"}


def device_identity(snapshot):
    devices = [{k.strip():v.strip() for k,v in row.items()}
               for row in csv.DictReader(snapshot.splitlines())]
    if len(devices) != 1:
        raise ValueError("ambiguous physical device snapshot")
    return tuple(devices[0][k] for k in ("name", "uuid", "driver_version", "power.limit [W]"))


def summarize_root(root, suffix="-r2", large_reference_suffix="-r3"):
    rows, receipts = [], []
    physical_device = None
    for s in (2048,8192):
        fixture = oracle = None
        for label in LABELS:
            first = ("screen-"+label+suffix if label.startswith("N") else
                     "references-"+label+(suffix if s == 2048 else large_reference_suffix))
            paths = [root/f"S{s}"/first/"result.json",
                     root/f"S{s}"/("confirm-"+label+suffix)/"result.json"]
            runs = []
            for path in paths:
                r = json.loads(path.read_text())
                if (r["shape"] != [1,s,32,256] or r["kv_heads"] != 2 or
                        r["dtype"] != "bf16" or r["causal"] is not True or
                        r["samples"] != SAMPLES or r["calls_per_sample"] != CALLS or
                        r["warmup"] != WARMUP or r["label"] != label or
                        r["numerics"] != "CPU-FP64-O/LSE+WITHIN-ARM-RAW-REPLAY/PASS"):
                    raise ValueError(f"wrong invocation/numerics: {path}")
                if fixture is None:
                    fixture, oracle = r["input_sha256"], r["oracle"]
                before, after = (device_identity(r[k]) for k in ("device_before","device_after_trace"))
                if physical_device is None:
                    physical_device = before
                if before != physical_device or after != physical_device:
                    raise ValueError("device/driver/power-limit identity changed")
                if r["input_sha256"] != fixture or r["oracle"] != oracle:
                    raise ValueError("cross-arm fixture/oracle changed")
                trace = path.parent / "kernel-timing-trace.json"
                if sha(trace) != r["trace_sha256"]:
                    raise ValueError("trace hash drift")
                trace_data = json.loads(trace.read_text())
                actual = inventory(trace_data, r["arm"], SAMPLES*CALLS)
                if actual["kernel_durations_us"] != r["kernel_durations_us"]:
                    raise ValueError("saved durations differ from trace")
                check_candidate_receipt(r)
                if any(len(r[p]["samples_us"]) != SAMPLES for p in PHASES):
                    raise ValueError("sample denominator changed")
                runs.append(r)
                receipts.append({"path":str(path.relative_to(root)), "result_sha256":sha(path),
                                 "power_trace_join":power_join(path,r,trace_data),
                                 **{k:r[k] for k in ("arm","label","shape","input_sha256","oracle",
                                       "extension_sha256","source_identity_sha256","script_sha256",
                                       "trace_sha256","output_sha256","lse_sha256","max_output_abs",
                                       "max_lse_abs","device_before","device_after_trace","power_monitor",
                                       "loaded_library_sha256","kernel_arguments","gpu_copy_memset_counts") if k in r}})
            if any(runs[0][k] != runs[1][k] for k in
                   ("output_sha256","lse_sha256","source_identity_sha256","kernel_name")):
                raise ValueError("between-run output or implementation changed")
            row = {"label":label, "seqlen":s, "runs":2,
                   "numerics":"CPU-FP64-O/LSE+RAW-REPLAY/PASS",
                   "kernel_arguments":runs[0]["kernel_arguments"], "kernel_name":runs[0]["kernel_name"]}
            for p in PHASES:
                row[p] = summarize([x for r in runs for x in r[p]["samples_us"]], runs[0]["useful_causal_flops"])
            rows.append(row)
        control = next(r for r in rows if r["label"] == "N80-LPT" and r["seqlen"] == s)
        for row in rows:
            if row["seqlen"] != s:
                continue
            for p in PHASES:
                row[p]["vs_control"] = "CONTROL" if row is control else compare(control[p], row[p])
                row[p]["median_speedup_vs_control"] = control[p]["median_us"] / row[p]["median_us"]
    return {"scope":"physical H800; NOT native PPU1.7/model", "source_parent":"ed150c9",
            "registered_primary_runs":24, "admitted_primary_runs":len(receipts),
            "shape_count":2, "rows":rows, "receipts":receipts,
            "excluded_from_primary_pool":[
                "cudnn-r3: valid preliminary pilot, not the two-pass balanced comparison",
                "BUSY, missing packaging/build tool, and old FlashInfer symbol-parser failures: no admitted timing"],
            "order_note":("S8192 first reference pass was deferred after a prelaunch BUSY refusal; recovered in references-r3. Candidate order reversed without dropping cells."
                          if large_reference_suffix != suffix else
                          "Common suffix for reference/screen/confirm passes; candidate order reversed without dropping cells.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--format", choices=("json","csv","table"), default="json")
    ap.add_argument("--suffix", default="-r2")
    ap.add_argument("--large-reference-suffix", default="-r3")
    args = ap.parse_args()
    result = summarize_root(args.root, args.suffix, args.large_reference_suffix)
    if args.format == "json":
        print(json.dumps(result, indent=2))
    elif args.format == "csv":
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(("arm","seqlen","phase","median_us","min_us","max_us","useful_causal_mfu_percent","numerics","decision"))
        for row in result["rows"]:
            for phase in PHASES:
                t = row[phase]
                writer.writerow((row["label"],row["seqlen"],phase,t["median_us"],t["min_us"],t["max_us"],
                                 t["useful_mfu_percent"],row["numerics"],t["vs_control"]))
        print(out.getvalue(), end="")
    else:
        for row in result["rows"]:
            k, f = (row[p] for p in ("instrumented_kernel_only","full_call_event_span"))
            print(f"S{row['seqlen']} {row['label']:12s} kernel={k['median_us']:.3f}us MFU={k['useful_mfu_percent']:.2f}% {k['vs_control']:10s} full={f['median_us']:.3f}us {f['vs_control']}")


if __name__ == "__main__":
    main()
