#!/usr/bin/env python3
"""Prove narrow candidate API rejections happen before any device kernel."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from bench_h800_causal_tuning import sha
from bench_ppu17_hopper_control import require_idle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    require_idle()
    args.out.mkdir(parents=True, exist_ok=False)
    info = json.loads(args.identity.read_text())
    sys.path.insert(0, info["extension_dir"])
    import torch
    extension = importlib.import_module("flash_attn_3._C")
    if sha(extension.__file__) != info["extension_sha256"]:
        raise ValueError("candidate binary changed")
    cases = []
    for name, dtype, d, causal in (("noncausal",torch.bfloat16,256,False),
                                   ("fp16",torch.float16,256,True),
                                   ("D64",torch.bfloat16,64,True)):
        qkv = [torch.ones((1,2,h,d), dtype=dtype, device="cpu").to("cuda") for h in (32,2,2)]
        cases.append((name,qkv,causal))
    torch.cuda.synchronize()
    verdicts = []
    with torch.inference_mode(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
        for name,(q,k,v),causal in cases:
            try:
                torch.ops.flash_attn_3.fwd(q,k,v,softmax_scale=1/16,is_causal=causal,num_splits=1,pack_gqa=False)
            except RuntimeError as e:
                if "causal experiment binary admits only BF16 D256 causal forward" not in str(e):
                    raise
                verdicts.append({"plant":name,"verdict":"EXPECTED_RED","reason":str(e)})
            else:
                raise AssertionError(f"{name} was silently admitted")
        torch.cuda.synchronize()
    require_idle()
    trace = args.out / "admission-trace.json"
    prof.export_chrome_trace(str(trace))
    if any(e.get("cat") == "kernel" for e in json.loads(trace.read_text())["traceEvents"]):
        raise AssertionError("rejection occurred only after device work")
    record = {"extension_sha256":info["extension_sha256"], "tests":verdicts,
              "device_kernel_launches":0, "trace_sha256":sha(trace)}
    (args.out / "result.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


if __name__ == "__main__":
    main()
