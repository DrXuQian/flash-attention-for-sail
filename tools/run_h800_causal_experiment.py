#!/usr/bin/env python3
"""Serial runner for the registered 2x2 H800 experiment (no auto-promotion)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bench_ppu17_hopper_control import require_idle

ROOT = Path(__file__).resolve().parents[1]
CONTROL_SHA = "a78aff6c443238cd888f53dc692bf1e017b9191cff137427677ff1da84640320"
CELLS = ((64, "LPT"), (80, "single"), (64, "single"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", type=Path, required=True)
    ap.add_argument("--build-root", type=Path, help="fresh build attempt; never overwrite a failed receipt")
    ap.add_argument("--control-dir", type=Path, required=True)
    ap.add_argument("--flashinfer-root", type=Path, required=True)
    ap.add_argument("--flashinfer-python", type=Path, required=True)
    ap.add_argument("--phase", choices=("references", "build", "screen", "confirm"), required=True)
    args = ap.parse_args()
    require_idle()
    args.out.mkdir(parents=True, exist_ok=True)
    build_root = args.build_root or args.out / "build"
    # Stay alive as the direct parent across cells. The idle gate excludes only
    # its own process and parent, so peer tasks can see this whole campaign.
    def run(cmd, log, env=None):
        require_idle()
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            raise FileExistsError(log)
        print("[causal campaign] "+" ".join(map(str,cmd)), flush=True)
        with log.open("x") as f:
            p = subprocess.run(list(map(str,cmd)), cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(log.read_text()[-7000:], flush=True)
        if p.returncode:
            raise RuntimeError(f"child failed rc={p.returncode}; evidence={log}")
        require_idle()

    if args.phase == "build":
        for n, scheduler in CELLS:
            label = f"N{n}-{scheduler}"
            run([sys.executable, ROOT / "tools/build_h800_causal_candidate.py", "--n", n,
                 "--scheduler", scheduler, "--backend", args.backend, "--out", build_root / label],
                build_root / f"{label}.log")
        return

    reference_identity = ROOT / "dev/ppu17/experiments/causal-tuning/reference-identity.json"
    if args.phase == "references":
        labels = ("flashinfer", "cudnn")
    elif args.phase == "screen":
        labels = ("N80-LPT", "N64-LPT", "N80-single", "N64-single")
    else:
        # Reverse all cells, not only a retrospectively cherry-picked winner.
        labels = ("N64-single", "N80-single", "N64-LPT", "N80-LPT", "cudnn", "flashinfer")
    for seqlen in (2048, 8192):
        for label in labels:
            env = {**os.environ, "OMP_NUM_THREADS":"8"}
            python = sys.executable
            extra = []
            identity = reference_identity
            if label == "N80-LPT":
                arm = "control"
                extra = ["--extension-dir", args.control_dir, "--expected-sha256", CONTROL_SHA]
            elif label.startswith("N"):
                arm = "candidate"
                identity = build_root / label / "identity.json"
                built = json.loads(identity.read_text())
                extra = ["--extension-dir", built["extension_dir"],
                         "--expected-sha256", built["extension_sha256"]]
            else:
                arm = label
                if label == "flashinfer":
                    python = args.flashinfer_python
                    extra = ["--flashinfer-source-root", args.flashinfer_root]
                    env.update(PYTHONPATH=str(args.flashinfer_root), FLASHINFER_CUDA_ARCH_LIST="9.0a",
                               FLASHINFER_WORKSPACE_BASE=str(args.out / "fi-cache"))
            cell = args.out / f"S{seqlen}" / f"{args.phase}-{label}"
            run([python, ROOT / "tools/bench_h800_causal_tuning.py", "--arm", arm,
                 "--label", label, "--seqlen", seqlen, "--out", cell,
                 "--oracle-cache", args.out / "oracles" / f"S{seqlen}.pt",
                 "--source-identity", identity, *extra], cell.with_suffix(".log"), env)


if __name__ == "__main__":
    main()
