"""W01 policy admission and native-code negative controls; no GPU calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "wg_gate", ROOT / "dev/ppu17/experiments/wg-scheduling/validate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class WGSchedulingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(f"/workspace/fa17-wg-policy-tests-{os.getpid()}")
        cls.out.mkdir(parents=True, exist_ok=False)

    def test_preprocessor_admission_and_negative_controls(self):
        common = ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_KV_TILE128=1",
                  "-DFLASHATTN_PPU17_INDEPENDENT_WG=1"]
        variants = [
            ("control", [], None), ("good", common, None),
            ("target", common[1:], "requires explicit PPU1.7"),
            ("value", common[:2] + ["-DFLASHATTN_PPU17_INDEPENDENT_WG=2"], "value 0 or 1"),
            ("kv", [common[0], common[2]], "requires explicit KV128"),
            ("exp", common + ["-DFLASHATTN_PPU17_SOFTMAX_OVERLAP=1"], "must not be composed"),
            ("tail", common + ["-DFLASHATTN_PPU17_Q_TAIL_MODE=1"], "must not be composed"),
        ]
        for name, flags, diagnostic in variants:
            with self.subTest(name=name):
                result = subprocess.run(["g++", "-std=c++17", "-x", "c++", "-fsyntax-only",
                                         "-I" + str(ROOT / "hopper"), *flags, "-"],
                                        input='#include "ppu17_wg_scheduling.h"\n',
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, diagnostic is None, result.stderr)
                if diagnostic: self.assertIn(diagnostic, result.stderr)

    def test_builder_rejects_missing_kv_before_creating_output(self):
        out = self.out / "forbidden"
        result = subprocess.run([sys.executable, str(ROOT / "tools/build_ppu17_standalone.py"),
                                 "--independent-wg", "--out", str(out)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --kv-tile128", result.stderr)
        self.assertFalse(out.exists())

    @unittest.skipUnless(os.environ.get("FA17_WG_CANDIDATE"), "exact W01 build not supplied")
    def test_described_build_and_codegen_negatives(self):
        candidate = Path(os.environ["FA17_WG_CANDIDATE"])
        meta = json.loads((candidate / "build.json").read_text())
        description = json.loads(subprocess.check_output([meta["executable"], "--describe"], text=True))
        self.assertEqual(description["wg_scheduling"], "independent")
        self.assertEqual((description["tile_m"], description["tile_n"]), (128, 128))
        gate.inspect_candidate((candidate / "executable.sass").read_text(),
                               (candidate / "shipping_fp16_d128.ptx").read_text())
        gate.codegen_negatives(candidate, Path(os.environ["FA17_WG_CONTROL"]))

    def test_data_protection_sites_preserved(self):
        gate.inspect_source()


if __name__ == "__main__":
    unittest.main()
