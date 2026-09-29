"""Registered KV128 geometry, actual selector and compile-only negative gates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_ppu17_softmax_overlap as gate


class KV128Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(f"/workspace/fa17-kv128-host-{os.getpid()}")
        cls.out.mkdir(parents=True, exist_ok=False)
        cls.base = ["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "hopper"),
                    str(ROOT / "dev/ppu17/experiments/kv128/selector.cpp")]

    def compile(self, name, flags, success=True):
        binary = self.out / name
        result = subprocess.run([*self.base, *flags, "-o", str(binary)], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return binary, result

    def test_actual_selector_full_denominator_and_single_axis(self):
        controls, _ = self.compile("control", [])
        candidate, _ = self.compile("candidate", ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_KV_TILE128=1"])
        def table(exe):
            rows = [tuple(map(int, line.split())) for line in subprocess.check_output([str(exe)], text=True).splitlines()]
            self.assertEqual(len(rows), 1536)
            self.assertEqual(len({r[:4] for r in rows}), 1536)
            return {r[:4]: r[4:] for r in rows}
        a, b = table(controls), table(candidate)
        changed = [key for key in a if a[key] != b[key]]
        self.assertEqual(changed, [(128, 128, 2, 0)])
        self.assertEqual(a[changed[0]], (128, 176, 1, 1))
        self.assertEqual(b[changed[0]], (128, 128, 1, 1))

    def test_wrong_target_and_mixed_experiments_reject(self):
        for name, flags, message in (
            ("wrong-target", ["-DFLASHATTN_PPU17_KV_TILE128=1"], "requires explicit PPU1.7"),
            ("wrong-value", ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_KV_TILE128=2"], "value of 0 or 1"),
            ("mixed", ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_KV_TILE128=1", "-DFLASHATTN_PPU17_SOFTMAX_OVERLAP=1"], "must not be composed")):
            _, result = self.compile(name, flags, success=False)
            self.assertIn(message, result.stderr)

    def test_builder_rejects_mixture_before_creating_build(self):
        out = self.out / "must-not-exist"
        result = subprocess.run([sys.executable, str(ROOT / "tools/build_ppu17_standalone.py"),
                                 "--out", str(out), "--kv-tile128", "--softmax-overlap"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not be composed", result.stderr)
        self.assertFalse(out.exists())

    @unittest.skipUnless(os.environ.get("FA17_KV128_EXE"), "compiled KV128 ELF not supplied")
    def test_exact_compiled_geometry_and_matrix_negative(self):
        exe = Path(os.environ["FA17_KV128_EXE"])
        manifest = json.loads((exe.parent / "build.json").read_text())
        description = json.loads(subprocess.check_output([str(exe), "--describe"], text=True))
        self.assertEqual(manifest["kv_tile"], 128)
        self.assertEqual(manifest["softmax_overlap"], "control")
        self.assertEqual((description["tile_m"], description["tile_n"], description["softmax_overlap"]),
                         (128, 128, "control"))
        text = (exe.parent / "executable.sass").read_text()
        result = gate.inspect_sass(text, kv_tile=128)
        self.assertEqual(result["matrix_sites"], {"64x128x16": 32})
        # Select the noncausal body to avoid mutating an unrelated causal body.
        body = gate.sass_body(text)
        first_pc = next(pc for pc, op in body if "HGMMA.64x128x16" in op)
        start = text.index(f"/*{first_pc:04x}*/", text.index("StaticPersistentTileScheduler"))
        bad = text[:start] + text[start:].replace("HGMMA.64x128x16", "NOT_AN_MMA", 1)
        with self.assertRaisesRegex(ValueError, "matrix denominator"):
            gate.inspect_sass(bad, kv_tile=128)


if __name__ == "__main__":
    unittest.main()
