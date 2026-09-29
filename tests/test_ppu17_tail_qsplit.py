"""Independent Q-tail experiment: actual selector/owners and fail-closed guards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_ppu17_softmax_overlap as gate


class TailQContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(f"/workspace/fa17-tail-contracts-{os.getpid()}")
        cls.out.mkdir(parents=True, exist_ok=False)
        cls.flags = ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_KV_TILE128=1"]

    def compile_selector(self, name, flags):
        target = self.out / name
        process = subprocess.run([
            "g++", "-std=c++17", "-O2", "-I" + str(ROOT / "hopper"),
            str(ROOT / "dev/ppu17/experiments/kv128/selector.cpp"),
            *flags, "-o", str(target)], capture_output=True, text=True)
        return target, process

    def test_actual_selector_changes_only_q_not_k_or_exp(self):
        tables = []
        for name, flags in (("control", self.flags),
                            ("m64", [*self.flags, "-DFLASHATTN_PPU17_Q_TAIL_MODE=1"])):
            exe, result = self.compile_selector(name, flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [tuple(map(int, s.split())) for s in
                    subprocess.check_output([str(exe)], text=True).splitlines()]
            self.assertEqual(len(rows), 1536)
            table = {row[:4]: row[4:] for row in rows}
            self.assertEqual(len(table), 1536)
            tables.append(table)
        changed = [key for key in tables[0] if tables[0][key] != tables[1][key]]
        self.assertEqual(changed, [(128, 128, 2, 0)])
        self.assertEqual(tables[0][changed[0]], (128, 128, 1, 1))
        self.assertEqual(tables[1][changed[0]], (64, 128, 1, 1))

    def test_unadmitted_hybrid_and_composed_exp_fail_compilation(self):
        for name, flags, message in (
            ("hybrid", [*self.flags, "-DFLASHATTN_PPU17_Q_TAIL_MODE=2"], "hybrid was rejected"),
            ("exp", [*self.flags, "-DFLASHATTN_PPU17_Q_TAIL_MODE=1",
                     "-DFLASHATTN_PPU17_SOFTMAX_OVERLAP=1"], "must not be composed"),
            ("no-kv", ["-DFLASHATTN_PPU17=1", "-DFLASHATTN_PPU17_Q_TAIL_MODE=1"], "requires explicit")):
            with self.subTest(name=name):
                _, result = self.compile_selector(name, flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejected_hybrid_not_exposed_by_builder(self):
        for name, flags, message in (
            ("hybrid", ["--kv-tile128", "--q-tail-mode", "hybrid"], "invalid choice"),
            ("missing-kv", ["--q-tail-mode", "m64"], "requires --kv-tile128")):
            out = self.out / name
            result = subprocess.run([sys.executable, str(ROOT / "tools/build_ppu17_standalone.py"),
                                     "--out", str(out), *flags], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertFalse(out.exists())

    @unittest.skipUnless(os.environ.get("FA17_TAIL_OWNERS"), "actual CuTe ownership executable not supplied")
    def test_actual_layout_and_five_plants(self):
        for mode in ("uniform", "hybrid"):
            for plant in ("none", "duplicate", "omit", "head", "origin", "denominator"):
                with self.subTest(mode=mode, plant=plant):
                    p = subprocess.run([os.environ["FA17_TAIL_OWNERS"], plant, mode],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0 if plant == "none" else 1, p.stdout + p.stderr)
                    if plant == "none":
                        self.assertIn("cells=7340032 rows=57344", p.stdout)
                        self.assertIn("prefix=0 halves=896" if mode == "uniform" else
                                      "prefix=440 halves=16", p.stdout)

    @unittest.skipUnless(os.environ.get("FA17_TAIL_EXE"), "M64 built executable not supplied")
    def test_exact_compiled_body_and_instruction_negative(self):
        exe = Path(os.environ["FA17_TAIL_EXE"])
        record = json.loads(subprocess.check_output([str(exe), "--describe"], text=True))
        self.assertEqual((record["q_tail_mode"], record["tile_m"], record["tile_n"],
                          record["softmax_overlap"]), ("m64", 64, 128, "control"))
        sass = (exe.parent / "executable.sass").read_text()
        actual = gate.inspect_sass(sass, kv_tile=128)
        self.assertEqual(actual["local_sites"], {})
        self.assertEqual(actual["early_old_P_or_O_writes"], 0)
        start = sass.index("StaticPersistentTileScheduler")
        bad = sass[:start] + sass[start:].replace("HGMMA.64x128x16", "NOT_AN_MMA", 1)
        with self.assertRaisesRegex(ValueError, "matrix denominator"):
            gate.inspect_sass(bad, kv_tile=128)


if __name__ == "__main__":
    unittest.main()
