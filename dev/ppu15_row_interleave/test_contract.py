#!/usr/bin/env python3
"""Host and real-native negatives; no device correctness claim."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "ppu15_bn128"))
import bench
from native import inspect_native
from native_schedule import compare, instructions, row_trace

CONTRACT = json.loads((HERE / "contract.json").read_text())


class ContractTests(unittest.TestCase):
    def test_only_schedule_varies(self):
        c, v = CONTRACT["configs"]
        for key in ("geometry_id", "bm", "bn", "warps", "shared_bytes"):
            self.assertEqual(c[key], v[key])
        self.assertNotIn("defines", c)
        self.assertEqual(v["defines"], ["FLASHATTN_PPU15_SOFTMAX_ROW_INTERLEAVE=1",
                                       "FLASH_NAMESPACE=flash_row_interleave"])
        self.assertTrue(CONTRACT["numeric"]["cross_control_raw_equality_required"])

    def test_real_validation_path_raw_negatives(self):
        # CPU stand-in for the C ABI output, calling the real admission code.
        # A one-ULP perturbation passes FP64 tolerances but must fail RAW-BIT.
        for dtype, storage in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            seq = 3
            q = torch.zeros(1, seq, 56, 128, dtype=storage)
            cpu = [q, q.clone(), torch.ones_like(q)]
            _, expected, expected_lse = bench.oracle(cpu, seq)

            class StandIn:
                def __init__(self, index, plant=None):
                    self.entry = {"id": index, "name": "control" if index == 0 else "row-interleave"}
                    self.plant = plant
                def prepare(self, *unused):
                    self.o = torch.empty_like(q)
                    self.lse = torch.empty(1, 56, seq)
                def launch(self):
                    self.o[0].copy_(expected)
                    self.lse[0].copy_(expected_lse)
                    if self.plant == "output":
                        self.o[0, 0, 0, 0] = torch.nextafter(self.o[0, 0, 0, 0], torch.tensor(float("inf"), dtype=storage))
                    if self.plant == "lse":
                        self.lse[0, 0, 0] = torch.nextafter(self.lse[0, 0, 0], torch.tensor(float("inf")))
                def release(self):
                    pass

            with patch.object(bench, "CONTRACT", CONTRACT):
                bench.validate([StandIn(0), StandIn(1)], cpu, cpu, dtype, seq)
                for plant in ("output", "lse"):
                    with self.subTest(dtype=dtype, plant=plant):
                        with self.assertRaisesRegex(AssertionError, "RAW-BIT equal"):
                            bench.validate([StandIn(0), StandIn(1, plant)], cpu, cpu, dtype, seq)

    def test_extended_instruction_encoding_is_not_dropped(self):
        source = ("Disassembly of section .text.kernel.test:\n"
                  "  10: 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f v.mma.test vreg[8:15], vreg[0:3]\n")
        parsed = instructions(source, "test")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(len(bytes.fromhex(parsed[0]["bytes"])), 16)

    @unittest.skipUnless(os.environ.get("FA15_ROW_BUILD"),
                         "FA15_ROW_BUILD missing: real full-kernel native admission not exercised")
    def test_real_native_and_three_negative_controls(self):
        build = Path(os.environ["FA15_ROW_BUILD"])
        read = lambda name, kind: (build / f"{name}.{kind}.txt").read_text()
        cr, ci = read("control", "resources"), read("control", "isa")
        vr, vi = read("row-interleave", "resources"), read("row-interleave", "isa")
        self.assertEqual(compare(CONTRACT, cr, ci, vr, vi)["status"], "PASS")
        # Keep candidate naming but replace its native body with the actual
        # control schedule: namespace alone must not satisfy this gate.
        renamed_cr = cr.replace("_ZN5flash", "_ZN20flash_row_interleave")
        renamed_ci = ci.replace("_ZN5flash", "_ZN20flash_row_interleave")
        with self.assertRaisesRegex(ValueError, "not exercised"):
            compare(CONTRACT, cr, ci, renamed_cr, renamed_ci)
        with self.assertRaisesRegex(ValueError, "spill"):
            compare(CONTRACT, cr, ci, vr.replace("STACK SIZE:0", "STACK SIZE:16", 1), vi)
        for config, resources, isa in ((CONTRACT["configs"][0], cr, ci), (CONTRACT["configs"][1], vr, vi)):
            for info in inspect_native(config, resources, isa).values():
                native = instructions(isa, info["symbol"])
                trace = row_trace(native)
                removed_pc = trace["recurrent_tile"]["events"][-1]["exp_pc"]
                with self.assertRaisesRegex(ValueError, "127/128"):
                    row_trace([ins for ins in native if ins["pc"] != removed_pc])

    @unittest.skipUnless(os.environ.get("FA15_ROW_BUILD") and os.environ.get("FA15_PARENT_NATIVE"),
                         "parent native artifact not supplied; do not claim default machine-code identity")
    def test_default_native_bytes_match_untouched_parent(self):
        current = Path(os.environ["FA15_ROW_BUILD"])
        parent = Path(os.environ["FA15_PARENT_NATIVE"])
        images = []
        for directory in (current, parent):
            resources = (directory / "control.resources.txt").read_text()
            isa = (directory / "control.isa.txt").read_text()
            info = inspect_native(CONTRACT["configs"][0], resources, isa)
            images.append({dtype: [i["bytes"] for i in instructions(isa, v["symbol"])]
                           for dtype, v in info.items()})
        self.assertEqual(images[0], images[1])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
