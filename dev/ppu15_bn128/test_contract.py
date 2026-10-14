#!/usr/bin/env python3
"""Host-only contract negatives. These do not certify device numerics."""
from pathlib import Path
import os
import unittest

import torch
from bench import CONTRACT, envelope, oracle, require_numeric, validate
from build import shipping_flags
from native import inspect_native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class ContractTests(unittest.TestCase):
    def test_inventory(self):
        self.assertEqual([(c["bm"], c["bn"], c["warps"]) for c in CONTRACT["configs"]],
                         [(128, 64, 4), (128, 128, 8), (64, 128, 4)])
        for c in CONTRACT["configs"]:
            self.assertEqual(c["shared_bytes"], 2*128*(c["bm"] + 2*c["bn"]))

    def test_shipping_flags(self):
        flags = shipping_flags()
        self.assertIn("-DUSE_AIU=1", flags)
        self.assertIn("--use_fast_math", flags)
        self.assertNotIn("-DUNFUSE_FMA", flags)

    def test_reference_is_full_k_and_correct_layout(self):
        # Uniform logits give exact mean(V); distinct heads/rows expose swaps.
        seq = 65
        q = torch.zeros(1, seq, 56, 128, dtype=torch.float16)
        k = torch.zeros_like(q)
        v = (torch.arange(seq).view(1, seq, 1, 1) +
             torch.arange(56).view(1, 1, 56, 1)).expand_as(q).to(q.dtype)
        rows, result, lse = oracle([q, k, v], seq)
        expected = v.double().mean(1)[0].expand(seq, 56, 128)
        torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(rows, list(range(seq)))
        self.assertEqual(tuple(lse.shape), (56, seq))

    def test_wrong_value_is_red_in_both_precisions(self):
        for dtype in ("fp16", "bf16"):
            want = torch.ones(17, 56, 128, dtype=torch.float64)
            require_numeric(want, want, dtype)
            got = want.clone(); got[-1, -1, -1] += 1
            with self.assertRaisesRegex(AssertionError, "mismatch"):
                require_numeric(got, want, dtype)

    def test_validation_pipeline_without_device(self):
        # Exercise real indexing, dtype conversion, LSE and replay checks with
        # a CPU stand-in, not evidence that any PPU body executed correctly.
        seq = 3
        q = torch.zeros(1, seq, 56, 128, dtype=torch.float16)
        cpu = [q, q.clone(), torch.arange(seq).view(1, seq, 1, 1).expand_as(q).to(q.dtype)]
        _, expected, expected_lse = oracle(cpu, seq)
        class CPUStandIn:
            def __init__(self, idx):
                self.entry = {"id": idx, "name": f"cpu-stand-in-{idx}"}
            def prepare(self, *unused):
                self.o = torch.empty_like(q)
                self.lse = torch.empty(1, 56, seq)
            def launch(self):
                self.o[0].copy_(expected)
                self.lse[0].copy_(expected_lse)
            def release(self):
                pass
        validate([CPUStandIn(0), CPUStandIn(1)], cpu, cpu, "fp16", seq)

    def test_unwritten_nan_is_red(self):
        want = torch.zeros(5, 128, dtype=torch.float64)
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            require_numeric(torch.full_like(want, float("nan")), want, "bf16")

    def test_envelope_does_not_promote_noisy_median(self):
        self.assertEqual(envelope([10, 12, 14], [8, 9, 11]), "UNRESOLVED")
        self.assertEqual(envelope([10, 11, 12], [7, 8, 9]), "CANDIDATE-FASTER")
        self.assertEqual(envelope([10, 11, 12], [13, 14, 15]), "CONTROL-FASTER")

    def test_cpp_config_geometry_matches_contract(self):
        # Native resource checks subsequently bind these source constants to
        # the exact emitted device types, not this source-only assertion.
        source = (HERE / "kernel.cu").read_text()
        self.assertIn("FA15_CONFIG == 2 ? 64 : 128", source)
        self.assertIn("FA15_CONFIG == 0 ? 64 : 128", source)
        self.assertIn("FA15_CONFIG == 1 ? 8 : 4", source)
        self.assertIn("false, false, T>", source)
        self.assertNotIn("compute_attn_1rowblock", source)
        self.assertNotIn("softmax_rescale_o", source)

    def test_native_resource_gate_negatives(self):
        c = CONTRACT["configs"][1]
        resources, isa = "(PPU 1.5)\n", ""
        for i, (type_name, mma_type) in enumerate((("half_t", "f16"), ("bfloat16_t", "bf16"))):
            symbol = f"flash_fwd_kernel_ILi128ELi128ELi128ELi8ELb0ELb0_{type_name}"
            resources += (f"Func {i} {symbol} RESOURCE INFO:\n"
                          "vreg_number:192\nsreg_number:128\nshared_memory_size:0\n"
                          "STACK SIZE:0\nprivate_en:0\n")
            isa += (f"Disassembly of section .text.kernel.{symbol}:\n"
                    f"  0: 00 00 00 00 00 00 00 00 v.mma.f32.{mma_type}.m16n16k16 vreg0\n")
        self.assertEqual(set(inspect_native(c, resources, isa)), {"fp16", "bf16"})
        for corrupted in (resources.replace("STACK SIZE:0", "STACK SIZE:16", 1),
                          resources.replace("(PPU 1.5)", "(PPU 1.0)"),
                          resources.replace("ELi8ELb0", "ELi4ELb0", 1)):
            with self.assertRaises(ValueError):
                inspect_native(c, corrupted, isa)
        with self.assertRaisesRegex(ValueError, "MMA"):
            inspect_native(c, resources, isa.replace("v.mma.f32.f16", "v.mul.f32.f16"))

    @unittest.skipUnless(os.environ.get("FA15_NATIVE_BUILD"),
                         "set FA15_NATIVE_BUILD to check actual compiler dumps, not synthetic samples")
    def test_real_native_artifact_negatives(self):
        build = Path(os.environ["FA15_NATIVE_BUILD"])
        for config in CONTRACT["configs"]:
            resources = (build / f"{config['name']}.resources.txt").read_text()
            isa = (build / f"{config['name']}.isa.txt").read_text()
            self.assertEqual(set(inspect_native(config, resources, isa)), {"fp16", "bf16"})
            with self.assertRaisesRegex(ValueError, "spill"):
                inspect_native(config, resources.replace("STACK SIZE:0", "STACK SIZE:16", 1), isa)
            with self.assertRaisesRegex(ValueError, "PPU1.5"):
                inspect_native(config, resources.replace("(PPU 1.5)", "(PPU 1.0)"), isa)
            with self.assertRaisesRegex(ValueError, "MMA"):
                inspect_native(config, resources, isa.replace("v.mma.", "v.fake."))
            old = f"ILi128ELi{config['bm']}ELi{config['bn']}ELi{config['warps']}"
            with self.assertRaisesRegex(ValueError, "geometry"):
                inspect_native(config, resources.replace(old, "ILi128ELi32ELi32ELi4"), isa)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
