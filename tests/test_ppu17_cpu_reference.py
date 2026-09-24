"""CPU-only oracle validation; no CUDA context or target kernel required."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("runner", Path(__file__).resolve().parents[1] / "tools/run_ppu17_forward.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CpuReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_uniform_scores_causal_prefix_gqa_and_tail(self):
        # Independent closed form: zero Q/K -> uniform (prefix) weights.
        # Each KV head has a different value sequence; GQA maps two Q heads
        # per KV head. Block size 2 makes the final query block incomplete.
        q = torch.zeros(1, 5, 4, 2)
        k = torch.zeros(1, 5, 2, 2)
        v = torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2)
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU reference forbidden")):
            for causal in (True, False):
                out, lse = runner.cpu_reference((q, k, v), causal=causal, query_block=2)
                for row in range(5):
                    count = row + 1 if causal else 5
                    for head in range(4):
                        expected = v[0, :count, head // 2].double().mean(0)
                        torch.testing.assert_close(out[0, row, head], expected)
                        self.assertAlmostEqual(lse[0, head, row].item(), torch.tensor(float(count)).double().log().item())
                self.assertEqual(out.device.type, "cpu")

    def test_nonuniform_reference_matches_literal_scalar_softmax(self):
        import math
        q = torch.tensor([[[[1., 2.]], [[-1., 1.]], [[2., -2.]]]])
        k = torch.tensor([[[[0., 2.]], [[1., 0.]], [[1., -1.]]]])
        v = torch.tensor([[[[1., 3.]], [[-2., 2.]], [[4., -1.]]]])
        for causal in (False, True):
            actual, lse = runner.cpu_reference((q, k, v), causal=causal, query_block=2)
            for i in range(3):
                js = range(i + 1 if causal else 3)
                scores = [sum(float(q[0, i, 0, d]) * float(k[0, j, 0, d]) for d in range(2)) / math.sqrt(2) for j in js]
                denom = sum(math.exp(x) for x in scores)
                for d in range(2):
                    expected = sum(math.exp(x) * float(v[0, j, 0, d]) for j, x in zip(js, scores)) / denom
                    self.assertAlmostEqual(actual[0, i, 0, d].item(), expected, places=12)
                self.assertAlmostEqual(lse[0, 0, i].item(), math.log(denom), places=12)

    def test_non_cpu_input_rejected_before_math(self):
        q = torch.empty((1, 1, 1, 64), device="meta")
        with self.assertRaisesRegex(ValueError, "CPU"):
            runner.cpu_reference((q, q, q), causal=True)


if __name__ == "__main__":
    unittest.main()
