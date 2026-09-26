"""Host-only timing/denominator negatives; no GPU/Torch import required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_ppu17_hopper_control as bench


class HardwareTimingTests(unittest.TestCase):
    def test_useful_causal_flops_and_dense_peak(self):
        self.assertEqual(bench.logical_flops(1, 2048, 32, 256, True), 68753031168)
        self.assertEqual(bench.PEAK_BF16_DENSE_TFLOPS, 756.5)
        self.assertAlmostEqual(bench.summarize([100], 68753031168)["target_70_latency_us"], 129.8329358285)

    def test_all_three_goal_verdicts_are_fixed(self):
        for samples, verdict in (([100, 110], "AT_LEAST_70"), ([140, 150], "BELOW_70"),
                                 ([120, 140], "UNRESOLVED_AT_70")):
            self.assertEqual(bench.summarize(samples, 68753031168)["verdict"], verdict)

    def test_missing_or_extra_gpu_kernel_is_red(self):
        kernel = {"cat": "kernel", "name": "FlashAttnFwdSm90", "ts": 1, "dur": 100}
        bench.kernel_samples({"traceEvents": [kernel]}, 1)
        for kernels in ([], [kernel, kernel]):
            with self.assertRaisesRegex(ValueError, "inventory"):
                bench.kernel_samples({"traceEvents": kernels}, 1)

    def test_gpu_reference_cannot_replace_attention_in_denominator(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            bench.kernel_samples({"traceEvents": [{"cat": "kernel", "name": "reference_gemm"}]}, 1)


if __name__ == "__main__":
    unittest.main()
