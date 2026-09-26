"""Host-only admission/inventory negatives for the official same-input A/B."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bench_h800_official_fa3 as ab


def event(name, ts=1):
    return {"cat": "kernel", "name": name, "ts": ts, "dur": 100}


class OfficialABTests(unittest.TestCase):
    def test_identity_does_not_alias_control(self):
        ab.check_identity("control", "a", "a", "a")
        ab.check_identity("official", "b", "b", "a")
        for args in (("control", "b", "b", "a"), ("official", "a", "a", "a"),
                     ("official", "b", "c", "a")):
            with self.assertRaises(ValueError):
                ab.check_identity(*args)

    def test_fixture_change_is_red(self):
        ab.check_inputs(["q", "k", "v"], ["q", "k", "v"])
        for actual in (["q", "v", "k"], ["q", "k"], ["q", "k", "other"]):
            with self.assertRaisesRegex(ValueError, "same-input"):
                ab.check_inputs(actual, ["q", "k", "v"])

    def test_larger_sequence_cannot_reuse_wrong_shape_or_input(self):
        reference = {"shape": [1, 4096, 32, 256], "kv_heads": 2, "dtype": "bf16",
                     "causal": True, "input_sha256": ["q", "k", "v"]}
        ab.check_sequence_inputs(4096, ["q", "k", "v"], reference)
        with self.assertRaisesRegex(ValueError, "shape/type"):
            ab.check_sequence_inputs(8192, ["q", "k", "v"], reference)
        with self.assertRaisesRegex(ValueError, "same-input admission"):
            ab.check_sequence_inputs(4096, ["q", "k", "other"], reference)
        for key, value in (("kv_heads", 32), ("dtype", "fp16"), ("causal", False)):
            with self.assertRaisesRegex(ValueError, "shape/type"):
                ab.check_sequence_inputs(4096, ["q", "k", "v"], {**reference, key: value})

    def test_causal_denominator_scales_quadratically_not_linearly(self):
        expected = {2048: 68753031168, 4096: 274945015808,
                    8192: 1099645845504, 16384: 4398314946560}
        for length in ab.SEQUENCE_LENGTHS:
            self.assertEqual(ab.logical_flops(1, length, 32, 256, True), expected[length])

    def test_busy_prelaunch_does_not_become_a_pass(self):
        with patch.object(ab, "require_idle", side_effect=RuntimeError("BUSY")):
            with self.assertRaisesRegex(RuntimeError, "BUSY"):
                ab.wait_before_first_launch(0)

    def test_prelaunch_wait_rechecks_instead_of_ignoring_busy(self):
        with patch.object(ab, "require_idle", side_effect=[RuntimeError("BUSY"), None]) as check:
            with patch.object(ab.time, "sleep"):
                ab.wait_before_first_launch(60)
        self.assertEqual(check.call_count, 2)

    def test_official_counter_zero_is_separate_not_attention(self):
        trace = {"traceEvents": [event("FillFunctor<int>"), event("FlashAttnFwdSm90", 2)]}
        result = ab.parse_trace(trace, 1, "official")
        self.assertEqual(result["attention_durations_us"], [100])
        self.assertEqual(result["auxiliary_kernel_total_us"], 100)
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            ab.parse_trace(trace, 1, "control")

    def test_missing_target_extra_reference_and_wrong_fill_fail(self):
        for names in ([], ["FlashAttnFwdSm90", "FlashAttnFwdSm90"],
                      ["FlashAttnFwdSm90", "reference_gemm"],
                      ["FlashAttnFwdSm90", "FillFunctor<float>"]):
            with self.assertRaises(ValueError):
                ab.parse_trace({"traceEvents": [event(n, i) for i, n in enumerate(names)]},
                               1, "official")


if __name__ == "__main__":
    unittest.main()
