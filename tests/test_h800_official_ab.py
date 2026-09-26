"""Host-only admission/inventory negatives for the official same-input A/B."""
from pathlib import Path
import sys
import unittest

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
