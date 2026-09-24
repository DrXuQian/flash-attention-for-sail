"""Host-only contracts. Run directly; importing FA/Torch is not required."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hopper"))
import ppu17_build as build


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_script("check_ppu17_source")
runner = load_script("run_ppu17_forward")


def ptx_fixture():
    body = "{ wgmma.mma_async.sync.aligned.test; cp.async.bulk.tensor.4d.shared::cluster.global; cp.async.bulk.tensor.4d.global.shared::cta; ret; }"
    return ".entry causal() " + body + "\n.entry noncausal() " + body


class Ppu17Contracts(unittest.TestCase):
    def test_all_six_units_are_real_sm90_instantiations(self):
        files = build.source_files()
        self.assertEqual(len(files), 6)
        self.assertEqual(len(set(files)), 6)
        for path in files:
            source = path.read_text()
            self.assertIn("run_mha_fwd_<90,", source)
            self.assertIn("cudaStream_t stream", source)

    def test_removed_unit_is_red_not_a_smaller_denominator(self):
        original = Path.is_file
        def planted(path):
            return False if path.name == "flash_fwd_hdim256_bf16_sm90.cu" else original(path)
        with patch.object(Path, "is_file", planted):
            with self.assertRaisesRegex(ValueError, "missing.*generated"):
                build.source_files()

    def test_flags_do_not_enable_legacy_or_disable_hopper(self):
        flags = build.compile_flags()
        self.assertIn("-gencode=arch=compute_90a,code=sm_90a", flags)
        self.assertIn("-DACOMPUTE_VERSION=10700", flags)
        self.assertFalse(any("USE_PPU" in x or "USE_AIU" in x or "DISABLE_SM90" in x for x in flags))
        self.assertIn("-DFLASHATTENTION_DISABLE_BACKWARD", flags)

    def test_each_unsupported_feature_request_fails(self):
        for feature in build.DISABLED:
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    build.check_environment({f"FLASH_ATTENTION_DISABLE_{feature}": "FALSE"})

    def test_required_dtype_dimension_and_arch_cannot_be_removed(self):
        for feature in ("SM90", "FP16", "HDIM64", "HDIM128", "HDIM256"):
            with self.assertRaises(ValueError):
                build.check_environment({f"FLASH_ATTENTION_DISABLE_{feature}": "TRUE"})

    def test_default_scope_and_simulation_are_explicit(self):
        build.check_environment({})
        build.check_environment({"FLASH_ATTENTION_PPU17_COMPILE_MODE": "simulation"})
        with self.assertRaises(ValueError):
            build.check_environment({"FLASH_ATTENTION_PPU17_COMPILE_MODE": "auto"})

    def test_wrong_dependency_no_fallback(self):
        with self.assertRaises(ValueError):
            build.cutlass_root(None)
        with self.assertRaises(ValueError):
            build.cutlass_root(ROOT / "hopper")

    def test_causal_workload_denominator(self):
        self.assertEqual(runner.logical_flops(1, 2048, 32, 256, True), 68753031168)
        self.assertEqual(runner.logical_flops(1, 2048, 32, 256, False), 137438953472)

    def test_two_live_bodies_pass(self):
        self.assertEqual(check.inspect_ptx(ptx_fixture())["entries"], 2)

    def test_one_empty_body_is_red_even_with_other_body_valid(self):
        planted = ptx_fixture().replace("wgmma.mma_async.sync.aligned.test;", "", 1)
        with self.assertRaisesRegex(ValueError, "missing wgmma"):
            check.inspect_ptx(planted)

    def test_tma_writeback_removed_is_red(self):
        planted = ptx_fixture().replace("cp.async.bulk.tensor.4d.global.shared::cta;", "", 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check.inspect_ptx(planted)

    def test_legacy_barrier_is_red(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            check.inspect_ptx(ptx_fixture() + "\nppu.bar.arrive 0, 128;\n")

    def test_one_missing_entry_is_red(self):
        with self.assertRaisesRegex(ValueError, "two forward"):
            check.inspect_ptx(ptx_fixture().split("\n")[0])

    def test_no_forward_definition_is_invented(self):
        # Public generated units stay upstream-owned; this port selects them.
        result = subprocess.run(["git", "diff", "--name-only", "--", "hopper/instantiations"],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "")

    def test_stale_extension_identity_fails_before_invocation(self):
        runner.validate_extension(SimpleNamespace(ppu17_backend="cutlass36-sm90-forward-v1"))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            runner.validate_extension(SimpleNamespace())

    def test_forward_is_invoked_once_with_single_kernel_features(self):
        interface = Mock()
        interface._flash_attn_forward.return_value = ("o", "lse", None, None)
        self.assertEqual(runner.invoke_once(interface, "q", "k", "v", causal=True), ("o", "lse", None, None))
        interface._flash_attn_forward.assert_called_once_with(
            "q", "k", "v", causal=True, num_splits=1, pack_gqa=False)

    def test_forward_failure_does_not_trigger_fallback_or_retry(self):
        interface = Mock()
        interface._flash_attn_forward.side_effect = RuntimeError("planted failure")
        with self.assertRaisesRegex(RuntimeError, "planted failure"):
            runner.invoke_once(interface, "q", "k", "v", causal=False)
        self.assertEqual(interface._flash_attn_forward.call_count, 1)

    def test_missing_generated_definition_is_not_skipped(self):
        check.check_undefined_symbols(" U cudaLaunchKernel\n U torch::empty()\n")
        with self.assertRaisesRegex(ValueError, "definition missing"):
            check.check_undefined_symbols(" U void run_mha_fwd_<90, half, 256>(params&, stream*)\n")


if __name__ == "__main__":
    unittest.main()
