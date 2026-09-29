"""Torch-free standalone contracts; optional built-ELF checks never call CUDA."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_ppu17_standalone as standalone


class StandaloneContracts(unittest.TestCase):
    def test_shipping_generated_unit_is_not_a_copy(self):
        self.assertEqual(standalone.SHIPPING.relative_to(ROOT).as_posix(),
                         "hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu")
        self.assertIn("run_mha_fwd_<90, cutlass::half_t, 128, 128, false, false, false, false>",
                      standalone.SHIPPING.read_text())

    def test_one_call_and_no_auxiliary_gpu_body(self):
        source = standalone.APPLICATION.read_text()
        code = re.sub(r"//[^\n]*", "", source)
        self.assertEqual(len(re.findall(r"run_mha_fwd_<", code)), 1)
        for forbidden in ("__global__", "cudaMemset", "cudaEvent", "<<<", "torch::"):
            self.assertNotIn(forbidden, code)
        self.assertIn("dout.upload(o)", code)
        self.assertIn("dlse.upload(lse)", code)

    def test_compiler_flags_have_one_authority(self):
        source = (ROOT / "tools/build_ppu17_standalone.py").read_text()
        self.assertIn("build.compile_flags(backend)", source)
        self.assertIn("build.check_compiler(nvcc", source)
        self.assertNotIn("-gencode=", source)
        self.assertIn("source/backend changed", source)

    def test_dependency_negative(self):
        standalone.inspect_dependencies("NEEDED libcudart.so.12\nNEEDED libcuda.so.1")
        for library in ("libtorch_cuda.so", "libc10.so", "libpython3.12.so", "libhggcrt.so"):
            with self.subTest(library=library), self.assertRaises(ValueError):
                standalone.inspect_dependencies("NEEDED " + library)

    def test_actual_cpu_oracle_and_negatives(self):
        out = Path(os.environ.get("FA17_HOST_TEST_OUT", "/workspace/fa17-standalone-host-tests"))
        out.mkdir(parents=True, exist_ok=True)
        binary = out / f"reference-{os.getpid()}"
        subprocess.run(["g++", "-std=c++17", "-O3", "-fopenmp",
                        str(ROOT / "dev/ppu17/standalone_host_test.cpp"), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], text=True, capture_output=True, check=True,
                                env={**os.environ, "OMP_NUM_THREADS": "2"})
        self.assertIn("last-output/NaN/extent negatives", result.stdout)

    @unittest.skipUnless(os.environ.get("FA17_STANDALONE_EXE"), "built ELF not supplied; host binary checks unavailable")
    def test_actual_executable_host_paths(self):
        exe = os.environ["FA17_STANDALONE_EXE"]
        record = json.loads(subprocess.check_output([exe, "--describe"], text=True))
        expected = dict(batch=1, seqlen_q=1024, seqlen_k=1024, heads=56, kv_heads=56,
                        dim=128, dtype="fp16", causal=False, layout="BSHD", tile_m=128, tile_n=176,
                        row_stride_elements=7168, head_stride_elements=128,
                        batch_stride_elements=7340032, num_splits=1, logical_flops=30064771072)
        build_record = json.loads((Path(exe).parent / "build.json").read_text())
        expected["tile_n"] = build_record.get("kv_tile", 176)
        if build_record.get("q_tail_mode") == "m64":
            expected["tile_m"] = 64
        self.assertIn(expected["tile_n"], (128, 176))
        self.assertEqual({key: record[key] for key in expected}, expected)
        self.assertIn("device=NOT_RUN", subprocess.check_output([exe, "--host-self-test"], text=True))
        # A repeated-launch switch cannot silently turn simulation into a benchmark.
        bad = subprocess.run([exe, "--iterations", "2"], text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unknown/incomplete option", bad.stderr)

    @unittest.skipUnless(os.environ.get("FA17_STANDALONE_EXE"), "built ELF not supplied; real link negative unavailable")
    def test_actual_missing_generated_unit_is_a_link_error(self):
        out = Path(os.environ["FA17_STANDALONE_EXE"]).resolve().parent
        build = json.loads((out / "build.json").read_text())
        links = [command for command in build["commands"] if "--cudart=shared" in command]
        self.assertEqual(len(links), 1)
        command = [item for item in links[0] if item != str(out / "shipping_fp16_d128.o")]
        self.assertEqual(len(command), len(links[0]) - 1)
        command[-1] = str(out / "negative-missing-generated")
        result = subprocess.run(command, capture_output=True,
                                env={**os.environ, "TMPDIR": str(out / "compiler-tmp")})
        (out / "negative-missing-generated.log").write_bytes(result.stdout + result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"run_mha_fwd_", result.stderr)
        self.assertIn(b"undefined reference", result.stderr)

    @unittest.skipUnless(os.environ.get("FA17_STANDALONE_EXE"), "built ELF not supplied; build-plan check unavailable")
    def test_ptx_and_disassembly_are_optional_and_not_silently_passed(self):
        out = Path(os.environ["FA17_STANDALONE_EXE"]).resolve().parent
        record = json.loads((out / "build.json").read_text())
        inspected = record["shipping_device_code"]["verdict"] == "PASS"
        self.assertEqual(any("--ptx" in command for command in record["commands"]), inspected)
        self.assertEqual(any("--dump-sass" in command for command in record["commands"]), inspected)
        if inspected:
            self.assertEqual(record["shipping_device_code"]["entries"], 2)
        else:
            self.assertEqual(record["shipping_device_code"]["verdict"], "SKIP")
            self.assertIn("not selected", record["shipping_device_code"]["reason"])


if __name__ == "__main__":
    unittest.main()
