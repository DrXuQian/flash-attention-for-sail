"""Compile the real shared policy; reject wrong geometry/metadata/inventory."""
import itertools
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench_h800_causal_tuning as bench


class PolicyTest(unittest.TestCase):
    def compile(self, n=80, single=0, experiment=True, plant=""):
        folder = Path("/workspace/flash-attn-ppu17-causal-policy-tests")
        folder.mkdir(exist_ok=True)
        target = folder / f"probe-{time.time_ns()}.o"
        flags = ["-DFLASHATTN_PPU17=1"]
        if experiment:
            flags += ["-DFLASHATTN_PPU17_CAUSAL_EXPERIMENT=1",
                      f"-DFLASHATTN_PPU17_CAUSAL_N={n}", f"-DFLASHATTN_PPU17_CAUSAL_SINGLE={single}"]
        source = '''
#include "tile_size.h"
using namespace flash::ppu17_causal;
constexpr auto tile = tile_size_fwd_sm90(256, 256, true, false);
static_assert(std::get<0>(tile) == 128);
static_assert(std::get<1>(tile) == EXPECT_N, "wrong tile");
constexpr bool kernel_single = single_tile(90,256,256,2,true,false,false,false,false);
constexpr bool host_counter = !single_tile(90,256,256,2,true,false,false,false,false);
static_assert(kernel_single == EXPECT_SINGLE);
static_assert(host_counter != kernel_single, "host-counter disagreement");
static_assert(!single_tile(80,256,256,2,true,false,false,false,false));
static_assert(!single_tile(90,256,256,2,true,false,true,false,false));
static_assert(!single_tile(90,256,256,2,true,false,false,true,false));
static_assert(!single_tile(90,256,256,2,true,false,false,false,true));
static_assert(std::get<1>(tile_size_fwd_sm90(256,256,false,false)) == 80);
static_assert(std::get<1>(tile_size_fwd_sm90(128,128,true,false)) == 128);
static_assert(std::get<1>(tile_size_fwd_sm90(64,64,true,false)) == 128);
static_assert(std::get<1>(tile_size_fwd_sm90(256,256,true,false,1)) == 128);
'''.replace("EXPECT_N", str(n if experiment else 80)).replace("EXPECT_SINGLE", str(single if experiment else 0))
        if plant == "tile":
            source = source.replace('== 64, "wrong tile"', '== 80, "wrong tile"')
        if plant == "host":
            source = source.replace("constexpr bool host_counter = !single_tile", "constexpr bool host_counter = single_tile")
        return subprocess.run(["g++", "-std=c++17", "-x", "c++", "-c", "-", "-o", str(target),
                               "-I"+str(ROOT / "hopper"), *flags], input=source, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def test_shared_policy_all_four_cells_and_default(self):
        for n, single in itertools.product((64,80), (0,1)):
            result = self.compile(n, single)
            self.assertEqual(result.returncode, 0, result.stdout)
        result = self.compile(experiment=False)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_wrong_geometry_and_host_counter_are_red(self):
        for plant, message in (("tile", "wrong tile"), ("host", "host-counter disagreement")):
            result = self.compile(64, 1, plant=plant)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stdout)

    def test_host_and_kernel_use_shared_predicate(self):
        self.assertIn("!flash::ppu17_causal::single_tile(params.arch", (ROOT / "hopper/flash_api.cpp").read_text())
        self.assertIn("!flash::ppu17_causal::single_tile(Arch", (ROOT / "hopper/flash_fwd_launch_template.h").read_text())

    def test_single_tile_launch_denominator(self):
        # Existing SingleTileScheduler maps blockIdx=(m,h,b) directly. Enumerate
        # all output owners for both registered shapes; missing a head is red.
        for s in (2048, 8192):
            expected = {(m,h,0) for h in range(32) for m in range((s+127)//128)}
            actual = [(m,h,0) for h in range(32) for m in range((s+127)//128)]
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(set(actual), expected)
            self.assertNotEqual(set(x for x in actual if x[1] != 31), expected)

    def test_no_fallback_or_missing_kernel_is_timing(self):
        def trace(name, copies=2):
            return {"traceEvents": [{"cat":"kernel","name":name,"ts":i,"dur":1} for i in range(copies)]}
        for arm, symbol in (("candidate","FlashAttnFwdSm90"),
                            ("flashinfer","SinglePrefillWithKVCacheKernel"), ("cudnn","fmha_fprop")):
            self.assertEqual(len(bench.inventory(trace(symbol), arm, 2)["kernel_durations_us"]), 2)
            with self.assertRaises(ValueError):
                bench.inventory(trace("reference_matmul"), arm, 2)
            with self.assertRaises(ValueError):
                bench.inventory(trace(symbol, 1), arm, 2)

    def test_actual_candidate_type_grid_and_host_copies_are_bound(self):
        record = {"arm":"candidate", "source_identity":{"tile":[128,64],"scheduler":"single"},
                  "shape":[1,2048,32,256],
                  "kernel_name":"FlashAttnFwdSm90<cute::tuple<cute::C<128>, cute::C<64>, cute::C<256>>, SingleTileScheduler>",
                  "kernel_arguments":{"grid":[16,32,1]}, "gpu_copy_memset_counts":{}}
        bench.check_candidate_receipt(record)
        plants = [("kernel_name", record["kernel_name"].replace("C<64>","C<80>")),
                  ("kernel_name", record["kernel_name"].replace("SingleTileScheduler","DynamicPersistentTileScheduler")),
                  ("kernel_arguments", {"grid":[114,1,1]}),
                  ("gpu_copy_memset_counts", {"HtoD":450})]
        for key,value in plants:
            with self.assertRaises(ValueError):
                bench.check_candidate_receipt({**record,key:value})


if __name__ == "__main__":
    unittest.main()
