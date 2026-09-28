"""Host-only postcondition negatives, optionally against real compiled bodies."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_ppu17_softmax_overlap as gate


def sass_fixture(before=90):
    # One known algorithmic denominator: 8 QK prologue, 8 QK+11 PV steady,
    # 11 PV drain; only the steady window placement changes.
    inst = []
    def qk():
        inst.extend(["HGMMA.64x176x16.F32 R24, gdesc[UR8], R24 ;"] * 8)
    def pv():
        inst.extend(f"HGMMA.64x128x16.F32 R112, R{176 + 4 * i}, gdesc[UR8].tnspB, R112 ;"
                    for i in range(11))
    qk(); inst.append("WARPGROUP.DEPBAR.LE gsb0, 0x0 ;")
    qk(); pv(); inst.append("WARPGROUP.DEPBAR.LE gsb0, 0x1 ;")
    inst.extend(["MUFU.EX2 R24, R24 ;"] * before)
    inst.append("WARPGROUP.DEPBAR.LE gsb0, 0x0 ;")
    inst.extend(["MUFU.EX2 R24, R24 ;"] * (90 - before))
    pv(); inst.append("WARPGROUP.DEPBAR.LE gsb0, 0x0 ;")
    return "Function : test_StaticPersistentTileScheduler\n" + "\n".join(
        f"/*{16 * i:04x}*/ {op} /* 0x0000000000000000 */" for i, op in enumerate(inst))


class SoftmaxOverlapGate(unittest.TestCase):
    def test_whole_window_positive(self):
        r = gate.inspect_sass(sass_fixture())
        self.assertEqual(r["exp2_before_wait0"], 90)
        self.assertEqual(r["old_P_words"], 44)
        self.assertEqual(r["live_O_words"], 64)

    def test_old_schedule_is_not_an_improvement(self):
        old = gate.inspect_sass(sass_fixture(19))
        with self.assertRaisesRegex(ValueError, "old schedule"):
            gate.require(old["exp2_before_wait0"] == 90, "old schedule retained")

    def test_missing_wait_red(self):
        text = sass_fixture().replace("WARPGROUP.DEPBAR.LE gsb0, 0x0 ;", "NOP ;", 1)
        with self.assertRaisesRegex(ValueError, "waits"):
            gate.inspect_sass(text)

    def test_one_exponent_omitted_red(self):
        text = sass_fixture().replace("MUFU.EX2 R24, R24 ;", "NOP ;", 1)
        with self.assertRaisesRegex(ValueError, "denominator"):
            gate.inspect_sass(text)

    def test_early_old_P_and_O_overwrite_red(self):
        for dest in (176, 219, 112, 175):
            text = sass_fixture().replace("MUFU.EX2 R24, R24 ;", f"MUFU.EX2 R{dest}, R24 ;", 1)
            with self.subTest(dest=dest), self.assertRaisesRegex(ValueError, "early old-P/O"):
                gate.inspect_sass(text)

    def test_missing_matrix_group_red(self):
        text = sass_fixture().replace("HGMMA.64x128x16.F32 R112, R176, gdesc[UR8].tnspB, R112 ;", "NOP ;", 1)
        with self.assertRaisesRegex(ValueError, "matrix denominator"):
            gate.inspect_sass(text)

    def test_wrong_or_duplicate_symbol_red(self):
        for text in (sass_fixture().replace("StaticPersistent", "DynamicPersistent"),
                     sass_fixture() + "\n" + sass_fixture()):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                gate.inspect_sass(text)

    def test_builder_is_opt_in_and_records_variant(self):
        source = (ROOT / "tools/build_ppu17_standalone.py").read_text()
        self.assertIn('"--softmax-overlap", action="store_true"', source)
        self.assertIn('if args.softmax_overlap:', source)
        self.assertIn('"softmax_overlap":', source)
        app = (ROOT / "dev/ppu17/standalone_fwd.cu").read_text()
        self.assertIn('kOverlapVariant', app)
        self.assertIn('completed_invocations=1', app)

    def test_private_slot_byte_accounting(self):
        rows = [{"inst": "vmem.ld.b32 vreg3, [0x805 + (vreg2 + %tid) * 0x4] @sreg[108:109]", "executed": 7},
                {"inst": "vmem.st.b32 vreg9, [0x805 + (vreg2 + %tid) * 0x4] @sreg[108:109]", "executed": 3},
                {"inst": "vmem.st.b32 vreg3, [0x0] @vreg[4:5]", "executed": 4}]
        memory = {"vmem_inst_read_bytes": 896, "vmem_inst_write_bytes": 512}
        result = gate.inspect_private_traffic(rows, memory)
        self.assertEqual((result["read_bytes"], result["write_bytes"], result["slot_count"]), (896, 384, 1))
        bad = copy.deepcopy(rows)
        bad[0]["executed"] -= 1
        with self.assertRaisesRegex(ValueError, "read-byte denominator"):
            gate.inspect_private_traffic(bad, memory)

    def test_unknown_private_read_must_not_be_hidden(self):
        rows = [{"inst": "vmem.ld.b32 vreg3, [0x0] @vreg[4:5]", "executed": 1}]
        with self.assertRaisesRegex(ValueError, "unclassified scalar read"):
            gate.inspect_private_traffic(rows, {"vmem_inst_read_bytes": 128, "vmem_inst_write_bytes": 0})

    @unittest.skipUnless(os.environ.get("FA17_OVERLAP_CANDIDATE_REPORT"), "PPU C03 report not supplied")
    def test_real_candidate_overlap_is_not_no_spill(self):
        import subprocess
        report_path = Path(os.environ["FA17_OVERLAP_CANDIDATE_REPORT"])
        raw = json.loads(report_path.read_bytes())
        result = gate.inspect_report(raw)
        self.assertEqual(result["exp2_before_wait0"], 90)
        self.assertEqual(result["private_traffic"]["slot_count"], 23)
        self.assertEqual(result["private_traffic"]["read_bytes"], 35323904)
        self.assertEqual(result["private_traffic"]["write_bytes"], 18964480)
        rejected = subprocess.run([sys.executable, str(ROOT / "tools/check_ppu17_softmax_overlap.py"),
                                   "report", str(report_path), "--require-before", "90",
                                   "--max-private-bytes", "0"], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("private traffic exceeds control budget", rejected.stderr)
        # Keep aggregate counts internally consistent: removing one executed
        # private read must still fail the independent memory-byte denominator.
        bad = copy.deepcopy(raw)
        inst = bad["ppu"][0]["instruction_statistics"]
        row = next(r for r in inst["source_view_data"]["data"] if r["inst"].startswith("vmem.ld.b32"))
        row["executed"] -= 1
        inst["executed_instructions"] -= 1
        next(pair for pair in inst["inst_histogram_data"] if pair[0] == "vmem.ld.b32")[1] -= 1
        with self.assertRaisesRegex(ValueError, "read-byte denominator"):
            gate.inspect_report(bad)

    @unittest.skipUnless(os.environ.get("FA17_OVERLAP_SASS"), "compiled candidate not supplied")
    def test_real_candidate_and_old_P_negative(self):
        path = Path(os.environ["FA17_OVERLAP_SASS"])
        text = path.read_text()
        result = gate.inspect_sass(text)
        self.assertEqual(result["exp2_before_wait0"], 90)
        import re
        body = gate.sass_body(text)
        first = next(op for _, op in body if "HGMMA.64x128x16" in op)
        p_reg = int(re.search(r", R(\d+), gdesc", first)[1])
        wait_pc = result["wait_pcs"][1]
        start = text.index(f"/*{int(wait_pc, 16):04x}*/", text.index("StaticPersistentTileScheduler"))
        prefix, tail = text[:start], text[start:]
        planted = re.sub(r"MUFU\.EX2 R\d+,", f"MUFU.EX2 R{p_reg},", tail, count=1)
        with self.assertRaisesRegex(ValueError, "early old-P/O"):
            gate.inspect_sass(prefix + planted)

    @unittest.skipUnless(os.environ.get("FA17_OVERLAP_REPORT"), "PPU baseline report not supplied")
    def test_real_report_and_coverage_negative(self):
        report = json.loads(Path(os.environ["FA17_OVERLAP_REPORT"]).read_bytes())
        base = gate.inspect_report(report)
        self.assertEqual(base["exp2_before_wait0"], 0)
        bad = copy.deepcopy(report)
        rows = bad["ppu"][0]["instruction_statistics"]["source_view_data"]["data"]
        row = next(r for r in rows if r["inst"].startswith("v.exp2.f32") and r["executed"] == 17920)
        row["executed"] -= 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            gate.inspect_report(bad)


if __name__ == "__main__":
    unittest.main()
