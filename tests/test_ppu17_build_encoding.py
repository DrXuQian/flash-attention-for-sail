"""Non-UTF8 vendor inputs must not conceal a real compiler/contract failure."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hopper"))
sys.path.insert(0, str(ROOT / "tools"))
import ppu17_build as build
from check_ppu17_source import inspect_ptx


class EncodingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ.get("FA17_ENCODING_TEST_OUT", "/workspace/fa17-build-encoding-tests")) / str(os.getpid())
        cls.out.mkdir(parents=True, exist_ok=False)

    def test_old_decode_failure_reproduced_at_byte_22_and_version_now_parses(self):
        root = self.out / "nonutf-header"
        header = root / "include/cutlass/version.h"
        header.parent.mkdir(parents=True)
        raw = b"//" + b" " * 20 + b"\x82\n#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 3\n#define CUTLASS_PATCH 0\n"
        header.write_bytes(raw)
        with self.assertRaises(UnicodeDecodeError) as old:
            header.read_text(encoding="utf-8")
        self.assertEqual(old.exception.start, 22)
        self.assertEqual(build.cutlass_version(root), (4, 3, 0))
        self.assertEqual(header.read_bytes(), raw)

    def test_nonutf_version_token_is_rejected_not_repaired(self):
        root = self.out / "damaged-macro"
        header = root / "include/cutlass/version.h"
        header.parent.mkdir(parents=True)
        header.write_bytes(b"#define CUTLASS_MAJOR 4\x82\n#define CUTLASS_MINOR 3\n#define CUTLASS_PATCH 0\n")
        with self.assertRaisesRegex(ValueError, "malformed CUTLASS version header"):
            build.cutlass_version(root)

    def test_whitespace_crlf_and_nonutf_trailing_comments(self):
        root = self.out / "commented-header"
        header = root / "include/cutlass/version.h"
        header.parent.mkdir(parents=True)
        header.write_bytes(b" \t# define CUTLASS_MAJOR 4 // \x82\r\n#define CUTLASS_MINOR 3\r\n#define CUTLASS_PATCH 0 /*\x82*/\r\n")
        self.assertEqual(build.cutlass_version(root), (4, 3, 0))

    def test_nonutf_compiler_failure_keeps_exit_code_message_and_raw_bytes(self):
        log = self.out / "compiler-error.log"
        command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'error: missing symbol \\x82\\n'); sys.exit(17)"]
        with self.assertRaises(RuntimeError) as got:
            build.run_logged(command, log)
        self.assertIn("rc=17", str(got.exception))
        self.assertIn("compiler-error.log", str(got.exception))
        self.assertIn("missing symbol \\x82", str(got.exception))
        self.assertEqual(log.read_bytes(), b"error: missing symbol \x82\n")

    def test_nonutf_warning_can_succeed_without_losing_bytes(self):
        log = self.out / "warning.log"
        command = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'warning: \\x82\\n')"]
        self.assertEqual(build.run_logged(command, log), b"warning: \x82\n")
        self.assertEqual(log.read_bytes(), b"warning: \x82\n")

    def test_zero_exit_without_compiled_object_is_still_rejected(self):
        # A real /usr/bin/true process accepts arguments and returns zero, but
        # does not compile. Decoding hardening must not turn it into admission.
        with self.assertRaisesRegex(RuntimeError, "without a target object"):
            build.check_compiler("/usr/bin/true", self.out / "empty-probe", source_check=True)

    def test_ptx_comment_bytes_do_not_remove_live_body_checks(self):
        body = b"{ wgmma.mma_async.sync.aligned.test; cp.async.bulk.tensor.4d.shared::cluster.global; cp.async.bulk.tensor.4d.global.shared::cta; ret; }"
        raw = b"// vendor \\path \x82\n.entry causal() " + body + b"\n.entry noncausal() " + body
        self.assertEqual(inspect_ptx(build.diagnostic_text(raw))["entries"], 2)
        damaged = raw.replace(b"wgmma.mma_async", b"wg\x82mma.mma_async", 1)
        with self.assertRaisesRegex(ValueError, "missing wgmma"):
            inspect_ptx(build.diagnostic_text(damaged))

    def test_utf8_text_is_preserved_and_unknown_bytes_are_not_guessed(self):
        text = "编译器 warning ✓"
        self.assertEqual(build.diagnostic_text(text.encode("utf-8")), text)
        self.assertEqual(build.diagnostic_text(b"a\x82b"), "a\\x82b")


if __name__ == "__main__":
    unittest.main()
