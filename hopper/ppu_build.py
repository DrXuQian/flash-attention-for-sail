"""PPU host/device split build for the FlashAttention-3 extension.

PyTorch's CUDA extension helper assumes that one CUDA facade owns both the
host and device translation units.  That assumption is false for PPU: host
code must see PyTorch's CUDA-compatible facade, while device code must see
HGCC's native headers.  Mixing both header families in one nvcc invocation
redefines half/bfloat16 types and is therefore rejected rather than patched
over with include-order tricks.
"""

import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from setuptools.command.build_ext import build_ext


class HGCCBuildExtension(build_ext):
    """Compile .cu with HGCC, .cpp with the host compiler, then HGCC-link."""

    source_root = Path(__file__).resolve().parent

    @staticmethod
    def _require(path, what):
        path = Path(path).resolve()
        if not path.exists():
            raise RuntimeError(f"missing {what}: {path}")
        return path

    @staticmethod
    def _ninja_command(arguments):
        # Ninja's $in is a list of shell words and must not be quoted as one.
        return " ".join(
            str(argument) if argument in ("$in", "$out") else shlex.quote(str(argument))
            for argument in arguments
        )

    def build_extensions(self):
        if "PPU_SDK" not in os.environ:
            raise RuntimeError("HGCCBuildExtension requires PPU_SDK")
        for extension in self.extensions:
            self._build_extension_hgcc(extension)

    def _build_extension_hgcc(self, extension):
        ppu_sdk = self._require(os.environ["PPU_SDK"], "PPU SDK")
        hgcc = self._require(ppu_sdk / "bin" / "hgcc", "HGCC compiler")
        ppu_include = self._require(ppu_sdk / "include", "PPU include directory")
        ppu_target_include = self._require(
            ppu_sdk / "targets" / "x86_64-linux" / "include",
            "PPU target include directory",
        )
        ppu_lib = self._require(ppu_sdk / "lib", "PPU runtime library directory")
        ppu_cuda_include = self._require(
            ppu_sdk / "CUDA_SDK" / "include", "PPU CUDA facade include directory"
        )
        ppu_cuda_lib = self._require(
            ppu_sdk / "CUDA_SDK" / "lib64", "PPU CUDA facade library directory"
        )

        cxx = shutil.which(os.environ.get("CXX", "c++"))
        ninja = shutil.which("ninja")
        if cxx is None or ninja is None:
            raise RuntimeError(f"build tools missing: CXX={cxx}, ninja={ninja}")

        torch_root = Path(torch.__path__[0]).resolve()
        torch_include = self._require(torch_root / "include", "PyTorch include directory")
        torch_api_include = self._require(
            torch_include / "torch" / "csrc" / "api" / "include",
            "PyTorch C++ API include directory",
        )
        torch_lib = self._require(torch_root / "lib", "PyTorch library directory")
        python_include = self._require(sysconfig.get_path("include"), "Python include directory")

        sources = [(self.source_root / source).resolve() for source in extension.sources]
        for source in sources:
            self._require(source, "extension source")

        build_temp = Path(self.build_temp).resolve() / "hgcc_objs"
        build_temp.mkdir(parents=True, exist_ok=True)
        extension_path = Path(self.get_ext_fullpath(extension.name)).resolve()
        extension_path.parent.mkdir(parents=True, exist_ok=True)

        common_includes = [Path(path).resolve() for path in extension.include_dirs]
        # Device TUs intentionally exclude all PyTorch/CUDA-facade headers.
        device_includes = common_includes + [ppu_include, ppu_target_include]
        # Host TUs use PyTorch's ABI and its PPU CUDA-compatible facade.
        host_includes = common_includes + [
            ppu_include,
            ppu_target_include,
            torch_include,
            torch_api_include,
            ppu_cuda_include,
            python_include,
        ]

        extra = extension.extra_compile_args if isinstance(extension.extra_compile_args, dict) else {}
        hgcc_flags = list(extra.get("hgcc", []))
        cxx_flags = list(extra.get("cxx", []))
        link_flags = list(getattr(extension, "extra_link_args", []) or [])

        pybind_flags = []
        for macro, attribute in (
            ("PYBIND11_COMPILER_TYPE", "_PYBIND11_COMPILER_TYPE"),
            ("PYBIND11_STDLIB", "_PYBIND11_STDLIB"),
            ("PYBIND11_BUILD_ABI", "_PYBIND11_BUILD_ABI"),
        ):
            value = getattr(torch._C, attribute, None)
            if value:
                pybind_flags.append(f'-D{macro}="{value}"')
        abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"

        hgcc_command = [
            "env", "-u", "LD_LIBRARY_PATH", str(hgcc),
            *hgcc_flags,
            abi_flag,
            *(f"-I{path}" for path in device_includes),
            "-c", "$in", "-o", "$out",
        ]
        cxx_command = [
            cxx,
            *cxx_flags,
            abi_flag,
            *pybind_flags,
            *(f"-I{path}" for path in host_includes),
            "-c", "$in", "-o", "$out",
        ]
        link_command = [
            "env", "-u", "LD_LIBRARY_PATH", str(hgcc),
            "-shared", "-o", "$out", "$in",
            f"-L{torch_lib}", f"-L{ppu_lib}", f"-L{ppu_cuda_lib}",
            "-ltorch_python", "-ltorch_cuda", "-ltorch_cpu", "-ltorch",
            "-lc10_cuda", "-lc10",
            "-lhggc_wrapper", "-lhggcrt1", "-lhggc", "-lhg_wrapper",
            "-lcudart", "-ldl",
            *link_flags,
        ]

        objects = []
        ninja_file = build_temp / "build.ninja"
        with ninja_file.open("w", encoding="utf-8") as handle:
            handle.write("ninja_required_version = 1.3\n\n")
            handle.write("rule hgcc_compile\n")
            handle.write(f"  command = {shlex.join(hgcc_command)}\n")
            handle.write("  description = HGCC $in\n\n")
            handle.write("rule cxx_compile\n")
            handle.write(f"  command = {shlex.join(cxx_command)}\n")
            handle.write("  description = CXX $in\n\n")
            handle.write("rule link\n")
            handle.write(f"  command = {self._ninja_command(link_command)}\n")
            handle.write("  description = LINK $out\n\n")

            for source in sources:
                relative = source.relative_to(self.source_root)
                object_path = build_temp / ("__".join(relative.with_suffix("").parts) + ".o")
                objects.append(object_path)
                rule = "hgcc_compile" if source.suffix == ".cu" else "cxx_compile"
                handle.write(f"build {object_path}: {rule} {source}\n")

            handle.write(f"\nbuild {extension_path}: link {' '.join(map(str, objects))}\n")
            handle.write(f"\ndefault {extension_path}\n")

        jobs = os.environ.get("MAX_JOBS", str(max(1, min(os.cpu_count() or 1, 16))))
        print(
            f"[FA3 HGCCBuildExtension] sources={len(sources)} jobs={jobs} "
            f"architectures=ppu_10,ppu_15 output={extension_path}"
        )
        subprocess.check_call([ninja, "-f", ninja_file.name, f"-j{jobs}"], cwd=build_temp)
        print(f"[FA3 HGCCBuildExtension] built {extension_path}")
