# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Copyright (c) 2023, Tri Dao.

import os
import re
import ast
import shlex
import shutil
import sysconfig
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch

PACKAGE_NAME = "flash_attn"
this_dir = os.path.dirname(os.path.abspath(__file__))

USE_PPU = 'PPU_SDK' in os.environ.keys()
SKIP_KERNEL_BUILD = os.getenv("FLASH_ATTENTION_SKIP_KERNEL_BUILD", "FALSE") == "TRUE"
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"


class HGCCBuildExtension(build_ext):
    """Compile PPU device TUs without admitting NVIDIA CUDA headers.

    A generic PyTorch installation exposes NVIDIA's CUDA facade. Feeding those
    headers to HGCC together with the PPU runtime creates two incompatible
    definitions of CUDA scalar and stream types. Host code still uses the
    PyTorch C++ ABI, while every .cu translation unit is deliberately compiled
    from the smaller, PPU-owned include graph.
    """

    def build_extensions(self):
        if not USE_PPU:
            raise RuntimeError(
                "This fork builds device code with HGCC; set PPU_SDK or set "
                "FLASH_ATTENTION_SKIP_KERNEL_BUILD=TRUE."
            )
        for ext in self.extensions:
            self._build_extension_hgcc(ext)

    @staticmethod
    def _require(path, what):
        path = Path(path).resolve()
        if not path.exists():
            raise RuntimeError(f"missing {what}: {path}")
        return path

    @staticmethod
    def _ninja_command(arguments):
        # Ninja expands $in to multiple shell words. Quoting that variable as
        # one argument makes the device linker see every object as one path.
        return " ".join(
            str(argument) if argument in ("$in", "$out") else shlex.quote(str(argument))
            for argument in arguments
        )

    def _build_extension_hgcc(self, ext):
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

        source_root = Path(this_dir).resolve()
        sources = [(source_root / source).resolve() for source in ext.sources]
        for source in sources:
            self._require(source, "extension source")

        build_temp = Path(self.build_temp).resolve() / "hgcc_objs"
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        common_includes = [Path(path).resolve() for path in ext.include_dirs]
        device_includes = common_includes + [ppu_include, ppu_target_include]
        host_includes = common_includes + [
            ppu_include,
            ppu_target_include,
            torch_include,
            torch_api_include,
            ppu_cuda_include,
            python_include,
        ]

        extra = ext.extra_compile_args if isinstance(ext.extra_compile_args, dict) else {}
        hgcc_flags = list(extra.get("hgcc", []))
        cxx_flags = list(extra.get("cxx", []))
        link_flags = list(getattr(ext, "extra_link_args", []) or [])

        pybind_flags = []
        for macro, attr in (
            ("PYBIND11_COMPILER_TYPE", "_PYBIND11_COMPILER_TYPE"),
            ("PYBIND11_STDLIB", "_PYBIND11_STDLIB"),
            ("PYBIND11_BUILD_ABI", "_PYBIND11_BUILD_ABI"),
        ):
            value = getattr(torch._C, attr, None)
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

        object_files = []
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
                relative = source.relative_to(source_root)
                object_name = "__".join(relative.with_suffix("").parts) + ".o"
                object_path = build_temp / object_name
                object_files.append(object_path)
                rule = "hgcc_compile" if source.suffix == ".cu" else "cxx_compile"
                handle.write(f"build {object_path}: {rule} {source}\n")

            handle.write(f"\nbuild {ext_path}: link {' '.join(map(str, object_files))}\n")
            handle.write(f"\ndefault {ext_path}\n")

        requested_jobs = os.environ.get("MAX_JOBS")
        if requested_jobs is None:
            requested_jobs = str(max(1, min(os.cpu_count() or 1, 16)))
        print(
            f"[HGCCBuildExtension] sources={len(sources)} jobs={requested_jobs} "
            f"architectures=ppu_10,ppu_15 output={ext_path}"
        )
        subprocess.check_call(
            [ninja, "-f", ninja_file.name, f"-j{requested_jobs}"], cwd=build_temp
        )
        print(f"[HGCCBuildExtension] built {ext_path}")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        return super().run()


# ============================================================================
# Extension module definition
# ============================================================================

ext_modules = []

dir_actlize = this_dir + "/csrc/actlize"
if not os.path.exists(dir_actlize):
    repo_actlize = os.path.dirname(this_dir) + "/actlize"
    if not os.path.exists(repo_actlize):
        raise RuntimeError(
            f"actlize does not exist: actlize must be fetched in advance as:\n"
            f" \"{repo_actlize}\" or \"{dir_actlize}\""
        )
    else:
        os.symlink(repo_actlize, dir_actlize)

if not SKIP_KERNEL_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

    # hgcc flags
    hgcc_flags = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-mllvm", "-ppu-max-vreg-count=256",
        "-mllvm", "-ppu-sink-matrix-addr=true",
        "-mllvm", "-ppu-max-alloca-byte-size=320",
        "-mllvm", "-ppu-sink-async-addr=true",
        "-mllvm", "-ppu-sink-load-addr=true",
        "-mllvm", "-ppu-sink-store-addr=true",
        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
        "-Xcompiler", "-fPIC",
        "-DSWITCH_TO_HGGCRT",
        "-DUSE_CLANG",
        "-DUSE_HGGC",
        "-DUSE_PPU",
        "-DUSE_AIU=1",
        "-DFLASHATTN_PPU_DEVICE_COMPILE",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=flash_attn_2_cuda",
    ]

    cxx_flags = [
        "-O3", "-std=c++17", "-fPIC", "-DUSE_PPU", "-DUSE_AIU=1",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=flash_attn_2_cuda",
    ]

    ext_modules.append(
        Extension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                "hgcc": hgcc_flags + ["-arch=ppu_10", "-arch=ppu_15"],
                "cxx": cxx_flags,
            },
            include_dirs=[
                str(Path(this_dir) / "csrc" / "flash_attn"),
                str(Path(this_dir) / "csrc" / "flash_attn" / "src"),
                str(Path(this_dir) / "csrc" / "actlize" / "include"),
            ],
        )
    )


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build", "csrc", "include", "tests", "dist",
            "docs", "benchmarks", "flash_attn.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": HGCCBuildExtension},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
