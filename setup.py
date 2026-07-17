# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Copyright (c) 2023, Tri Dao.

import sys
import functools
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch

PACKAGE_NAME = "flash_attn"
this_dir = os.path.dirname(os.path.abspath(__file__))

USE_PPU = 'PPU_SDK' in os.environ.keys()
SKIP_KERNEL_BUILD = os.getenv("FLASH_ATTENTION_SKIP_KERNEL_BUILD", "FALSE") == "TRUE"
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ============================================================================
# PPU HGCC Build Extension
# ============================================================================

class HGCCBuildExtension(build_ext):
    """Custom build extension that uses hgcc for .cu, c++ for .cpp."""

    def build_extensions(self):
        if not os.environ.get("MAX_JOBS"):
            import psutil
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            max_num_jobs_memory = int(free_memory_gb / 9)
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        for ext in self.extensions:
            self._build_extension_hgcc(ext)

    def _build_extension_hgcc(self, ext):
        import ninja  # noqa: F401

        ppu_sdk = os.environ.get("PPU_SDK", "")
        hgcc = os.path.join(ppu_sdk, "bin", "hgcc")
        torch_dir = torch.__path__[0]

        sources = [os.path.join(this_dir, s) for s in ext.sources]
        include_dirs = [os.path.abspath(d) for d in ext.include_dirs]

        output_dir = os.path.join(self.build_temp, "hgcc_objs")
        os.makedirs(output_dir, exist_ok=True)

        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)

        torch_include = os.path.join(torch_dir, "include")
        torch_include_csrc = os.path.join(torch_dir, "include", "torch", "csrc", "api", "include")
        python_include = subprocess.check_output(
            [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
        ).decode().strip()

        ppu_sdk_inc = os.path.join(ppu_sdk, "include")
        ppu_targets_inc = os.path.join(ppu_sdk, "targets", "x86_64-linux", "include")
        all_includes = include_dirs + [torch_include, torch_include_csrc, python_include, ppu_sdk_inc, ppu_targets_inc]
        include_flags = [f"-I{d}" for d in all_includes]

        # Read extra_compile_args from extension
        extra_compile_args = ext.extra_compile_args if hasattr(ext, 'extra_compile_args') else {}
        if isinstance(extra_compile_args, dict):
            hgcc_flags = extra_compile_args.get('hgcc', [])
            cxx_flags = extra_compile_args.get('cxx', [])
        else:
            hgcc_flags = []
            cxx_flags = list(extra_compile_args)

        # Build ninja file
        max_jobs = int(os.environ.get("MAX_JOBS", "4"))
        ninja_file = os.path.join(output_dir, "build.ninja")
        obj_files = []

        with open(ninja_file, "w") as f:
            f.write("ninja_required_version = 1.3\n\n")

            f.write(f"rule hgcc_compile\n")
            f.write(f"  command = {hgcc} {' '.join(hgcc_flags)} {' '.join(include_flags)} -c $in -o $out\n")
            f.write(f"  description = HGCC $in\n\n")

            cxx_compiler = "c++"
            cxx_include_flags = include_flags
            f.write(f"rule cxx_compile\n")
            f.write(f"  command = {cxx_compiler} {' '.join(cxx_flags)} {' '.join(cxx_include_flags)} -c $in -o $out\n")
            f.write(f"  description = CXX $in\n\n")

            torch_lib_dir = os.path.join(torch_dir, "lib")
            ppu_lib_dir = os.path.join(ppu_sdk, "lib")
            link_libs = f"-L{torch_lib_dir} -L{ppu_lib_dir} -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda -lhggc_wrapper -lhg_wrapper"
            f.write(f"rule link\n")
            f.write(f"  command = {hgcc} -shared -o $out $in {link_libs}\n")
            f.write(f"  description = LINK $out\n\n")

            for src in sources:
                basename = os.path.splitext(os.path.basename(src))[0]
                obj = os.path.join(output_dir, basename + ".o")
                obj_files.append(obj)

                if src.endswith(".cu"):
                    f.write(f"build {obj}: hgcc_compile {src}\n")
                else:
                    f.write(f"build {obj}: cxx_compile {src}\n")

            f.write(f"\nbuild {ext_path}: link {' '.join(obj_files)}\n")
            f.write(f"\ndefault {ext_path}\n")

        print(f"\n[HGCCBuildExtension] Building {ext.name} with {len(sources)} sources, max_jobs={max_jobs}")
        subprocess.check_call(
            ["ninja", "-f", ninja_file, f"-j{max_jobs}"],
        )
        print(f"[HGCCBuildExtension] Built {ext_path}")


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
        "-U__HGGC_NO_HALF_OPERATORS__",
        "-U__HGGC_NO_HALF_CONVERSIONS__",
        "-U__HGGC_NO_HALF2_OPERATORS__",
        "-U__HGGC_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-arch=ppu_10",
        "-arch=ppu_15",
        "-DUSE_PPU",
        "-DUSE_AIU=1",
        "-DSWITCH_TO_HGGCRT",
        "-DUSE_CLANG",
        "-DUSE_HGGC",
        "-mllvm", "-ppu-max-vreg-count=256",
        "-mllvm", "-ppu-sink-matrix-addr=true",
        "-mllvm", "-ppu-max-alloca-byte-size=320",
        "-mllvm", "-ppu-sink-async-addr=true",
        "-mllvm", "-ppu-sink-load-addr=true",
        "-mllvm", "-ppu-sink-store-addr=true",
        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
    ]

    # cxx flags (same as original)
    cxx_flags = ["-O3", "-std=c++17", "-fPIC", "-DUSE_PPU", "-DUSE_AIU=1"]

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
                "hgcc": hgcc_flags,
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
