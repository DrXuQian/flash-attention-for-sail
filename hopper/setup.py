# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.
# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# PPU build using hgcc.

import sys
import os
import re
import ast
import itertools
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch

if 'PPU_SDK' in os.environ.keys():
    from ppu_build import HGCCBuildExtension

PACKAGE_NAME = "flash_attn_3"
this_dir = os.path.dirname(os.path.abspath(__file__))

USE_PPU = 'PPU_SDK' in os.environ.keys()
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"

SKIP_KERNEL_BUILD = os.getenv("FLASH_ATTENTION_SKIP_KERNEL_BUILD", "FALSE") == "TRUE"

FA3_HLLM_BUILD = os.getenv("FA3_HLLM_BUILD", "0") == "1"
FA3_HLLM_USE_ADDR = os.getenv("FA3_HLLM_USE_ADDR", "0") == "1"

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_SPLIT = os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE"
DISABLE_PAGEDKV = os.getenv("FLASH_ATTENTION_DISABLE_PAGEDKV", "FALSE") == "TRUE"
DISABLE_APPENDKV = os.getenv("FLASH_ATTENTION_DISABLE_APPENDKV", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("FLASH_ATTENTION_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_SOFTCAP = os.getenv("FLASH_ATTENTION_DISABLE_SOFTCAP", "FALSE") == "TRUE"
DISABLE_PACKGQA = os.getenv("FLASH_ATTENTION_DISABLE_PACKGQA", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("FLASH_ATTENTION_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("FLASH_ATTENTION_DISABLE_FP8", "FALSE") == "TRUE"
DISABLE_VARLEN = os.getenv("FLASH_ATTENTION_DISABLE_VARLEN", "FALSE") == "TRUE"
DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"
DISABLE_HDIM64 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM96 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM96", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM192 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM192", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("FLASH_ATTENTION_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM8x" if USE_PPU else "FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"
DISABLE_SM90 = os.getenv("FLASH_ATTENTION_DISABLE_SM90", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"
DISABLE_SM80 = os.getenv("FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"
DISABLE_SM89 = os.getenv("FLASH_ATTENTION_DISABLE_SM89", "FALSE") == "TRUE"
ENABLE_VCOLMAJOR = os.getenv("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") == "TRUE"


# ============================================================================
# Source file generation
# ============================================================================

ext_modules = []

dir_actlize = this_dir + "/../csrc/actlize"
if not os.path.exists(dir_actlize):
    repo_actlize = os.path.dirname(this_dir) + "/actlize"
    if not os.path.exists(repo_actlize):
        raise RuntimeError(f"actlize does not exist: \"{repo_actlize}\" or \"{dir_actlize}\"")
    else:
        os.symlink(repo_actlize, dir_actlize)

def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return ["--threads", nvcc_threads]

if not SKIP_CUDA_BUILD:
    print(f"\n\ntorch.__version__  = {torch.__version__}\n\n")

    repo_dir = Path(this_dir).parent
    actlize_dir = repo_dir / "csrc" / "actlize"

    feature_args = (
        []
        + (["-DFLASHATTENTION_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DFLASHATTENTION_DISABLE_PAGEDKV"] if DISABLE_PAGEDKV else [])
        + (["-DFLASHATTENTION_DISABLE_SPLIT"] if DISABLE_SPLIT else [])
        + (["-DFLASHATTENTION_DISABLE_APPENDKV"] if DISABLE_APPENDKV else [])
        + (["-DFLASHATTENTION_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DFLASHATTENTION_DISABLE_SOFTCAP"] if DISABLE_SOFTCAP else [])
        + (["-DFLASHATTENTION_DISABLE_PACKGQA"] if DISABLE_PACKGQA else [])
        + (["-DFLASHATTENTION_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DFLASHATTENTION_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DFLASHATTENTION_DISABLE_VARLEN"] if DISABLE_VARLEN else [])
        + (["-DFLASHATTENTION_DISABLE_CLUSTER"] if DISABLE_CLUSTER else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM96"] if DISABLE_HDIM96 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM192"] if DISABLE_HDIM192 else [])
        + (["-DFLASHATTENTION_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DFLASHATTENTION_DISABLE_SM8x"] if DISABLE_SM8x else [])
        + ["-DFLASHATTENTION_DISABLE_SM86"]
        + (["-DFLASHATTENTION_DISABLE_SM90"] if DISABLE_SM90 else [])
        + (["-DFLASHATTENTION_ENABLE_VCOLMAJOR"] if ENABLE_VCOLMAJOR else [])
        + ["-DUSE_PPU", "-DUSE_AIU=1"]
        + (["-DFA3_HLLM_BUILD"] if FA3_HLLM_BUILD else [])
        + (["-DFA3_HLLM_USE_ADDR"] if FA3_HLLM_USE_ADDR else [])
    )

    DTYPE_FWD_SM80 = ["bf16"] + (["fp16"] if not DISABLE_FP16 else []) + (["e4m3"] if not DISABLE_FP8 else [])
    DTYPE_BWD = ["bf16"] + (["fp16"] if not DISABLE_FP16 else [])
    HEAD_DIMENSIONS_BWD = (
        []
        + ([64] if not DISABLE_HDIM64 else [])
        + ([96] if not DISABLE_HDIM96 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([192] if not DISABLE_HDIM192 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    HEAD_DIMENSIONS_FWD_SM80 = HEAD_DIMENSIONS_BWD
    SPLIT = [""] + (["_split"] if not DISABLE_SPLIT else [])
    PAGEDKV = [""] + (["_paged"] if not DISABLE_PAGEDKV else [])
    SOFTCAP = [""] + (["_softcap"] if not DISABLE_SOFTCAP else [])
    SOFTCAP_ALL = [""] if DISABLE_SOFTCAP else ["_softcapall"]
    PACKGQA = [""] + (["_packgqa"] if not DISABLE_PACKGQA else [])

    sources_fwd_sm80 = [f"instantiations/flash_fwd_hdim{hdim}_{dtype}{paged}{split}{softcap}{packgqa}_sm80.cu"
                        for hdim, dtype, split, paged, softcap, packgqa in itertools.product(HEAD_DIMENSIONS_FWD_SM80, DTYPE_FWD_SM80, SPLIT, PAGEDKV, SOFTCAP_ALL, PACKGQA)
                        if not ((packgqa and split) or (dtype == "e4m3" and hdim == 96))]
    sources_bwd_sm80 = [f"instantiations/flash_bwd_hdim{hdim}_{dtype}{softcap}_sm80.cu"
                        for hdim, dtype, softcap in itertools.product(HEAD_DIMENSIONS_BWD, DTYPE_BWD, SOFTCAP)]

    if DISABLE_BACKWARD:
        sources_bwd_sm80 = []

    sources = (
        ["flash_api.cpp"]
        + (sources_fwd_sm80 if not DISABLE_SM8x else [])
        + (sources_bwd_sm80 if not DISABLE_SM8x else [])
    )
    if not DISABLE_SPLIT:
        sources += ["flash_fwd_combine.cu"]
    sources += ["flash_prepare_scheduler.cu"]

    include_dirs = [
        str(Path(this_dir)),
        str(actlize_dir / "include"),
    ]

    os.environ["HGGC_ENABLE_COMPRESS"] = "1"

    hgcc_flags = [
        "-O3", "-std=c++17",
        "--ftemplate-backtrace-limit=0",
        "--use_fast_math",
        "--resource-usage",
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        "-mllvm", "-ppu-max-vreg-count=256",
        "-mllvm", "-ppu-sink-matrix-addr=true",
        "-mllvm", "-ppu-max-alloca-byte-size=320",
        "-mllvm", "-ppu-sink-async-addr=true",
        "-mllvm", "-ppu-sink-load-addr=true",
        "-mllvm", "-ppu-sink-store-addr=true",
        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
        "-mllvm", "-ppu-volatile-yield=false",
        "-mllvm", "-sort-copy-before-coalesce=true",
        "-Xfatbin",
        "--compress-all",
    ]

    if USE_PPU:
        # These are nvcc-driver-only fatbin options.  The split PPU build calls
        # HGCC directly and HGCC emits the two requested device images itself.
        hgcc_flags = [flag for flag in hgcc_flags if flag not in ("-Xfatbin", "--compress-all")]
        hgcc_flags += [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler", "-fPIC",
            "-DSWITCH_TO_HGGCRT",
            "-DUSE_CLANG",
            "-DUSE_HGGC",
            "-DFLASHATTN_PPU_DEVICE_COMPILE",
            "-DTORCH_API_INCLUDE_EXTENSION_H",
            "-DTORCH_EXTENSION_NAME=_C",
        ]

    cc_flag = []
    cc_flag.append("-arch=ppu_10")
    cc_flag.append("-arch=ppu_15")

    extension_type = Extension if USE_PPU else CUDAExtension
    ext_modules.append(
        extension_type(
            name=f"{PACKAGE_NAME}._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                ("hgcc" if USE_PPU else "nvcc"): ([] if USE_PPU else nvcc_threads_args()) + hgcc_flags + cc_flag + feature_args,
                "cxx": ["-O3", "-std=c++17", "-fPIC", "-DPy_LIMITED_API=0x03090000",
                        "-DTORCH_API_INCLUDE_EXTENSION_H", "-DTORCH_EXTENSION_NAME=_C"] + feature_args,
            },
            py_limited_api=True,
        )
    )


# ============================================================================
# Package setup
# ============================================================================

def get_package_version():
    with open(Path(this_dir) / "__init__.py", "r") as f:
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


class PerSourceBuildExtension(BuildExtension):
    """
    BuildExtension that adds -mllvm -ppu-register-margin=2 only for backward
    headdim 128 CUDA sources. This is done by post-processing the ninja build
    file to add per-source cuda_post_cflags overrides.
    """

    def build_extensions(self):
        import torch.utils.cpp_extension as torch_ext

        original_write_ninja_file = torch_ext._write_ninja_file

        def write_ninja_file_with_register_margin(path, *args, **kwargs):
            original_write_ninja_file(path, *args, **kwargs)
            self._patch_ninja_register_margin(path)

        torch_ext._write_ninja_file = write_ninja_file_with_register_margin
        try:
            super().build_extensions()
        finally:
            torch_ext._write_ninja_file = original_write_ninja_file

    @staticmethod
    def _patch_ninja_register_margin(path):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            lines = f.read().splitlines()

        new_lines = []
        margin_flags = "-mllvm -ppu-register-margin=2"
        for line in lines:
            new_lines.append(line)
            # Match build statements for cuda_compile rules (e.g. cuda_compile,
            # cuda_compile_sm80, cuda_compile_sm80_sm90, cuda_compile_sm100).
            # Ninja line format: "build <obj>: <rule> <src>"
            # After split(): ["build", "<obj>:", "<rule>", "<src>", ...]
            if line.startswith("build "):
                parts = line.split()
                if len(parts) >= 4 and parts[2].startswith("cuda_compile"):
                    src = parts[3]
                    if "flash_bwd_hdim128_" in src and src.endswith(".cu"):
                        new_lines.append(
                            f"  cuda_post_cflags = $cuda_post_cflags {margin_flags}"
                        )

        with open(path, "w") as f:
            f.write("\n".join(new_lines))



setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=("build", "csrc", "include", "tests", "dist", "docs", "benchmarks")
    ),
    py_modules=["flash_attn_interface"],
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention 3: Fast and Memory-Efficient Exact Attention",
    long_description=open("../README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand,
              "build_ext": HGCCBuildExtension if USE_PPU else PerSourceBuildExtension},
    python_requires=">=3.8",
    install_requires=["torch", "einops", "packaging", "ninja"],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
