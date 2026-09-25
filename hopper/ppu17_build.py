"""Single source of truth for the explicitly selected PPU1.7 forward build.

The CUDA-compatible 10700 toolchain must actually select SM90a. A successful
exit from an old HGGC CUDA wrapper is insufficient: SDK2.1.1 accepts sm_90a
but targets PPU1.0. compiler_target.cu rejects that case before any FA build.
Importing this module requires neither Torch nor an SDK.
"""

import os
import re
import subprocess
import time
from pathlib import Path

HOPPER = Path(__file__).resolve().parent
ROOT = HOPPER.parent
HEAD_DIMS = (64, 128, 256)
DTYPES = ("bf16", "fp16")
DISABLED = (
    "BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
    "PACKGQA", "FP8", "VARLEN", "CLUSTER", "HDIM96", "HDIM192", "SM8x", "SM86",
)


def compile_flags():
    return [
        "-std=c++17", "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
        "-DFLASHATTN_PPU17=1", "-DACOMPUTE_VERSION=10700",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        *(f"-DFLASHATTENTION_DISABLE_{feature}" for feature in DISABLED),
    ]


def source_files():
    sources = [HOPPER / f"instantiations/flash_fwd_hdim{d}_{t}_sm90.cu"
               for d in HEAD_DIMS for t in DTYPES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"missing PPU1.7 generated source units: {missing}")
    return sources


def cutlass_root(value):
    if not value:
        raise ValueError("set CUTLASS_PPU17_ROOT to the PPU CUTLASS 3.6.0 tree")
    root = Path(value).resolve()
    version = root / "include/cutlass/version.h"
    if not version.is_file():
        raise ValueError(f"not a CUTLASS root: {root}")
    text = version.read_text()
    actual = tuple(int(re.search(rf"#define CUTLASS_{part}\s+(\d+)", text)[1])
                   for part in ("MAJOR", "MINOR", "PATCH"))
    if actual != (3, 6, 0):
        raise ValueError(f"expected CUTLASS 3.6.0, got {actual}")
    for header in ("ppu/ppu_include_10700.hpp", "cute/arch/mma_sm90_gmma.hpp",
                   "cute/atom/copy_traits_sm90_tma.hpp"):
        if not (root / "include" / header).is_file():
            raise ValueError(f"missing PPU1.7 backend header: {header}")
    return root


def check_environment(env):
    # Reject conflicting requests instead of making switches silently dead.
    for feature in DISABLED:
        key = f"FLASH_ATTENTION_DISABLE_{feature}"
        if key in env and env[key] != "TRUE":
            raise ValueError(f"PPU1.7 initial scope requires {key}=TRUE")
    for key in ("FLASH_ATTENTION_DISABLE_SM90", "FLASH_ATTENTION_DISABLE_FP16",
                "FLASH_ATTENTION_DISABLE_HDIM64", "FLASH_ATTENTION_DISABLE_HDIM128",
                "FLASH_ATTENTION_DISABLE_HDIM256"):
        if env.get(key, "FALSE") != "FALSE":
            raise ValueError(f"PPU1.7 build manifest requires {key}=FALSE")
    if env.get("FA3_HLLM_BUILD", "0") != "0" or env.get("FA3_HLLM_USE_ADDR", "0") != "0":
        raise ValueError("PPU1.7 initial source integration does not admit the HLLM ABI")
    if env.get("FLASH_ATTENTION_ENABLE_VCOLMAJOR", "FALSE") != "FALSE":
        raise ValueError("PPU1.7 source integration does not admit V-colmajor")
    if env.get("FLASH_ATTENTION_ENABLE_QSA", "FALSE") != "FALSE":
        raise ValueError("PPU1.7 source integration does not admit QSA")
    if env.get("FLASH_ATTENTION_PPU17_COMPILE_MODE", "native") not in ("native", "simulation"):
        raise ValueError("PPU1.7 compile mode must be native or simulation")


def check_compiler(compiler, output, *, source_check=False):
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise ValueError(f"compiler does not exist: {compiler}")
    output = Path(output).resolve() / f"probe-{time.time_ns()}-{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    flags = compile_flags()
    if source_check:
        flags += ["-DFLASHATTN_PPU17_SOURCE_CHECK=1"]
    cmd = [str(compiler), *flags, "-c", str(ROOT / "dev/ppu17/compiler_target.cu"),
           "-o", str(output / "compiler_target.o")]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "TMPDIR": str(output)})
    (output / "compiler-target.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("compiler failed the real SM90a/PPU1.7 target contract; "
                           f"see {output / 'compiler-target.log'}\n{result.stdout[-4000:]}")
    obj = output / "compiler_target.o"
    if not obj.is_file() or obj.stat().st_size == 0:
        raise RuntimeError("compiler returned success without a target object")


def setup_extension():
    # This path does NOT define USE_PPU/USE_AIU and does not import the
    # legacy HGCC host/device splitter. The 10700 backend uses CUDA APIs.
    check_environment(os.environ)
    backend = cutlass_root(os.environ.get("CUTLASS_PPU17_ROOT"))
    simulation = os.environ.get("FLASH_ATTENTION_PPU17_COMPILE_MODE", "native") == "simulation"
    if simulation:
        if not os.environ.get("CUDA_HOME"):
            raise ValueError("simulation input build requires an explicit CUDA_HOME")
        cuda_home = Path(os.environ["CUDA_HOME"]).resolve()
    else:
        if not os.environ.get("PPU_SDK"):
            raise ValueError("native PPU1.7 build requires PPU_SDK")
        cuda_home = Path(os.environ["PPU_SDK"]).resolve() / "CUDA_SDK"
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    from setuptools import setup
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    class PPU17BuildExtension(BuildExtension):
        def build_extensions(self):
            check_compiler(cuda_home / "bin/nvcc", Path(self.build_temp) / "ppu17-target",
                           source_check=simulation)
            super().build_extensions()

    macros = [flag for flag in compile_flags() if flag.startswith("-D")]
    setup(
        name="flash_attn_3",
        version="3.0.0b1+ppu17" + (".simulation" if simulation else ""),
        packages=["flash_attn_3"],
        package_dir={"flash_attn_3": str(HOPPER)},
        ext_modules=[CUDAExtension(
            "flash_attn_3._C",
            sources=[str(HOPPER / "flash_api.cpp"), str(HOPPER / "flash_prepare_scheduler.cu"),
                     *(str(p) for p in source_files())],
            include_dirs=[str(HOPPER), str(backend / "include")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17", *macros],
                                "nvcc": compile_flags()},
            libraries=["cuda"],
        )],
        cmdclass={"build_ext": PPU17BuildExtension},
        python_requires=">=3.9",
    )
