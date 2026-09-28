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
SUPPORTED_CUTLASS = ((3, 6, 0), (4, 3, 0))
DISABLED = (
    "BACKWARD", "SPLIT", "PAGEDKV", "APPENDKV", "LOCAL", "SOFTCAP",
    "PACKGQA", "FP8", "VARLEN", "CLUSTER", "HDIM96", "HDIM192", "SM8x", "SM86",
)


def compile_flags(backend=None):
    flags = [
        "-std=c++17", "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a", "--use_fast_math",
        "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
        "-DFLASHATTN_PPU17=1", "-DACOMPUTE_VERSION=10700",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        *(f"-DFLASHATTENTION_DISABLE_{feature}" for feature in DISABLED),
    ]
    if backend is not None:
        major, minor, patch = cutlass_version(backend)
        flags += [f"-DFLASHATTN_PPU17_EXPECTED_CUTLASS_VERSION={major * 100 + minor * 10 + patch}"]
    return flags


def source_files():
    sources = [HOPPER / f"instantiations/flash_fwd_hdim{d}_{t}_sm90.cu"
               for d in HEAD_DIMS for t in DTYPES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"missing PPU1.7 generated source units: {missing}")
    return sources


def cutlass_version(root):
    root = Path(root)
    version = root / "include/cutlass/version.h"
    if not version.is_file():
        raise ValueError(f"not a CUTLASS root: {root}")
    # Vendor headers can contain non-UTF8 copyright/comments. Version macros
    # are ASCII tokens: parse bytes, without guessing the comment encoding or
    # dropping a malformed byte from a semantic token.
    raw = version.read_bytes()
    matches = [re.search(rb"(?m)^[ \t]*#[ \t]*define[ \t]+CUTLASS_" + part.encode("ascii") +
                         rb"[ \t]+([0-9]+)(?=[ \t]*(?://|/\*|\r?$))", raw)
               for part in ("MAJOR", "MINOR", "PATCH")]
    if not all(matches):
        raise ValueError(f"malformed CUTLASS version header: {version}")
    return tuple(int(match[1]) for match in matches)


def diagnostic_text(raw):
    """Display arbitrary tool bytes without hiding them or masking exit codes."""
    return raw.decode("utf-8", errors="backslashreplace")


def run_logged(command, log, *, env=None):
    """Keep verbatim bytes; a nonzero tool exit remains a hard failure."""
    log = Path(log)
    with log.open("wb") as handle:
        result = subprocess.run([str(item) for item in command], env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    raw = log.read_bytes()
    if result.returncode:
        raise RuntimeError(f"{log.name}: command failed (rc={result.returncode}); raw log: {log}\n"
                           + diagnostic_text(raw[-6000:]))
    return raw


def cutlass_root(value):
    if not value:
        raise ValueError("set CUTLASS_PPU17_ROOT to a PPU CUTLASS 3.6.0 or 4.3.0 tree")
    root = Path(value).resolve()
    actual = cutlass_version(root)
    if actual not in SUPPORTED_CUTLASS:
        raise ValueError(f"expected PPU CUTLASS 3.6.0 or 4.3.0, got {actual}")
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
    try:
        run_logged(cmd, output / "compiler-target.log", env={**os.environ, "TMPDIR": str(output)})
    except RuntimeError as error:
        raise RuntimeError(f"compiler failed the real SM90a/PPU1.7 target contract; {error}") from error
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

    flags = compile_flags(backend)
    macros = [flag for flag in flags if flag.startswith("-D")]
    backend_version = ".".join(map(str, cutlass_version(backend)))
    print(f"[PPU1.7 backend] cutlass={backend_version} root={backend} "
          f"mode={'simulation' if simulation else 'native'}", flush=True)
    setup(
        name="flash_attn_3",
        version="3.0.0b1+ppu17.cutlass" + backend_version + (".simulation" if simulation else ""),
        packages=["flash_attn_3"],
        package_dir={"flash_attn_3": str(HOPPER)},
        ext_modules=[CUDAExtension(
            "flash_attn_3._C",
            sources=[str(HOPPER / "flash_api.cpp"), str(HOPPER / "flash_prepare_scheduler.cu"),
                     *(str(p) for p in source_files())],
            include_dirs=[str(HOPPER), str(backend / "include")],
            extra_compile_args={"cxx": ["-O3", "-std=c++17", *macros],
                                "nvcc": flags},
            libraries=["cuda"],
        )],
        cmdclass={"build_ext": PPU17BuildExtension},
        python_requires=">=3.9",
    )
