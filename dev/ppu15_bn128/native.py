"""Fail-closed native resource/type checks; static counts are NOT timings."""
from collections import Counter
import re


def inspect_native(config, resources, disassembly):
    if "(PPU 1.5)" not in resources:
        raise ValueError("resource dump is not PPU1.5")
    chunks = re.split(r"^Func \d+ (\S+) RESOURCE INFO:\s*$", resources, flags=re.M)
    if len(chunks) != 5:
        raise ValueError("expected exactly two emitted kernel bodies (FP16 and BF16)")
    expected_geometry = f"ILi128ELi{config['bm']}ELi{config['bn']}ELi{config['warps']}ELb0ELb0"
    result = {}
    for symbol, text in zip(chunks[1::2], chunks[2::2]):
        if "flash_fwd_kernel" not in symbol or expected_geometry not in symbol:
            raise ValueError("native kernel type is not the registered shipping geometry")
        dtype = "bf16" if "bfloat16_t" in symbol else "fp16" if "half_t" in symbol else None
        if not dtype or dtype in result:
            raise ValueError("native dtype coverage is missing or duplicated")
        def number(key):
            found = re.findall(rf"^{re.escape(key)}:(\d+)\s*$", text, re.M)
            if len(found) != 1:
                raise ValueError(f"resource field not unique: {key}")
            return int(found[0])
        stack, private = number("STACK SIZE"), number("private_en")
        if stack != 0 or private != 0:
            raise ValueError(f"spill/private-memory gate: {dtype} stack={stack} private={private}")
        marker = f"Disassembly of section .text.kernel.{symbol}:"
        if disassembly.count(marker) != 1:
            raise ValueError("native body not found exactly once")
        body = disassembly.split(marker, 1)[1].split("Disassembly of section", 1)[0]
        opcodes = Counter(re.findall(
            r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)+([a-z][a-z0-9_.]+)\b", body, re.M))
        mma = "v.mma.f32." + ("bf16" if dtype == "bf16" else "f16") + ".m16n16k16"
        families = {op for op in opcodes if op.startswith("v.mma.")}
        if families != {mma}:
            raise ValueError(f"unexpected/missing native MMA precision: {families}, expected {mma}")
        result[dtype] = {"symbol": symbol, "vector_registers": number("vreg_number"),
                         "scalar_registers": number("sreg_number"), "stack_bytes": stack,
                         "static_shared_bytes": number("shared_memory_size"),
                         "dynamic_shared_bytes": config["shared_bytes"],
                         "static_instruction_sites": sum(opcodes.values()),
                         "static_opcodes": dict(sorted(opcodes.items())), "admission": "PASS"}
    return result
