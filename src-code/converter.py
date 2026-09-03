"""
convert to a custom n730 binary

File layout:
  [MAGIC 8b][VERSION 4b][HEADER_SIZE 4b][JSON_HEADER Nb][PADDING to 4KB]
  [LAYER_0_DATA][PADDING to 4KB]
  [LAYER_1_DATA][PADDING to 4KB]
  ...

Each LAYER_DATA block:
  [LAYER_MAGIC 4b][LAYER_IDX 4b][PRECISION 1b][ROWS 4b][COLS 4b]
  [SCALE f32][ZERO_POINT f32][DATA_SIZE 4b][QUANTIZED_BYTES ...]

Usage:
    python converter.py --model deepseek-ai/deepseek-r1-distill-qwen-1.5b \\
                        --sensitivity sensitivity_map.json \\
                        --output model.n730

    python converter.py --synthetic --sensitivity sensitivity_map.json \\
                        --output model.n730
"""

import argparse
import json
import math
import struct
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

MAGIC         = b"N730\x00\x01\x00\x00"   # 8 bytes: file magic
LAYER_MAGIC   = b"LYR\x00"                 # 4 bytes: layer block magic
FORMAT_VERSION = 1
PAGE_SIZE     = 4096                        # 4KB alignment for DMA

PRECISION_BITS = {"INT2": 2, "INT4": 4, "INT8": 8, "FP16": 16}
PRECISION_ID   = {"INT2": 2, "INT4": 4, "INT8": 8, "FP16": 16}

def quantize_int8(weights: np.ndarray) -> tuple[bytes, float, float]:
    """Symmetric INT8: scale to [-127, 127], pack as uint8 with offset 128."""
    flat = weights.flatten().astype(np.float32)
    abs_max = float(np.max(np.abs(flat)))
    if abs_max < 1e-9:
        return bytes(flat.size), 1.0, 0.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
    # Store as uint8 (offset by 128) for unsigned DMA friendliness
    packed = (quantized.astype(np.int16) + 128).astype(np.uint8)
    return packed.tobytes(), scale, 128.0


def quantize_int4(weights: np.ndarray) -> tuple[bytes, float, float]:
    """
    INT4: scale to [0, 15], pack two values per byte (low nibble / high nibble).
    zero_point is the offset (typically 8 for symmetric).
    """
    flat = weights.flatten().astype(np.float32)
    w_min = float(np.min(flat))
    w_max = float(np.max(flat))
    if abs(w_max - w_min) < 1e-9:
        n_bytes = math.ceil(flat.size / 2)
        return bytes(n_bytes), 1.0, 0.0

    scale = (w_max - w_min) / 15.0
    zero_point = round(-w_min / scale)
    zero_point = int(np.clip(zero_point, 0, 15))

    quantized = np.clip(np.round(flat / scale) + zero_point, 0, 15).astype(np.uint8)

    # Pack two 4-bit values per byte
    if len(quantized) % 2 != 0:
        quantized = np.append(quantized, 0)
    packed = (quantized[0::2] & 0x0F) | ((quantized[1::2] & 0x0F) << 4)
    return packed.tobytes(), scale, float(zero_point)


def quantize_int2(weights: np.ndarray) -> tuple[bytes, float, float]:
    """
    INT2: only 4 values representable [0,1,2,3]. Pack 4 values per byte.
    Extremely aggressive — only safe for low-sensitivity middle layers.
    """
    flat = weights.flatten().astype(np.float32)
    w_min = float(np.min(flat))
    w_max = float(np.max(flat))
    if abs(w_max - w_min) < 1e-9:
        n_bytes = math.ceil(flat.size / 4)
        return bytes(n_bytes), 1.0, 0.0

    scale = (w_max - w_min) / 3.0
    zero_point = round(-w_min / scale)
    zero_point = int(np.clip(zero_point, 0, 3))

    quantized = np.clip(np.round(flat / scale) + zero_point, 0, 3).astype(np.uint8)

    # Pad to multiple of 4
    pad = (4 - len(quantized) % 4) % 4
    if pad:
        quantized = np.append(quantized, np.zeros(pad, dtype=np.uint8))

    # Pack 4 x 2-bit values per byte
    packed = (
        (quantized[0::4] & 0x03) |
        ((quantized[1::4] & 0x03) << 2) |
        ((quantized[2::4] & 0x03) << 4) |
        ((quantized[3::4] & 0x03) << 6)
    )
    return packed.tobytes(), scale, float(zero_point)


def quantize_fp16(weights: np.ndarray) -> tuple[bytes, float, float]:
    """FP16: straight cast, no quantization loss beyond float16 precision."""
    flat = weights.flatten().astype(np.float16)
    return flat.tobytes(), 1.0, 0.0


QUANTIZERS = {
    "INT2": quantize_int2,
    "INT4": quantize_int4,
    "INT8": quantize_int8,
    "FP16": quantize_fp16,
}

def pad_to_page(data: bytes) -> bytes:
    """Pad bytes to the next 4KB boundary."""
    remainder = len(data) % PAGE_SIZE
    if remainder == 0:
        return data
    return data + bytes(PAGE_SIZE - remainder)


def serialize_layer(
    layer_idx: int,
    precision: str,
    weights: np.ndarray,
    layer_name: str,
) -> tuple[bytes, dict]:

    shape = weights.shape
    rows = shape[0]
    cols = shape[1] if len(shape) > 1 else 1

    quantize_fn = QUANTIZERS[precision]
    quantized_bytes, scale, zero_point = quantize_fn(weights)

    precision_id = PRECISION_ID[precision]

    header = struct.pack(
        ">4sIBIIffi",         # big-endian: magic, idx, prec, rows, cols, scale, zp, data_size
        LAYER_MAGIC,
        layer_idx,
        precision_id,
        rows,
        cols,
        scale,
        zero_point,
        len(quantized_bytes),
    )

    block = header + quantized_bytes
    block_padded = pad_to_page(block)

    metadata = {
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "precision": precision,
        "rows": rows,
        "cols": cols,
        "scale": scale,
        "zero_point": zero_point,
        "original_bytes": weights.size * 4,
        "stored_bytes": len(quantized_bytes),
        "block_bytes": len(block_padded),
    }

    return block_padded, metadata

def convert(
    weight_layers: list,           # list of (name, np.ndarray)
    sensitivity_map: dict,
    output_path: Path,
    verbose: bool = True,
):
    # Build a lookup from layer name → precision
    precision_lookup = {
        p["layer_name"]: p["assigned_precision"]
        for p in sensitivity_map["profiles"]
    }

    total = len(weight_layers)
    seek_table = []       # will be embedded in the file header
    layer_blocks = []     # raw bytes, accumulated before writing

    print(f"\n  Converting {total} layers → {output_path.name}\n")

    t0 = time.time()
    bar_width = 40

    current_offset = 0    # tracks byte offset *after* header (filled in later)

    total_original = 0
    total_stored = 0

    for idx, (name, weights) in enumerate(weight_layers):
        if weights.ndim == 1:
            precision = "FP16"
        else:
            precision = precision_lookup.get(name, "INT4")

        block, meta = serialize_layer(idx, precision, weights, name)
        meta["file_offset"] = current_offset   # where this block starts in the data section
        current_offset += len(block)

        seek_table.append(meta)
        layer_blocks.append(block)

        total_original += meta["original_bytes"]
        total_stored += meta["stored_bytes"]

        if verbose:
            filled = int(bar_width * (idx + 1) / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            ratio = meta["original_bytes"] / max(meta["stored_bytes"], 1)
            print(
                f"  [{bar}] {idx+1:>3}/{total}  "
                f"[{precision:<4}]  {ratio:4.1f}×  {name[:45]}"
            )

    elapsed = time.time() - t0

    file_header = {
        "format": "N730",
        "version": FORMAT_VERSION,
        "project": "N730",
        "model_id": sensitivity_map["model_id"],
        "total_layers": total,
        "total_params": sensitivity_map["total_params"],
        "original_mb": round(total_original / (1024**2), 2),
        "stored_mb": round(total_stored / (1024**2), 2),
        "compression": round(total_original / max(total_stored, 1), 2),
        "page_size": PAGE_SIZE,
        "seek_table": seek_table,
    }

    header_json = json.dumps(file_header, indent=2).encode("utf-8")
    header_size = len(header_json)
    preamble = MAGIC + struct.pack(">II", FORMAT_VERSION, header_size)
    preamble_padded = pad_to_page(preamble + header_json)
    data_start = len(preamble_padded)
    for entry in file_header["seek_table"]:
        entry["file_offset"] += data_start
    header_json = json.dumps(file_header, indent=2).encode("utf-8")
    header_size = len(header_json)
    preamble = MAGIC + struct.pack(">II", FORMAT_VERSION, header_size)
    preamble_padded = pad_to_page(preamble + header_json)

    # Write the file
    with open(output_path, "wb") as f:
        f.write(preamble_padded)
        for block in layer_blocks:
            f.write(block)

    final_size_mb = output_path.stat().st_size / (1024**2)
    compression = total_original / max(total_stored, 1)

    print(f"\n  Done in {elapsed:.1f}s")
    print()
    print("═" * 60)
    print(f"  N730 CONVERSION COMPLETE")
    print("═" * 60)
    print(f"  Model          : {sensitivity_map['model_id']}")
    print(f"  Output file    : {output_path}")
    print(f"  Original size  : {total_original / (1024**2):.1f} MB  (FP32)")
    print(f"  N730 size      : {final_size_mb:.1f} MB  (with alignment padding)")
    print(f"  Compression    : {compression:.1f}×")
    print(f"  Layers         : {total}")
    print(f"  Page alignment : {PAGE_SIZE} bytes  (DMA-ready)")
    print("═" * 60)

    return file_header

def inspect_n730(path: Path):
    """Read and print the header of a .n730 file. Useful for debugging."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Not a valid .n730 file (bad magic: {magic!r})")
        version, header_size = struct.unpack(">II", f.read(8))
        header_json = f.read(header_size)

    header = json.loads(header_json)
    print(f"\n  N730 File Inspector")
    print(f"  {'─'*40}")
    print(f"  Format version : {version}")
    print(f"  Model          : {header['model_id']}")
    print(f"  Total layers   : {header['total_layers']}")
    print(f"  Original size  : {header['original_mb']} MB")
    print(f"  Stored size    : {header['stored_mb']} MB")
    print(f"  Compression    : {header['compression']}×")
    print(f"\n  First 5 seek table entries:")
    for entry in header["seek_table"][:5]:
        print(
            f"    [{entry['precision']:<4}] offset={entry['file_offset']:>12,}  "
            f"{entry['rows']}×{entry['cols']}  {entry['layer_name']}"
        )
    print(f"  ...")
    print(f"  Last entry:")
    last = header["seek_table"][-1]
    print(
        f"    [{last['precision']:<4}] offset={last['file_offset']:>12,}  "
        f"{last['rows']}×{last['cols']}  {last['layer_name']}"
    )


def read_layer(path: Path, layer_idx: int, header: dict = None) -> np.ndarray:
    """
    Read and dequantize a single layer from a .n730 file.
    This is what the Phase 3 scheduler will call — O(1) seek, no full file scan.
    """
    if header is None:
        with open(path, "rb") as f:
            f.read(8)   # magic
            version, header_size = struct.unpack(">II", f.read(8))
            header = json.loads(f.read(header_size))

    entry = header["seek_table"][layer_idx]
    precision = entry["precision"]

    with open(path, "rb") as f:
        f.seek(entry["file_offset"])
        # Read block header
        layer_magic = f.read(4)
        if layer_magic != LAYER_MAGIC:
            raise ValueError(f"Bad layer magic at offset {entry['file_offset']}")
        idx, prec_id, rows, cols, scale, zero_point, data_size = struct.unpack(
            ">IBIIffi", f.read(4+1+4+4+4+4+4)
        )
        raw_bytes = f.read(data_size)

    # Dequantize
    total_elements = rows * cols

    if precision == "INT8":
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        weights = (arr - zero_point) * scale

    elif precision == "INT4":
        packed = np.frombuffer(raw_bytes, dtype=np.uint8)
        lo = (packed & 0x0F).astype(np.float32)
        hi = ((packed >> 4) & 0x0F).astype(np.float32)
        interleaved = np.empty(lo.size + hi.size, dtype=np.float32)
        interleaved[0::2] = lo
        interleaved[1::2] = hi
        weights = (interleaved[:total_elements] - zero_point) * scale

    elif precision == "INT2":
        packed = np.frombuffer(raw_bytes, dtype=np.uint8)
        b0 = (packed & 0x03).astype(np.float32)
        b1 = ((packed >> 2) & 0x03).astype(np.float32)
        b2 = ((packed >> 4) & 0x03).astype(np.float32)
        b3 = ((packed >> 6) & 0x03).astype(np.float32)
        interleaved = np.empty(b0.size * 4, dtype=np.float32)
        interleaved[0::4] = b0
        interleaved[1::4] = b1
        interleaved[2::4] = b2
        interleaved[3::4] = b3
        weights = (interleaved[:total_elements] - zero_point) * scale

    elif precision == "FP16":
        weights = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)

    else:
        raise ValueError(f"Unknown precision: {precision}")

    return weights.reshape(rows, cols)

def generate_synthetic_weights(sensitivity_map: dict) -> list:
    layers = []
    rng = np.random.default_rng(42)
    MAX_PARAMS = 1_000_000  # cap per layer for synthetic mode
    for profile in sensitivity_map["profiles"]:
        params = min(profile["param_count"], MAX_PARAMS)
        cols = min(1024, params)
        rows = max(1, params // cols)
        std = profile.get("weight_std") or 0.02
        w = rng.normal(0, std, (rows, cols)).astype(np.float32)
        layers.append((profile["layer_name"], w))
    return layers

def main():
    parser = argparse.ArgumentParser(
        description="N730 Format Converter"
    )
    parser.add_argument("--model", type=str, help="HuggingFace model path or hub ID")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic weights (shapes from sensitivity map)")
    parser.add_argument("--sensitivity", type=str, default="sensitivity_map.json", help="Path to sensitivity_map.json")
    parser.add_argument("--output", type=str, default="model.n730", help="Output .n730 file path")
    parser.add_argument("--inspect", type=str, help="Inspect an existing .n730 file and exit")
    parser.add_argument("--read-layer", type=int, help="Read and dequantize a specific layer index (requires --inspect)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--force-incompatible", action="store_true",
                         help="Convert anyway even if model_compat.py flags the architecture as unsupported")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════╗")
    print("║  Converter stage to get .n730 binary   ║")
    print("╚══════════════════════════════════════╝")

    if args.inspect:
        inspect_n730(Path(args.inspect))
        if args.read_layer is not None:
            print(f"\n  Reading layer {args.read_layer}...")
            weights = read_layer(Path(args.inspect), args.read_layer)
            print(f"  Shape: {weights.shape}")
            print(f"  Mean:  {weights.mean():.6f}")
            print(f"  Std:   {weights.std():.6f}")
            print(f"  Min:   {weights.min():.6f}  Max: {weights.max():.6f}")
        return
    with open(args.sensitivity) as f:
        sensitivity_map = json.load(f)
    print(f"\n  Sensitivity map : {args.sensitivity}")
    print(f"  Model           : {sensitivity_map['model_id']}")
    print(f"  Layers          : {sensitivity_map['total_layers']}")

    if args.synthetic:
        print("\n  Mode: SYNTHETIC weights")
        weight_layers = generate_synthetic_weights(sensitivity_map)
    elif args.model:
        print(f"\n  Mode: REAL MODEL ({args.model})")
        try:
            import model_compat
            compat = model_compat.check_repo(args.model)
        except Exception as e:
            compat = None
            print(f"  (compat check skipped: {e})")
        if compat is not None:
            print("\n  " + compat.describe().replace("\n", "\n  "))
            if not compat.ok:
                print(
                    "\n  Refusing to convert: this architecture doesn't match what "
                    "N730's kernels assume (RMSNorm + GQA + rotate_half RoPE + SwiGLU).\n"
                    "  Converting anyway would produce a .n730 file that loads but "
                    "generates garbage. Pass --force-incompatible to override."
                )
                if not getattr(args, "force_incompatible", False):
                    sys.exit(1)
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        weight_layers = [
            (name, param.detach().numpy())
            for name, param in model.named_parameters()
            if param.ndim >= 2
            or (param.ndim == 1 and "bias" in name and "norm" not in name.lower())
        ]
    else:
        print("\n  No model specified — using synthetic weights from sensitivity map.")
        weight_layers = generate_synthetic_weights(sensitivity_map)

    output_path = Path(args.output)
    convert(weight_layers, sensitivity_map, output_path, verbose=not args.quiet)

    print(f"\n  To inspect: python converter.py --inspect {output_path}")
    print(f"  To read layer 0: python converter.py --inspect {output_path} --read-layer 0\n")


if __name__ == "__main__":
    main()