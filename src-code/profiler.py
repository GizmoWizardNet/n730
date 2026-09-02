"""
N730 Precision Profiler
=======================

Scores every layer in a transformer model for sensitivity to quantization.
Critical layers keep higher precision (INT8), safe layers get crushed (INT2/INT4).

essentially useless for anything else

Usage:
    python profiler.py --model <path_or_hf_id> --output sensitivity_map.json
    python profiler.py --synthetic --layers 32 --output sensitivity_map.json
"""

import argparse
import json
import math
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


# ─── Data structures ──────────────────────────────────────────────────────────

PRECISION_LEVELS = {
    "INT2": 2,
    "INT4": 4,
    "INT8": 8,
    "FP16": 16,
}

@dataclass
class LayerProfile:
    layer_idx: int
    layer_name: str
    param_count: int           # total parameters in this layer
    weight_std: float          # standard deviation of weights (spread)
    weight_kurtosis: float     # kurtosis: how "spiky" the distribution is
    outlier_ratio: float       # fraction of weights > 3 std devs from mean
    sensitivity_score: float   # 0.0 (safe to crush) → 1.0 (critical, keep precision)
    assigned_precision: str    # INT2 / INT4 / INT8 / FP16
    vram_bytes_original: int   # bytes at FP32
    vram_bytes_n730: int       # bytes at assigned precision
    compression_ratio: float   # how much we shrank it


@dataclass
class SensitivityMap:
    model_id: str
    total_layers: int
    total_params: int
    profiles: list
    vram_original_mb: float
    vram_n730_mb: float
    overall_compression: float
    profiler_version: str = "0.1.0"
    project: str = "N730"


# ─── Core scoring logic ───────────────────────────────────────────────────────

def compute_sensitivity(weights: np.ndarray, layer_idx: int, total_layers: int) -> float:
    """
    Assigns a sensitivity score 0→1 to a weight tensor.

    Three signals combined:
      1. Outlier ratio  — layers with extreme weights are harder to quantize
      2. Kurtosis       — spiky distributions lose more info when rounded
      3. Position bias  — first and last ~15% of layers are empirically more critical
                          (embeddings and final projection matter most)
    """
    # Always float64 — large tensors (lm_head, embed_tokens) overflow float32 arithmetic
    flat = weights.flatten().astype(np.float64)

    # Sample large tensors to keep memory sane (>4M params → sample 4M)
    if flat.size > 4_000_000:
        rng = np.random.default_rng(seed=layer_idx)
        flat = rng.choice(flat, size=4_000_000, replace=False)

    # Signal 1: outlier ratio
    std = float(np.std(flat))
    mean = float(np.mean(flat))
    if std < 1e-9:
        outlier_ratio = 0.0
    else:
        outliers = np.abs(flat - mean) > (3.0 * std)
        outlier_ratio = float(np.mean(outliers))

    # Signal 2: kurtosis (excess)
    if std < 1e-9:
        kurtosis = 0.0
    else:
        normalised = np.clip((flat - mean) / std, -1e6, 1e6)
        kurtosis = float(np.mean(normalised ** 4)) - 3.0  # excess kurtosis

    # Signal 3: position bias
    relative_pos = layer_idx / max(total_layers - 1, 1)
    # U-shaped curve: high at start and end, low in the middle
    position_bias = 1.0 - math.sin(math.pi * relative_pos) ** 2

    # Combine: weights tuned to match empirical quantization research
    outlier_score = min(outlier_ratio * 20.0, 1.0)          # 5% outliers → score 1.0
    kurtosis_score = min(max(kurtosis / 10.0, 0.0), 1.0)    # kurtosis 10 → score 1.0

    sensitivity = (
        0.45 * outlier_score +
        0.30 * kurtosis_score +
        0.25 * position_bias
    )
    return float(np.clip(sensitivity, 0.0, 1.0))


def assign_precision(sensitivity: float, layer_name: str = "") -> str:
    """
    INT8 floor for all layers.

    Why: this implementation uses per-tensor quantization (one scale/zp per matrix).
    Per-tensor INT4 has ~5-10x higher error than group-INT4 (e.g. llama.cpp Q4_0),
    making it too lossy for a 1.5B model where every parameter counts.
    22 of 28 layers were falling to INT4 and compounding error fatally.

    Per-tensor INT8 gives <0.5% weight error and fits easily in 2GB VRAM because
    only one weight matrix is resident at a time (streamed). Peak usage ~88 MB.

    To safely use INT4 in the future: add group quantization (group_size=128)
    so each group gets its own scale — that matches llama.cpp Q4_0 quality.
    """
    return "INT8"


def profile_weight_tensor(
    weights: np.ndarray,
    layer_idx: int,
    layer_name: str,
    total_layers: int,
) -> LayerProfile:
    param_count = int(weights.size)  # full count before any sampling

    # Always float64; sample large tensors (lm_head can be 500M+ params)
    flat = weights.flatten().astype(np.float64)
    if flat.size > 4_000_000:
        rng = np.random.default_rng(seed=layer_idx)
        flat = rng.choice(flat, size=4_000_000, replace=False)

    std = float(np.std(flat))
    mean = float(np.mean(flat))

    if std < 1e-9:
        kurtosis = 0.0
        outlier_ratio = 0.0
    else:
        normalised = np.clip((flat - mean) / std, -1e6, 1e6)
        kurtosis = float(np.mean(normalised ** 4)) - 3.0
        outlier_ratio = float(np.mean(np.abs(flat - mean) > (3.0 * std)))

    sensitivity = compute_sensitivity(weights, layer_idx, total_layers)
    precision = assign_precision(sensitivity, layer_name)
    bits = PRECISION_LEVELS[precision]

    vram_original = param_count * 4          # FP32 = 4 bytes
    vram_n730 = math.ceil(param_count * bits / 8)
    compression = vram_original / max(vram_n730, 1)

    return LayerProfile(
        layer_idx=layer_idx,
        layer_name=layer_name,
        param_count=param_count,
        weight_std=round(std, 6),
        weight_kurtosis=round(kurtosis, 4),
        outlier_ratio=round(outlier_ratio, 6),
        sensitivity_score=round(sensitivity, 4),
        assigned_precision=precision,
        vram_bytes_original=vram_original,
        vram_bytes_n730=vram_n730,
        compression_ratio=round(compression, 2),
    )


# ─── Synthetic model generator (for testing without downloading a model) ──────

def generate_synthetic_layers(num_layers: int, hidden_size: int = 4096):
    """
    Generates weight tensors that realistically mimic a transformer:
    - First layers: tighter distributions (embedding-like)
    - Middle layers: wider, more Gaussian
    - Last layers: spiky with outliers (common in real LLMs)
    """
    layers = []

    for i in range(num_layers):
        relative = i / max(num_layers - 1, 1)

        # Simulate different weight characteristics per position
        if relative < 0.1:
            # Early layers: tight, embedding-like
            w = np.random.normal(0, 0.02, (hidden_size, hidden_size // 4)).astype(np.float32)
        elif relative > 0.88:
            # Final layers: spiky with outliers (common in real models)
            w = np.random.normal(0, 0.04, (hidden_size, hidden_size // 4)).astype(np.float32)
            # Inject outliers
            n_outliers = int(w.size * 0.06)
            idx = np.random.choice(w.size, n_outliers, replace=False)
            w.flat[idx] = np.random.choice([-1, 1], n_outliers) * np.random.uniform(0.3, 0.8, n_outliers)
        else:
            # Middle layers: standard Gaussian
            w = np.random.normal(0, 0.035, (hidden_size, hidden_size // 4)).astype(np.float32)

        layers.append((f"transformer.layer.{i}.weight", w))

    return layers


# ─── Real model loader ────────────────────────────────────────────────────────

def load_model_layers(model_path: str):
    """
    Loads weight tensors from a HuggingFace model (local path or hub ID).
    Yields (name, numpy_array) for each linear layer weight.
    """
    try:
        from transformers import AutoModelForCausalLM
        import torch
    except ImportError:
        raise RuntimeError("transformers and torch required for real model loading")

    print(f"  Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    layers = []
    for name, param in model.named_parameters():
        if param.ndim >= 2:  # only weight matrices, skip biases/norms
            layers.append((name, param.detach().numpy()))

    return layers


# ─── Main profiler ────────────────────────────────────────────────────────────

def run_profiler(
    model_id: str,
    weight_layers: list,
    verbose: bool = True,
) -> SensitivityMap:
    total_layers = len(weight_layers)
    profiles = []

    total_params = 0
    total_vram_original = 0
    total_vram_n730 = 0

    print(f"\n  Profiling {total_layers} layers...\n")

    bar_width = 40
    t0 = time.time()

    for idx, (name, weights) in enumerate(weight_layers):
        profile = profile_weight_tensor(weights, idx, name, total_layers)
        profiles.append(profile)

        total_params += profile.param_count
        total_vram_original += profile.vram_bytes_original
        total_vram_n730 += profile.vram_bytes_n730

        if verbose:
            filled = int(bar_width * (idx + 1) / total_layers)
            bar = "█" * filled + "░" * (bar_width - filled)
            precision_tag = {
                "INT2": "INT2 ██",
                "INT4": "INT4 ████",
                "INT8": "INT8 ████████",
                "FP16": "FP16 ████████████████",
            }[profile.assigned_precision]
            print(
                f"  [{bar}] {idx+1:>3}/{total_layers}  "
                f"score={profile.sensitivity_score:.2f}  "
                f"{precision_tag:<20}  {name[:40]}"
            )

    elapsed = time.time() - t0
    vram_orig_mb = total_vram_original / (1024 ** 2)
    vram_n730_mb = total_vram_n730 / (1024 ** 2)
    overall_compression = vram_orig_mb / max(vram_n730_mb, 0.001)

    print(f"\n  Done in {elapsed:.1f}s")

    return SensitivityMap(
        model_id=model_id,
        total_layers=total_layers,
        total_params=total_params,
        profiles=[asdict(p) for p in profiles],
        vram_original_mb=round(vram_orig_mb, 2),
        vram_n730_mb=round(vram_n730_mb, 2),
        overall_compression=round(overall_compression, 2),
    )


def print_summary(smap: SensitivityMap):
    profiles = smap.profiles
    precision_counts = {"INT2": 0, "INT4": 0, "INT8": 0, "FP16": 0}
    for p in profiles:
        precision_counts[p["assigned_precision"]] += 1

    print("\n" + "═" * 60)
    print(f"  N730 SENSITIVITY REPORT — {smap.model_id}")
    print("═" * 60)
    print(f"  Total layers profiled : {smap.total_layers}")
    print(f"  Total parameters      : {smap.total_params:,}")
    print(f"  Original size (FP32)  : {smap.vram_original_mb:.1f} MB")
    print(f"  N730 size             : {smap.vram_n730_mb:.1f} MB")
    print(f"  Compression ratio     : {smap.overall_compression:.1f}×")
    print()
    print("  Precision breakdown:")
    for prec, count in precision_counts.items():
        pct = count / max(smap.total_layers, 1) * 100
        bar = "█" * int(pct / 3)
        print(f"    {prec:<5}  {count:>3} layers  ({pct:4.1f}%)  {bar}")
    print()

    # Highlight the most and least critical layers
    sorted_profiles = sorted(profiles, key=lambda p: p["sensitivity_score"], reverse=True)
    print("  Most critical layers (protected):")
    for p in sorted_profiles[:3]:
        print(f"    [{p['assigned_precision']}] score={p['sensitivity_score']:.2f}  {p['layer_name'][:55]}")
    print()
    print("  Most compressible layers (aggressive quantization):")
    for p in sorted_profiles[-3:]:
        print(f"    [{p['assigned_precision']}] score={p['sensitivity_score']:.2f}  {p['layer_name'][:55]}")
    print("═" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N730 Precision Profiler"
    )
    parser.add_argument("--model", type=str, help="HuggingFace model path or hub ID")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic model (no download needed)")
    parser.add_argument("--layers", type=int, default=32, help="Number of layers for synthetic mode")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden size for synthetic mode")
    parser.add_argument("--output", type=str, default="sensitivity_map.json", help="Output JSON path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-layer output")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════╗")
    print("║   N730 Profiler yay dangummit        ║")
    print("╚══════════════════════════════════════╝\n")

    if args.synthetic:
        print(f"  Mode: TEST ({args.layers} layers, hidden={args.hidden})")
        model_id = f"synthetic-{args.layers}L-h{args.hidden}"
        layers = generate_synthetic_layers(args.layers, args.hidden)
    elif args.model:
        print(f"  Mode: REAL BOI ({args.model})")
        model_id = args.model
        layers = load_model_layers(args.model)
    else:
        print("  No model specified. Running in test mode (--layers 32).")
        print("  Use --model <path> to profile a real model.\n")
        model_id = "synthetic-32L-h4096"
        layers = generate_synthetic_layers(32, 4096)

    smap = run_profiler(model_id, layers, verbose=not args.quiet)
    print_summary(smap)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(asdict(smap), f, indent=2)

    print(f"\n  Sensitivity map saved → {out_path}\n")


if __name__ == "__main__":
    main()