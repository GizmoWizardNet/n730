"""
N730 Scheduler (C++ accelerated now vroom vroom)
==================================

Architecture:
    Python (this file)          C++ (n730core.dll/.so)
    ─────────────────────       ──────────────────────────
    Seek table management   →   n730_open()  / n730_close()
    Prefetch thread logic   →   n730_read_layer()  ← THE HOT PATH
    Pipeline / queuing      →   n730_layer_elements()
    Stats / reporting           dequant_int2/4/8/fp16()

Usage:
    python scheduler_cpp.py --model deepseek-r1-1.5b.n730 --benchmark
    python scheduler_cpp.py --model deepseek-r1-1.5b.n730 --layer 42
    python scheduler.py --model deepseek-r1-1.5b.n730 --benchmark --simulate-gpu-ms 50
"""

import ctypes
import json
import os
import platform
import struct
import sys
import threading
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


# ─── Load the C++ core ───────────────────────────────────────────────────────

def _load_n730core() -> ctypes.CDLL:
    """
    Find and load n730core shared library.
    Looks next to this script first, then PATH.
    """
    here = Path(__file__).parent

    if platform.system() == "Windows":
       candidates = [
            here / "cpp" / "n730core.dll",
            here / "n730core.dll",
            Path("n730core.dll")
        ]
    else:
        candidates = [here / "n730core.so", Path("n730core.so")]

    for path in candidates:
        if path.exists():
            lib = ctypes.CDLL(str(path))
            _setup_signatures(lib)
            return lib

    names = [str(c) for c in candidates]
    raise FileNotFoundError(
        f"Cannot find n730core library. Expected one of:\n  " +
        "\n  ".join(names) +
        f"\n\nBuild it first:\n"
        f"  Linux/Mac : g++ -O3 -march=native -shared -fPIC -o n730core.so n730core.cpp\n"
        f"  Windows   : g++ -O3 -march=native -shared -o n730core.dll n730core.cpp"
    )


def _setup_signatures(lib: ctypes.CDLL):
    """Set ctypes argument and return types for every exported function."""

    # int64_t n730_open(const char* path)
    lib.n730_open.argtypes = [ctypes.c_char_p]
    lib.n730_open.restype  = ctypes.c_int64

    # void n730_close(int64_t handle)
    lib.n730_close.argtypes = [ctypes.c_int64]
    lib.n730_close.restype  = None

    # int32_t n730_read_layer(handle, file_offset, out_buffer, out_rows, out_cols, out_prec_id)
    lib.n730_read_layer.argtypes = [
        ctypes.c_int64,                          # handle
        ctypes.c_int64,                          # file_offset
        ctypes.POINTER(ctypes.c_float),          # out_buffer
        ctypes.POINTER(ctypes.c_int32),          # out_rows
        ctypes.POINTER(ctypes.c_int32),          # out_cols
        ctypes.POINTER(ctypes.c_int32),          # out_prec_id
    ]
    lib.n730_read_layer.restype = ctypes.c_int32

    # int32_t n730_layer_elements(handle, file_offset)
    lib.n730_layer_elements.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.n730_layer_elements.restype  = ctypes.c_int32

    # const char* n730_version()
    lib.n730_version.argtypes = []
    lib.n730_version.restype  = ctypes.c_char_p


# ─── Error codes ─────────────────────────────────────────────────────────────

N730_ERRORS = {
    -1: "BAD_MAGIC — not a valid .n730 file",
    -2: "FILE — I/O error",
    -3: "ALLOC — out of memory",
    -4: "BAD_LAYER — corrupt layer block",
    -5: "BAD_PREC — unknown precision id",
    -6: "NULL — null pointer",
}

def check_err(code: int, context: str = ""):
    if code < 0:
        msg = N730_ERRORS.get(code, f"unknown error {code}")
        raise RuntimeError(f"n730core error in {context}: {msg}")


# ─── Data structures ─────────────────────────────────────────────────────────

import numpy as np

@dataclass
class LayerBuffer:
    layer_idx: int
    layer_name: str
    weights: np.ndarray      # float32 (rows, cols)
    precision: str
    load_time_ms: float      # C++ read + dequant combined


@dataclass
class SchedulerStats:
    total_layers: int = 0
    total_bytes_read: int = 0
    total_load_ms: float = 0.0
    gpu_wait_ms: float = 0.0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    peak_ram_layers: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / max(total, 1)

    @property
    def avg_load_ms(self) -> float:
        return self.total_load_ms / max(self.total_layers, 1)

    @property
    def throughput_mb_s(self) -> float:
        if self.total_load_ms < 1:
            return 0.0
        return (self.total_bytes_read / 1024**2) / (self.total_load_ms / 1000)


# ─── Core scheduler ──────────────────────────────────────────────────────────

class N730Scheduler:
    """
    C++-accelerated layer scheduler for .n730 files.

    The C++ core (n730core.dll/.so) handles all file I/O and dequantization.
    This Python class handles the pipeline logic: prefetching, threading,
    stats tracking, and the public API.
    """

    def __init__(self, model_path: str, prefetch: int = 4):
        self.model_path = Path(model_path)
        self.prefetch = prefetch
        self.stats = SchedulerStats()

        # Load C++ core
        self._lib = _load_n730core()
        print(f"  C++ core  : {self._lib.n730_version().decode()}")

        # Open persistent file handle in C++ (stays open for model lifetime)
        self._handle = self._lib.n730_open(str(self.model_path).encode())
        if self._handle < 0:
            check_err(int(self._handle), "n730_open")

        # Load seek table from Python (header parsing stays in Python — readable)
        self._load_header()

        # Pre-allocate a reusable output buffer for the largest layer
        max_elements = max(
            e["rows"] * e["cols"] for e in self._seek_table
        )
        # Each worker thread needs its own buffer — allocate per-thread in stream()
        self._max_elements = max_elements

    def _load_header(self):
        with open(self.model_path, "rb") as f:
            magic = f.read(8)
            if magic != b"N730\x00\x01\x00\x00":
                raise ValueError("Not a valid .n730 file")
            version, header_size = struct.unpack(">II", f.read(8))
            header = json.loads(f.read(header_size))

        self._header = header
        self._seek_table = header["seek_table"]

        print(f"  Model     : {header['model_id']}")
        print(f"  Layers    : {header['total_layers']}")
        print(f"  Size      : {header['stored_mb']} MB  ({header['compression']}× compressed)")
        print(f"  Prefetch  : {self.prefetch} layers")

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_handle") and self._handle > 0:
            self._lib.n730_close(self._handle)

    def _read_layer_cpp(self, entry: dict) -> LayerBuffer:
        """
        Call into C++ to read + dequantize one layer.
        This is where the 74ms → ~1ms speedup lives.
        """
        n_elements = entry["rows"] * entry["cols"]

        # Allocate output buffer (numpy array backed by C-contiguous float32 memory)
        out = np.empty(n_elements, dtype=np.float32)
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        rows_out  = ctypes.c_int32(0)
        cols_out  = ctypes.c_int32(0)
        prec_out  = ctypes.c_int32(0)

        t0 = time.perf_counter()
        rc = self._lib.n730_read_layer(
            self._handle,
            ctypes.c_int64(entry["file_offset"]),
            out_ptr,
            ctypes.byref(rows_out),
            ctypes.byref(cols_out),
            ctypes.byref(prec_out),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        check_err(rc, f"n730_read_layer (layer {entry['layer_idx']})")

        weights = out.reshape(rows_out.value, cols_out.value)

        return LayerBuffer(
            layer_idx=entry["layer_idx"],
            layer_name=entry["layer_name"],
            weights=weights,
            precision=entry["precision"],
            load_time_ms=elapsed_ms,
        )

    def stream(self, start: int = 0, end: Optional[int] = None) -> Iterator[LayerBuffer]:
        """
        Stream layers using a background prefetch thread.
        The prefetch thread calls C++ for each layer; the caller (GPU) just
        pulls from the ready queue.

        NOTE: C++ file handle is not thread-safe for concurrent seeks.
        The prefetch thread gets exclusive access; the main thread only
        reads from the Python queue.
        """
        if end is None:
            end = len(self._seek_table)

        ready_q: queue.Queue = queue.Queue(maxsize=self.prefetch + 1)

        def prefetch_thread():
            for i in range(start, end):
                entry = self._seek_table[i]
                layer = self._read_layer_cpp(entry)

                self.stats.total_layers += 1
                self.stats.total_load_ms += layer.load_time_ms
                self.stats.total_bytes_read += entry["stored_bytes"]

                ready_q.put(layer)

                depth = ready_q.qsize()
                if depth > self.stats.peak_ram_layers:
                    self.stats.peak_ram_layers = depth

            ready_q.put(None)

        t = threading.Thread(target=prefetch_thread, daemon=True)
        t.start()

        while True:
            t_wait = time.perf_counter()
            layer = ready_q.get()
            wait_ms = (time.perf_counter() - t_wait) * 1000

            if layer is None:
                break

            if wait_ms < 2.0:
                self.stats.prefetch_hits += 1
            else:
                self.stats.prefetch_misses += 1
                self.stats.gpu_wait_ms += wait_ms

            yield layer

        t.join()

    def get_layer(self, layer_idx: int) -> LayerBuffer:
        """Random-access single layer. O(1) seek."""
        return self._read_layer_cpp(self._seek_table[layer_idx])

    def print_stats(self):
        s = self.stats
        print("\n" + "═" * 56)
        print("  N730 SCHEDULER STATS  (C++ accelerated)")
        print("═" * 56)
        print(f"  Layers streamed   : {s.total_layers}")
        print(f"  Data read         : {s.total_bytes_read / 1024**2:.1f} MB")
        print(f"  Avg load time     : {s.avg_load_ms:.2f} ms/layer  (read+dequant)")
        print(f"  Throughput        : {s.throughput_mb_s:.1f} MB/s")
        print(f"  Prefetch hit rate : {s.hit_rate * 100:.1f}%")
        print(f"  GPU wait total    : {s.gpu_wait_ms:.1f} ms")
        print(f"  Peak RAM layers   : {s.peak_ram_layers}")
        if s.prefetch_misses == 0:
            print(f"  Pipeline stalls   : 0  ✓ GPU never waited")
        else:
            print(f"  Pipeline stalls   : {s.prefetch_misses}")
            if s.hit_rate < 0.5:
                print(f"  → Disk still the limit. Try --prefetch {self.prefetch + 4}")
        print("═" * 56)


# ─── Simulated forward pass ──────────────────────────────────────────────────

def simulated_forward_pass(layer: LayerBuffer, activations: np.ndarray,
                           simulate_ms: float = 0.0) -> np.ndarray:
    if simulate_ms > 0:
        time.sleep(simulate_ms / 1000)
    w = layer.weights
    in_dim = w.shape[1]
    if activations.shape[0] != in_dim:
        activations = np.ones(in_dim, dtype=np.float32) * 0.01
    return np.tanh(w @ activations)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def run_benchmark(model_path: str, prefetch: int, n_layers: Optional[int],
                  simulate_gpu_ms: float = 0.0):
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║  N730 Scheduler                      ║")
    print(f"╚══════════════════════════════════════╝\n")

    scheduler = N730Scheduler(model_path, prefetch=prefetch)
    total = len(scheduler._seek_table)
    end = min(n_layers, total) if n_layers else total

    gpu_label = f" + {simulate_gpu_ms:.0f}ms GPU" if simulate_gpu_ms else ""
    print(f"\n  Benchmarking {end}/{total} layers  prefetch={prefetch}{gpu_label}\n")

    activations = np.ones(1024, dtype=np.float32) * 0.01
    bar_width = 40
    t_total = time.perf_counter()

    for layer in scheduler.stream(end=end):
        activations = simulated_forward_pass(layer, activations, simulate_gpu_ms)

        done = layer.layer_idx + 1
        filled = int(bar_width * done / end)
        bar = "█" * filled + "░" * (bar_width - filled)
        hit_pct = scheduler.stats.hit_rate * 100
        status = "✓" if hit_pct >= 80 else ("~" if hit_pct >= 40 else "⚠")
        print(
            f"  [{bar}] {done:>3}/{end}  "
            f"[{layer.precision:<4}]  "
            f"{layer.load_time_ms:5.2f}ms  "
            f"hit={hit_pct:.0f}%{status}",
            end="\r"
        )

    elapsed = time.perf_counter() - t_total
    print(f"\n\n  Total wall time: {elapsed:.2f}s")
    scheduler.print_stats()


def read_single_layer(model_path: str, layer_idx: int):
    print(f"\n╔══════════════════════════════════════╗")
    print(f"║  N730 Scheduler                      ║")
    print(f"╚══════════════════════════════════════╝\n")
    scheduler = N730Scheduler(model_path, prefetch=1)
    print(f"\n  Reading layer {layer_idx}...")
    layer = scheduler.get_layer(layer_idx)
    print(f"  Name      : {layer.layer_name}")
    print(f"  Precision : {layer.precision}")
    print(f"  Shape     : {layer.weights.shape}")
    print(f"  Mean/Std  : {layer.weights.mean():.6f} / {layer.weights.std():.6f}")
    print(f"  Min/Max   : {layer.weights.min():.4f} / {layer.weights.max():.4f}")
    print(f"  Load time : {layer.load_time_ms:.3f} ms  (C++ read+dequant)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="N730 Scheduler")
    parser.add_argument("--model", type=str, default="model.n730")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--simulate-gpu-ms", type=float, default=0.0,
                        help="Simulate GPU compute time per layer to test pipeline balance")
    args = parser.parse_args()

    if args.layer is not None:
        read_single_layer(args.model, args.layer)
    else:
        run_benchmark(args.model, args.prefetch, args.layers,
                      getattr(args, "simulate_gpu_ms", 0.0))