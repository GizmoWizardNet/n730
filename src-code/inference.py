"""
Usage:
  python inference.py --model model_file.n730 \
    --hf-model hf-model-id --interactive
"""
import os
if os.name == "nt":
    _cuda_bin = os.environ.get("CUDA_BIN_DIR", r"D:\Applications\cuda\bin")
    if hasattr(os, "add_dll_directory") and os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)
import argparse
import ctypes
import json
import struct
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

try:
    from scheduler import N730Scheduler
    SCHED = "C++"
except ImportError:
    from scheduler import N730Scheduler
    SCHED = "Python"

def _load_cuda_lib() -> ctypes.CDLL:
    here = Path(__file__).parent
    search_dirs = [here, here / "cpp"]
    names = ["n730_cuda.dll", "n730_cuda.so"]
    for name in names:
        for dir in search_dirs:
            p = dir / name
            if p.exists():
                lib = ctypes.CDLL(str(p))
                _setup_cuda_sigs(lib)
                return lib
    raise FileNotFoundError(
        "n730_cuda.dll/.so not found. Build with:\n"
        "  nvcc -O3 -arch=sm_35 --shared -lcublas -o n730_cuda.dll n730_cuda.cu\n"
        "  Linux: nvcc -O3 -arch=sm_35 -shared -Xcompiler -fPIC -lcublas "
        "-o n730_cuda.so n730_cuda.cu"
    )

def _setup_cuda_sigs(lib):
    vp  = ctypes.c_void_p
    fp  = ctypes.POINTER(ctypes.c_float)
    bp  = ctypes.POINTER(ctypes.c_uint8)
    i32 = ctypes.c_int32
    f32 = ctypes.c_float
    pp  = ctypes.POINTER(ctypes.c_void_p)

    lib.n730_cuda_init.argtypes = [i32,i32,i32,i32,i32,i32,pp]
    lib.n730_cuda_init.restype  = i32

    lib.n730_cuda_destroy.argtypes = [vp]
    lib.n730_cuda_destroy.restype  = None

    lib.n730_load_activations.argtypes = [vp, fp, i32, i32]
    lib.n730_load_activations.restype  = i32

    lib.n730_get_activations.argtypes = [vp, fp, i32, i32]
    lib.n730_get_activations.restype  = i32

    lib.n730_upload_weight.argtypes = [vp, bp, i32, i32, f32, f32]
    lib.n730_upload_weight.restype  = i32

    lib.n730_upload_norm_weight.argtypes = [fp, i32, pp]
    lib.n730_upload_norm_weight.restype  = i32

    lib.n730_free_device_buf.argtypes = [vp]
    lib.n730_free_device_buf.restype  = None

    lib.n730_rmsnorm.argtypes = [vp, vp, i32, f32]
    lib.n730_rmsnorm.restype  = i32

    lib.n730_linear.argtypes = [vp, vp, i32, i32, i32]
    lib.n730_linear.restype  = i32

    # New: linear from arbitrary device buffer (no norm_buf dependency)
    lib.n730_linear_from_buf.argtypes = [vp, fp, fp, i32, i32, i32]
    lib.n730_linear_from_buf.restype  = i32

    lib.n730_residual_add.argtypes = [vp, vp, i32]
    lib.n730_residual_add.restype  = i32

    lib.n730_add_bias.argtypes = [vp, fp, fp, i32, i32]
    lib.n730_add_bias.restype  = i32

    lib.n730_swiglu.argtypes = [vp, vp, i32, i32]
    lib.n730_swiglu.restype  = i32

    lib.n730_apply_rope.argtypes = [vp, vp, vp, i32, i32, i32, i32]
    lib.n730_apply_rope.restype  = i32

    lib.n730_rope_precompute.argtypes = [i32, i32, f32, pp, pp]
    lib.n730_rope_precompute.restype  = i32

    lib.n730_softmax_scores.argtypes = [vp, i32, i32, i32, i32]
    lib.n730_softmax_scores.restype  = i32

    # New: full GPU attention forward (no CPU roundtrip)
    lib.n730_attention_forward.argtypes = [
        vp,   # ctx
        fp,   # d_q
        fp,   # d_k_cache
        fp,   # d_v_cache
        fp,   # d_out
        i32,  # seq_q
        i32,  # seq_total
        i32,  # n_heads
        i32,  # n_kv_heads
        i32,  # head_dim
        i32,  # cache_offset
    ]
    lib.n730_attention_forward.restype = i32

    lib.n730_device_alloc.argtypes = [i32, pp]
    lib.n730_device_alloc.restype  = i32

    lib.n730_device_free.argtypes = [vp]
    lib.n730_device_free.restype  = None

    lib.n730_memcpy_d2d.argtypes = [vp, vp, i32]
    lib.n730_memcpy_d2d.restype  = i32

    lib.n730_memcpy_h2d.argtypes = [vp, fp, i32]
    lib.n730_memcpy_h2d.restype  = i32

    lib.n730_memcpy_d2h.argtypes = [fp, vp, i32]
    lib.n730_memcpy_d2h.restype  = i32

    lib.n730_sync.argtypes = []
    lib.n730_sync.restype  = i32

    lib.n730_cuda_version.argtypes = []
    lib.n730_cuda_version.restype  = ctypes.c_char_p


def check(rc, where=""):
    if rc < 0:
        raise RuntimeError(f"CUDA kernel error {rc} in {where}")

class ModelConfig:
    def __init__(self, path=None, hf_path=None):
        self.hidden_size             = 1536
        self.num_hidden_layers       = 28
        self.num_attention_heads     = 12
        self.num_kv_heads            = 2
        self.head_dim                = 128
        self.intermediate_size       = 8960
        self.vocab_size              = 151936
        self.rms_norm_eps            = 1e-6
        self.rope_theta              = 10000.0
        self.max_position_embeddings = 131072

        if path and Path(path).exists():
            with open(path) as f:
                c = json.load(f)
            self._apply(c)

        if hf_path:
            c = None
            local_cfg = Path(hf_path) / "config.json"
            if local_cfg.exists():
                with open(local_cfg) as f:
                    c = json.load(f)
            else:
                try:
                    from huggingface_hub import hf_hub_download
                    resolved = hf_hub_download(repo_id=hf_path, filename="config.json")
                    with open(resolved) as f:
                        c = json.load(f)
                except Exception as e:
                    print(f"  WARNING: could not resolve config.json for "
                          f"hf_path={hf_path!r} ({e}); falling back to "
                          f"built-in defaults (hidden={self.hidden_size}, "
                          f"heads={self.num_attention_heads}, "
                          f"vocab={self.vocab_size}) — this will likely "
                          f"crash or silently corrupt output if the actual "
                          f"model doesn't match those shapes.")
            if c is not None:
                self._apply(c)
                try:
                    import model_compat
                    compat = model_compat.check(c)
                    if not compat.ok:
                        print(f"  WARNING: {compat.architecture} is not a supported "
                              f"architecture — output is likely garbage:")
                        for r in compat.reasons:
                            print(f"    ✗ {r}")
                    for w in compat.warnings:
                        print(f"  NOTE: {w}")
                except Exception:
                    pass  # compat check is best-effort, never block inference on it

    def _apply(self, c):
        self.hidden_size         = c.get("hidden_size",             self.hidden_size)
        self.num_hidden_layers   = c.get("num_hidden_layers",       self.num_hidden_layers)
        self.num_attention_heads = c.get("num_attention_heads",     self.num_attention_heads)
        self.num_kv_heads        = c.get("num_key_value_heads",     self.num_kv_heads)
        self.intermediate_size   = c.get("intermediate_size",       self.intermediate_size)
        self.vocab_size          = c.get("vocab_size",              self.vocab_size)
        self.rms_norm_eps        = c.get("rms_norm_eps",            self.rms_norm_eps)
        self.rope_theta          = c.get("rope_theta",              self.rope_theta)
        self.max_position_embeddings = c.get("max_position_embeddings", self.max_position_embeddings)
        self.head_dim            = self.hidden_size // self.num_attention_heads

class DeviceBuf:
    """Owns a VRAM allocation. Freed on garbage collection."""
    def __init__(self, lib, n_floats):
        self._lib = lib
        self._ptr = ctypes.c_void_p(0)
        check(lib.n730_device_alloc(n_floats, ctypes.byref(self._ptr)), "device_alloc")
        self.n = n_floats

    @property
    def ptr(self): return self._ptr

    def fp(self):
        """Return as POINTER(c_float) for linear_from_buf calls."""
        return ctypes.cast(self._ptr, ctypes.POINTER(ctypes.c_float))

    def __del__(self):
        if self._ptr:
            self._lib.n730_device_free(self._ptr)
            self._ptr = ctypes.c_void_p(0)

class VRAMKVCache:
    def __init__(self, lib, n_layers, n_kv_heads, head_dim, max_seq=512):
        self._lib   = lib
        self.n_layers = n_layers
        self.n_kv   = n_kv_heads
        self.d      = head_dim
        self.max_seq = max_seq
        self.seq_len = 0

        elems = max_seq * n_kv_heads * head_dim
        self.k_bufs = [DeviceBuf(lib, elems) for _ in range(n_layers)]
        self.v_bufs = [DeviceBuf(lib, elems) for _ in range(n_layers)]

        ram_mb = 2 * n_layers * elems * 4 / 1024**2
        print(f"  KV cache VRAM: {ram_mb:.1f} MB ({max_seq} token max)")

    def ptr_k(self, layer): return self.k_bufs[layer].ptr
    def ptr_v(self, layer): return self.v_bufs[layer].ptr
    def fp_k(self, layer):  return self.k_bufs[layer].fp()
    def fp_v(self, layer):  return self.v_bufs[layer].fp()

    def clear(self):
        self.seq_len = 0

    @property
    def vram_mb(self):
        elems = self.max_seq * self.n_kv * self.d
        return 2 * self.n_layers * elems * 4 / 1024**2

class N730CudaTransformer:

    def __init__(self, model_path: str, hf_path: Optional[str],
                 cfg: ModelConfig, prefetch: int = 8):
        self.cfg  = cfg
        self.lib  = _load_cuda_lib()
        print(f"  {self.lib.n730_cuda_version().decode()}")
        print(f"  rope_theta = {cfg.rope_theta:.0f}  max_pos = {cfg.max_position_embeddings}")
        if cfg.rope_theta < 100000:
            print(f"  WARNING: rope_theta={cfg.rope_theta} looks wrong for this model!")
            print(f"  WARNING: Use --config or pass --hf-model to auto-load correct value.")

        print(f"  Loading weights ({SCHED} scheduler)...")
        self._load_weights(model_path, prefetch)

        self.norm_w: Dict[str, np.ndarray] = {}
        self._load_norms(hf_path)

        max_w = max(e["rows"] * e["cols"] for e in self._seek_table)
        ctx_ptr = ctypes.c_void_p(0)
        check(self.lib.n730_cuda_init(
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.head_dim,
            cfg.vocab_size,
            512,
            max_w,
            ctypes.byref(ctx_ptr),
        ), "cuda_init")
        self.ctx = ctx_ptr

        self.d_norm: Dict[str, ctypes.c_void_p] = {}
        self._upload_norms()
        self.d_bias: Dict[str, Optional[ctypes.c_void_p]] = {}

        self.d_cos = ctypes.c_void_p(0)
        self.d_sin = ctypes.c_void_p(0)
        check(self.lib.n730_rope_precompute(
            cfg.max_position_embeddings,
            cfg.head_dim,
            ctypes.c_float(cfg.rope_theta),
            ctypes.byref(self.d_cos),
            ctypes.byref(self.d_sin),
        ), "rope_precompute")

        h, d = cfg.hidden_size, cfg.head_dim
        inter = cfg.intermediate_size
        kv = cfg.num_kv_heads
        nh = cfg.num_attention_heads
        self.d_q        = DeviceBuf(self.lib, 512 * nh * d)
        self.d_k        = DeviceBuf(self.lib, 512 * kv * d)
        self.d_v        = DeviceBuf(self.lib, 512 * kv * d)
        self.d_gate     = DeviceBuf(self.lib, 512 * inter)
        self.d_up       = DeviceBuf(self.lib, 512 * inter)
        self.d_attn_out = DeviceBuf(self.lib, 512 * h)   # attention output + o_proj output
        self.d_mlp_out  = DeviceBuf(self.lib, 512 * h)   # down_proj output

        # KV cache in VRAM
        self.kv = VRAMKVCache(self.lib, cfg.num_hidden_layers,
                              cfg.num_kv_heads, cfg.head_dim)

    def _load_weights(self, model_path: str, prefetch: int):
        sched = N730Scheduler(model_path, prefetch=prefetch)
        self._sched = sched
        self._seek_table = sched._seek_table
        self._name_to_entry: Dict[str, dict] = {
            e["layer_name"]: e for e in self._seek_table
        }
        self._model_path = model_path
        # Cache for embed/lm_head (CPU side only — these are looked up by name not index)
        self._float_cache: Dict[str, np.ndarray] = {}
        self._validate_model_layout()
        print(f"  Weight index built ({len(self._seek_table)} layers)")

    def _validate_model_layout(self):
        """Reject incomplete/incompatible files before stale GPU buffers can be used."""
        cfg = self.cfg
        required = ["model.embed_tokens.weight"]
        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"
            required.extend([
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx}.mlp.gate_proj.weight",
                f"{pfx}.mlp.up_proj.weight",
                f"{pfx}.mlp.down_proj.weight",
            ])
        missing = [name for name in required if name not in self._name_to_entry]
        if missing:
            preview = ", ".join(missing[:4])
            suffix = " ..." if len(missing) > 4 else ""
            raise ValueError(f"Model file is missing {len(missing)} required tensors: "
                             f"{preview}{suffix}")

    def _get_raw_weight(self, name: str):
        """Read raw quantized bytes for a layer directly from .n730 file."""
        entry = self._name_to_entry.get(name)
        if entry is None:
            return None, None
        with open(self._model_path, "rb") as f:
            f.seek(entry["file_offset"])
            f.read(4)  # magic
            idx, prec_id, rows, cols, scale, zp, dsz = struct.unpack(
                ">IBIIffi", f.read(25))
            raw = f.read(dsz)
        return raw, entry

    def _load_norms(self, hf_path):
        if not hf_path:
            print("  Norm weights: identity (--hf-model for real quality)")
            return
        print(f"  Loading norm weights...")
        try:
            import torch
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(
                hf_path, dtype=torch.float32,
                device_map="cpu", low_cpu_mem_usage=True)
            n = 0
            for name, p in m.named_parameters():
                if "norm" in name.lower() and p.ndim == 1:
                    self.norm_w[name] = p.detach().numpy().astype(np.float32)
                    n += 1
            del m
            print(f"  Loaded {n} norm vectors")
        except Exception as e:
            print(f"  Norm load failed: {e}")

    def _upload_norms(self):
        cfg = self.cfg
        for name, w in self.norm_w.items():
            arr = w.astype(np.float32)
            ptr = ctypes.c_void_p(0)
            fp  = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rc  = self.lib.n730_upload_norm_weight(fp, len(arr), ctypes.byref(ptr))
            if rc == 0:
                self.d_norm[name] = ptr
        ones = np.ones(cfg.hidden_size, dtype=np.float32)
        ptr = ctypes.c_void_p(0)
        fp  = ones.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.n730_upload_norm_weight(fp, cfg.hidden_size, ctypes.byref(ptr))
        self.d_norm["__identity__"] = ptr
        print(f"  Uploaded {len(self.d_norm)} norm vectors to VRAM")

    def _norm_ptr(self, name: str) -> ctypes.c_void_p:
        return self.d_norm.get(name, self.d_norm["__identity__"])

    def _upload_layer_weight(self, name: str) -> bool:
        """Upload one weight matrix to GPU d_weights buffer via pinned staging."""
        raw, entry = self._get_raw_weight(name)
        if raw is None or entry is None:
            return False
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        bp = raw_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        prec_id = {"INT4": 4, "INT8": 8, "FP16": 16, "INT2": 2}.get(
            entry["precision"], 8)
        n_elem = entry["rows"] * entry["cols"]
        rc = self.lib.n730_upload_weight(
            self.ctx, bp,
            ctypes.c_int32(prec_id),
            ctypes.c_int32(n_elem),
            ctypes.c_float(entry["scale"]),
            ctypes.c_float(entry["zero_point"]),
        )
        return rc == 0

    def _bias_ptr(self, name: str) -> Optional[ctypes.c_void_p]:
        if name in self.d_bias:
            return self.d_bias[name]

        raw, entry = self._get_raw_weight(name)
        if raw is None or entry is None:
            self.d_bias[name] = None
            return None

        if entry["precision"] != "FP16":
            self.d_bias[name] = None
            return None

        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        ptr = ctypes.c_void_p(0)
        fp = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = self.lib.n730_upload_norm_weight(fp, len(arr), ctypes.byref(ptr))
        self.d_bias[name] = ptr if rc == 0 else None
        return self.d_bias[name]

    def _add_bias_if_present(self, weight_name: str, d_out, seq: int, dim: int):
        """weight_name is the *.weight name; looks up the matching *.bias."""
        bias_name = weight_name[:-len(".weight")] + ".bias"
        bp = self._bias_ptr(bias_name)
        if bp is not None:
            check(self.lib.n730_add_bias(self.ctx, d_out, bp, seq, dim),
                  f"add_bias({bias_name})")

    def _get_float_weight(self, name: str) -> Optional[np.ndarray]:
        """Get a weight as float32 numpy array — only used for embed/lm_head."""
        if name in self._float_cache:
            return self._float_cache[name]
        entry = self._name_to_entry.get(name)
        if entry is None:
            return None
        layer = self._sched.get_layer(entry["layer_idx"])
        w = layer.weights.astype(np.float32)
        self._float_cache[name] = w
        return w

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        cfg    = self.cfg
        token_ids = np.asarray(token_ids, dtype=np.int32)
        seq    = len(token_ids)
        offset = self.kv.seq_len

        if seq == 0:
            raise ValueError("forward requires at least one token")
        if offset + seq > self.kv.max_seq:
            raise ValueError(f"KV cache capacity exceeded: {offset + seq} tokens "
                             f"requested, maximum is {self.kv.max_seq}")

        # ── Embed: CPU lookup → upload to VRAM ────────────────────────────
        embed_w = self._get_float_weight("model.embed_tokens.weight")
        x_cpu = embed_w[token_ids].astype(np.float32) if embed_w is not None \
                else np.zeros((seq, cfg.hidden_size), np.float32)
        x_ptr = x_cpu.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        check(self.lib.n730_load_activations(self.ctx, x_ptr, seq, cfg.hidden_size),
              "load_activations")

        # ── Transformer layers — all math on GPU ──────────────────────────
        for i in range(cfg.num_hidden_layers):
            pfx = f"model.layers.{i}"

            # ── Attention block ────────────────────────────────────────────
            # RMSNorm → d_norm_buf  (d_activations untouched for residual)
            check(self.lib.n730_rmsnorm(
                self.ctx,
                self._norm_ptr(f"{pfx}.input_layernorm.weight"),
                seq, ctypes.c_float(cfg.rms_norm_eps),
            ), "rmsnorm_attn")

            # Q projection: d_norm_buf → d_q
            if self._upload_layer_weight(f"{pfx}.self_attn.q_proj.weight"):
                check(self.lib.n730_linear(
                    self.ctx, self.d_q.ptr, seq,
                    cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim,
                ), "q_proj")
                self._add_bias_if_present(
                    f"{pfx}.self_attn.q_proj.weight", self.d_q.fp(),
                    seq, cfg.num_attention_heads * cfg.head_dim)

            # K projection: d_norm_buf → d_k
            if self._upload_layer_weight(f"{pfx}.self_attn.k_proj.weight"):
                check(self.lib.n730_linear(
                    self.ctx, self.d_k.ptr, seq,
                    cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim,
                ), "k_proj")
                self._add_bias_if_present(
                    f"{pfx}.self_attn.k_proj.weight", self.d_k.fp(),
                    seq, cfg.num_kv_heads * cfg.head_dim)

            # V projection: d_norm_buf → d_v
            if self._upload_layer_weight(f"{pfx}.self_attn.v_proj.weight"):
                check(self.lib.n730_linear(
                    self.ctx, self.d_v.ptr, seq,
                    cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim,
                ), "v_proj")
                self._add_bias_if_present(
                    f"{pfx}.self_attn.v_proj.weight", self.d_v.fp(),
                    seq, cfg.num_kv_heads * cfg.head_dim)

            # RoPE on Q and K — in place
            check(self.lib.n730_apply_rope(
                self.d_q.ptr, self.d_cos, self.d_sin,
                seq, cfg.num_attention_heads, cfg.head_dim, offset,
            ), "rope_q")
            check(self.lib.n730_apply_rope(
                self.d_k.ptr, self.d_cos, self.d_sin,
                seq, cfg.num_kv_heads, cfg.head_dim, offset,
            ), "rope_k")

            # Append K/V to KV cache (device-to-device copy into cache slot)
            kv_elem    = seq * cfg.num_kv_heads * cfg.head_dim
            cache_off  = offset * cfg.num_kv_heads * cfg.head_dim  # in floats
            check(self.lib.n730_memcpy_d2d(
                ctypes.c_void_p(self.kv.ptr_k(i).value + cache_off * 4),
                self.d_k.ptr, kv_elem,
            ), "kv_append_k")
            check(self.lib.n730_memcpy_d2d(
                ctypes.c_void_p(self.kv.ptr_v(i).value + cache_off * 4),
                self.d_v.ptr, kv_elem,
            ), "kv_append_v")

            total_seq = offset + seq

            # ── Full GPU attention (Q@K^T + softmax + @V) — no CPU ─────────
            # d_attn_out receives (seq, n_heads * head_dim) result
            check(self.lib.n730_attention_forward(
                self.ctx,
                self.d_q.fp(),           # Q (already RoPE'd)
                self.kv.fp_k(i),         # full K cache for this layer
                self.kv.fp_v(i),         # full V cache
                self.d_attn_out.fp(),    # output
                seq, total_seq,
                cfg.num_attention_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                offset,
            ), "attention_forward")

            # O-projection: d_attn_out → d_mlp_out (reuse as temp), then residual
            # Upload o_proj weight → d_weights, then GEMM from d_attn_out → d_mlp_out
            if self._upload_layer_weight(f"{pfx}.self_attn.o_proj.weight"):
                check(self.lib.n730_linear_from_buf(
                    self.ctx,
                    self.d_attn_out.fp(),   # input:  (seq, hidden)
                    self.d_mlp_out.fp(),    # output: (seq, hidden)
                    seq,
                    cfg.hidden_size,        # in_dim  = n_heads * head_dim = hidden
                    cfg.hidden_size,        # out_dim = hidden
                ), "o_proj")
                self._add_bias_if_present(
                    f"{pfx}.self_attn.o_proj.weight", self.d_mlp_out.fp(),
                    seq, cfg.hidden_size)
                # Residual: d_activations += d_mlp_out (o_proj result)
                check(self.lib.n730_residual_add(
                    self.ctx, self.d_mlp_out.ptr, seq,
                ), "residual_attn")
            else:
                # No o_proj weights — still add raw attention output to residual
                check(self.lib.n730_residual_add(
                    self.ctx, self.d_attn_out.ptr, seq,
                ), "residual_attn_raw")

            # ── MLP block ──────────────────────────────────────────────────
            # RMSNorm → d_norm_buf
            check(self.lib.n730_rmsnorm(
                self.ctx,
                self._norm_ptr(f"{pfx}.post_attention_layernorm.weight"),
                seq, ctypes.c_float(cfg.rms_norm_eps),
            ), "rmsnorm_mlp")

            # Gate projection: d_norm_buf → d_gate
            if self._upload_layer_weight(f"{pfx}.mlp.gate_proj.weight"):
                check(self.lib.n730_linear(
                    self.ctx, self.d_gate.ptr, seq,
                    cfg.hidden_size, cfg.intermediate_size,
                ), "gate_proj")
                self._add_bias_if_present(
                    f"{pfx}.mlp.gate_proj.weight", self.d_gate.fp(),
                    seq, cfg.intermediate_size)

            # Up projection: d_norm_buf → d_up
            if self._upload_layer_weight(f"{pfx}.mlp.up_proj.weight"):
                check(self.lib.n730_linear(
                    self.ctx, self.d_up.ptr, seq,
                    cfg.hidden_size, cfg.intermediate_size,
                ), "up_proj")
                self._add_bias_if_present(
                    f"{pfx}.mlp.up_proj.weight", self.d_up.fp(),
                    seq, cfg.intermediate_size)

            # SwiGLU: d_gate = silu(d_gate) * d_up  — stays on GPU
            check(self.lib.n730_swiglu(
                self.d_gate.ptr, self.d_up.ptr,
                seq, cfg.intermediate_size,
            ), "swiglu")

            # Down projection: d_gate → d_attn_out (reuse as temp)
            # Upload down_proj weight → d_weights, GEMM from d_gate → d_attn_out
            if self._upload_layer_weight(f"{pfx}.mlp.down_proj.weight"):
                check(self.lib.n730_linear_from_buf(
                    self.ctx,
                    self.d_gate.fp(),       # input:  (seq, intermediate_size)
                    self.d_attn_out.fp(),   # output: (seq, hidden)
                    seq,
                    cfg.intermediate_size,  # in_dim
                    cfg.hidden_size,        # out_dim
                ), "down_proj")
                self._add_bias_if_present(
                    f"{pfx}.mlp.down_proj.weight", self.d_attn_out.fp(),
                    seq, cfg.hidden_size)
                # Residual: d_activations += d_attn_out (down_proj result)
                check(self.lib.n730_residual_add(
                    self.ctx, self.d_attn_out.ptr, seq,
                ), "residual_mlp")

        # Update KV cache position
        self.kv.seq_len += seq

        # ── Final norm + LM head on CPU ───────────────────────────────────
        x_out = np.zeros(seq * cfg.hidden_size, np.float32)
        xp = x_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        check(self.lib.n730_get_activations(self.ctx, xp, seq, cfg.hidden_size),
              "get_activations")
        last = x_out.reshape(seq, cfg.hidden_size)[-1]

        # Final RMSNorm
        w = self.norm_w.get("model.norm.weight",
                            np.ones(cfg.hidden_size, np.float32))
        rms = np.sqrt((last.astype(np.float64)**2).mean() + cfg.rms_norm_eps)
        last = (last / rms * w).astype(np.float32)

        # LM head
        lmh = self._get_float_weight("lm_head.weight")
        # Most Llama-family checkpoints tie the output embedding to
        # model.embed_tokens.weight, so named_parameters() stores it once.
        if lmh is None:
            lmh = embed_w
        if lmh is not None:
            return last @ lmh.T
        raise ValueError("Model has neither lm_head.weight nor model.embed_tokens.weight")


# ─── Sampling ─────────────────────────────────────────────────────────────────

def sample(logits: np.ndarray, temp: float = 0.7, top_p: float = 0.9) -> int:
    if temp <= 0: return int(np.argmax(logits))
    logits = (logits.astype(np.float32) - logits.max()) / temp
    probs  = np.exp(logits); probs /= probs.sum()
    idx    = np.argsort(probs)[::-1]
    cum    = np.cumsum(probs[idx])
    cut    = int(np.searchsorted(cum, top_p)) + 1
    idx    = idx[:cut]; p = probs[idx]; p /= p.sum()
    return int(np.random.choice(idx, p=p))

class Tokenizer:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.eos = self.tok.eos_token_id
        print(f"  Tokenizer: vocab={self.tok.vocab_size} eos={self.eos}")
    def encode(self, t): return np.array(self.tok.encode(t), dtype=np.int32)
    def encode_chat(self, prompt):
        """Encode a user turn for instruction-tuned checkpoints when possible."""
        if getattr(self.tok, "chat_template", None):
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            # Transformers versions differ: some return a token list while
            # newer releases return a BatchEncoding mapping.
            if hasattr(ids, "get") and ids.get("input_ids") is not None:
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return np.asarray(ids, dtype=np.int32)
        return self.encode(prompt)
    def decode(self, ids): return self.tok.decode(list(ids), skip_special_tokens=False)

class N730Generator:
    def __init__(self, model_path, hf_path, config_path=None, prefetch=8):
        print(f"\n╔═══════════════════════════════════════════╗")
        print(f"║  N730 CUDA Inference                        ║")
        print(f"╚═══════════════════════════════════════════╝\n")
        self.cfg = ModelConfig(config_path, hf_path=hf_path)
        self.transformer = N730CudaTransformer(
            model_path, hf_path, self.cfg, prefetch)
        self.tok = Tokenizer(hf_path or
                             "deepseek-ai/deepseek-r1-distill-qwen-1.5b")
        print(f"\n  ✓ Ready on GT 730\n")

    def generate(self, prompt, max_tokens=200, temp=0.7, top_p=0.9):
        self.transformer.kv.clear()
        ids = self.tok.encode_chat(prompt)
        print(f"\n  You: {prompt}")
        print(f"  N730: ", end="", flush=True)
        generated = []
        t0 = time.perf_counter()
        logits = self.transformer.forward(ids)
        tok = sample(logits, temp, top_p)
        generated.append(tok)
        print(self.tok.decode([tok]), end="", flush=True)
        prefill_s = time.perf_counter() - t0
        t_dec = time.perf_counter()
        for _ in range(max_tokens - 1):
            if tok == self.tok.eos: break
            logits = self.transformer.forward(np.array([tok], np.int32))
            tok = sample(logits, temp, top_p)
            generated.append(tok)
            print(self.tok.decode([tok]), end="", flush=True)
        n = max(len(generated)-1, 1)
        tps = n / max(time.perf_counter()-t_dec, 0.001)
        print(f"\n\n  [{len(generated)} tokens | prefill {prefill_s:.1f}s | "
              f"{tps:.2f} tok/s | KV {self.transformer.kv.vram_mb:.1f}MB VRAM]\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True)
    ap.add_argument("--hf-model",    default=None)
    ap.add_argument("--config",      default=None)
    ap.add_argument("--prompt",      default="Hello")
    ap.add_argument("--max-tokens",  type=int,   default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p",       type=float, default=0.9)
    ap.add_argument("--prefetch",    type=int,   default=8)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    gen = N730Generator(args.model, args.hf_model, args.config, args.prefetch)

    if args.interactive:
        print("  Type 'quit' to exit\n")
        while True:
            try:
                p = input("  You: ").strip()
                if p.lower() in ("quit","exit","q"): break
                if p: gen.generate(p, args.max_tokens, args.temperature, args.top_p)
            except KeyboardInterrupt:
                print("\n  Bye."); break
    else:
        gen.generate(args.prompt, args.max_tokens, args.temperature, args.top_p)

if __name__ == "__main__":
    main()