import argparse, ctypes, math, os, struct, sys
import numpy as np
from pathlib import Path

if os.name == "nt":
    _cuda_bin = os.environ.get("CUDA_BIN_DIR", r"D:\Applications\cuda\bin")
    if hasattr(os, "add_dll_directory") and os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)
sys.path.insert(0, str(Path(__file__).parent))

from inference import (
    _load_cuda_lib, _setup_cuda_sigs, ModelConfig, DeviceBuf, check
)
from scheduler import N730Scheduler


def pull(lib, ctx, ptr, n):
    if n == 0: return np.array([])
    buf = np.zeros(n, np.float32)
    bp  = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.n730_memcpy_d2h(bp, ptr, n)
    return buf

def compare(name, cuda_arr, ref_arr, tol=1e-2):
    diff = np.abs(cuda_arr.flatten() - ref_arr.flatten())
    rms  = float(np.sqrt((diff**2).mean()))
    mx   = float(diff.max())
    ok   = rms < tol
    sym  = "✓" if ok else "✗ MISMATCH"
    print(f"  {sym}  {name:<44} rms={rms:.5f}  max={mx:.5f}")
    return ok

def get_raw_weight(model_path, entry):
    with open(model_path, "rb") as f:
        f.seek(entry["file_offset"])
        f.read(4)
        idx, prec_id, rows, cols, scale, zp, dsz = struct.unpack(">IBIIffi", f.read(25))
        raw = f.read(dsz)
    return raw, entry

def dequant_numpy(raw, entry):
    src = np.frombuffer(raw, dtype=np.uint8)
    n   = entry["rows"] * entry["cols"]
    s   = entry["scale"]; zp = entry["zero_point"]
    if entry["precision"] == "INT8":
        w = (src.astype(np.float32) - zp) * s
    elif entry["precision"] == "INT4":
        lo = (src & 0x0F).astype(np.float32)
        hi = ((src >> 4) & 0x0F).astype(np.float32)
        interleaved = np.empty(len(src)*2, np.float32)
        interleaved[0::2] = lo; interleaved[1::2] = hi
        w = (interleaved[:n] - zp) * s
    else:
        w = src.astype(np.float32)
    return w.reshape(entry["rows"], entry["cols"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--config",   default=None)
    ap.add_argument("--layers",   type=int, default=1,
                    help="How many transformer layers to test (default 1)")
    args = ap.parse_args()

    print("\n══ N730 Full Layer Diagnostic ══\n")

    cfg = ModelConfig(args.config, hf_path=args.hf_model)
    lib = _load_cuda_lib()
    print(f"  {lib.n730_cuda_version()}")
    print(f"  rope_theta={cfg.rope_theta}  rms_norm_eps={cfg.rms_norm_eps}\n")

    from scheduler import N730Scheduler
    sched = N730Scheduler(args.model, prefetch=1)
    name_to_entry = {e["layer_name"]: e for e in sched._seek_table}

    print("  Loading HF reference model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model, dtype=torch.float32,
        device_map="cpu", low_cpu_mem_usage=True)
    sd = {k: v.detach().float().numpy() for k, v in hf.named_parameters()}
    print(f"  HF params loaded: {len(sd)}\n")

    max_w = max(e["rows"] * e["cols"] for e in sched._seek_table)
    ctx_ptr = ctypes.c_void_p(0)
    check(lib.n730_cuda_init(
        cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim,
        cfg.vocab_size, 512, max_w, ctypes.byref(ctx_ptr)), "init")
    ctx = ctx_ptr

    # Upload norm weights
    d_norm = {}
    for name, w in sd.items():
        if "norm" in name.lower() and w.ndim == 1:
            arr = w.astype(np.float32)
            ptr = ctypes.c_void_p(0)
            fp  = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if lib.n730_upload_norm_weight(fp, len(arr), ctypes.byref(ptr)) == 0:
                d_norm[name] = ptr
    ones = np.ones(cfg.hidden_size, np.float32)
    ptr  = ctypes.c_void_p(0)
    fp   = ones.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.n730_upload_norm_weight(fp, cfg.hidden_size, ctypes.byref(ptr))
    d_norm["__identity__"] = ptr

    def norm_ptr(name):
        return d_norm.get(name, d_norm["__identity__"])

    d_cos = ctypes.c_void_p(0); d_sin = ctypes.c_void_p(0)
    check(lib.n730_rope_precompute(
        cfg.max_position_embeddings, cfg.head_dim,
        ctypes.c_float(cfg.rope_theta),
        ctypes.byref(d_cos), ctypes.byref(d_sin)), "rope_precompute")

    tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    token_ids = np.array(tok.encode("2+2="), dtype=np.int32)
    seq = len(token_ids)
    print(f"  Test tokens: {token_ids} (seq={seq})\n")

    h = cfg.hidden_size; nh = cfg.num_attention_heads
    nkv = cfg.num_kv_heads; d = cfg.head_dim
    inter = cfg.intermediate_size
    grp   = nh // nkv

    d_q        = DeviceBuf(lib, 512*nh*d)
    d_k        = DeviceBuf(lib, 512*nkv*d)
    d_v        = DeviceBuf(lib, 512*nkv*d)
    d_attn_out = DeviceBuf(lib, 512*h)
    d_mlp_out  = DeviceBuf(lib, 512*h)
    d_gate     = DeviceBuf(lib, 512*inter)
    d_up       = DeviceBuf(lib, 512*inter)
    d_k_cache  = DeviceBuf(lib, 512*nkv*d)
    d_v_cache  = DeviceBuf(lib, 512*nkv*d)

    def upload_weight(name):
        entry = name_to_entry.get(name)
        if entry is None: return False
        raw, ent = get_raw_weight(args.model, entry)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bp  = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        pid = {"INT4":4,"INT8":8,"FP16":16,"INT2":2}.get(ent["precision"],8)
        return lib.n730_upload_weight(ctx, bp, pid,
            ent["rows"]*ent["cols"],
            ctypes.c_float(ent["scale"]),
            ctypes.c_float(ent["zero_point"])) == 0

    def get_w(name):
        entry = name_to_entry.get(name)
        if entry is None: return None
        raw, ent = get_raw_weight(args.model, entry)
        return dequant_numpy(raw, ent)

    def rms_norm_numpy(x, w, eps):
        rms = np.sqrt((x.astype(np.float64)**2).mean(axis=-1, keepdims=True) + eps)
        return (x / rms * w).astype(np.float32)

    def rope_numpy(x, n_heads, head_dim, offset=0):
        sl = x.shape[0]
        x  = x.reshape(sl, n_heads, head_dim).copy()
        h2 = head_dim // 2
        for s in range(sl):
            pos = s + offset
            for i in range(h2):
                freq  = 1.0 / (cfg.rope_theta ** (2*i / head_dim))
                angle = pos * freq
                c, sv = math.cos(angle), math.sin(angle)
                # These slices are views.  Keep copies of both components before
                # writing either half, otherwise the second assignment would use
                # the already-rotated x0 rather than the original value.
                x0 = x[s,:,i].copy(); x1 = x[s,:,i+h2].copy()
                x[s,:,i]    = x0*c - x1*sv
                x[s,:,i+h2] = x0*sv + x1*c
        return x.reshape(sl, n_heads*head_dim)

    # ── Initialize activations with embeddings ─────────────────────────────
    embed_ref = sd["model.embed_tokens.weight"][token_ids].astype(np.float32)
    x_ptr = embed_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    check(lib.n730_load_activations(ctx, x_ptr, seq, h), "load_act")

    # ── Run N reference layers in parallel ─────────────────────────────────
    x_ref = embed_ref.copy()

    all_ok = True
    for layer_i in range(args.layers):
        pfx = f"model.layers.{layer_i}"
        print(f"── Layer {layer_i} ──────────────────────────────────────────────────────")

        # ── Attention block ──────────────────────────────────────────────────
        norm1_w = sd[f"{pfx}.input_layernorm.weight"]
        norm1   = rms_norm_numpy(x_ref, norm1_w, cfg.rms_norm_eps)

        check(lib.n730_rmsnorm(ctx, norm_ptr(f"{pfx}.input_layernorm.weight"),
                               seq, ctypes.c_float(cfg.rms_norm_eps)), "rmsnorm1")

        # Q/K/V
        q_w = get_w(f"{pfx}.self_attn.q_proj.weight")
        k_w = get_w(f"{pfx}.self_attn.k_proj.weight")
        v_w = get_w(f"{pfx}.self_attn.v_proj.weight")
        q_ref = norm1 @ q_w.T; k_ref = norm1 @ k_w.T; v_ref = norm1 @ v_w.T

        upload_weight(f"{pfx}.self_attn.q_proj.weight")
        check(lib.n730_linear(ctx, d_q.ptr, seq, h, nh*d), "q_proj")
        upload_weight(f"{pfx}.self_attn.k_proj.weight")
        check(lib.n730_linear(ctx, d_k.ptr, seq, h, nkv*d), "k_proj")
        upload_weight(f"{pfx}.self_attn.v_proj.weight")
        check(lib.n730_linear(ctx, d_v.ptr, seq, h, nkv*d), "v_proj")

        # The projections must agree before attention.  A tiny error here is
        # amplified by softmax, so do not hide it behind the output check.
        all_ok &= compare(f"L{layer_i} q_proj", pull(lib, ctx, d_q.ptr, seq*nh*d), q_ref)
        all_ok &= compare(f"L{layer_i} k_proj", pull(lib, ctx, d_k.ptr, seq*nkv*d), k_ref)
        all_ok &= compare(f"L{layer_i} v_proj", pull(lib, ctx, d_v.ptr, seq*nkv*d), v_ref)

        # RoPE
        q_rope = rope_numpy(q_ref, nh,  d, offset=0)
        k_rope = rope_numpy(k_ref, nkv, d, offset=0)
        check(lib.n730_apply_rope(d_q.ptr, d_cos, d_sin, seq, nh,  d, 0), "rope_q")
        check(lib.n730_apply_rope(d_k.ptr, d_cos, d_sin, seq, nkv, d, 0), "rope_k")

        all_ok &= compare(f"L{layer_i} q_rope", pull(lib, ctx, d_q.ptr, seq*nh*d), q_rope)
        all_ok &= compare(f"L{layer_i} k_rope", pull(lib, ctx, d_k.ptr, seq*nkv*d), k_rope)

        # KV cache
        check(lib.n730_memcpy_d2d(d_k_cache.ptr, d_k.ptr, seq*nkv*d), "kv_k")
        check(lib.n730_memcpy_d2d(d_v_cache.ptr, d_v.ptr, seq*nkv*d), "kv_v")

        # Attention
        check(lib.n730_attention_forward(
            ctx,
            ctypes.cast(d_q.ptr,       ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(d_k_cache.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(d_v_cache.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(d_attn_out.ptr,ctypes.POINTER(ctypes.c_float)),
            seq, seq, nh, nkv, d, 0), "attn_fwd")
        attn_cuda = pull(lib, ctx, d_attn_out.ptr, seq*nh*d).reshape(seq, nh*d)

        # Reference attention
        q_r = q_rope.reshape(seq,nh,d); k_r = k_rope.reshape(seq,nkv,d)
        v_r = v_ref.reshape(seq,nkv,d)
        k_r = np.repeat(k_r,grp,axis=1); v_r = np.repeat(v_r,grp,axis=1)
        sc  = np.einsum("shd,thd->sht",q_r,k_r) / math.sqrt(d)
        mask = np.tril(np.ones((seq,seq)))
        sc   = np.where(mask[None,:,:].transpose(1,0,2)>0, sc, -1e4)
        sc  -= sc.max(axis=-1,keepdims=True)
        wa   = np.exp(sc); wa /= wa.sum(axis=-1,keepdims=True)+1e-9
        attn_ref = np.einsum("sht,thd->shd",wa,v_r).reshape(seq,nh*d).astype(np.float32)

        # Scores are stored head-major by the CUDA implementation.
        score_cuda = np.empty(nh * seq * seq, dtype=np.float32)
        lib.n730_get_scores(ctx, score_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nh, seq, seq)
        all_ok &= compare(f"L{layer_i} attn probabilities", score_cuda.reshape(nh, seq, seq).transpose(1, 0, 2), wa)

        all_ok &= compare(f"L{layer_i} attn out", attn_cuda, attn_ref)

        # O-proj + residual
        o_w   = get_w(f"{pfx}.self_attn.o_proj.weight")
        o_ref = attn_ref @ o_w.T
        upload_weight(f"{pfx}.self_attn.o_proj.weight")
        check(lib.n730_linear_from_buf(ctx,
            ctypes.cast(d_attn_out.ptr, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(d_mlp_out.ptr,  ctypes.POINTER(ctypes.c_float)),
            seq, h, h), "o_proj")
        o_cuda = pull(lib, ctx, d_mlp_out.ptr, seq*h).reshape(seq, h)
        all_ok &= compare(f"L{layer_i} o_proj", o_cuda, o_ref)

        check(lib.n730_residual_add(ctx, d_mlp_out.ptr, seq), "resid1")
        resid1_buf = np.zeros(seq*h, np.float32)
        check(lib.n730_get_activations(ctx,
            resid1_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), seq, h), "get_act")
        resid1_cuda = resid1_buf.reshape(seq, h)
        resid1_ref  = (x_ref + o_ref).astype(np.float32)
        all_ok &= compare(f"L{layer_i} resid1", resid1_cuda, resid1_ref)

        print(f"  MLP block:")
        norm2_w = sd[f"{pfx}.post_attention_layernorm.weight"]
        norm2   = rms_norm_numpy(resid1_ref, norm2_w, cfg.rms_norm_eps)

        check(lib.n730_rmsnorm(ctx, norm_ptr(f"{pfx}.post_attention_layernorm.weight"),
                               seq, ctypes.c_float(cfg.rms_norm_eps)), "rmsnorm2")

        # Gate + Up projections
        gate_w = get_w(f"{pfx}.mlp.gate_proj.weight")
        up_w   = get_w(f"{pfx}.mlp.up_proj.weight")
        gate_ref = norm2 @ gate_w.T
        up_ref   = norm2 @ up_w.T

        upload_weight(f"{pfx}.mlp.gate_proj.weight")
        check(lib.n730_linear(ctx, d_gate.ptr, seq, h, inter), "gate_proj")
        upload_weight(f"{pfx}.mlp.up_proj.weight")
        check(lib.n730_linear(ctx, d_up.ptr,   seq, h, inter), "up_proj")

        gate_cuda = pull(lib, ctx, d_gate.ptr, seq*inter).reshape(seq, inter)
        up_cuda   = pull(lib, ctx, d_up.ptr,   seq*inter).reshape(seq, inter)

        all_ok &= compare(f"L{layer_i} gate_proj", gate_cuda, gate_ref)
        all_ok &= compare(f"L{layer_i} up_proj",   up_cuda,   up_ref)

        # SwiGLU reference: silu(gate) * up
        silu_ref  = gate_ref / (1.0 + np.exp(-gate_ref))
        swiglu_ref = (silu_ref * up_ref).astype(np.float32)

        check(lib.n730_swiglu(d_gate.ptr, d_up.ptr, seq, inter), "swiglu")
        swiglu_cuda = pull(lib, ctx, d_gate.ptr, seq*inter).reshape(seq, inter)
        all_ok &= compare(f"L{layer_i} swiglu", swiglu_cuda, swiglu_ref)

        # Down projection
        down_w   = get_w(f"{pfx}.mlp.down_proj.weight")
        down_ref = swiglu_ref @ down_w.T

        upload_weight(f"{pfx}.mlp.down_proj.weight")
        check(lib.n730_linear_from_buf(ctx,
            ctypes.cast(d_gate.ptr,     ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(d_attn_out.ptr, ctypes.POINTER(ctypes.c_float)),
            seq, inter, h), "down_proj")
        down_cuda = pull(lib, ctx, d_attn_out.ptr, seq*h).reshape(seq, h)
        all_ok &= compare(f"L{layer_i} down_proj", down_cuda, down_ref)

        check(lib.n730_residual_add(ctx, d_attn_out.ptr, seq), "resid2")
        resid2_buf = np.zeros(seq*h, np.float32)
        check(lib.n730_get_activations(ctx,
            resid2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), seq, h), "get_act2")
        resid2_cuda = resid2_buf.reshape(seq, h)
        resid2_ref  = (resid1_ref + down_ref).astype(np.float32)
        all_ok &= compare(f"L{layer_i} resid2 (full layer out)", resid2_cuda, resid2_ref)

        # Advance reference state for next layer
        x_ref = resid2_ref

    print(f"\n══ {'ALL PASS' if all_ok else 'FAILURES DETECTED'} ══\n")
    lib.n730_cuda_destroy(ctx)

if __name__ == "__main__":
    main()
