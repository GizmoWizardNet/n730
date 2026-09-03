"""
model_compat.py — one source of truth for "will N730 actually work on this
HF model", used by converter.py, inference.py, diagnose_cancer.py, and
setup.py so they can never silently disagree with each other.

N730's CUDA kernels hardcode a specific block shape:
    RMSNorm -> GQA self-attention (rotate_half RoPE) -> RMSNorm -> SwiGLU MLP
That shape is shared by the "Llama family" (Llama 2/3, Mistral, Qwen2/2.5,
StableLM 2, SmolLM/SmolLM2, Yi, and most fine-tunes of those). It is NOT
shared by GPT-2/Falcon (LayerNorm + learned-abs-pos + fused qkv), Phi
(partial rotary), Gemma (GeGLU + extra norms + soft-capping), Mixtral
(MoE routing), or anything using ALiBi.

Rather than silently feeding an incompatible model through the pipeline and
producing confident-looking garbage, every entry point calls check() first
and refuses (or warns) with a specific, actionable reason.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    "StableLmForCausalLM",
}

# architectures whose config.json satisfies SUPPORTED_ARCHITECTURES on paper
# but which are known to break one of N730's hardcoded assumptions anyway.
KNOWN_INCOMPATIBLE_NOTES = {
    "MixtralForCausalLM": "uses MoE routing (num_local_experts) — N730 has no router/expert-dispatch kernel.",
    "Phi3ForCausalLM": "uses partial_rotary_factor + fused qkv_proj — different attention layout.",
    "GemmaForCausalLM": "uses GeGLU MLP + extra pre/post feedforward norms, not plain SwiGLU.",
    "Gemma2ForCausalLM": "uses GeGLU + logit soft-capping + sliding/full attention alternation.",
    "FalconForCausalLM": "uses fused qkv + parallel attn/MLP — different block structure entirely.",
    "GPT2LMHeadModel": "uses LayerNorm + learned absolute position embeddings, not RMSNorm+RoPE.",
    "GPTNeoXForCausalLM": "uses LayerNorm + (usually) interleaved RoPE — different norm & rotation.",
}


@dataclass
class CompatResult:
    ok: bool
    architecture: str = "unknown"
    reasons: list[str] = field(default_factory=list)   # hard blockers
    warnings: list[str] = field(default_factory=list)  # works, but flag it

    def describe(self) -> str:
        lines = [f"architecture: {self.architecture}"]
        if self.ok:
            lines.append("compatible: yes")
        else:
            lines.append("compatible: NO")
        for r in self.reasons:
            lines.append(f"  ✗ {r}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


def check(config: dict) -> CompatResult:
    archs = config.get("architectures") or []
    arch = archs[0] if archs else config.get("model_type", "unknown")
    res = CompatResult(ok=True, architecture=arch)

    if arch not in SUPPORTED_ARCHITECTURES:
        note = KNOWN_INCOMPATIBLE_NOTES.get(arch)
        reason = f"'{arch}' is not in the supported family (Llama/Qwen2/Mistral/StableLM2)."
        if note:
            reason += f" Specifically: {note}"
        res.ok = False
        res.reasons.append(reason)
        return res  # no point checking finer details on an unsupported arch

    hidden_act = config.get("hidden_act", "silu")
    if hidden_act not in ("silu", "swish"):
        res.ok = False
        res.reasons.append(
            f"hidden_act={hidden_act!r} — N730's MLP kernel is hardcoded SwiGLU (silu), "
            f"this model's activation won't compute the same function."
        )

    if config.get("rope_interleaved"):
        res.ok = False
        res.reasons.append(
            "rope_interleaved=true — this model uses GPT-NeoX-style interleaved RoPE; "
            "N730's kernel implements the rotate_half convention only."
        )

    partial_rotary = config.get("partial_rotary_factor", 1.0)
    if partial_rotary != 1.0:
        res.ok = False
        res.reasons.append(
            f"partial_rotary_factor={partial_rotary} — N730 always rotates the full head_dim, "
            f"this model only rotates a fraction of it."
        )

    if config.get("num_local_experts") or config.get("num_experts"):
        res.ok = False
        res.reasons.append("MoE config detected (num_local_experts/num_experts) — no expert-routing kernel in N730.")

    hidden_size = config.get("hidden_size")
    n_heads = config.get("num_attention_heads")
    explicit_head_dim = config.get("head_dim")
    if hidden_size and n_heads:
        implicit = hidden_size // n_heads
        if explicit_head_dim and explicit_head_dim != implicit:
            res.ok = False
            res.reasons.append(
                f"head_dim={explicit_head_dim} in config but hidden_size/num_attention_heads="
                f"{implicit} — N730 always derives head_dim as hidden_size//num_attention_heads, "
                f"so this model needs an explicit head_dim override that the pipeline doesn't plumb through yet."
            )

    if config.get("sliding_window"):
        res.warnings.append(
            f"sliding_window={config['sliding_window']} set in config — N730 does full (non-windowed) "
            f"attention, so behavior will diverge from the reference model on long sequences."
        )

    if config.get("tie_word_embeddings"):
        res.warnings.append("tie_word_embeddings=true — handled (lm_head falls back to embed_tokens), just noting it.")

    return res


def check_repo(repo_id_or_path: str) -> CompatResult:
    """Resolve a local dir OR a HF Hub repo id to config.json, then check()."""
    import json
    from pathlib import Path

    local_cfg = Path(repo_id_or_path) / "config.json"
    if local_cfg.exists():
        with open(local_cfg) as f:
            return check(json.load(f))

    from huggingface_hub import hf_hub_download
    resolved = hf_hub_download(repo_id=repo_id_or_path, filename="config.json")
    with open(resolved) as f:
        return check(json.load(f))