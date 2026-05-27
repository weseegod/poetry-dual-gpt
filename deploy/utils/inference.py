"""
v4.2.3 Production Inference — Lục Bát đối thơ.
Used by doitho backend. Thin wrapper around src/generation.py.

ALL generation logic comes from src/generation.py — the single canonical
implementation used by eval, CLI, and production. This file only adds:
  - Model loading (load_model)
  - Prompt builders (auto_tag, build_doi_tho_prompt, build_doi_tho_from_lines)
  - Production decode wrapper (decode_doi_tho)

Exports:
  load_model(checkpoint_path, tokenizer_path, device) → (model, tokenizer)
  generate(prompt, model, tokenizer, ...) → dict(luc, bat, lines, token_ids)  [from generation.py]
  auto_tag(prompt) → formatted prompt string
  build_doi_tho_prompt(line6, line8) → formatted couplet prompt
  build_doi_tho_from_lines(lines) → formatted multi-couplet prompt
  decode_doi_tho(tokenizer, new_token_ids, ...) → list of line strings        [wraps generation.py]
"""

import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

# Support both package contexts:
#   poetry-dual-gpt/:          from src.generation import ...
#   doitho/utils/:             from generation import ...  (packed flat)
try:
    from generation import generate as _generate, decode_response
except ImportError:
    from src.generation import generate as _generate, decode_response

try:
    from tones import get_luc_bat_tags, get_doi_tho_tags, get_tram_bong_tag
except ImportError:
    from src.tones import get_luc_bat_tags, get_doi_tho_tags, get_tram_bong_tag

try:
    from model import PoetryDuelGPT
except ImportError:
    from src.model import PoetryDuelGPT


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """Load model and tokenizer."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = PoetryDuelGPT(
        vocab_size=ckpt["vocab_size"],
        n_embd=cfg["n_embd"], n_head=cfg["n_head"],
        n_layer=cfg["n_layer"], block_size=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
    )
    # Key remap for old checkpoints
    old = ckpt["model_state_dict"]
    new_s = {}
    for k, v in old.items():
        nk = k.replace("qkv_proj", "qkv").replace("out_proj", "out")
        nk = nk.replace("causal_mask", "mask")
        nk = nk.replace(".ffn.fc1.", ".ffn.net.0.").replace(".ffn.fc2.", ".ffn.net.2.")
        nk = {"token_embedding.weight": "tok_emb.weight",
              "position_embedding.weight": "pos_emb.weight",
              "ln_final.weight": "ln_f.weight", "ln_final.bias": "ln_f.bias",
              "lm_head.weight": "head.weight"}.get(nk, nk)
        new_s[nk] = v
    model.load_state_dict(new_s, strict=False)
    model.to(device).eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════
#  PROMPT BUILDERS (production-specific wrappers)
# ═══════════════════════════════════════════════════════════

def auto_tag(prompt: str) -> str:
    """Wrap single 6-syl line with [LUC_BAT] + rhyme/tone/trambong tags."""
    p = prompt.strip()
    if p.startswith("[LUC_BAT]"):
        return p
    rhyme, tone, trambong = get_luc_bat_tags(p)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    return f"[LUC_BAT] {tags} {p}" if tags else f"[LUC_BAT] {p}"


def build_doi_tho_prompt(line6: str, line8: str) -> str:
    """Build đối thơ prompt from a full couplet (6+8).
    
    Uses get_doi_tho_tags — rhyme from pos8 of Bát line (chain rhyme),
    matching the training format and src/generation.py.
    """
    rhyme, tone = get_doi_tho_tags(line6, line8)
    trambong = get_tram_bong_tag(line8)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    return f"<|start|> [LUC_BAT] {tags} {line6} <|linebreak|> {line8} <|reply|>"


def build_doi_tho_from_lines(lines: list[str]) -> str:
    """Build prompt from alternating 6/8 syllable lines."""
    if len(lines) < 2:
        return auto_tag(lines[0]) if lines else ""
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1, s2 = len(lines[i].split()), len(lines[i + 1].split())
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    if not couplets:
        return auto_tag(lines[-1]) if lines else ""
    last_a, last_b = couplets[-1]
    rhyme, tone = get_doi_tho_tags(last_a, last_b)
    trambong = get_tram_bong_tag(last_b)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    input_lines = []
    for a, b in couplets:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    return f"<|start|> [LUC_BAT] {tags} {input_str} <|reply|>"


# ═══════════════════════════════════════════════════════════
#  DECODE (thin production wrapper around generation.py)
# ═══════════════════════════════════════════════════════════

def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=False,
                   max_lines=2, is_tn=False):
    """
    Production wrapper around generation.decode_response.
    Returns cleaned lines, optionally enforcing syllable count.
    """
    lines = decode_response(tokenizer, new_token_ids, enforce_syllables=enforce_syllables)

    # Production: enforce max_lines + syllable targets
    targets = (7, 7) if is_tn else (6, 8)
    if len(lines) > max_lines:
        t1, t2 = targets
        for i in range(len(lines) - 1):
            s1 = len(lines[i].split())
            s2 = len(lines[i + 1].split())
            if s1 == t1 and s2 == t2:
                lines = lines[i:i + 2]
                break
        else:
            lines = lines[:max_lines]

    return lines


# ═══════════════════════════════════════════════════════════
#  RE-EXPORT generate from src/generation.py
# ═══════════════════════════════════════════════════════════

# src/generation.py::generate(model, tokenizer, prompt, ...)
# Old doitho callers use: generate(prompt, model, tokenizer, ...)
# This wrapper reorders arguments for backward compatibility.
@torch.no_grad()
def generate(prompt, model, tokenizer, **kwargs):
    """
    Generate đối thơ response. Wraps src/generation.generate().
    Accepts (prompt, model, tokenizer) for backward compat with doitho.
    Remaps max_tokens → max_new for src/generation.py compatibility.
    """
    # Remap doitho's 'max_tokens' → generation.py's 'max_new'
    if 'max_tokens' in kwargs:
        kwargs['max_new'] = kwargs.pop('max_tokens')
    tokens, text = _generate(model, tokenizer, prompt, **kwargs)

    # Build result dict in the format doitho expects
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in tokens:
        if t == lb_id:
            if chunk:
                decoded = tokenizer.decode(chunk)
                decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
                decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
                lines.append(decoded.strip(',.-;:!?()[]{}<> \\t'))
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        decoded = tokenizer.decode(chunk)
        decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
        decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
        lines.append(decoded.strip(',.-;:!?()[]{}<> \\t'))
    lines = [l for l in lines if l]

    return {
        "luc": lines[0] if len(lines) > 0 else "",
        "bat": lines[1] if len(lines) > 1 else "",
        "lines": lines,
        "token_ids": tokens,
        "raw": tokenizer.decode(tokens).replace("<|end|>", "").strip(),
    }
