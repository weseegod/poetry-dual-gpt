"""
v4.1 Production Inference — Lục Bát đối thơ.
Used by doitho backend. Single file, no training deps.

Exports:
  load_model(checkpoint_path, tokenizer_path, device) → (model, tokenizer)
  generate(prompt, model, tokenizer, ...) → (luc_line, bat_line)
  auto_tag(prompt) → formatted prompt string
"""

import re
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# Support both direct import (from utils/) and relative import (from package)
try:
    from .model import PoetryDuelGPT
except ImportError:
    from model import PoetryDuelGPT

try:
    from .tones import get_luc_bat_tags, get_rhyme_group
except ImportError:
    from tones import get_luc_bat_tags, get_rhyme_group


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, tokenizer_path: str, device: str = "cpu"):
    """Load model and tokenizer. Caches in module globals for reuse."""
    global _model, _tokenizer
    if "_model" in globals() and _model is not None:
        return _model, _tokenizer

    _tokenizer = Tokenizer.from_file(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    _model = PoetryDuelGPT(
        vocab_size=ckpt["vocab_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.to(device).eval()
    return _model, _tokenizer


# ═══════════════════════════════════════════════════════════════
#  PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════

def auto_tag(prompt: str) -> str:
    """
    Wrap single 6-syl line with [LUC_BAT] + rhyme/tone/trambong tags.
    For couplet input, use build_doi_tho_prompt() instead.
    """
    p = prompt.strip()
    if p.startswith("[LUC_BAT]"):
        return p
    rhyme, tone, trambong = get_luc_bat_tags(p)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    return f"[LUC_BAT] {tags} {p}" if tags else f"[LUC_BAT] {p}"


def build_doi_tho_prompt(line6: str, line8: str) -> str:
    """
    Build đối thơ prompt from a full couplet.
    Format: <|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH]
              6-syl <|linebreak|> 8-syl <|reply|>
    """
    rhyme, tone, trambong = get_luc_bat_tags(line6 + ' ' + line8)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    return f"<|start|> [LUC_BAT] {tags} {line6} <|linebreak|> {line8} <|reply|>"


def build_doi_tho_from_lines(lines: list[str]) -> str:
    """
    Build prompt from a list of alternating 6/8 syllable lines.
    E.g. [l6, l8, l6, l8] → last couplet used for tags.
    """
    if len(lines) < 2:
        return auto_tag(lines[0]) if lines else ""
    
    # Group into couplets
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1, s2 = len(lines[i].split()), len(lines[i+1].split())
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    
    if not couplets:
        return auto_tag(lines[-1]) if lines else ""
    
    # Use last couplet for tags
    last_a, last_b = couplets[-1]
    rhyme, tone, trambong = get_luc_bat_tags(last_a + ' ' + last_b)
    tags = ' '.join(t for t in [rhyme, tone, trambong] if t)
    
    # Build input with all couplets
    input_lines = []
    for a, b in couplets:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    
    return f"<|start|> [LUC_BAT] {tags} {input_str} <|reply|>"


# ═══════════════════════════════════════════════════════════════
#  GENERATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    prompt: str,
    model: PoetryDuelGPT,
    tokenizer: Tokenizer,
    temperature: float = 0.75,
    top_k: int = 50,
    top_p: float = None,
    max_tokens: int = 64,
) -> dict:
    """
    Generate đối thơ response from a formatted prompt.
    
    Args:
        prompt: formatted prompt string (from build_doi_tho_prompt or auto_tag)
        model: loaded PoetryDuelGPT model
        tokenizer: ByteLevel BPE tokenizer
        temperature: sampling temperature (0.75 default)
        top_k: top-k filtering (50 default)
        top_p: nucleus filtering (None = off)
        max_tokens: max tokens to generate
    
    Returns:
        dict with 'luc' (6-syl line), 'bat' (8-syl line), 'raw_tokens'
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    device = next(model.parameters()).device
    
    # Parse target rhyme for constraint
    target_rhyme = None
    rhyme_match = re.search(r'\[RHYME:([^\]]+)\]', prompt)
    if rhyme_match:
        target_rhyme = rhyme_match.group(1)

    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_tokens = []

    for _ in range(max_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        logits[:, pad_id] = float("-inf")

        # Repetition penalty
        for prev in new_tokens[-16:]:
            logits[:, prev] -= 1.2

        # P1: Rhyme constraint at pos6 of output Lục line
        if target_rhyme is not None:
            # Count syllables since last <|linebreak|> or <|reply|>
            last_delim = -1
            for delim_id in [lb_id, tokenizer.token_to_id("<|reply|>")]:
                if delim_id in new_tokens:
                    pos = max(i for i, t in enumerate(new_tokens) if t == delim_id)
                    last_delim = max(last_delim, pos)
            after_delim = new_tokens[last_delim + 1:] if last_delim >= 0 else new_tokens
            decoded_after = tokenizer.decode(after_delim)
            current_syl_count = len(decoded_after.strip().split()) if decoded_after.strip() else 0

            if current_syl_count == 5:  # 6th syllable (0-indexed)
                candidate_k = min(100, logits.size(-1))
                _, topk_idx = torch.topk(logits, candidate_k)
                matching, non_matching = [], []
                for tid in topk_idx[0]:
                    tid_i = tid.item()
                    if tid_i in (end_id, pad_id, lb_id):
                        continue
                    decoded = tokenizer.decode([tid_i]).strip()
                    if not decoded:
                        continue
                    if get_rhyme_group(decoded) == target_rhyme:
                        matching.append(tid_i)
                    else:
                        non_matching.append(tid_i)
                if matching:
                    for tid_i in non_matching:
                        logits[:, tid_i] = float("-inf")

        # Top-k
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        # Top-p (nucleus)
        if top_p is not None:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            logits[:, sorted_idx[mask]] = float("-inf")

        # Sample
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if next_id == end_id:
            break
        new_tokens.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    # Decode output — split on linebreaks
    lines = []
    chunk = []
    for t in new_tokens:
        if t == lb_id:
            if chunk:
                lines.append(tokenizer.decode(chunk).strip())
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        lines.append(tokenizer.decode(chunk).strip())

    return {
        "luc": lines[0] if len(lines) > 0 else "",
        "bat": lines[1] if len(lines) > 1 else "",
        "lines": lines,
        "token_ids": new_tokens,
        "raw": tokenizer.decode(new_tokens).replace("<|end|>", "").strip(),
    }


def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=False, max_lines=2, is_tn=False):
    """
    Decode generated token IDs into lines, splitting on <|linebreak|>.
    v4.1: Lục Bát only. No T2a re-split.
    """
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in new_token_ids:
        if t == lb_id:
            if chunk:
                lines.append(tokenizer.decode(chunk).strip())
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        lines.append(tokenizer.decode(chunk).strip())

    if not lines:
        return []

    targets = (6, 8)

    if enforce_syllables:
        for i, line in enumerate(lines):
            words = line.split()
            target = targets[i % 2]
            if len(words) > target:
                words = words[:target]
            lines[i] = ' '.join(words)

    if len(lines) > max_lines:
        t1, t2 = targets
        for i in range(len(lines) - 1):
            s1 = len(lines[i].split())
            s2 = len(lines[i+1].split())
            if s1 == t1 and s2 == t2:
                lines = lines[i:i+2]
                break
        else:
            lines = lines[:max_lines]

    return lines
