"""
v4.2.3 Production Inference — Lục Bát đối thơ.
Used by doitho backend. Single file, no training deps.

Key v4.2.3 improvements over v4.1:
  - Soft rhyme constraint (logit boost +2.0 instead of hard masking)
  - Top-p nucleus sampling (top_p=0.92)
  - Repetition penalty (-1.2) in all paths
  - Control token suppression prevents BPE artifact leaks
  - Strict linebreak syllable enforcement (6+8 for production)

Exports:
  load_model(checkpoint_path, tokenizer_path, device) → (model, tokenizer)
  generate(prompt, model, tokenizer, ...) → dict(luc, bat, lines, token_ids)
  auto_tag(prompt) → formatted prompt string
  build_doi_tho_prompt(line6, line8) → formatted couplet prompt
  build_doi_tho_from_lines(lines) → formatted multi-couplet prompt
  decode_doi_tho(tokenizer, new_token_ids, ...) → list of line strings
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
    top_p: float = 0.92,
    max_tokens: int = 64,
) -> dict:
    """
    v4.2.3: Generate đối thơ response from a formatted prompt.
    
    Uses soft rhyme (logit boost +2.0), soft linebreak bias, top-p nucleus
    sampling, and only suppresses <|pad|> and <|start|> — the model naturally
    won't generate control tokens during content generation.
    
    Args:
        prompt: formatted prompt string (from build_doi_tho_prompt or auto_tag)
        model: loaded PoetryDuelGPT model
        tokenizer: ByteLevel BPE tokenizer
        temperature: sampling temperature (0.75 default)
        top_k: top-k filtering (50 default)
        top_p: nucleus filtering (0.92 default)
        max_tokens: max tokens to generate
    
    Returns:
        dict with 'luc' (6-syl line), 'bat' (8-syl line), 'lines', 'token_ids'
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    start_id = tokenizer.token_to_id("<|start|>")
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    reply_id = tokenizer.token_to_id("<|reply|>")
    device = next(model.parameters()).device
    
    # v4.2.3: Only suppress <|pad|> and <|start|> — these should never appear.
    # Do NOT suppress [RHYME:*], [TONE:*], [TRAMBONG:*], <|reply|> — the model
    # won't generate them naturally (they only appear in the prompt prefix).
    # Aggressive suppression redistributes probability onto rare ByteLevel BPE
    # characters (e.g. ')', '(', '.', ',') causing visible garbage in output.
    suppress_ids = {pad_id}
    if start_id is not None:
        suppress_ids.add(start_id)
    
    # Parse target rhyme for constraint
    target_rhyme = None
    rhyme_match = re.search(r'\[RHYME:([^\]]+)\]', prompt)
    if rhyme_match:
        target_rhyme = rhyme_match.group(1)

    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_tokens = []
    lb_emitted = False  # track if first linebreak has been emitted
    
    for _ in range(max_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        
        # Suppress only <|pad|> and <|start|> — bare minimum safety
        for tid in suppress_ids:
            logits[:, tid] = float("-inf")

        # v4.2.3: Soft linebreak BIAS instead of hard FORCE.
        # Bias <|linebreak|> at position ~6, bias <|end|> at position ~8.
        # The model can still override if it has a strong semantic preference.
        after = new_tokens
        if lb_emitted:
            # In second line: bias <|end|> around syllable 8
            last_lb = max(i for i, t in enumerate(new_tokens) if t == lb_id)
            after = new_tokens[last_lb + 1:]
            decoded_after = tokenizer.decode(after)
            syl_count2 = len(decoded_after.strip().split()) if decoded_after.strip() else 0
            if syl_count2 < 6:
                logits[:, end_id] = float("-inf")  # don't stop before 6 syl
            elif 6 <= syl_count2 < 8:
                logits[:, end_id] += 1.0  # soft bias toward stopping
                logits[:, lb_id] = float("-inf")  # no extra linebreaks
            elif syl_count2 >= 8:
                logits[:, end_id] += 3.0  # strong bias to stop at 8
                logits[:, lb_id] = float("-inf")
        else:
            # In first line: bias <|linebreak|> around syllable 5-7
            decoded_after = tokenizer.decode(after)
            syl_count = len(decoded_after.strip().split()) if decoded_after.strip() else 0
            if syl_count < 4:
                logits[:, lb_id] = float("-inf")
                logits[:, end_id] = float("-inf")
            elif 4 <= syl_count < 6:
                logits[:, lb_id] += 1.0  # soft bias
                logits[:, end_id] = float("-inf")
            elif syl_count >= 6:
                logits[:, lb_id] += 3.0  # strong bias after 6
                logits[:, end_id] = float("-inf")

        # Repetition penalty — penalize tokens from recent output (last 16)
        for prev in new_tokens[-16:]:
            logits[:, prev] -= 1.2

        # P2 v4.2.3: Soft rhyme constraint at pos6 of output Lục line
        if target_rhyme is not None:
            # Count syllables since last delimiter
            last_delim = -1
            for delim_id in [lb_id, reply_id]:
                if delim_id is not None and delim_id in new_tokens:
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
                    # v4.2.3 soft: boost matching, allow override if semantically better
                    rhyme_logit_boost = 2.0
                    for tid_i in matching:
                        logits[:, tid_i] += rhyme_logit_boost
                    
                    # Safety valve: if model is very uncertain about ALL candidates
                    # (flat distribution), fall back to hard masking to prevent randomness
                    probs_check = F.softmax(logits.clone(), dim=-1)
                    max_prob = probs_check.max().item()
                    if max_prob < 0.03:
                        for tid_i in non_matching:
                            logits[:, tid_i] = float("-inf")

        # Top-k
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        # Top-p (nucleus) — v4.2.3: enabled by default
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
        if next_id == lb_id:
            lb_emitted = True
        new_tokens.append(next_id)
        idx = torch.cat((idx, torch.tensor([[next_id]], device=device)), dim=1)

    # Decode output — split on linebreaks, then clean
    lines = []
    chunk = []
    for t in new_tokens:
        if t == lb_id:
            if chunk:
                decoded = tokenizer.decode(chunk)
                # Clean: remove control tokens + trailing punctuation
                decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
                decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
                lines.append(decoded.strip(',.-;:!?()[]{}<> \t'))
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        decoded = tokenizer.decode(chunk)
        decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
        decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
        lines.append(decoded.strip(',.-;:!?()[]{}<> \t'))
    lines = [l for l in lines if l]  # remove empty

    return {
        "luc": lines[0] if len(lines) > 0 else "",
        "bat": lines[1] if len(lines) > 1 else "",
        "lines": lines,
        "token_ids": new_tokens,
        "raw": tokenizer.decode(new_tokens).replace("<|end|>", "").strip(),
    }


def decode_doi_tho(tokenizer, new_token_ids, enforce_syllables=False, max_lines=2, is_tn=False):
    """
    v4.2.3: Decode generated token IDs into cleaned lines.
    Splits on <|linebreak|>, strips control tokens and trailing punctuation.
    """
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in new_token_ids:
        if t == lb_id:
            if chunk:
                decoded = tokenizer.decode(chunk)
                decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
                decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
                lines.append(decoded.strip(',.-;:!?()[]{}<> \t'))
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        decoded = tokenizer.decode(chunk)
        decoded = decoded.replace('<|end|>', '').replace('<|reply|>', '')
        decoded = decoded.replace('<|start|>', '').replace('<|pad|>', '')
        lines.append(decoded.strip(',.-;:!?()[]{}<> \t'))
    lines = [l for l in lines if l]

    if not lines:
        return []

    targets = (7, 7) if is_tn else (6, 8)

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
