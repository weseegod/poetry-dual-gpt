"""
v4.2: Canonical generation module — single source of truth for all generation paths.

Used by: src/sample.py (CLI), client/server.py (API), evaluate/eval_rules.py (benchmark)

Features:
  - Soft rhyme constraint (P2): logit boost instead of hard masking
  - Repetition penalty: -1.2 for recent 16 tokens
  - Top-k + Top-p filtering
  - Generate-and-rerank (P4): pick best of N candidates
  - BPE artifact detection v2 (P5): dictionary-based syllable validity
  - Unified prompt building with <|start|> + [TRAMBONG:NH/HN]
"""

import re
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional, Set

ROOT = Path(__file__).parent.parent

# Import from siblings (handles both PYTHONPATH=src and PYTHONPATH=.)
try:
    from tones import (
        get_luc_bat_tags, get_doi_tho_tags, get_rhyme_group, get_tram_bong_tag
    )
except ImportError:
    from src.tones import (
        get_luc_bat_tags, get_doi_tho_tags, get_rhyme_group, get_tram_bong_tag
    )


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS — token IDs (matching poetry_bpe.model)
# ═══════════════════════════════════════════════════════════════

PAD_ID = 0
START_ID = 1
REPLY_ID = 2
END_ID = 3
LB_ID = 9


# ═══════════════════════════════════════════════════════════════
#  P5: BPE ARTIFACT DETECTION v2 — dictionary-based
# ═══════════════════════════════════════════════════════════════

# Lazy-loaded set of valid Vietnamese syllables from training corpus
_valid_syllables: Optional[Set[str]] = None


def _build_valid_syllables() -> Set[str]:
    """Extract all unique syllables from doi_tho_corpus.txt."""
    corpus_path = ROOT / "data" / "doi_tho_corpus.txt"
    syllables: Set[str] = set()
    
    # Also include a static fallback set of very common Vietnamese syllables
    # in case corpus isn't available
    common = set("""
        em anh tôi ta mình nó họ chúng ta các những này kia đó đây
        là và với cho của nhưng nên vì tại bởi trong ngoài trên dưới
        có không đã đang sẽ vẫn cũng mới từng chưa hãy đừng
        một hai ba bốn năm sáu bảy tám chín mười
        người nhà cửa đường sông núi biển trời đất
        nắng mưa gió mây trăng sao hoa lá cây cỏ
        xuân hạ thu đông ngày đêm sáng chiều tối
        cha mẹ con cái anh chị em cháu
        yêu thương nhớ mong buồn vui giận hờn
        đi đến về ra vào lên xuống qua lại
        ăn uống ngủ nghỉ nói cười khóc hát
        xinh đẹp tốt xấu cao thấp dài ngắn
        non nước quê hương làng xóm đồng ruộng
        thơ ca nhạc họa đàn tranh sáo
        vàng xanh đỏ tím trắng đen hồng
    """.split())
    
    try:
        if corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Extract content portion (skip control tokens)
                    # Format: <|start|> [LUC_BAT] [RHYME:...] ... <|reply|> ... <|end|>
                    # Get everything between <|reply|> and <|end|>
                    parts = line.split("<|reply|>")
                    content = parts[-1] if len(parts) > 1 else line
                    content = content.replace("<|end|>", "").replace("<|linebreak|>", " ")
                    # Also get input portion
                    input_part = parts[0] if len(parts) > 1 else ""
                    input_part = re.sub(r'\[[^\]]+\]', ' ', input_part)
                    input_part = input_part.replace("<|start|>", "").replace("<|linebreak|>", " ")
                    content = input_part + " " + content
                    # Extract words
                    for word in content.split():
                        word = word.strip('.,!?;:-"\'()[] \t')
                        if word and len(word) >= 2:
                            syllables.add(word.lower())
    
    except Exception:
        pass
    
    # Merge with common set
    syllables.update(common)
    
    # Remove control tokens that might have been picked up
    control = {'pad', 'start', 'end', 'linebreak', 'reply', 'luc_bat', 'that_ngon',
               'doi_tho', 'trambong', 'rhyme', 'tone'}
    syllables = {s for s in syllables if s not in control and not s.startswith('<') 
                 and not s.startswith('[')}
    
    return syllables


def get_valid_syllables() -> Set[str]:
    global _valid_syllables
    if _valid_syllables is None:
        _valid_syllables = _build_valid_syllables()
    return _valid_syllables


def is_bpe_artifact(syllable: str) -> bool:
    """
    P5: Check if a syllable is a BPE subword fragment.
    Returns True if syllable is NOT a real Vietnamese morpheme.
    """
    if not syllable or len(syllable) < 2:
        return True
    
    # Strip punctuation
    clean = syllable.strip('.,!?;:-"\'()[] \t\n\r')
    if not clean:
        return False  # pure punctuation is fine
    
    # Must contain at least one Vietnamese character
    has_vn_char = any(c in "aăâeêioôơuưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỵỷỹ"
                      "AĂÂEÊIOÔƠUƯYÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỴỶỸ"
                      for c in clean.lower() if c.isalpha())
    if not has_vn_char:
        return False  # numbers, pure symbols
    
    # Check against known Vietnamese syllables
    valid = get_valid_syllables()
    if clean.lower() not in valid:
        # Check if it looks like a BPE continuation (starts with common subword patterns)
        if re.match(r'^[bcdđghklmnpqrstvx][aăâeêioôơuưy]', clean.lower()) and len(clean) <= 3:
            return True  # short, starts with consonant + vowel — likely subword
        # Allow: common punctuation-attached words
        return True
    
    return False


def count_bpe_artifacts(syllables: List[str]) -> int:
    """Count how many syllables in the list are BPE artifacts."""
    return sum(1 for s in syllables if is_bpe_artifact(s))


# ═══════════════════════════════════════════════════════════════
#  P4: CANDIDATE SCORING FOR RERANKING
# ═══════════════════════════════════════════════════════════════

def score_candidate(text: str) -> float:
    """
    Score a generated output. Higher = better semantic quality.
    No external model needed.
    """
    syls = text.strip().split()
    n = len(syls)
    if n == 0:
        return -100.0
    
    score = 0.0
    
    # 1. Lexical diversity: unique/total ratio → weight 2.0
    unique_ratio = len(set(syls)) / n
    score += unique_ratio * 2.0
    
    # 2. BPE artifact penalty → -1.5 per artifact
    bpe_count = count_bpe_artifacts(syls)
    score -= bpe_count * 1.5
    
    # 3. Adjacent repeat penalty → -2.0 per repeat
    adj_repeats = sum(1 for i in range(n - 1) if syls[i] == syls[i + 1])
    score -= adj_repeats * 2.0
    
    # 4. Length bonus: penalize too-short outputs (< 10 syllables)
    if n < 10:
        score -= (10 - n) * 0.3
    
    # 5. Bigram novelty: % of bigrams not in last 8 tokens
    if n >= 3:
        recent = set(syls[max(0, n-8):])
        bigram_novel = sum(1 for i in range(n - 1) 
                          if syls[i] not in recent or syls[i+1] not in recent)
        score += (bigram_novel / max(n - 1, 1)) * 1.0
    
    return score


# ═══════════════════════════════════════════════════════════════
#  PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════

def build_prompt(user_input: str, *, max_context_couplets: int = 1,
                 include_trambong: bool = True) -> str:
    """
    v4.2: Build a fully-formatted Lục Bát prompt from user input.
    
    Always includes: <|start|> [LUC_BAT] [RHYME:X] [TONE:BBBBBB] [TRAMBONG:NH/HN]
    
    Handles:
      - Single 6-syl line → [LUC_BAT] prompt
      - Couplet (6+8) → đối thơ format  
      - Multi-couplet → last N couplets as context
    
    Returns formatted prompt string ending with <|reply|>.
    """
    user_input = user_input.lower()
    lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
    
    # Single line → simple Lục Bát prompt
    if len(lines) == 1:
        return _build_single_line_prompt(lines[0], include_trambong=include_trambong)
    
    # Multi-line → try to group into couplets
    couplets = _group_couplets(lines)
    
    if not couplets:
        return _build_single_line_prompt(lines[-1], include_trambong=include_trambong)
    
    # Keep last N couplets as context
    couplets = couplets[-max_context_couplets:]
    
    return _build_couplet_prompt(couplets, include_trambong=include_trambong)


def _group_couplets(lines: List[str]) -> List[Tuple[str, str]]:
    """Group lines into Lục Bát (6+8) couplets."""
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1 = len(lines[i].split())
        s2 = len(lines[i + 1].split())
        if s1 == 6 and s2 == 8:
            couplets.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return couplets


def _build_single_line_prompt(line: str, *, include_trambong: bool = True) -> str:
    """Build prompt for a single 6-syl line."""
    p = line.strip()
    
    # Already formatted?
    if p.startswith("[LUC_BAT]") or p.startswith("<|start|>"):
        if "<|reply|>" in p:
            return p
        return p + " <|reply|>"
    
    rhyme, tone, trambong = get_luc_bat_tags(p)
    tags_parts = [rhyme, tone]
    if include_trambong and trambong:
        tags_parts.append(trambong)
    tags = ' '.join(t for t in tags_parts if t)
    
    if tags:
        return f"<|start|> [LUC_BAT] {tags} {p} <|reply|>"
    return f"<|start|> [LUC_BAT] {p} <|reply|>"


def _build_couplet_prompt(couplets: List[Tuple[str, str]], *,
                          include_trambong: bool = True) -> str:
    """Build đối thơ prompt from couplets."""
    last_a, last_b = couplets[-1]
    
    # Extract tags from last input couplet
    rhyme_tag, tone_tag = get_doi_tho_tags(last_a, last_b)
    
    # Trầm-Bổng: from last couplet's 8-syl line, or default to NH
    trambong_tag = ""
    if include_trambong:
        trambong_tag = get_tram_bong_tag(last_b)
        if not trambong_tag:
            trambong_tag = "[TRAMBONG:NH]"  # most common pattern
    
    # Build input lines
    input_lines = []
    for a, b in couplets:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    
    # Build tags
    tags_parts = [rhyme_tag, tone_tag]
    if include_trambong and trambong_tag:
        tags_parts.append(trambong_tag)
    tags = ' '.join(t for t in tags_parts if t)
    
    return f"<|start|> [LUC_BAT] {tags} {input_str} <|reply|>"


# ═══════════════════════════════════════════════════════════════
#  GENERATION — canonical single function
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt: str, *,
             max_new: int = 64,
             temperature: float = 0.75,
             top_k: int = 50,
             top_p: float = 0.92,
             repetition_penalty: float = 1.2,
             rhyme_constraint: bool = True,
             rhyme_mode: str = "soft",       # "soft" (v4.2) or "hard" (v4.1)
             rhyme_logit_boost: float = 2.0,  # soft mode only
             device: str = "cpu") -> Tuple[List[int], str]:
    """
    Canonical autoregressive generation. Used by CLI, server, and eval.
    
    Returns (new_token_ids, decoded_text).
    """
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    reply_id = tokenizer.token_to_id("<|reply|>")
    
    # Use module constants as fallback
    if end_id is None: end_id = END_ID
    if pad_id is None: pad_id = PAD_ID
    if lb_id is None: lb_id = LB_ID
    if reply_id is None: reply_id = REPLY_ID
    
    # P1/P2: Parse target rhyme from prompt
    target_rhyme = None
    rhyme_syl_idx = 5  # default: Lục Bát pos6 of output (0-indexed: 5)
    rhyme_match = re.search(r'\[RHYME:([^\]]+)\]', prompt)
    if rhyme_match:
        target_rhyme = rhyme_match.group(1)
    
    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    new_tokens: List[int] = []
    for _ in range(max_new):
        # Forward pass on cropped context
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        
        # Suppress <|pad|>
        logits[:, pad_id] = float("-inf")
        
        # Repetition penalty — penalize tokens from recent output (last 16)
        if repetition_penalty:
            for prev in new_tokens[-16:]:
                logits[:, prev] -= repetition_penalty
        
        # P2: Rhyme constraint at target position in output line
        if rhyme_constraint and target_rhyme is not None:
            # Count syllables generated since last <|linebreak|> (or <|reply|>)
            last_delim = -1
            for delim_id in [lb_id, reply_id]:
                if delim_id in new_tokens:
                    pos = max(i for i, t in enumerate(new_tokens) if t == delim_id)
                    last_delim = max(last_delim, pos)
            
            after_delim = new_tokens[last_delim + 1:] if last_delim >= 0 else new_tokens
            decoded_after = tokenizer.decode(after_delim) if after_delim else ""
            current_syl_count = len(decoded_after.strip().split()) if decoded_after.strip() else 0
            
            if current_syl_count == rhyme_syl_idx:
                # Find matching and non-matching candidates in top-k*2
                candidate_k = min(top_k * 2 if top_k else 100, logits.size(-1))
                _, topk_idx = torch.topk(logits, candidate_k)
                matching, non_matching = [], []
                for tid in topk_idx[0]:
                    tid_i = tid.item()
                    if tid_i in (end_id, pad_id, lb_id, reply_id):
                        continue
                    decoded = tokenizer.decode([tid_i]).strip()
                    if not decoded:
                        continue
                    if get_rhyme_group(decoded) == target_rhyme:
                        matching.append(tid_i)
                    else:
                        non_matching.append(tid_i)
                
                if rhyme_mode == "hard":
                    # v4.1 behavior: force rhyme if any candidate exists
                    if matching:
                        for tid_i in non_matching:
                            logits[:, tid_i] = float("-inf")
                else:
                    # v4.2 soft: boost matching, allow override if semantically better
                    for tid_i in matching:
                        logits[:, tid_i] += rhyme_logit_boost
                    
                    # Safety valve: if model is very uncertain (flat distribution)
                    # and matching candidates exist, fall back to hard masking
                    probs_check = F.softmax(logits.clone(), dim=-1)
                    max_prob = probs_check.max().item()
                    if max_prob < 0.03 and matching:
                        for tid_i in non_matching:
                            logits[:, tid_i] = float("-inf")
        
        # Top-k filtering
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        
        # Top-p (nucleus) filtering
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
    
    # Decode
    text = tokenizer.decode(new_tokens)
    return new_tokens, text


# ═══════════════════════════════════════════════════════════════
#  P4: GENERATE-AND-RERANK
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_best(model, tokenizer, prompt: str, *,
                  n_candidates: int = 5,
                  **kwargs) -> Tuple[List[int], str]:
    """
    P4: Generate N candidates, return the one with best semantic score.
    """
    best_score = -float('inf')
    best_tokens, best_text = [], ""
    
    for _ in range(n_candidates):
        tokens, text = generate(model, tokenizer, prompt, **kwargs)
        score = score_candidate(text)
        if score > best_score:
            best_score = score
            best_tokens, best_text = tokens, text
    
    return best_tokens, best_text


# ═══════════════════════════════════════════════════════════════
#  RESPONSE DECODING — linebreak-aware
# ═══════════════════════════════════════════════════════════════

def decode_response(tokenizer, new_token_ids: List[int], *,
                    enforce_syllables: bool = False,
                    max_lines: int = 2) -> List[str]:
    """
    Decode generated tokens into lines, splitting on <|linebreak|>.
    
    v4.2: enforce_syllables defaults to False — metrics should reflect raw quality.
    No T2a TN re-split. No P3 truncation by default.
    
    Returns list of line strings (typically 2 for a couplet).
    """
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    if lb_id is None:
        lb_id = LB_ID
    
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
    
    # Clean: strip <|end|>, <|reply|>, and excess whitespace/punctuation
    lines = [l.replace('<|end|>', '').replace('<|reply|>', '').strip(',.-;:!? \t')
             for l in lines]
    lines = [l for l in lines if l]  # remove empty lines
    
    # Detect genre from first line syllable count
    if lines:
        s1 = len(lines[0].split())
        targets = (6, 8) if s1 <= 7 else (7, 7)
    else:
        targets = (6, 8)
    
    # Optional syllable enforcement (off by default)
    if enforce_syllables:
        for i, line in enumerate(lines):
            words = line.split()
            target = targets[i % 2]
            if len(words) > target:
                words = words[:target]
            lines[i] = ' '.join(words)
    
    # Overgeneration fix: if too many lines, find first valid couplet
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
