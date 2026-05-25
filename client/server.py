"""
FastAPI backend for PoetryDuel-GPT.
Loads model + tokenizer, exposes /chat endpoint.
Usage:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Add src/ to path so we can import model + sample
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

from model import PoetryDuelGPT
from tones import get_luc_bat_tags, get_that_ngon_tags, get_doi_tho_tags, get_rhyme_group


# ── Config ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CHECKPOINT = ROOT / "checkpoints" / "doi_tho_best.pt"
if not CHECKPOINT.exists():
    CHECKPOINT = ROOT / "checkpoints" / "best.pt"
if not CHECKPOINT.exists():
    CHECKPOINT = ROOT / "checkpoints" / "final.pt"
TOKENIZER_PATH = ROOT / "tokenizer" / "poetry_bpe.model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────

app = FastAPI(title="PoetryDuel-GPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals set on startup
model = None
tokenizer = None
model_info = {}


def load():
    global model, tokenizer, model_info

    tok_path = str(TOKENIZER_PATH)
    if not Tokenizer:
        raise RuntimeError("tokenizers not installed")
    tokenizer = Tokenizer.from_file(tok_path)

    ckpt_path = str(CHECKPOINT)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Remap old checkpoint keys → current model attribute names
    key_map = {
        "token_embedding.weight": "tok_emb.weight",
        "position_embedding.weight": "pos_emb.weight",
        "ln_final.weight": "ln_f.weight",
        "ln_final.bias": "ln_f.bias",
        "lm_head.weight": "head.weight",
    }
    old_state = ckpt["model_state_dict"]
    new_state = {}
    for key, val in old_state.items():
        # Block-level remapping
        # old: blocks.N.attn.qkv_proj.weight  →  blocks.N.attn.qkv.weight
        # old: blocks.N.attn.out_proj.weight  →  blocks.N.attn.out.weight
        # old: blocks.N.ffn.fc1.weight        →  blocks.N.ffn.net.0.weight
        # old: blocks.N.ffn.fc2.weight        →  blocks.N.ffn.net.2.weight
        k = key.replace("qkv_proj", "qkv") \
               .replace("out_proj", "out") \
               .replace("causal_mask", "mask") \
               .replace(".ffn.fc1.", ".ffn.net.0.") \
               .replace(".ffn.fc2.", ".ffn.net.2.")
        # Top-level remapping
        k = key_map.get(k, k)
        new_state[k] = val

    model = PoetryDuelGPT(
        vocab_size=ckpt["vocab_size"],
        n_embd=ckpt["model_config"]["n_embd"],
        n_head=ckpt["model_config"]["n_head"],
        n_layer=ckpt["model_config"]["n_layer"],
        block_size=ckpt["model_config"]["block_size"],
        dropout=ckpt["model_config"].get("dropout", 0.1),
    )
    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE).eval()

    model_info = {
        "loaded": True,
        "vocab_size": ckpt["vocab_size"],
        "n_embd": ckpt["model_config"]["n_embd"],
        "n_layer": ckpt["model_config"]["n_layer"],
        "n_head": ckpt["model_config"]["n_head"],
        "params": sum(p.numel() for p in model.parameters()),
        "step": ckpt["step"],
        "device": DEVICE,
    }


# ── Request/Response models ─────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.75
    top_k: int = 50
    top_p: float | None = 0.92
    max_tokens: int = 64


class ChatResponse(BaseModel):
    response: str
    prompt: str


def _parse_couplets(user_input: str):
    """Parse user input into (couplets_list, num_input_couplets).
    Supports both Lục Bát (6+8) and Thất Ngôn (7+7).
    Normalizes to lowercase to match training data."""
    user_input = user_input.lower()
    lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1, s2 = len(lines[i].split()), len(lines[i+1].split())
        # Accept Lục Bát (6+8) or Thất Ngôn (7+7) couplets
        if (s1 == 6 and s2 == 8) or (s1 == 7 and s2 == 7):
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    if not couplets and lines:
        # Single line without a pair — treat as 1-couplet input
        return [], 0
    return couplets, len(couplets)


def _build_doi_tho_prompt(couplets, max_context=1):
    """Build [DOI_THO] prompt from last N couplets with genre token."""
    if not couplets:
        return ""
    ctx = couplets[-max_context:]
    last_a, last_b = ctx[-1]
    s1_last = len(last_a.split())
    if s1_last == 7:
        rhyme_tag, tone_tag = get_doi_tho_tags_tn(last_a)
        genre_token = "[THAT_NGON]"
    else:
        rhyme_tag, tone_tag = get_doi_tho_tags(last_a, last_b)
        genre_token = "[LUC_BAT]"
    input_lines = []
    for a, b in ctx:
        input_lines.append(a)
        input_lines.append(b)
    input_str = " <|linebreak|> ".join(input_lines)
    tags = f"{rhyme_tag} {tone_tag}".strip()
    tag_part = f"{genre_token}"
    if tags:
        tag_part += f" {tags}"
    return f"<|start|> {tag_part} {input_str} <|reply|>"


def _auto_tag_doi_tho(user_input: str, max_context: int = 1) -> str:
    """Backward-compat: wrap input as [DOI_THO] prompt with genre token."""
    couplets, _ = _parse_couplets(user_input)
    if not couplets:
        lines = [l.strip() for l in user_input.strip().split('\n') if l.strip()]
        if lines:
            line = lines[-1]
            syl_count = len(line.split())
            if syl_count == 7:
                rhyme_tag, tone_tag = get_doi_tho_tags_tn(line)
                genre_token = "[THAT_NGON]"
            else:
                rhyme_tag, tone_tag = get_doi_tho_tags(line, line)
                genre_token = "[LUC_BAT]"
            tags = f"{rhyme_tag} {tone_tag}".strip()
            tag_part = f"{genre_token}"
            if tags:
                tag_part += f" {tags}"
            return f"<|start|> {tag_part} {line} <|reply|>"
        return ""
    return _build_doi_tho_prompt(couplets, max_context)


@torch.no_grad()
def generate(prompt: str, temperature=0.75, top_k=50, top_p=0.92, max_tokens=64, is_doi_tho=False):
    """
    Generate đối thơ response. Each input couplet gets its own independent response:
      1 couplet in → 1 couplet out  (2 lines)
      2 couplets in → 2 couplets out (4 lines, C1→C3, C2→C4)
      N couplets in → N couplets out
    
    P1: Rhyme constraint at rhyme position in 2nd output line.
    """
    import re
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    lb_id = tokenizer.token_to_id("<|linebreak|>")
    
    # Parse input
    couplets, num_input = _parse_couplets(prompt)
    if num_input == 0:
        num_input = 1  # single line = 1 couplet
    
    # Collect input lines
    input_lines = []
    for six, eight in couplets:
        input_lines.extend([six, eight])
    if not input_lines:
        input_lines = [l.strip() for l in prompt.strip().split('\n') if l.strip()]
    
    # Generate one response per input couplet (independent duels)
    all_new_tokens = []
    for turn in range(num_input):
        # Get this turn's input couplet
        if turn < len(couplets):
            c6, c8 = couplets[turn]
        else:
            # Single line: use it as both
            c6 = c8 = input_lines[-1]
        
        # Build prompt for just this couplet
        prompt_str = _build_doi_tho_prompt([(c6, c8)], max_context=1)
        if not prompt_str:
            break
        
        is_tn = '[THAT_NGON]' in prompt_str
        
        # P1: Parse target rhyme from prompt
        target_rhyme = None
        rhyme_syl_idx = None
        rhyme_match = re.search(r'\[RHYME:([^\]]+)\]', prompt_str)
        if rhyme_match:
            target_rhyme = rhyme_match.group(1)
            if '[THAT_NGON]' in prompt_str:
                rhyme_syl_idx = 6  # Thất Ngôn: 7th syllable
            else:
                rhyme_syl_idx = 5  # Lục Bát: 6th syllable
        
        # Encode and generate
        ids = tokenizer.encode(prompt_str).ids
        idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        
        new_tokens = []
        for _ in range(max_tokens):
            logits, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / temperature
            logits[:, pad_id] = float("-inf")
            
            # Repetition penalty — penalize tokens from recent output
            for prev in new_tokens[-16:]:
                logits[:, prev] -= 1.2
            
            # P1: Rhyme constraint at rhyme position in 2nd line
            if target_rhyme is not None and rhyme_syl_idx is not None:
                if lb_id in new_tokens:
                    last_lb = max(i for i, t in enumerate(new_tokens) if t == lb_id)
                    after_lb = new_tokens[last_lb + 1:]
                    decoded_after = tokenizer.decode(after_lb)
                    current_syl_count = len(decoded_after.strip().split()) if decoded_after.strip() else 0
                else:
                    current_syl_count = 0
                if current_syl_count == rhyme_syl_idx:
                    candidate_k = min(top_k * 2 if top_k else 100, logits.size(-1))
                    _, topk_idx = torch.topk(logits, candidate_k)
                    matching = []
                    non_matching = []
                    for tid in topk_idx[0]:
                        tid_i = tid.item()
                        if tid_i in (end_id, pad_id, lb_id):
                            continue
                        decoded = tokenizer.decode([tid_i]).strip()
                        if not decoded:
                            continue
                        rg = get_rhyme_group(decoded)
                        if rg == target_rhyme:
                            matching.append(tid_i)
                        else:
                            non_matching.append(tid_i)
                    # Only mask if at least one matching candidate exists
                    if matching:
                        for tid_i in non_matching:
                            logits[:, tid_i] = float("-inf")
            
            # T2a: For Thất Ngôn, suppress premature <|linebreak|> in 1st output line
            if is_tn and lb_id not in new_tokens:
                decoded = tokenizer.decode(new_tokens)
                syl_count = len(decoded.strip().split()) if decoded.strip() else 0
                if syl_count < 7:
                    logits[:, lb_id] = float("-inf")
            
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p is not None:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                logits[:, sorted_idx[mask]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            if next_id == end_id:
                break
            new_tokens.append(next_id)
            idx = torch.cat((idx, torch.tensor([[next_id]], device=DEVICE)), dim=1)
        
        all_new_tokens.extend(new_tokens)
        # Add linebreak between turns (separates responses in final output)
        if turn < num_input - 1:
            all_new_tokens.append(lb_id)
    
    return all_new_tokens


def _decode_doi_tho(tok, new_token_ids, enforce_syllables=True, max_lines=2):
    """Decode tokens, splitting on <|linebreak|> positions.
    Enforces syllable pattern (P3) and fixes overgeneration (P2: > max_lines).
    Detects genre (Lục Bát vs Thất Ngôn) from first line syllable count."""
    lb_id = tok.token_to_id("<|linebreak|>")
    lines = []
    chunk = []
    for t in new_token_ids:
        if t == lb_id:
            if chunk:
                lines.append(tok.decode(chunk).strip())
            chunk = []
        else:
            chunk.append(t)
    if chunk:
        lines.append(tok.decode(chunk).strip())
    
    # Detect genre from first line syllable count
    if lines:
        s1 = len(lines[0].split())
        targets = (6, 8) if s1 <= 7 else (7, 7)
    else:
        targets = (6, 8)
    
    # P3: Enforce syllable count per line
    if enforce_syllables:
        for i, line in enumerate(lines):
            words = line.split()
            target = targets[i % 2]
            if len(words) > target:
                words = words[:target]
            lines[i] = ' '.join(words)
    
    # P2: Fix overgeneration — if too many lines, find first valid couplet
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


# ── Routes ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "PoetryDuel-GPT API",
        "model": model_info,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # Detect multi-line: newlines or pipe separator
    raw = req.prompt.replace("|", "\n")
    is_doi_tho = "\n" in raw

    new_ids = generate(
        raw,
        req.temperature,
        req.top_k,
        req.top_p,
        req.max_tokens,
        is_doi_tho=is_doi_tho,
    )

    # Decode with proper linebreak handling, capitalize first letter
    lines = _decode_doi_tho(tokenizer, new_ids)
    lines = [l[0].upper() + l[1:] if l else l for l in lines]
    response = "\n".join(lines)

    return ChatResponse(response=response, prompt=req.prompt)


# ── Startup ────────────────────────────────────────

@app.on_event("startup")
def startup():
    print(f"🚀 Loading PoetryDuel-GPT on {DEVICE}...")
    try:
        load()
        m = model_info
        print(f"✅ Model loaded: {m['params']/1e6:.1f}M params, vocab={m['vocab_size']}")
        print(f"   Checkpoint step: {m['step']}")
    except FileNotFoundError:
        print("⚠️  Checkpoint not found! Run src/train.py first.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
