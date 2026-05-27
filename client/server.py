"""
FastAPI backend for PoetryDuel-GPT.
Loads model + tokenizer, exposes /chat endpoint.
Usage:  uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path

# Add src/ to path so we can import model + generation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

from model import PoetryDuelGPT

# v4.2: canonical generation module (replaces all local generation code)
from generation import (
    build_prompt, generate, decode_response,
    PAD_ID, END_ID, LB_ID, REPLY_ID,
)


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


# ── v4.2: All generation logic delegated to src/generation.py ──
# build_prompt(), generate(), decode_response() are imported above.


@torch.no_grad()
def generate_doi_tho(raw_input: str, temperature=0.75, top_k=50, top_p=0.92,
                     max_tokens=64) -> list:
    """
    v4.2: Generate đối thơ response. Each input couplet gets its own response.
    Uses the unified generate() from src/generation.py.
    
    Returns list of token IDs with linebreaks between couplet responses.
    """
    raw_input = raw_input.lower()
    lines = [l.strip() for l in raw_input.strip().split('\n') if l.strip()]
    
    # Group into couplets
    couplets = []
    i = 0
    while i + 1 < len(lines):
        s1, s2 = len(lines[i].split()), len(lines[i+1].split())
        if (s1 == 6 and s2 == 8):
            couplets.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    
    if not couplets and lines:
        # Single line: wrap as one couplet
        couplets = [(lines[-1], lines[-1])]
    
    if not couplets:
        return []
    
    all_tokens = []
    for turn, (c6, c8) in enumerate(couplets):
        # Build couplet input string
        if c6 == c8:
            # Single line input
            couplet_input = c6
        else:
            couplet_input = f"{c6}\n{c8}"
        
        prompt = build_prompt(couplet_input, include_trambong=True)
        tokens, _ = generate(model, tokenizer, prompt,
                            max_new=max_tokens, temperature=temperature,
                            top_k=top_k, top_p=top_p, device=DEVICE,
                            rhyme_mode="soft")
        all_tokens.extend(tokens)
        
        # Add linebreak between couplet responses
        if turn < len(couplets) - 1:
            all_tokens.append(LB_ID)
    
    return all_tokens


# ── Routes ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "PoetryDuel-GPT API v4.2",
        "model": model_info,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    raw = req.prompt.replace("|", "\n")

    new_ids = generate_doi_tho(
        raw,
        req.temperature,
        req.top_k,
        req.top_p,
        req.max_tokens,
    )

    # Decode with unified decode_response
    lines = decode_response(tokenizer, new_ids, enforce_syllables=False)
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
