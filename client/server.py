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
from tones import get_luc_bat_tags, get_that_ngon_tags


# ── Config ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
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


@torch.no_grad()
def generate(prompt: str, temperature=0.75, top_k=50, top_p=0.92, max_tokens=64):
    """Autoregressive generation (same logic as sample.py)."""
    end_id = tokenizer.token_to_id("<|end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

    # Auto-wrap genre + rhyme/tone tags
    if not prompt.startswith("["):
        syl = len(prompt.split())
        if syl == 7:
            link2, doi_am = get_that_ngon_tags(prompt)
            extras_parts = [t for t in [link2, doi_am] if t]
            tag = f"[THAT_NGON] {' '.join(extras_parts)}" if extras_parts else "[THAT_NGON]"
        else:
            rhyme, tone = get_luc_bat_tags(prompt)
            extras = f"{rhyme} {tone}".strip()
            tag = f"[LUC_BAT] {extras}" if extras else "[LUC_BAT]"
        prompt = f"{tag} {prompt}"

    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    new_tokens = []
    for _ in range(max_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        logits[:, pad_id] = float("-inf")

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

    return new_tokens


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

    new_ids = generate(
        req.prompt,
        req.temperature,
        req.top_k,
        req.top_p,
        req.max_tokens,
    )

    # Decode only the NEW tokens, strip punctuation artifacts
    response = tokenizer.decode(new_ids)
    response = response.replace("<|end|>", "").replace("<|start|>", "").strip()
    response = response.lstrip(", .;:-")

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
