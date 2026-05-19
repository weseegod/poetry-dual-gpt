"""
GPT-style Autoregressive Transformer for Vietnamese poetry.
Built from scratch with torch.nn — zero HuggingFace wrappers.

Architecture (decoder-only):
  TokenEmbed + PositionEmbed
    → [TransformerBlock × n_layer]
      ├── Causal MultiHeadAttention (Q·K^T/√d, masked)
      └── FeedForward (Linear→GELU→Linear)
    → LayerNorm → LM Head → logits

Specs: n_embd=384  n_head=6  n_layer=6  block_size=256
"""

import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  MULTI-HEAD CAUSAL SELF-ATTENTION
# ═══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Q, K, V each project the input into n_head subspaces (64 dim each).
    Causal mask prevents token i from attending to token j > i.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd, self.n_head = n_embd, n_head
        self.head_dim = n_embd // n_head

        # Combined QKV projection (faster than 3 separate linears)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)

        # Lower-triangular mask: True = allowed to attend
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        # Project → split into Q, K, V → reshape for multi-head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.drop_attn(F.softmax(attn, dim=-1))

        # Weighted sum → combine heads → project out
        out = attn @ v                                              # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)       # (B, T, C)
        return self.drop_out(self.out(out))


# ═══════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """Expand n_embd → 4*n_embd → GELU → contract back."""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK (Pre-Norm)
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-norm: apply LayerNorm BEFORE attention/FFN, then add residual.
      x = x + Attention( LN(x) )
      x = x + FFN( LN(x) )
    Pre-norm is more stable for training than post-norm.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # attention with residual
        x = x + self.ffn(self.ln2(x))   # FFN with residual
        return x


# ═══════════════════════════════════════════════════════════════
#  FULL MODEL
# ═══════════════════════════════════════════════════════════════

class PoetryDuelGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Embeddings: token meaning + position in sequence
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Output: normalize → project to vocabulary logits
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embedding
        # (saves params + improves training: "predicting a word" ≈ "word's meaning")
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        print(f"PoetryDuelGPT: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size

        # Token embedding + positional embedding
        tok = self.tok_emb(idx)                                  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))   # (T, C)
        x = tok + pos                                            # broadcast add

        # Transformer blocks
        x = self.blocks(x)

        # LM head → logits
        logits = self.head(self.ln_f(x))                         # (B, T, vocab)

        loss = None
        if targets is not None:
            # Cross-entropy on EVERY position simultaneously
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation: sample one token at a time."""
        for _ in range(max_new_tokens):
            # Crop to window size
            cond = idx[:, -self.block_size:]

            # Forward → logits at last position
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Sample from softmax distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ═══════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Try to load actual vocab size, fallback to 10581
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(__file__).parent.parent / "tokenizer" / "poetry_bpe.model"))
        V = tok.get_vocab_size()
    except Exception:
        V = 10581

    m = PoetryDuelGPT(vocab_size=V)
    x = torch.randint(0, V, (2, 64))
    logits, loss = m(x, targets=x)
    print(f"Logits: {logits.shape} | Loss: {loss.item():.2f} (expected ~{math.log(V):.1f})")

    gen = m.generate(torch.randint(0, V, (1, 10)), max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generate: input=10 → output={gen.shape[1]} tokens ✓")
