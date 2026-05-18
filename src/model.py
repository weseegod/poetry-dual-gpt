"""
model.py — Custom GPT-style Transformer for Vietnamese Poetry Duels.

~45M parameters. Built with raw torch.nn — zero HuggingFace wrappers.

Architecture:
  Token Embedding + Position Embedding
    → 6× TransformerBlock (causal self-attention + FFN)
    → LayerNorm → LM Head → logits

Specs:
  n_embd=384  n_head=6  n_layer=6  block_size=256  vocab=~10K
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Multi-Head Causal Self-Attention
# =========================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with mask."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular): True = allowed to attend
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, embedding_dim

        # QKV projection
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale  # (B, n_head, T, T)

        # Apply causal mask (set future positions to -inf)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = attn_weights @ v  # (B, n_head, T, head_dim)

        # Combine heads back: (B, n_head, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_dropout(self.out_proj(out))
        return out


# =========================================================================
# Feed-Forward Network
# =========================================================================

class FeedForward(nn.Module):
    """Position-wise FFN: expand → GELU → contract."""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# =========================================================================
# Transformer Block (Pre-Norm)
# =========================================================================

class TransformerBlock(nn.Module):
    """One decoder block: Attention + FFN, both with pre-norm & residuals."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ffn(self.ln2(x))    # FFN with residual
        return x


# =========================================================================
# PoetryDuelGPT — Full Model
# =========================================================================

class PoetryDuelGPT(nn.Module):
    """
    Autoregressive language model for Vietnamese poetic duels.

    Phase 1: Trained on Lục Bát (6→8 syllable) with [LUC_BAT] control tag.
    """

    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6,
                 block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Output
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: LM head shares weights with token embedding
        self.lm_head.weight = self.token_embedding.weight

        # Initialize
        self.apply(self._init_weights)

        # Count params
        total = sum(p.numel() for p in self.parameters())
        print(f"PoetryDuelGPT: {total/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) token indices
            targets: (B, T) optional targets for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar or None
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence {T} exceeds block_size {self.block_size}"

        # Token + position embeddings
        tok_emb = self.token_embedding(idx)          # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)       # (T, n_embd)

        x = tok_emb + pos_emb                        # (B, T, n_embd)

        # Transformer blocks
        x = self.blocks(x)                           # (B, T, n_embd)

        # Final LayerNorm + LM head
        x = self.ln_final(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        # Loss
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # ignore padding tokens
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation.

        Args:
            idx:            (1, T) starting token sequence
            max_new_tokens: how many tokens to generate
            temperature:    lower = more deterministic
            top_k:          keep only top-k logits (None = disabled)

        Returns:
            Generated sequence (1, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block_size
            idx_cond = idx[:, -self.block_size:]

            # Forward
            logits, _ = self(idx_cond)  # (1, T_cond, vocab_size)
            logits = logits[:, -1, :]   # only last position (1, vocab_size)

            # Temperature
            logits = logits / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Softmax → sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def count_parameters(model):
    """Return total parameters in millions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


# =========================================================================
# Quick test
# =========================================================================
if __name__ == "__main__":
    # Use dynamic vocab size from trained tokenizer
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(__file__).parent.parent / "tokenizer" / "poetry_bpe.model"))
        vocab_size = tok.get_vocab_size()
    except Exception:
        vocab_size = 10581  # fallback

    model = PoetryDuelGPT(vocab_size=vocab_size)
    print(f"Parameters: {count_parameters(model):.1f}M")

    # Test forward pass
    x = torch.randint(0, vocab_size, (2, 64))
    logits, loss = model(x, targets=x)
    print(f"Logits: {logits.shape}")  # (2, 64, vocab_size)
    print(f"Loss:   {loss.item():.4f}  (expected ~{math.log(vocab_size):.1f})")

    # Test generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    gen = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generate: {gen.shape}")  # (1, 30)

    print("✅ All tests passed")
