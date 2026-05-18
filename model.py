# model.py
# =========
# Purpose: The core — a custom autoregressive Transformer built from scratch
# using raw PyTorch (nn.Module). No HuggingFace wrappers.
#
# =========================================================================
# CONCEPT: What is a Language Model?
# =========================================================================
# A language model learns P(token_t | token_0, token_1, ..., token_{t-1})
# — the probability of the next token given all previous tokens.
#
# "Autoregressive" means we generate one token at a time, feeding each
# newly generated token back as input for the next step.
#
# "Causal" means the model can only look at past tokens, never future ones.
# This is enforced by a triangular mask in the attention mechanism.
#
# =========================================================================
# CONCEPT: The Transformer Architecture (decoder-only, like GPT)
# =========================================================================
#
#   Input tokens (B, T)
#        │
#        ▼
#   ┌─────────────────────────┐
#   │ Token Embedding         │  maps token ID → vector of size n_embd
#   │ + Position Embedding    │  adds position info (word order matters!)
#   └─────────────────────────┘
#        │
#        ▼
#   ┌─────────────────────────┐
#   │ Transformer Block 1     │
#   │  ├─ LayerNorm           │  normalizes activations (stable training)
#   │  ├─ Multi-Head Causal   │
#   │  │    Self-Attention    │  the "magic" — tokens look at each other
#   │  ├─ Residual (+)        │  skip connection: helps gradients flow
#   │  ├─ LayerNorm           │
#   │  ├─ FeedForward (FFN)   │  2-layer MLP, processes each position
#   │  └─ Residual (+)        │
#   └─────────────────────────┘
#        │ ... repeat n_layer times (6 in our case) ...
#        ▼
#   ┌─────────────────────────┐
#   │ Final LayerNorm         │
#   └─────────────────────────┘
#        │
#        ▼
#   ┌─────────────────────────┐
#   │ LM Head (Linear)        │  projects hidden state → vocab_size logits
#   └─────────────────────────┘
#        │
#        ▼
#   Output logits (B, T, vocab_size)
#        │
#        ▼
#   Cross-entropy loss vs targets (target = input shifted right by 1)
#
# =========================================================================
# CONCEPT: Self-Attention (the core innovation)
# =========================================================================
#
# Every token "looks at" every other token (that comes before it) and
# decides how relevant each is. This is done via Query, Key, Value:
#
#   For each token:
#     Q = token × W_q    "What am I looking for?"
#     K = token × W_k    "What do I contain?"
#     V = token × W_v    "What information do I carry?"
#
#   Attention score(i→j) = Q_i · K_j / sqrt(d_k)
#     (how much token i should attend to token j)
#
#   Attention weights = softmax(scores)  → probabilities summing to 1
#
#   Output_i = Σ_j (weight_i→j × V_j)
#     (weighted sum of all previous tokens' values)
#
# "Multi-head" means we do this in parallel with different W_q/W_k/W_v
# for each head, then concatenate results. Each head learns different
# types of relationships (syntax, rhyme, tone, meaning...).
#
# =========================================================================
# CONCEPT: Causal Mask
# =========================================================================
# A lower-triangular matrix that prevents token at position i from
# attending to any token at position > i (the future).
#
#   mask = [[T, F, F, F],     Position 0 sees: [0]
#           [T, T, F, F],     Position 1 sees: [0,1]
#           [T, T, T, F],     Position 2 sees: [0,1,2]
#           [T, T, T, T]]     Position 3 sees: [0,1,2,3]
#
# Before softmax, we set masked positions to -infinity so they become 0.
#
# =========================================================================
# CONCEPT: Feed-Forward Network (FFN)
# =========================================================================
# After attention gathers information from other tokens, the FFN
# processes each position independently through two linear layers:
#
#   FFN(x) = Linear_2( GELU( Linear_1(x) ) )
#
# Linear_1 expands: n_embd → 4×n_embd (e.g., 384 → 1536)
# Linear_2 contracts: 4×n_embd → n_embd (e.g., 1536 → 384)
#
# This gives the model capacity to "think" about what attention gathered.
#
# =========================================================================
# CONCEPT: Residual Connections
# =========================================================================
# Instead of:   x = F(x)
# We use:       x = x + F(x)
#
# Why? In deep networks, gradients can vanish. The "+ x" provides a
# "gradient highway" straight back to early layers. This is why we can
# stack 6, 12, or even 96 layers.
#
# =========================================================================
# CONCEPT: Layer Normalization ("Pre-Norm")
# =========================================================================
# Normalizes the activations across the embedding dimension for each
# token independently. This keeps values in a stable range.
#
# "Pre-Norm" means we apply LayerNorm BEFORE the sublayer (attention/FFN),
# not after. This is more stable for training large models.
#
#   Pre-Norm:   x = x + Attention(LayerNorm(x))
#   Post-Norm:  x = LayerNorm(x + Attention(x))   ← older style, less stable
#
# =========================================================================
# CONCEPT: Weight Tying
# =========================================================================
# The LM head and token embedding can share the same weight matrix.
# This makes intuitive sense: a token's "meaning" vector (embedding)
# should be similar to the vector that predicts that token (LM head).
# It also saves (vocab_size × n_embd) ≈ 4.6M parameters.
#
# =========================================================================
# SPEC SHEET (from README)
# =========================================================================
# | Param            | Value  | Notes                              |
# |------------------|--------|------------------------------------|
# | vocab_size       | 12,000 | Custom BPE                         |
# | n_embd           | 384    | Embedding/hidden dimension         |
# | n_head           | 6      | Attention heads (64 dim each)      |
# | n_layer          | 6      | Transformer blocks                 |
# | block_size       | 256    | Max context length                 |
# | Total params     | ~45M   | Fit on single GPU                  |
#
# =========================================================================
# IMPLEMENTATION PLAN — 5 classes to build:
# =========================================================================
#
# Class 1: MultiHeadAttention
#   - __init__: create Q,K,V linear projections, output projection, causal mask
#   - forward: project → reshape → attention → mask → softmax → weighted sum → output
#
# Class 2: FeedForward
#   - __init__: two linear layers (n_embd → 4*n_embd → n_embd)
#   - forward: Linear1 → GELU → Linear2
#
# Class 3: TransformerBlock
#   - __init__: LayerNorm × 2, MultiHeadAttention, FeedForward
#   - forward: x = x + attn(ln1(x)), x = x + ffn(ln2(x))
#
# Class 4: PoetryDuelGPT (the full model)
#   - __init__: token_emb, pos_emb, blocks (×n_layer), final_ln, lm_head
#   - forward: embed → add position → blocks → ln → lm_head → logits
#   - generate: autoregressive loop (implemented later in sample.py)
#
# Class 5 (utility): count_parameters
#   - Sum all parameter tensors, return in millions
#
# =========================================================================
# QUICK REFERENCE: PyTorch operations you'll need
# =========================================================================
# nn.Linear(in, out, bias=False)         — linear projection (matrix multiply)
# nn.Embedding(num, dim)                  — lookup table for token/position IDs
# nn.LayerNorm(dim)                       — layer normalization
# F.softmax(x, dim=-1)                    — softmax along last dimension
# torch.tril(torch.ones(N, N))            — lower triangular matrix (causal mask)
# x.view(B, T, n_head, head_dim)          — reshape (preserving data)
# x.transpose(1, 2)                       — swap dimensions for multi-head
# x.masked_fill(mask, value)              — fill positions where mask=True
# F.cross_entropy(logits, targets)        — standard loss for classification
# torch.multinomial(probs, num_samples)   — sample from probability distribution
# register_buffer('name', tensor)         — non-parameter tensor (moves with .to())

# --- YOUR CODE BELOW ---
