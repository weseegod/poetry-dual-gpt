# sample.py
# ==========
# Purpose: Generate poetry from a trained model — the fun part!
#
# =========================================================================
# CONCEPT: Autoregressive Generation
# =========================================================================
# The model generates one token at a time:
#
#   1. Start with a prompt: "[LUC_BAT] Thân em như chẽn lúa đòng đòng,"
#   2. Tokenize it: [1, 4, token(Thân), token(em), ...]  → tensor (1, T0)
#   3. Forward pass: model(current_sequence) → logits (1, T0, vocab_size)
#   4. Take logits at the LAST position: logits[0, -1, :]  → (vocab_size,)
#   5. Apply sampling strategy: temperature → top-k → top-p → softmax → sample
#   6. Append the sampled token to the sequence
#   7. Repeat steps 3-6 until we hit <|end|> or max_new_tokens
#   8. Decode the full sequence back to text
#
# This is the same algorithm GPT-4 uses, just at much smaller scale.
#
# =========================================================================
# CONCEPT: Temperature Sampling
# =========================================================================
# Controls randomness by scaling logits before softmax:
#
#   scaled_logits = logits / temperature
#   probabilities = softmax(scaled_logits)
#
# temperature = 0.01 → almost deterministic (always picks most likely)
# temperature = 0.75 → balanced (good default for poetry)
# temperature = 1.0  → natural distribution
# temperature = 2.0  → very random/creative (may produce nonsense)
#
# Why does this work? Smaller temperature makes high logits even higher,
# so softmax concentrates probability on the top few tokens.
#
# =========================================================================
# CONCEPT: Top-K Sampling
# =========================================================================
# Keep only the k most likely tokens, set all others to -infinity.
# Then softmax redistributes probability among the survivors.
#
# top_k = 50: only consider the 50 most likely next tokens
# top_k = None: consider all 12,000 tokens (usually too noisy)
#
# This prevents the model from picking very unlikely tokens that
# would derail the generation.
#
# Implementation:
#   values, indices = torch.topk(logits, k)
#   logits[logits < values[:, -1:]] = float('-inf')
#
# =========================================================================
# CONCEPT: Top-P (Nucleus) Sampling
# =========================================================================
# More adaptive than top-k. Keep the smallest set of tokens whose
# cumulative probability ≥ p.
#
# Example: if top_p = 0.9 and token probs are [0.5, 0.2, 0.15, 0.1, 0.05],
# keep [0.5, 0.2, 0.15] because 0.5+0.2+0.15 = 0.85 < 0.9, and adding
# 0.1 would exceed 0.9? Actually we keep adding until we CROSS p.
# So we keep [0.5, 0.2, 0.15, 0.1] (cumsum = 0.95 ≥ 0.9).
#
# Implementation:
#   1. Sort logits descending
#   2. Compute softmax
#   3. Compute cumulative sum
#   4. Mask where cumsum > top_p (but keep at least one token)
#   5. Re-scatter to original order
#
# =========================================================================
# CONCEPT: Vietnamese Poetic Rule Verification
# =========================================================================
# After generation, we check if the output follows Vietnamese poetic rules.
# This is what makes this project impressive to recruiters!
#
# 1. SYLLABLE COUNT CHECK
#    Lục Bát:  prompt 6 syllables → response 8 syllables
#    Tứ Tuyệt: prompt 7 syllables → response 7 syllables
#    Count syllables: split by spaces, Vietnamese syllables are single
#    orthographic words (each syllable = one written word)
#
# 2. TONE ALIGNMENT (Luật Bằng-Trắc)
#    Vietnamese has 6 tones grouped into 2 categories:
#
#    BẰNG (level) tones:
#      - Ngang (a): no diacritic, mid-level
#      - Huyền (à): grave accent, low-falling
#
#    TRẮC (sharp) tones:
#      - Sắc (á): acute accent, high-rising
#      - Nặng (ạ): dot below, low-constricted
#      - Hỏi (ả): hook, falling-rising
#      - Ngã (ã): tilde, broken-rising
#
#    For Lục Bát, the required pattern is:
#      Line 1 (6 syllables):  - B - T - B     (positions 2,4,6)
#      Line 2 (8 syllables):  - B - T - B - B (positions 2,4,6,8)
#      where B = Bằng tone, T = Trắc tone
#      Position 1 in each line is free (any tone).
#
#    Plus, the 6th syllable of line 1 must rhyme with the 6th syllable
#    of line 2 (vần). We can optionally check this.
#
# 3. RHYME CHECK (Vần) — advanced, optional
#    In Lục Bát: syllable 6 of the 6-word line rhymes with syllable 6
#    of the 8-word line. Rhyme in Vietnamese is based on the final
#    (rime) part of the syllable, not just the last letter.
#
# =========================================================================
# IMPLEMENTATION PLAN
# =========================================================================
#
# 1. load_model(checkpoint_path, device)
#    - Load checkpoint, reconstruct model, load weights
#
# 2. Sampling utilities:
#    - sample_with_temperature(logits, temperature)
#    - sample_top_k(logits, top_k)
#    - sample_top_p(logits, top_p)
#
# 3. generate(model, tokenizer, prompt, ...)
#    - The main generation loop (steps 1-8 above)
#    - Handle context window cropping (if seq > block_size, keep last block_size)
#    - Stop on <|end|> token
#
# 4. Vietnamese rule checking:
#    - count_syllables(text) — split by space
#    - get_tone_type(syllable) — classify Bằng/Trắc from diacritics
#    - check_syllable_count(prompt, response, expected) — verify count
#    - check_tone_alignment(line, genre) — verify B-T pattern
#
# 5. evaluate_generation(prompt, generated_text, genre)
#    - Parse generated output, extract response, run all checks
#
# 6. main() — CLI: single prompt mode and interactive mode
#    - Interactive mode: user types a line, model responds, repeat
#
# =========================================================================
# EXPECTED OUTPUT FORMAT
# =========================================================================
# [Input Prompt]: [LUC_BAT] Thân em như chẽn lúa đòng đòng,
# [Model Rebuttal]: Phất phơ dưới ngọn nắng hồng ban mai.
# ==================================================
# * Metric Evaluation *
# Syllable Verification: PASS (6-word prompt -> 8-word response)
# Tone Map Alignment: Bằng - Trắc Match Confirmed.

# --- YOUR CODE BELOW ---
