# tokenizer/train_bpe.py
# ======================
# Purpose: Train a BPE tokenizer specifically for Vietnamese poetry.
#
# What is a tokenizer?
#   A tokenizer converts raw text into numbers (token IDs) that the model
#   can process. It also converts generated token IDs back into text.
#
# What is BPE (Byte-Pair Encoding)?
#   BPE starts with individual characters, then repeatedly merges the most
#   frequent adjacent pair of symbols. This creates subword units.
#   Example: "người" might become ["ng", "ười"] or stay whole if common.
#   Benefit: handles rare words by breaking them into known subwords.
#
# Why custom tokenizer instead of a pre-built one?
#   - Vietnamese has unique diacritics (ton dấu) that generic tokenizers
#     often split incorrectly
#   - Poetry has domain-specific vocabulary
#   - We need special control tokens: <|start|>, <|reply|>, <|end|>,
#     [LUC_BAT], [TU_TUYET], [THAT_NGON_BAT_CU], <|pad|>
#
# Special tokens we need:
#   0: <|pad|>           - padding (also acts as unknown token)
#   1: <|start|>         - marks beginning of a training example
#   2: <|reply|>         - separates prompt from reply
#   3: <|end|>           - marks end of training example
#   4: [LUC_BAT]         - genre tag: 6-8 syllable couplet
#   5: [TU_TUYET]        - genre tag: 4x7 syllable poem
#   6: [THAT_NGON_BAT_CU] - genre tag: 8x7 syllable poem
#
# What you'll learn:
#   - How BPE algorithm works (frequency-based merging)
#   - How to use HuggingFace `tokenizers` library
#   - Pre-tokenization (byte-level), normalization (NFKC)
#   - Post-processing templates for sequence packing
#   - How vocab size affects model size (embedding = vocab_size × n_embd)
#
# Target: vocab_size = 12,000 tokens
#
# Implementation plan:
#   1. Define special tokens list (must be first in vocab!)
#   2. Initialize BPE tokenizer with unknown token = <|pad|>
#   3. Set normalizer (NFKC unicode normalization)
#   4. Set pre-tokenizer (ByteLevel, handles all Unicode)
#   5. Configure BPE trainer (vocab_size, min_frequency, special_tokens)
#   6. Train on the preprocessed corpus (from data/poetry_corpus.txt)
#   7. Set post-processor template for sequence formatting
#   8. Enable padding (to block_size=256) and truncation
#   9. Save tokenizer to tokenizer/poetry_bpe.model
#
# API reference (HuggingFace tokenizers):
#   from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
#   Tokenizer(models.BPE(unk_token="<|pad|>"))
#   trainer = trainers.BpeTrainer(vocab_size=12000, special_tokens=[...])
#   tokenizer.train([corpus_path], trainer)
#   tokenizer.save("tokenizer/poetry_bpe.model")
#   tokenizer.post_processor = processors.TemplateProcessing(...)
#   tokenizer.enable_padding(pad_id=0, pad_token="<|pad|>", length=256)
#
# After training, you can test it:
#   from tokenizers import Tokenizer
#   tok = Tokenizer.from_file("tokenizer/poetry_bpe.model")
#   print(tok.encode("<|start|> [LUC_BAT] Trăm năm").ids)
#   # → [1, 4, ...token IDs for "Trăm", "năm"...]
#   print(tok.get_vocab_size())  # → 12000

# --- YOUR CODE BELOW ---
