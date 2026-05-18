"""
train_bpe.py — Train a Byte-Pair Encoding tokenizer for Vietnamese poetry.

Produces tokenizer/poetry_bpe.model with vocab_size=12,000.
Special tokens: <|pad|> <|start|> <|reply|> <|end|> [LUC_BAT] [TU_TUYET] [THAT_NGON_BAT_CU]
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

ROOT = Path(__file__).parent.parent
CORPUS_PATH = ROOT / "data" / "poetry_corpus.txt"
OUTPUT_PATH = ROOT / "tokenizer" / "poetry_bpe.model"

# Order matters: pad must be index 0
SPECIAL_TOKENS = [
    "<|pad|>",            # 0 — padding / unknown
    "<|start|>",          # 1 — beginning of training example
    "<|reply|>",          # 2 — separates prompt from reply
    "<|end|>",            # 3 — end of training example
    "[LUC_BAT]",          # 4 — genre: 6-8 couplet
    "[TU_TUYET]",         # 5 — genre: 4×7 poem
    "[THAT_NGON_BAT_CU]", # 6 — genre: 8×7 poem
]


def train_tokenizer(corpus_path=None, output_dir=None, vocab_size=12000):
    """Train BPE tokenizer and save to disk."""
    corpus_path = Path(corpus_path) if corpus_path else CORPUS_PATH
    output_dir = Path(output_dir) if output_dir else OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "poetry_bpe.model"

    print(f"Training BPE tokenizer (vocab_size={vocab_size:,})")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {output_path}")

    # BPE model with byte-level pre-tokenization
    tokenizer = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tokenizer.normalizer = None  # Vietnamese diacritics must NOT be normalized away
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train on corpus
    tokenizer.train([str(corpus_path)], trainer)

    # Save
    tokenizer.save(str(output_path))
    print(f"\n✅ Tokenizer saved: {output_path}")
    print(f"   Vocab size: {tokenizer.get_vocab_size():,}")

    # Verify special tokens are at correct indices
    for i, tok in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(tok)
        assert actual_id == i, f"{tok} should be index {i}, got {actual_id}"
    print("   Special token indices verified ✅")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for PoetryDuel-GPT")
    parser.add_argument("--corpus", type=str, default=None, help="Path to poetry_corpus.txt")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=12000)
    args = parser.parse_args()

    train_tokenizer(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )
