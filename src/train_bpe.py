"""
Train a BPE tokenizer on Vietnamese poetry corpus.
Saves tokenizer/poetry_bpe.model (~10K vocab).

Special tokens (indices must stay fixed):
  0: <|pad|>              1: <|start|>
  2: <|reply|>             3: <|end|>
  4: [LUC_BAT]             5: [TU_TUYET]
  6: [THAT_NGON_BAT_CU]
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "data" / "poetry_corpus.txt"
OUTPUT = ROOT / "tokenizer" / "poetry_bpe.model"

# Order matters: pad=0, start=1, reply=2, end=3, genre tags 4-6
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|start|>",
    "<|reply|>",
    "<|end|>",
    "[LUC_BAT]",
    "[TU_TUYET]",
    "[THAT_NGON_BAT_CU]",
    "[THAT_NGON]",          # thất ngôn (7-syllable, any line count)
]


def train(corpus=None, output_dir=None, vocab_size=12000):
    corpus = Path(corpus or CORPUS)
    out_dir = Path(output_dir or OUTPUT.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "poetry_bpe.model"

    print(f"Training BPE (vocab={vocab_size:,}) on {corpus}")

    # BPE model: characters → frequent subword pairs → merge repeatedly
    tok = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tok.normalizer = None  # keep Vietnamese diacritics intact
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tok.train([str(corpus)], trainer)
    tok.save(str(out_path))

    # Verify special token indices
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"{t} != index {i}"

    print(f"Saved → {out_path}  (vocab={tok.get_vocab_size():,})")
    return tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--vocab_size", type=int, default=12000)
    args = p.parse_args()
    train(args.corpus, args.output_dir, args.vocab_size)
