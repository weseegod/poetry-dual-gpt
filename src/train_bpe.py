"""
Train a BPE tokenizer on Vietnamese poetry corpus.
Saves tokenizer/poetry_bpe.model (~11K vocab).

Special tokens (indices must stay fixed):
  0-7:   Core (<|pad|>, <|start|>, <|reply|>, <|end|>, genre tags)
  8-148: Rhyme groups     [RHYME:a] ... [RHYME:...]   (141 tokens)
  149-212: Tone patterns   [TONE:BBBBBB] ... [TONE:TTTTTT]   (64 tokens)
  213-340: Đối âm patterns [DOIAM:BBBBBBB] ... [DOIAM:TTTTTTT] (128 tokens)
  341-342: Link2           [LINK2:B], [LINK2:T]   (2 tokens)
"""

import argparse, re
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "resources" / "poetry_corpus.txt"
OUTPUT = ROOT / "tokenizer" / "poetry_bpe.model"


def collect_rhyme_groups(corpus_path):
    """Collect all valid Vietnamese rhyme groups from corpus."""
    rhymes = set()
    with open(corpus_path) as f:
        for line in f:
            for m in re.finditer(r'\[RHYME:(\w+)\]', line):
                r = m.group(1)
                if all(c.isalpha() for c in r) and len(r) >= 1:
                    rhymes.add(r)
    return sorted(rhymes)


def collect_tone_patterns(corpus_path):
    """Collect all 6-position tone patterns from corpus."""
    tones = set()
    with open(corpus_path) as f:
        for line in f:
            for m in re.finditer(r'\[TONE:([BT]+)\]', line):
                tones.add(m.group(1))
    return sorted(tones)


def collect_doi_am_patterns(corpus_path):
    """Collect all 7-position tone patterns for Thất Ngôn đối âm."""
    # Inline get_tone to avoid import issues when run from any directory
    TRAC = set('áắấéếíóốớúứýạặậẹệịọộợụựỵảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹ'
               'ÁẮẤÉẾÍÓỐỚÚỨÝẠẶẬẸỆỊỌỘỢỤỰỴẢẲẨẺỂỈỎỔỞỦỬỶÃẴẪẼỄĨÕỖỠŨỮỸ')
    def get_tone(syl):
        for ch in syl:
            if ch in TRAC: return 'T'
        return 'B'
    
    patterns = set()
    with open(corpus_path) as f:
        for line in f:
            if '[THAT_NGON]' in line and '<|reply|>' in line:
                parts = line.split('<|reply|>')
                if len(parts) >= 2:
                    resp = parts[1].split('<|end|>')[0].strip()
                    tones = ''.join(get_tone(s) for s in resp.split()[:7])
                    if len(tones) == 7:
                        patterns.add(tones)
    return sorted(patterns)


def build_special_tokens(corpus_path):
    """Build the complete SPECIAL_TOKENS list."""
    core = [
        "<|pad|>",              # 0
        "<|start|>",            # 1
        "<|reply|>",            # 2
        "<|end|>",              # 3
        "[LUC_BAT]",            # 4
        "[TU_TUYET]",           # 5
        "[THAT_NGON_BAT_CU]",   # 6
        "[THAT_NGON]",          # 7
        "[DOI_THO]",            # 8  — couplet-to-couplet poetry duel
        "<|linebreak|>",        # 9  — separates lines within input/output
    ]
    
    # Rhyme groups (R1 fix)
    rhyme_groups = collect_rhyme_groups(corpus_path)
    rhyme_tokens = [f"[RHYME:{r}]" for r in rhyme_groups]
    
    # Tone patterns (R2 fix)
    tone_pats = collect_tone_patterns(corpus_path)
    tone_tokens = [f"[TONE:{t}]" for t in tone_pats]
    
    # Đối âm patterns (đối âm fix)
    doi_am_pats = collect_doi_am_patterns(corpus_path)
    doi_am_tokens = [f"[DOIAM:{d}]" for d in doi_am_pats]
    
    # Link2
    link_tokens = ["[LINK2:B]", "[LINK2:T]"]
    
    all_tokens = core + rhyme_tokens + tone_tokens + doi_am_tokens + link_tokens
    print(f"Core: {len(core)} | Rhyme: {len(rhyme_tokens)} | Tone: {len(tone_tokens)}")
    print(f"Đối âm: {len(doi_am_tokens)} | Link2: {len(link_tokens)}")
    print(f"Total special tokens: {len(all_tokens)}")
    
    return all_tokens


def train(corpus=None, output_dir=None, vocab_size=12000):
    corpus = Path(corpus or CORPUS)
    out_dir = Path(output_dir or OUTPUT.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "poetry_bpe.model"

    special_tokens = build_special_tokens(corpus)

    print(f"\nTraining BPE (vocab={vocab_size:,}) on {corpus}")

    tok = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tok.train([str(corpus)], trainer)
    tok.save(str(out_path))

    # Verify indices for tokens we care about
    for i, t in enumerate(special_tokens):
        actual = tok.token_to_id(t)
        if actual != i:
            print(f"⚠️  Index mismatch: {t} → {actual} (expected {i})")
    
    # Key checks
    key_tokens = ["<|pad|>", "<|start|>", "<|reply|>", "<|end|>",
                  "[LUC_BAT]", "[THAT_NGON]", "[DOI_THO]", "<|linebreak|>",
                  "[RHYME:ong]", "[TONE:BBBTTB]", "[DOIAM:BBBBBBB]", "[LINK2:B]"]
    print(f"\nVerification:")
    for t in key_tokens:
        tid = tok.token_to_id(t)
        nsub = len(tok.encode(t).ids)
        print(f"  {t:25s} → id={tid:5d}  subwords={nsub}  {'✅ SINGLE' if nsub==1 else '❌ FRAGMENTED'}")

    print(f"\nSaved → {out_path}  (vocab={tok.get_vocab_size():,})")
    return tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--vocab_size", type=int, default=12000)
    args = p.parse_args()
    train(args.corpus, args.output_dir, args.vocab_size)
