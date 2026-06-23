# 🎵 Rhyme & Tone Conditioning — Reference

> v2.1 — Current implementation. All tags are single-token, all conditioning is active.

---

## Current Format (v2.1)

```
<|start|> [DOI_THO] [RHYME:X] [TONE:XXXXXX]
  line6 <|linebreak|> line8 <|reply|>
  line6_out <|linebreak|> line8_out <|end|>
```

| Tag | Source | Meaning |
|-----|--------|---------|
| `[RHYME:X]` | Position **8** of 8-syl line | Chain rhyme — output pos 6 should match |
| `[TONE:XXXXXX]` | 6-syl line tone pattern | Expected B-T-B-B on positions 2,4,6,8 |
| `<\|linebreak\|>` | — | Separates 6-syl and 8-syl lines (token id=9) |

All control tokens are **single special tokens** in the 12,000-word BPE tokenizer:
- `[RHYME:ong]` = token id 78
- `[RHYME:a]` = token id 10
- `[TONE:BBBTTB]` = token id 153
- `<|linebreak|>` = token id 9
- `[DOI_THO]` = token id 8

---

## Lục Bát Rules

### Rule 1: Internal Rhyme (vần lưng)

```
Line 1 (6 syl): Thân em  như thể  bông sen
                [1] [2]  [3] [4]  [5]  [6]
                                          ↑ rhyme of THIS line (pos 6)
Line 2 (8 syl): Trong đầm mà chẳng hôi tanh mùi bùn
                [1]   [2]  [3]  [4]  [5] [6]   [7] [8]
                                          ↑ MUST rhyme with pos 6 of line 1
                                                                  ↑ carries forward (chain rhyme)

For đối thơ: [RHYME:X] is extracted from pos 8 of the last 8-syl line.
This is the "chain rhyme" that the NEXT couplet's pos 6 must match.
```

**Vietnamese rhyme groups** (vần):

| Group | Examples |
|-------|----------|
| `a` | ta, ba, xa, nhà, gà, la, đà |
| `en` | sen, chen, đen, khen, len, phen |
| `ang` | trang, làng, đàng, vàng, sáng, nàng |
| `ong` | trong, lòng, sông, đồng, chồng, bông |
| `ơ` | thơ, chờ, bơ, ngơ, mơ, lơ |
| `iêu` | nhiêu, phiêu, tiêu, siêu, chiều |
| `inh` | tình, mình, hình, đình, linh, xinh |
| `an` | tan, ban, lan, ngàn, đàn, tàn |
| `uôn` | buồn, muộn, luôn, khuôn, nguồn |

### Rule 2: Tone Rules (luật bằng trắc)

```
6-syl line (Lục):  Position: 1    2    3    4    5    6
                   Tone:     -    B    T    T    B    B
                             ↑    ↑    ↑    ↑    ↑    ↑
                            free even sharp sharp even even

8-syl line (Bát):   Position: 1    2    3    4    5    6    7    8
                    Tone:     -    B    T    T    B    B    T    B
                              ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
                             free even sharp sharp even even sharp even

B = Bằng (level):  ngang (no mark), huyền ( ` )
T = Trắc (sharp):  sắc ( ´ ), nặng ( . ), hỏi ( ˀ ), ngã ( ~ )
```

### Rule 3: Syllable Count

- First line of each couplet: exactly **6** syllables
- Second line of each couplet: exactly **8** syllables
- Enforced by v3 post-processing (truncation to target)

---

## How Conditioning Works

During training, the model sees the tags as part of the input context:

```
Input:  <|start|> [DOI_THO] [RHYME:a] [TONE:TBTTBB] khóc than kể hết niềm tây ...
Target:          [DOI_THO] [RHYME:a] [TONE:TBTTBB] khóc than kể hết ...

The model learns via attention: when [RHYME:a] is in context,
generate words that rhyme with 'a'. The tags condition the model,
but the model doesn't need to predict them — we inject them at inference.
```

At inference, `auto_tag_doi_tho()` extracts `[RHYME:X]` and `[TONE:XXXXXX]` from the user's input couplet, injects them into the prompt, and the model generates a response conditioned on those tags.

---

## Current Accuracy (v2.1, step 4400/10000)

| Rule | Accuracy | Random baseline |
|------|----------|-----------------|
| R1: Internal rhyme | ~50% | 0.6% |
| R2: Tone pattern | ~88% | 6.2% |
| R3: Syllable count | ~71% → 100% (v3 enforced) | 6.7% |
| Chain rhyme (đối thơ) | ~32% | — |
| Stress test (valid output) | 79% | — |

Full evaluation: `evaluate/eval_rules.py`, `evaluate/eval_doi_tho.py`

---

## Implementation Files

| File | Role |
|------|------|
| `src/tones.py` | Tone classification + rhyme extraction (get_tone, get_rhyme_group, get_doi_tho_tags) |
| `src/preprocess_doi_tho.py` | Generate training corpus with tags injected |
| `src/sample.py` | CLI: auto_tag_doi_tho(), decode_doi_tho() |
| `client/server.py` | API: _auto_tag_doi_tho(), _decode_doi_tho() |
| `data/doi_tho_corpus.txt` | Training corpus (998K pairs, window=1 + window=2) |
| `tokenizer/poetry_bpe.model` | 12K BPE tokenizer, single-token control tags |
| `tests/test_tones.py` | Unit tests for tone/rhyme |

---

## Known Limitations

- Compound vowel nuclei (uô, ươ, iê) — ~90% correct rhyme group detection
- Low chain rhyme accuracy (32%) — will improve with more training (currently 44% done)
- BPE collapse on rare vocabulary — fixed by lowercase normalization in v3
- No Thất Ngôn support — not needed for current đối thơ scope
- No rhythm/grammatical/semantic conditioning — needs external tools (future)
