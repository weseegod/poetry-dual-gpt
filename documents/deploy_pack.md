# 📦 Deploy: poetry-dual-gpt → doitho

How the trained model gets from `checkpoints/` to `doitho.net`.

---

## Architecture

```
poetry-dual-gpt/                    doitho/
├── checkpoints/                    ├── utils/
│   └── doi_tho_best.pt  ──pack──►  │   ├── doitho.pt          (380MB)
├── tokenizer/                      │   ├── poetry_bpe.model   (884KB)
│   └── poetry_bpe.model ──pack──►  │   ├── model.py
├── src/                            │   ├── tones.py
│   ├── model.py          ──pack──►  │   └── inference.py
│   └── tones.py          ──pack──►  ├── backend/
└── deploy/utils/                   │   ├── config.py   → CHECKPOINT = utils/doitho.pt
    └── inference.py      ──pack──►  │   └── services/
                                    │       └── model.py → imports from utils/
                                    └── frontend/
```

**5 files are packed.** The doitho project imports from `utils/` as a flat module — no `src/` nesting, no training dependencies.

---

## Pack & Deploy

### 1. Update the checkpoint

Before packing, ensure the default checkpoint is the one you want to ship:

```bash
# Point default to the version you want to ship:
cp checkpoints/doi_tho_best_v4.2.3.pt checkpoints/doi_tho_best.pt
cp tokenizer/poetry_bpe_v4.2.3.model tokenizer/poetry_bpe.model
```

### 2. Run the pack script

```bash
cd poetry-dual-gpt
bash scripts/pack_for_doitho.sh
```

This:
- Copies 5 files to `deploy/_pack_tmp/`
- Verifies checkpoint structure, tokenizer IDs, model forward pass
- Creates `deploy/doitho_utils.zip`

### 3. Deploy to doitho

```bash
unzip -o deploy/doitho_utils.zip -d ../doitho/utils/
```

### 4. Run tests

```bash
cd ../doitho
python -m pytest backend/tests/ -x -q
```

Key test file: `backend/tests/test_v4_1_inference.py` — verifies:
- Model loads without error
- Prompt building produces correct format
- Generation produces 6+8 syllable output (3/3 times)
- No control token artifacts leak into output
- Edge cases don't crash

---

## What Each Packed File Does

| File | Source | Role |
|------|--------|------|
| `doitho.pt` | `checkpoints/doi_tho_best.pt` | 31.5M parameter checkpoint. Contains `model_state_dict`, `model_config`, `vocab_size`. |
| `poetry_bpe.model` | `tokenizer/poetry_bpe.model` | 12,000-token ByteLevel BPE tokenizer. Token IDs: 0=pad, 1=start, 2=reply, 3=end, 9=linebreak. |
| `model.py` | `src/model.py` | `PoetryDuelGPT` class. Architecture: n_embd=512, 8 layers, 8 heads, block_size=256. Config loaded from checkpoint dict at runtime. |
| `tones.py` | `src/tones.py` | Vietnamese tone classification (Bằng/Trắc), rhyme group extraction, diacritic detection (ngang/huyền), Trầm-Bổng tag generation. |
| `inference.py` | `deploy/utils/inference.py` | Production inference module. Self-contained — does NOT import from `src/generation.py`. Contains its own `generate()`, prompt builders, and `decode_doi_tho()`. |

---

## Why `inference.py` Is Separate from `src/generation.py`

The production inference module (`deploy/utils/inference.py`) is intentionally self-contained:

1. **No training dependencies** — doitho doesn't need `src/dataset.py`, `src/train.py`, `src/preprocess.py`
2. **Strict linebreak enforcement** — production forces exact 6+8 syllables (users expect correct output). Eval mode in `src/generation.py` reports raw quality by default.
3. **Single-file deployment** — zero cross-file imports within `utils/`. All prompt building + generation + decoding in one file.
4. **Hard linebreak control** — production suppresses `<|linebreak|>` before syllable 6, forces it at syllable 6, suppresses `<|end|>` before syllable 8 in output line 2. This guarantees users always see well-formed 6+8 couplets.

**When updating `src/generation.py`**, check if the production `inference.py` needs the same changes. Key differences:

| Feature | `src/generation.py` (eval) | `deploy/utils/inference.py` (prod) |
|---------|---------------------------|-----------------------------------|
| Linebreak mode | Passive (split after generation) | **Active** (force at exact positions) |
| Syllable enforcement | Off by default (raw quality) | **Always ON** |
| Rhyme constraint | Soft (boost +2.0) | Soft (boost +2.0) — synced v4.2.3 |
| Control token suppression | None | Suppresses all [TAG] and <\|control\|> |
| Repetition penalty | -1.2, last 16 tokens | -1.2, last 16 tokens |
| Top-k | 50 | 50 |
| Top-p | 0.92 | 0.92 — synced v4.2.3 |

---

## Checklist: Before Deploying a New Version

- [ ] `checkpoints/doi_tho_best.pt` is the correct checkpoint
- [ ] `tokenizer/poetry_bpe.model` has correct control token IDs (0,1,2,3,9)
- [ ] `src/model.py` is compatible with the checkpoint (key names match)
- [ ] `deploy/utils/inference.py` generation logic matches current best practices
- [ ] `bash scripts/pack_for_doitho.sh` runs without errors
- [ ] `cd ../doitho && python -m pytest backend/tests/ -x -q` all pass
- [ ] Manual test: generate a poem, verify it reads like Vietnamese
- [ ] `src/tones.py` functions work with current tokenizer

---

## Tokenizer ID Contract (DO NOT BREAK)

These IDs are hard-coded in training, generation, and server code:

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<\|pad\|>` | Padding (ignored in loss) |
| 1 | `<\|start\|>` | Sequence start |
| 2 | `<\|reply\|>` | Separator between input and output |
| 3 | `<\|end\|>` | Sequence end (model generates this to stop) |
| 9 | `<\|linebreak\|>` | Line separator (model generates this between lines) |
| 4 | `[LUC_BAT]` | Genre tag |
| 213 | `[TRAMBONG:NH]` | Trầm-Bổng tag (Ngang→Huyền) |
| 214 | `[TRAMBONG:HN]` | Trầm-Bổng tag (Huyền→Ngang) |
| 10+ | `[RHYME:*]` | Rhyme group tags |
| 147+ | `[TONE:*]` | Tone pattern tags |
| 215+ | Content | Vietnamese syllables and BPE subwords |

**Any new tokenizer must preserve these exact IDs.** Retraining with changed IDs requires retraining the model from scratch.
