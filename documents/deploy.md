# 🚀 Deploy: poetry-dual-gpt → doitho

End-to-end workflow from training to production.

---

## Architecture

```
poetry-dual-gpt/                          doitho/
├── src/
│   ├── generation.py  ── CANONICAL ──►  utils/generation.py
│   ├── model.py             ──pack──►   utils/model.py
│   └── tones.py             ──pack──►   utils/tones.py
├── deploy/utils/
│   └── inference.py   (wrapper) ──►     utils/inference.py
├── checkpoints/
│   └── doi_tho_best.pt      ──pack──►   utils/doitho.pt
└── tokenizer/
    └── poetry_bpe.model     ──pack──►   utils/poetry_bpe.model
```

**One `generate()` function.** `src/generation.py` is the single source of truth.  
Eval, CLI, and doitho all call the exact same code. `deploy/utils/inference.py`  
is a thin wrapper — imports `generate`, adds `load_model` + prompt builders.

---

## End-to-End Pipeline

After any model change (new training, new prompts, code change), run these 7 steps:

### Step 1 — Verify prompts

```bash
python evaluate/prompts.py
```
Must show: `✅ All prompts valid!` with 116 couplets, all 6+8 syllables.

### Step 2 — Rule evaluation

```bash
PYTHONPATH=. python evaluate/eval_rules.py
```
Check `All5` ≥ 88%. If it drops significantly, investigate before proceeding.

### Step 3 — Quality evaluation

```bash
PYTHONPATH=. python evaluate/eval_quality.py
```
Check: no empty outputs, lexical diversity > 0.85, BPE artifacts < 3%.

### Step 4 — Pack for doitho

```bash
bash scripts/pack_for_doitho.sh
```
Creates `deploy/doitho_utils.zip` with 6 files. Verifies checkpoint structure,  
tokenizer IDs, and model forward pass.

### Step 5 — Deploy to doitho

```bash
unzip -o deploy/doitho_utils.zip -d ../doitho/utils/
```

### Step 6 — Run doitho tests

```bash
cd ../doitho && python -m pytest backend/tests/test_v4_1_inference.py -v
```
Must show: `12 passed`.

### Step 7 — Verify eval == doitho

```python
# Same seed, same prompt, same model → must produce identical token IDs
import torch
torch.manual_seed(42)

# Eval
from src.generation import generate, build_prompt, decode_response
prompt = build_prompt("lục_line\nbát_line", include_trambong=True)
eval_ids, _ = generate(model, tok, prompt, device='cpu', rhyme_mode='soft')

# Doitho (via inference wrapper → same generation.generate)
torch.manual_seed(42)
from inference import build_doi_tho_prompt, generate as doitho_gen
doitho_p = build_doi_tho_prompt(l6, l8)
doitho_r = doitho_gen(doitho_p, model, tok, ...)

assert eval_ids == doitho_r['token_ids']  # Must be True
```

If this assertion fails, eval and doitho are using different code paths —  
**stop and fix before shipping**.

---

## Quick Sanity Check (one-liner)

```bash
cd poetry-dual-gpt && \
  python evaluate/prompts.py && \
  PYTHONPATH=. python evaluate/eval_rules.py && \
  PYTHONPATH=. python evaluate/eval_quality.py && \
  bash scripts/pack_for_doitho.sh && \
  unzip -o deploy/doitho_utils.zip -d ../doitho/utils/ && \
  cd ../doitho && python -m pytest backend/tests/test_v4_1_inference.py -v
```

---

## What Each Packed File Does

| File | Source | Role |
|------|--------|------|
| `doitho.pt` | `checkpoints/doi_tho_best.pt` | 31.5M checkpoint: `model_state_dict`, `model_config`, `vocab_size` |
| `poetry_bpe.model` | `tokenizer/poetry_bpe.model` | 12,000 BPE tokens. IDs: 0=pad, 1=start, 2=reply, 3=end, 9=linebreak |
| `model.py` | `src/model.py` | `PoetryDuelGPT` class. Config loaded from checkpoint at runtime |
| `tones.py` | `src/tones.py` | Tone, rhyme, diacritic utilities |
| `generation.py` | `src/generation.py` | **Canonical `generate()`.** Same code as eval and CLI |
| `inference.py` | `deploy/utils/inference.py` | Thin wrapper: `load_model`, prompt builders, `decode_doi_tho` |

## Why `inference.py` Is a Thin Wrapper

`deploy/utils/inference.py` no longer contains its own `generate()` function.  
It imports `generate` and `decode_response` from `src/generation.py` — the  
identical module used by evaluation and CLI. This guarantees:

1. **Eval = Production.** Same code path, same results.
2. **No drift.** Any improvement to `src/generation.py` automatically benefits doitho.
3. **Single source of truth.** One `generate()` function for all callers.

The only production-specific additions are:
- `load_model()` — checkpoint loading with key remapping
- `auto_tag()`, `build_doi_tho_prompt()`, `build_doi_tho_from_lines()` — prompt builders
- `decode_doi_tho()` — thin wrapper around `decode_response` with max_lines enforcement

---

## Tokenizer ID Contract (DO NOT BREAK)

These IDs are hard-coded in training, generation, and server code:

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<\|pad\|>` | Padding (ignored in loss) |
| 1 | `<\|start\|>` | Sequence start |
| 2 | `<\|reply\|>` | Separator between input and output |
| 3 | `<\|end\|>` | Sequence end (model generates this to stop) |
| 9 | `<\|linebreak\|>` | Line separator |
| 4 | `[LUC_BAT]` | Genre tag |
| 213 | `[TRAMBONG:NH]` | Trầm-Bổng (Ngang→Huyền) |
| 214 | `[TRAMBONG:HN]` | Trầm-Bổng (Huyền→Ngang) |
| 10+ | `[RHYME:*]` | Rhyme group tags |
| 147+ | `[TONE:*]` | Tone pattern tags |
| 215+ | Content | Vietnamese syllables and BPE subwords |

**Any new tokenizer must preserve these exact IDs.**

---

## Pre-Deploy Checklist

- [ ] `checkpoints/doi_tho_best.pt` is the correct checkpoint
- [ ] `tokenizer/poetry_bpe.model` has correct control token IDs (0,1,2,3,9)
- [ ] `src/model.py` is compatible with checkpoint (key names match)
- [ ] `src/generation.py` is the canonical generate function (eval = prod = CLI)
- [ ] `deploy/utils/inference.py` imports `generate` from `src/generation.py` (not own copy)
- [ ] `build_doi_tho_prompt` and `build_prompt` produce identical format
- [ ] `evaluate/prompts.py` passes verification (116 valid couplets)
- [ ] `eval_rules.py` All5 ≥ 88%
- [ ] `eval_quality.py` no empty outputs, BPE < 3%
- [ ] `bash scripts/pack_for_doitho.sh` runs without errors
- [ ] `cd ../doitho && pytest backend/tests/test_v4_1_inference.py` all pass
- [ ] Step 7: eval token IDs == doitho token IDs (same seed)
