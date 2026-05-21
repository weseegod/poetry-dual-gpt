# 📊 Evaluation

Run per-rule evaluation on novel prompts:

```bash
cd poetry-dual-gpt
PYTHONPATH=. python3 evaluate/eval_rules.py
```

Outputs:
- `documents/rule_evaluation.md` — human-readable report
- `evaluate/rule_evaluation.json` — raw data

Requires:
- Trained checkpoints in `checkpoints/`
- Matching tokenizer at `tokenizer/poetry_bpe.model`
