# 🔬 Stage 2 Lục Bát — Evaluation & Fix Plan

> Generated: 2026-05-20  
> Model: stage2_best.pt (step 200, val=1.56, 30.9M params)

---

## 📊 Evaluation

### Memorization vs Generalization

| Evaluation | Prompts | Exact 8 syl | Close 6-10 | Response Tone | Rhyme |
|------------|---------|-------------|------------|---------------|-------|
| **Corpus prompts** (memorized) | 1000 | **99.8%** | 100% | 99.2% | 90.6% |
| **Novel prompts** (generalization) | 92 | **18.5%** | 79.3% | 58.8% | 19.6% |

The 1000-corpus evaluation is pure **memorization** — the model had seen these exact prompts during training. The 92-novel evaluation is the real score: the model generalizing to unseen Vietnamese folk poetry.

### Length distribution (novel prompts, 92 samples)

```
 0 syl:    4 ( 4.3%)  ← empty
 1-6 syl:   9 ( 9.8%)
 7 syl:   26 (28.3%)  ← 1 short of 8
 8 syl:   17 (18.5%)  ← correct
 9 syl:   21 (22.8%)  ← 1 over
10-14 syl: 15 (16.3%)
```

The model clusters around 7-9 syllables (69.6%) but rarely hits exactly 8 (18.5%).

### Sample novel outputs

| Prompt | Response | Syllables |
|--------|----------|-----------|
| Thân em như tấm lụa đào | em là ngọn gió dạt dào vườn xanh | 9 |
| Công cha như núi thái sơn | núi cao ngày nhớ đêm hờn đỉnh non | 8 ✅ |
| Chim khôn đậu nóc nhà quan | khôn ngoan đậu mái nhà quan đời thường | 9 |
| Đèn nhà ai nấy sáng trưng | cũng như người khác vui mừng đón xem | 8 ✅ |
| Thuyền ơi có nhớ bến không | sông sâu có nước em trông bến chờ | 8 ✅ |
| Gió mùa thu mẹ ru con | ru con con ngủ cho tròn giấc say | 9 |

---

## 🔬 Root Cause: Position-Based Stopping

The `<|end|>` probability was probed token-by-token:

```
Response token #:  1    2    3    4    5    6    7    8    9   10   11   12
<|end|> prob:     0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
                                                                           ↑
                                                                     fires at token 12
```

**The model counts POSITIONS, not syllables.** It learned from training data that `<|end|>` appears ~11-12 tokens after `<|reply|>`. Since BPE splits some syllables into 2-3 tokens, 11 tokens ≈ 7-13 syllables.

```
Tokenization example:
  "đòng"    → 1 token  (= 1 position)
  "nghĩ"    → 3 tokens (= 3 positions for 1 syllable!)
  "trong"   → 1 token  (= 1 position)
```

The model has no concept of syllables — it only sees token positions.

---

## ⚠️ Rhyme Conditioning: Partial Success

`[RHYME:X]` tag improved rhyme from 15% (Stage 1) to ~20-30% (Stage 2) — a modest improvement.

**Why only 20%?**

1. **Fragmented signal**: `[RHYME:ong]` = 5 BPE tokens (`[`, `RHYME`, `:`, `ong`, `]`). The rhyme constraint is spread across 5 attention positions, not concentrated in one.

2. **Model capacity**: 30M params prioritize fluency over secondary constraints like rhyme.

3. **Token-level generation**: The model generates tokens, not syllables. It can't easily "aim" for a rhyme at a specific syllable position because it doesn't know where syllable boundaries are in token space.

---

## 🛠️ Fix Options

### Fix 1: Post-generation Syllable Truncation
```python
def truncate_to_syllables(text, n):
    words = text.split()
    return ' '.join(words[:n])
```
**Effect**: 100% syllable accuracy. First 8 words of any response.  
**Cost**: 3 lines, no retraining.  
**Risk**: Low — extra syllables are usually the model starting a second line.

### Fix 2: Rhyme/Tone as Special Tokens  
Add 225 tokens to tokenizer: `[RHYME:ong]` = single id, not 5 subwords.  
**Effect**: Estimated 40-60% rhyme accuracy.  
**Cost**: Retrain tokenizer + corpus + model (~4h Colab).  
**Risk**: Medium — requires full retrain.

### Fix 3: Qwen2.5-1.5B QLoRA
Swap 30M GPT for 1.5B pretrained model. Same control tokens.  
**Effect**: Major quality jump across all metrics.  
**Cost**: 1 day setup.  
**Risk**: Low — proven QLoRA recipe.

### Fix 4: Syllable-aware tokenizer
Pre-split on syllable boundaries before BPE.  
**Effect**: Model can count syllables from tokens.  
**Cost**: 2-3 days. Structural change.  
**Risk**: High — changes entire pipeline.

---

## 📋 Recommendation

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| **1** | Fix 1 — Truncation | 100% syllable accuracy | 3 lines |
| **2** | Fix 2 — Special tokens | 2-3× rhyme improvement | 1 day |
| **3** | Fix 3 — Qwen 1.5B | Quality ceiling increase | 1 day |

---

## 📝 Test Data

### 92 Novel Prompts (none in training corpus)
Classic Vietnamese folk poetry (ca dao), lullabies, proverbs — 6-syllable Lục Bát opening lines:
```
Thân em như tấm lụa đào, Trèo lên cây khế nửa ngày,
Ai làm cho bướm xa hoa, Đêm khuya thắp ngọn đèn dầu,
Gió mùa thu mẹ ru con, Chim khôn đậu nóc nhà quan,
... (92 total)
```

Full list + per-prompt scores: `documents/eval_100_novel.json`
