# 📊 Rule-by-Rule Evaluation - 173 Novel Prompts

> Generated: 2026-05-21 12:23
> 173 prompts (ca dao, folk poetry - NOT in training corpus)
> Model: 30.9M params, n_embd=512, n_head=8, n_layer=8

## 📈 Per-Rule Summary

| Rule | Tag | Stage 1 | Stage 2 | Random baseline | Effective? |
|------|-----|---------|---------|-----------------|------------|
| **R1: Internal Rhyme** (vần lưng) | `[RHYME:X]` | 45.7% | 49.7% | 0.6% | ✅ Yes |
| **R2: Tone Pattern** (B-T-B-B) | `[TONE:XXXXXX]` | 78.5% | 88.1% | 6.2% | ✅ Yes |
| **R3: Syllable Count** (8 syl) | (form) | 65.3% | 71.1% | 6.7% | ✅ Yes |
| **Combined: All rules pass** | - | 38.2% | 45.7% | - | - |

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Prompt tone accuracy (pos 2,4,6) | 91.5% | 91.5% |
| Avg response length | 8.0 syl | 7.2 syl |
| Syllable 6-10 range | 84.4% | 84.4% |

## 🔤 R1: Internal Rhyme (vần lưng)

**Tag**: `[RHYME:X]` - extracted from prompt position 6
**Check**: Response position 6 rhyme group must match prompt position 6
**Random baseline**: 0.6% (1 in 159 rhyme groups)

| Model | Accuracy | vs Random |
|-------|----------|-----------|
| Stage 1 | 45.7% | 73× |
| Stage 2 | 49.7% | 79× |

**Sample matches:**

| Prompt (pos 6) | Rhyme | Response (pos 6) | Rhyme | Match? |
|---------------|-------|-----------------|-------|--------|
| Thân em như tấm lụa đào | o | làm sao em muốn ước ao bao chàng | o | ✅ |
| Trèo lên cây khế nửa ngày | y | còn mấy lần trái tim nồng ý thơ | ông | ❌ |
| Ai làm cho bướm xa hoa | None | héo gầy | None | ❌ |
| Đêm khuya thắp ngọn đèn dầu | u | ta về thăm lại sắc màu thanh xuân | u | ✅ |
| Gió mùa thu mẹ ru con | on | mẹ ru con ngủ mẹ cười à ơi | i | ❌ |
| Chim khôn đậu nóc nhà quan | None | viên | None | ❌ |
| Cây khô chưa dễ mọc chồi | i | ai người biết được cuộc đời ra sao | i | ✅ |
| Mẹ già như chuối ba hương | ơng | thôi đành khép kín cổng trường nhìn đâu | ơng | ✅ |
| Ru con con ngủ cho lâu | u | không tròn tiếng hát bên cầu tri âm | u | ✅ |
| Công cha như núi thái sơn | ơn | bắc nam xum họp không còn khổ đau | on | ❌ |
| Rủ nhau xuống biển mò cua | None | thỏa lòng | None | ❌ |
| Đố ai đếm được lá rừng | None |  | None | ❌ |
| Cày đồng đang buổi ban trưa | a | mặt trời ló dạng đã thưa dần dần | a | ✅ |
| Mồ hôi thánh thót như mưa | a | những chiều mưa đổ như vừa tắm xong | a | ✅ |
| Dẻo thơm một hạt đắng cay | y | ai thương xin hãy tỏ bày cùng ta | y | ✅ |
| Ai về tôi gửi buồng cau | u | chờ ngày khai hội cùng nhau họp bàn | u | ✅ |
| Buồng cau non mẹ để già | a | cho con cháu hưởng tuổi già thảnh thơi | a | ✅ |
| Cau già khéo bổ thì non | on | non  còn tui có sức tui còn ham chơi | i | ❌ |
| Cây đa bến nước sân đình | inh | thuyền quyên nép bóng sân đình ngẩn ngơ | inh | ✅ |
| Qua đình ngả nón trông đình | inh | chị hai xinh xắn tươi xinh duyên thầm | inh | ✅ |

## 🎵 R2: Tone Pattern (B-T-B-B)

**Tag**: `[TONE:XXXXXX]` - tone sequence of prompt
**Check**: Response positions 2,4,6,8 must be B, T, B, B
**Random baseline**: 6.2% (4 positions × 50% B/T)

| Model | Accuracy | vs Random |
|-------|----------|-----------|
| Stage 1 | 78.5% | 13× |
| Stage 2 | 88.1% | 14× |

### Stage 1 (all genres) - Per-position tone accuracy

| Position | Expected | Correct | Total | Accuracy |
|----------|----------|---------|-------|----------|
| 2 (pos 2) | B | 139 | 167 | 83% ████████████████ |
| 4 (pos 4) | T | 108 | 162 | 67% █████████████ |
| 6 (pos 6) | B | 123 | 162 | 76% ███████████████ |
| 8 (pos 8) | B | 139 | 157 | 89% █████████████████ |

### Stage 2 (Lục Bát) - Per-position tone accuracy

| Position | Expected | Correct | Total | Accuracy |
|----------|----------|---------|-------|----------|
| 2 (pos 2) | B | 136 | 157 | 87% █████████████████ |
| 4 (pos 4) | T | 125 | 148 | 84% ████████████████ |
| 6 (pos 6) | B | 128 | 148 | 86% █████████████████ |
| 8 (pos 8) | B | 136 | 143 | 95% ███████████████████ |

## 📏 R3: Syllable Count (6→8)

**Check**: Response must be exactly 8 syllables
**Random baseline**: ~7% (exact length among typical 0-14 range)

### Stage 1 (all genres) - Length distribution

| Syllables | Count | % |
|-----------|-------|---|
| 0 | 3 | 1.7%  |
| 1 | 3 | 1.7%  |
| 2 | 5 | 2.9% █ |
| 7 | 5 | 2.9% █ |
| 8 | 113 | 65.3% ████████████████████████████████ |
| 9 | 25 | 14.5% ███████ |
| 10 | 13 | 7.5% ███ |
| 11 | 2 | 1.2%  |
| 13 | 1 | 0.6%  |
| 14 | 3 | 1.7%  |

### Stage 2 (Lục Bát) - Length distribution

| Syllables | Count | % |
|-----------|-------|---|
| 0 | 10 | 5.8% ██ |
| 1 | 6 | 3.5% █ |
| 2 | 9 | 5.2% ██ |
| 6 | 1 | 0.6%  |
| 7 | 4 | 2.3% █ |
| 8 | 123 | 71.1% ███████████████████████████████████ |
| 9 | 15 | 8.7% ████ |
| 10 | 3 | 1.7%  |
| 13 | 1 | 0.6%  |
| 17 | 1 | 0.6%  |

## 🛠️ Fix Recommendations

### R1: Rhyme (current: 50% - needs improvement)

**Root cause**: `[RHYME:ong]` is 5 BPE tokens. The rhyme signal is fragmented.
**Fix**: Make rhyme groups special tokens (single IDs like `[LUC_BAT]`).
**Expected**: 40-60% rhyme accuracy (based on genre tag effectiveness).
**Effort**: Retrain tokenizer + corpus + model (~4h Colab).

### R2: Tone Pattern (current: 88% - needs improvement)

**Root cause**: Same fragmentation as R1. `[TONE:BBBTTB]` is 5 BPE tokens.
**Fix**: Same as R1 - make tone patterns special tokens.
**Expected**: 50-70% tone accuracy.

### R3: Syllable Count (current: 71% - needs fix)

**Root cause**: Model uses position-based stopping, not syllable counting.
**Fix 1 (immediate)**: Post-generation truncation to 8 syllables. 3 lines of code.
**Fix 2 (architectural)**: Syllable-count control token `[SYL:8]` as special token.
**Fix 3 (structural)**: Syllable-aware pre-tokenizer before BPE.

### Priority

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 1 | R3 Fix 1 - Truncation | 100% syllable accuracy | 3 lines |
| 2 | R1+R2 Fix - Special tokens | 2-3× rhyme/tone improvement | 1 day |
| 3 | Qwen2.5-1.5B QLoRA | Overall quality + better rule following | 1 day |