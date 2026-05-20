# 🎵 Phase 9: Rhyme & Tone Conditioning

## 📊 Implementation Status

### Lục Bát (6→8)

| # | Rule | Priority | Status | How |
|---|------|----------|--------|-----|
| 1 | **Internal rhyme** (vần lưng) | HIGH | ✅ Implemented | `[RHYME:X]` — extracted from position 6 of prompt |
| 2 | **Tone pattern** (B-T-B / B-T-B-B) | RECOMMENDED | ✅ Implemented | `[TONE:XXXXXX]` — tone sequence of prompt |
| 3 | **Rhythm** (2/2/2, 4/2) | RECOMMENDED | ❌ Skipped | Needs word segmentation, subjective |

### Thất Ngôn (7→7)

| # | Rule | Priority | Status | How |
|---|------|----------|--------|-----|
| 4 | **Đối âm** (tonal contrast between lines) | HIGH | ✅ Learned | Model learns from B-T-B ↔ T-B-T patterns in data |
| 5 | **2nd syllable tone** | HIGH | ✅ Implemented | `[LINK2:X]` — tone of position 2, conditions contrast |
| 6 | **End rhyme** (position 7) | RECOMMENDED | ❌ Skipped | Single-syllable constraint, low impact |
| 7 | **Grammatical parallelism** | HIGH | ❌ Skipped | Needs POS tagger (underthesea/pyvi) |
| 8 | **Semantic parallelism** | HIGH | ❌ Skipped | Needs semantic model, research-level |

### What the model sees during training

```
[LUC_BAT] [RHYME:ong] [TONE:BBBTTB] Thân em như chẽn lúa đòng <|reply|> ...
[THAT_NGON] [LINK2:B] Lom khom dưới núi tiều vài chú <|reply|> ...
```

The model learns: "When I see `[RHYME:ong]`, my 6th syllable should rhyme with 'ong'."
Same mechanism as `[LUC_BAT]` → "generate 8-syllable response" — proven to work.

---

## Table of Contents

1. [Overview](#overview)
2. [Lục Bát Rules](#lục-bát-rules)
3. [Thất Ngôn Rules](#thất-ngôn-rules)
4. [Implementation Strategy](#implementation-strategy)
5. [Step 1: Tone Classification](#step-1-tone-classification)
6. [Step 2: Rhyme Extraction](#step-2-rhyme-extraction)
7. [Step 3: Training Data Injection](#step-3-training-data-injection)
8. [Step 4: Generation-Time Conditioning](#step-4-generation-time-conditioning)
9. [Step 5: Evaluation Metrics](#step-5-evaluation-metrics)
10. [What We Can't Automate (Yet)](#what-we-cant-automate-yet)
11. [Files Affected](#files-affected)

---

## Overview

Currently the model sees only:
```
<|start|> [LUC_BAT] prompt_six_syl reply_eight_syl <|end|>
```

After Phase 9, it will see:
```
<|start|> [LUC_BAT] [RHYME:en] [TONE:BTB] prompt_six_syl <|reply|> reply_eight_syl <|end|>
```

The model learns: *"When I see `[RHYME:en]` and a 6-syl prompt ending in 'sen', my response's 6th syllable should rhyme with 'en'."*

---

## Lục Bát Rules

### Rule 1: Internal Rhyme (vần lưng) — HIGH PRIORITY

```
Line 1 (6 syl): Thân em  như thể  bông sen
                [1] [2]  [3] [4]  [5]  [6]
                                          ↑ rhyme HERE
Line 2 (8 syl): Trong đầm mà chẳng hôi tanh mùi bùn
                [1]   [2]  [3]  [4]  [5] [6]   [7] [8]
                                          ↑ MUST rhyme with pos 6 of line 1
                                                                  ↑ carries forward

Rule: syllable 6 of line 1 MUST share tone/rhyme with syllable 6 of line 2.
      syllable 8 of line 2 carries the rhyme to the NEXT couplet (if any).
```

**Vietnamese rhyme groups** (vần) are based on the final sound:

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
| `ôi` | tôi, trôi, ngồi, chồi, khôi |

### Rule 2: Tone Rules (luật bằng trắc) — RECOMMENDED

```
Lục Bát tone pattern:

Line 1 (6 syl):  Position: 1    2    3    4    5    6
                 Tone:     -    B    T    T    B    B
                           ↑    ↑    ↑    ↑    ↑    ↑
                          free even sharp sharp even even

Line 2 (8 syl):  Position: 1    2    3    4    5    6    7    8
                 Tone:     -    B    T    T    B    B    T    B
                           ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
                          free even sharp sharp even even sharp even

B = Bằng (level):  ngang (no mark), huyền ( ` )
T = Trắc (sharp):  sắc ( ´ ), nặng ( . ), hỏi ( ˀ ), ngã ( ~ )

Famous example:
  Đói lòng ăn nửa trái sim,        (positions 2,4,6: B-T-B)
  Uống lưng bát nước đi tìm người thương.  (positions 2,4,6,8: B-T-B-B)
```

The pattern `B-T-B` for line 1 and `B-T-B-B` for line 2 is the standard, though poets occasionally deviate for effect.

### Rule 3: Rhythm (nhịp điệu) — RECOMMENDED

```
2/2/2 rhythm:
  Thân em // như thể // bông sen,
  Trong đầm // mà chẳng // hôi tanh mùi bùn.

4/2 rhythm:
  Trăm năm trong cõi // người ta,
  Chữ tài chữ mệnh // khéo là ghét nhau.

2/4 rhythm:
  Gió đưa // cành trúc la đà,
  Tiếng chuông // Trấn Võ canh gà Thọ Xương.
```

Lục bát typically uses 2/2/2 or 4/2 rhythm. Breaking at the wrong boundary sounds unnatural.

---

## Thất Ngôn Rules

Thất ngôn is a 7-syllable form with **parallel couplets** (đối). The two lines mirror and contrast each other.

### Rule 1: Tonal Parallelism (Đối Âm) — HIGH PRIORITY

**Đối âm** means corresponding positions in the two lines must have **opposite** tones (Bằng ↔ Trắc). This creates the musical contrast that defines a good parallel couplet.

```
Line 1:  Lom  khom  dưới  núi   tiều  vài  chú
         (-)  B     (-)   T     (-)   B    (-)
Line 2:  Lác  đác  bên   sông  chợ   mấy  nhà
         (-)  T     (-)   B     (-)   T    (-)

Positions 2,4,6: Line1 = B-T-B, Line2 = T-B-T  ← CONTRAST ✓
In this excellent couplet, ALL positions (not just 2,4,6) contrast.
```

**Important distinction:**

| Concept | What it controls | Rule |
|---------|-----------------|------|
| **Đối âm** (tonal parallelism) | Contrast **between** two lines | Corresponding positions should have opposite tones (B ↔ T) |
| **Nhị tứ lục phân minh** | Tone pattern **inside** each line | Positions 2, 4, 6 must strictly follow B-T-B or T-B-T |

In a strong parallel couplet, the two lines contrast at positions 2, 4, and 6 naturally — because one line follows B-T-B while the other follows T-B-T. But đối âm (contrast) is the rule between lines; nhị tứ lục is the rule within each line.

**How we implement this:** We don't directly enforce per-position contrast (that's too rigid). Instead, we give the model `[LINK2:X]` to condition the 2nd syllable tone, which is the strongest signal. The model learns the contrast pattern from training data.

### Rule 2: 2nd Syllable Tone Linking — HIGH PRIORITY

```
Đã bấy lâu nay bác tới nhà,
Trẻ thì đi vắng, chợ thời xa.

Line 1, pos 2: "bấy" → BẰNG
Line 2, pos 2: "thì" → BẰNG  ← MUST match
```

The 2nd syllable of the output line must share the same tone as the 2nd syllable of the input line.

### Rule 3: End Rhyme — RECOMMENDED (not implemented)

```
Tiếng suối trong như tiếng hát xa,
Trăng lồng cổ thụ, bóng lồng hoa.

Line 1, pos 7: "xa" → BẰNG
Line 2, pos 7: "hoa" → BẰNG  ← same tone
```

### Rule 4: Grammatical + Semantic Parallelism — HIGH PRIORITY (but hard to automate)

```
Lom khom (verb)  dưới núi (location)  tiều vài chú (noun phrase)
Lác đác (verb)  bên sông (location)  chợ mấy nhà (noun phrase)

Each position pair shares grammatical category.
Meanings complement (mountain vs river, people vs market).
```

**This requires a Vietnamese POS tagger and is NOT automatable with our current tools.** Mark as "manual verification only" for now.

---

## Implementation Strategy

We'll implement rules we CAN automate (tone, rhyme, syllable count) and leave semantic/grammatical rules for manual evaluation.

### What we CAN inject into training data:

| Rule | Lục Bát | Thất Ngôn | How |
|------|---------|-----------|-----|
| Internal rhyme (vần lưng) | ✅ | N/A | `[RHYME:X]` from position 6 of prompt |
| Tone pattern (B-T-B / B-T-B-B) | ✅ | N/A | `[TONE:XXXXXX]` from prompt's tones |
| Tone linking (2nd syllable) | N/A | ✅ | `[LINK2:X]` from position 2 of prompt |
| Đối âm (tonal contrast between lines) | N/A | Learned | Model learns from L1=B-T-B, L2=T-B-T patterns in data |
| Rhythm pattern | ❌ | N/A | Needs word segmentation |
| End rhyme | N/A | Not implemented | Single-syllable constraint, low impact |
| Grammatical parallelism | ❌ | ❌ | Needs POS tagger |
| Semantic parallelism | ❌ | ❌ | Needs semantic model |

### Control token format:

```
For Lục Bát:
  [LUC_BAT] [RHYME:en] [TONE:BTB]    → rhyme group + prompt tone pattern

For Thất Ngôn:
  [THAT_NGON] [LINK2:B]            → 2nd-syl tone for tonal contrast (đối âm)
```

The model sees these tags and learns: *"When generating after `[RHYME:en]`, my response's 6th syllable should sound like 'en'."*

---

## Step 1: Tone Classification

Create `src/tones.py` — reusable tone utilities.

### 1.1 Vietnamese tones

```python
# src/tones.py

# Bằng (level): ngang, huyền
BANG_VOWELS = set(
    "aăâeêioôơuưy"                              # ngang (no diacritic)
    "àằầèềìòồờùừỳ"                              # huyền ( ` )
    "AĂÂEÊIOÔƠUƯY"
    "ÀẰẦÈỀÌÒỒỜÙỪỲ"
)

# Trắc (sharp): sắc, nặng, hỏi, ngã
TRAC_VOWELS = set(
    "áắấéếíóốớúứý"                              # sắc ( ´ )
    "ạặậẹệịọộợụựỵ"                              # nặng ( . )
    "ảẳẩẻểỉỏổởủửỷ"                              # hỏi ( ̉ )
    "ãẵẫẽễĩõỗỡũữỹ"                              # ngã ( ~ )
    "ÁẮẤÉẾÍÓỐỚÚỨÝ"
    "ẠẶẬẸỆỊỌỘỢỤỰỴ"
    "ẢẲẨẺỂỈỎỔỞỦỬỶ"
    "ÃẴẪẼỄĨÕỖỠŨỮỸ"
)

def get_tone(syllable: str) -> str:
    """Return 'B' (bằng) or 'T' (trắc) for a syllable."""
    for ch in syllable:
        if ch in TRAC_VOWELS:
            return "T"
        if ch in BANG_VOWELS:
            return "B"
    return "B"  # default (no marked vowels = ngang = bằng)

def get_tone_sequence(line: str) -> str:
    """Return tone string for a line, e.g. 'BBTBBT'."""
    syls = line.strip().split()
    return "".join(get_tone(s) for s in syls)

def check_luc_bat_tones(line6: str, line8: str) -> dict:
    """
    Check Lục Bát tone rules.
    Expected: line6 = pos2=B, pos4=T, pos6=B
              line8 = pos2=B, pos4=T, pos6=B, pos8=B
    """
    s6 = line6.strip().split()
    s8 = line8.strip().split()

    result = {"ok": True, "details": []}

    # Line 1: positions 2,4,6
    expected_6 = {2: "B", 4: "T", 6: "B"}
    for pos, want in expected_6.items():
        if pos > len(s6):
            break
        got = get_tone(s6[pos - 1])
        ok = got == want
        if not ok:
            result["ok"] = False
        result["details"].append(f"L1_p{pos}={got}({want}){'✓' if ok else '✗'}")

    # Line 2: positions 2,4,6,8
    expected_8 = {2: "B", 4: "T", 6: "B", 8: "B"}
    for pos, want in expected_8.items():
        if pos > len(s8):
            break
        got = get_tone(s8[pos - 1])
        ok = got == want
        if not ok:
            result["ok"] = False
        result["details"].append(f"L2_p{pos}={got}({want}){'✓' if ok else '✗'}")

    return result

def check_that_ngon_246_contrast(line1: str, line2: str) -> dict:
    """
    Check Thất Ngôn 2-4-6 contrast rule.
    Positions 2,4,6 must contrast: if L1=B, L2=T; if L1=T, L2=B.
    """
    s1 = line1.strip().split()
    s2 = line2.strip().split()

    result = {"ok": True, "details": []}
    for pos in [2, 4, 6]:
        if pos > len(s1) or pos > len(s2):
            break
        t1, t2 = get_tone(s1[pos - 1]), get_tone(s2[pos - 1])
        ok = t1 != t2
        if not ok:
            result["ok"] = False
        result["details"].append(f"p{pos}={t1}vs{t2}{'✓' if ok else '✗'}")
    return result
```

---

## Step 2: Rhyme Extraction

### 2.1 Vietnamese rhyme grouping

A syllable's rhyme is its **final sound** (vần). Extract by finding the last vowel + everything after it, after stripping tone marks.

```python
# src/tones.py (continued)

# Tone mark → base vowel mapping
TONE_TO_BASE = {}
for base, toned in [
    ("a", "àáảãạ"), ("ă", "ằắẳẵặ"), ("â", "ầấẩẫậ"),
    ("e", "èéẻẽẹ"), ("ê", "ềếểễệ"),
    ("i", "ìíỉĩị"), ("o", "òóỏõọ"), ("ô", "ồốổỗộ"),
    ("ơ", "ờớởỡợ"), ("u", "ùúủũụ"), ("ư", "ừứửữự"),
    ("y", "ỳýỷỹỵ"),
]:
    for t in toned:
        TONE_TO_BASE[t] = base
        TONE_TO_BASE[t.upper()] = base.upper()

VOWELS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY")

def get_rhyme_group(syllable: str) -> str:
    """
    Extract rhyme group from a syllable.
    'sen' → 'en', 'buồn' → 'uôn', 'thương' → 'ương'
    """
    # Step 1: Strip tone mark → base letters
    base = "".join(TONE_TO_BASE.get(c, c) for c in syllable).lower()

    # Step 2: Find last vowel position
    last_vowel_idx = -1
    for i, c in enumerate(base):
        if c in VOWELS:
            last_vowel_idx = i

    if last_vowel_idx < 0:
        return base  # no vowel found, return as-is

    # Step 3: Return everything from last vowel onward
    return base[last_vowel_idx:]

def get_luc_bat_rhyme_tag(line6: str) -> str:
    """
    Get the rhyme tag for a 6-syl Lục Bát prompt.
    Returns the rhyme group of the 6th syllable.
    """
    syls = line6.strip().split()
    if len(syls) >= 6:
        rhyme = get_rhyme_group(syls[5])
        return f"[RHYME:{rhyme}]"
    return ""
```

### 2.2 Examples

```python
get_rhyme_group("sen")        → "en"
get_rhyme_group("tanh")       → "anh"
get_rhyme_group("buồn")       → "uôn"
get_rhyme_group("thương")     → "ương"
get_rhyme_group("ta")         → "a"
get_rhyme_group("nhau")       → "au"
get_rhyme_group("tiếng")      → "iêng"
get_rhyme_group("sông")       → "ông"

# Do "sen" and "chen" rhyme?  YES — both are "en"
# Do "sen" and "sông" rhyme?  NO  — "en" ≠ "ông"
```

---

## Step 3: Training Data Injection

Modify `src/preprocess.py` to inject control tokens into training pairs.

### 3.1 Updated pair format

**Lục Bát:**
```
<|start|> [LUC_BAT] [RHYME:en] [TONE:BTB] sen_six_syl <|reply|> tanh_eight_syl <|end|>
```

**Thất Ngôn:**
```
<|start|> [THAT_NGON] [LINK2:T] seven_syl_prompt <|reply|> seven_syl_reply <|end|>
```

### 3.2 Modified `make_pairs()` for Lục Bát

```python
def make_pairs(lines, genre):
    rule = GENRE_RULES[genre]
    tag = rule["tag"]
    p_target = rule["prompt_syl"]
    r_target = rule["reply_syl"]

    pairs = []
    for i in range(0, len(lines) - 1, 2):
        if i + 1 >= len(lines):
            break
        prompt, reply = lines[i], lines[i + 1]

        if count_syllables(prompt) != p_target or count_syllables(reply) != r_target:
            continue

        # ── NEW: Inject control tokens ──
        extras = ""

        if genre == "lục bát":
            # Rhyme tag from 6th syllable of prompt
            rhyme = get_luc_bat_rhyme_tag(prompt)
            # Tone pattern of prompt
            tone_seq = get_tone_sequence(prompt)
            tone_tag = f"[TONE:{tone_seq[:6]}]" if len(tone_seq) >= 6 else ""
            extras = f" {rhyme} {tone_tag}".strip()

        elif genre == "bảy chữ":
            # Tone of 2nd syllable (for tonal contrast)
            syls = prompt.split()
            if len(syls) >= 2:
                link2 = get_tone(syls[1])
                extras += f" [LINK2:{link2}]"

        pairs.append(f"{START} {tag}{f' {extras}' if extras else ''} {prompt} {REPLY} {reply} {END}")
    return pairs
```

### 3.3 Modified `make_pairs_song_that()`

```python
def make_pairs_song_that(lines):
    pairs = []
    i = 0
    while i + 3 < len(lines):
        l1, l2 = lines[i], lines[i + 1]       # 7-syl couplet
        l3, l4 = lines[i + 2], lines[i + 3]   # 6→8 Lục Bát

        # 7-7 pair → [THAT_NGON] with linking
        if count_syllables(l1) == 7 and count_syllables(l2) == 7:
            extras = ""
            s1 = l1.split()
            if len(s1) >= 2:
                extras += f" [LINK2:{get_tone(s1[1])}]"
            pairs.append(f"{START} [THAT_NGON]{extras} {l1} {REPLY} {l2} {END}")

        # 6-8 pair → [LUC_BAT] with rhyme
        if count_syllables(l3) == 6 and count_syllables(l4) == 8:
            rhyme = get_luc_bat_rhyme_tag(l3)
            tone_seq = get_tone_sequence(l3)
            tone_tag = f"[TONE:{tone_seq[:6]}]" if len(tone_seq) >= 6 else ""
            pairs.append(f"{START} [LUC_BAT] {rhyme} {tone_tag} {l3} {REPLY} {l4} {END}")

        i += 4
    return pairs
```

### 3.4 New training data examples

```
<|start|> [LUC_BAT] [RHYME:en] [TONE:BBTTBB] Thân em như thể bông sen <|reply|> Trong đầm mà chẳng hôi tanh mùi bùn <|end|>
<|start|> [LUC_BAT] [RHYME:im] [TONE:BBTBTB] Đói lòng ăn nửa trái sim <|reply|> Uống lưng bát nước đi tìm người thương <|end|>
<|start|> [THAT_NGON] [LINK2:B] Tiếng suối trong như tiếng hát xa <|reply|> Trăng lồng cổ thụ bóng lồng hoa <|end|>
<|start|> [THAT_NGON] [LINK2:T] Lom khom dưới núi tiều vài chú <|reply|> Lác đác bên sông chợ mấy nhà <|end|>
```

---

## Step 4: Generation-Time Conditioning

### 4.1 Server.py — auto-inject control tokens

```python
# client/server.py — generate()

# Auto-wrap genre if not tagged
if not prompt.startswith("["):
    syl = len(prompt.split())
    tag = "[THAT_NGON]" if syl == 7 else "[LUC_BAT]"
    prompt = f"{tag} {prompt}"

# ── NEW: Inject rhyme/tone tags ──
if "[LUC_BAT]" in prompt and "[RHYME:" not in prompt:
    line = prompt.replace("[LUC_BAT]", "").strip()
    syls = line.split()
    if len(syls) >= 6:
        from tones import get_rhyme_group, get_tone_sequence
        rhyme = get_rhyme_group(syls[5])
        tone_seq = get_tone_sequence(line)
        tone_part = f" [TONE:{tone_seq[:6]}]" if len(tone_seq) >= 6 else ""
        prompt = prompt.replace("[LUC_BAT]", f"[LUC_BAT] [RHYME:{rhyme}]{tone_part}")

if "[THAT_NGON]" in prompt and "[LINK2:" not in prompt:
    line = prompt.replace("[THAT_NGON]", "").strip()
    syls = line.split()
    extras = ""
    if len(syls) >= 2:
        extras += f" [LINK2:{get_tone(syls[1])}]"
    if extras:
        prompt = prompt.replace("[THAT_NGON]", f"[THAT_NGON]{extras}")
```

### 4.2 Sample.py — same injection

```python
# src/sample.py — in generate() or before calling it
if "[LUC_BAT]" in prompt and "[RHYME:" not in prompt:
    # ... same injection logic ...
```

### 4.3 How this helps the model

During training, the model learned:
```
[LUC_BAT] [RHYME:en] ... bông sen <|reply|> ... hôi tanh ...
             ↑                                              ↑
          "en" rhyme                                  position 6 = "tanh" → "anh"
```

During generation:
```
User: "Thân em như thể bông sen"
  → auto-injects: "[LUC_BAT] [RHYME:en] [TONE:BBTTBB] Thân em như thể bông sen"
  → model sees [RHYME:en] and biases position 6 of response toward "en" rhyme
  → model sees [TONE:BBTTBB] and biases positions 2,4,6 toward B-T-B pattern
```

The control tokens act as *soft constraints* — the model learns the correlation but isn't forced. Higher temperature weakens the constraint, lower temperature strengthens it.

---

## Step 5: Evaluation Metrics

### 5.1 What to measure

After training with rhyme conditioning, evaluate:

```python
def evaluate_rhyme(response, expected_rhyme_group):
    """Check if response's 6th syllable rhymes with expected group."""
    syls = response.split()
    if len(syls) >= 6:
        return get_rhyme_group(syls[5]) == expected_rhyme_group
    return False

def evaluate_luc_bat_full(prompt, response):
    """Full Lục Bát evaluation."""
    return {
        "syllables_ok": count_syllables(response) == 8,
        "rhyme_ok": evaluate_rhyme(response, get_rhyme_group(prompt.split()[5])),
        "tones_ok": check_luc_bat_tones(prompt, response)["ok"],
    }
```

### 5.2 Success targets

| Metric | Before Phase 9 | After Phase 9 (target) |
|--------|---------------|----------------------|
| Lục Bát rhyme accuracy | ~30% | 60-70% |
| Lục Bát tone correctness | ~60% | 80-90% |
| Thất Ngôn 2-4-6 contrast | ~20% | 50-60% |
| Thất Ngôn 2nd-syl matching | random | 60-70% |

---

## What We Can't Automate (Yet)

### Grammatical parallelism

```
Lom khom (động từ)  dưới núi (trạng ngữ)  tiều vài chú (danh từ)
Lác đác (động từ)  bên sông (trạng ngữ)  chợ mấy nhà (danh từ)
```

This requires a Vietnamese POS tagger. Options:
- **underthesea** library: `from underthesea import pos_tag`
- **pyvi**: `from pyvi import ViPosTagger`

Even with POS tagging, part-of-speech doesn't guarantee semantic parallelism ("mountain" vs "river" are both locations, but a POS tagger just sees two nouns).

### Semantic parallelism

```
Image of people on mountain  ↔  image of market by river
   (contrast: nature vs commerce)
```

Requires word embeddings or a semantic model. This is a research-level problem.

### Rhythm classification

```
2/2/2:  Thân em // như thể // bông sen
4/2:    Trăm năm trong cõi // người ta
```

Could be approximated with a Vietnamese word segmentation tool, but the rhythm is subjective. Not worth implementing until the core features work.

---

## Files Affected

```
New:
  src/tones.py                   ← tone classification + rhyme extraction

Modified:
  src/preprocess.py              ← inject control tokens into pairs
  src/sample.py                  ← auto-inject for generation
  client/server.py               ← auto-inject for chat UI
  tests/test_tones.py            ← unit tests for tone/rhyme functions

Regenerated:
  data/poetry_corpus.txt         ← new format with [RHYME:...] [TONE:...] [LINK2:...]
  tokenizer/poetry_bpe.model     ← retrain (control tokens are BPE-learned subwords)
```

---

## Implementation Order

```
Step 1: Create src/tones.py (tone + rhyme utilities)
Step 2: Write tests/test_tones.py
Step 3: Update preprocess.py (inject control tokens)
Step 4: Regenerate corpus + retrain tokenizer
Step 5: Update server.py + sample.py (auto-inject for generation)
Step 6: Train model
Step 7: Evaluate rhyme/tone accuracy
```

---

## Quick Reference Card

```
LỤC BÁT:
  [RHYME:X]  → position 6 of response must rhyme with X
  [TONE:XXXXXX] → positions 2,4,6 should be B-T-B

THẤT NGÔN:
  [LINK2:X]  → position 2 of response shares tone X (for đối âm — tonal contrast)

Tones: B = Bằng (ngang, huyền)  |  T = Trắc (sắc, nặng, hỏi, ngã)
Rhyme:  everything from last vowel onward after stripping tone marks
```

---

## 🔧 Build Status

| Step | What | File |
|------|------|------|
| 1 | Tone classification | `src/tones.py` ✅ |
| 2 | Rhyme extraction | `src/tones.py` ✅ |
| 3 | Training data injection | `src/preprocess.py` ✅ |
| 4 | Generation auto-inject | `src/sample.py`, `client/server.py` ✅ |
| 5 | Tokenizer (10,922 vocab) | `tokenizer/poetry_bpe.model` ✅ |
| 6 | Corpus with tags (942K pairs) | `data/poetry_corpus.txt` ✅ |
| 7 | Tests (74) | `tests/test_tones.py` ✅ |
| 8 | Train + evaluate | ⏳ Run Colab |

### Known limitations

- Compound vowel nuclei (uô, ươ, iê) — heuristic gets ~90% correct, misses ~10%
- `[ENDTONE]` skipped — single-syllable constraint, low expected impact
- Grammatical/semantic parallelism — needs POS tagger (future work)
- Đối âm (tonal contrast) — not enforced via token, model learns from data patterns
