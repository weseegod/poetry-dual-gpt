# 🗺️ Phase 4 Roadmap: From 14.8M Lục Bát → Full Poetry Engine

> This roadmap covers improvements 11, 12, 9, 10 from `documents/improvements.md`.
> Read each section → implement → ask me to verify. I won't touch the code.

---

## 📋 Phase 4 Order (and why)

```
11. Data cleaning pipeline      ← Fix garbage data FIRST
12. Multi-genre support         ← Add all genres on clean foundation
 9. Rhyme conditioning          ← Enhance quality on diverse data
10. Two-stage training          ← Final: pretrain all → fine-tune Lục Bát
```

| Step | Depends on | Why this order |
|------|-----------|----------------|
| 11 | Nothing | Can't build on dirty data |
| 12 | 11 | Clean data enables correct genre filtering |
| 9 | 12 | Need multi-genre pairs for rhyme extraction |
| 10 | 11, 12 | Need clean all-genre corpus for stage 1 |

---

## 11. Data Cleaning Pipeline

### 11.1 What's wrong with the CSV

```
Current data/poems_dataset.csv issues:
  - HTML artifacts: <br>, <p>, &nbsp;, <div>
  - Unicode inconsistencies: NFC vs NFD (tổ → tổ)
  - Duplicate poems (same content, different IDs)
  - Wrong genre labels (6-syl poem tagged as "thất ngôn")
  - Empty content fields
  - Poems with < 4 lines (can't form even one couplet)
  - Garbled rows from scraping errors
```

### 11.2 What to build

Create a new file: `src/clean_data.py`

```
Input:  data/poems_dataset.csv
Output: data/poems_dataset_clean.csv
```

The pipeline processes poems in stages, each stage is a function:

```
read_csv()
  → remove_empty()          drop rows with null/empty content
  → clean_html()            strip <br>, <p>, &nbsp;, etc.
  → normalize_unicode()     NFC normalization (tổ → tổ)
  → validate_genre()        check line count matches claimed genre
  → remove_duplicates()     dedupe by normalized content hash
  → filter_min_lines()      keep poems ≥ 4 lines
  → save()
```

### 11.3 Stage-by-stage guide

#### Stage 1: `remove_empty(df)`
```python
# Drop rows where content is NaN or whitespace-only
df = df[df["content"].notna()]
df = df[df["content"].str.strip() != ""]
print(f"After removing empty: {len(df)} poems")
return df
```

#### Stage 2: `clean_html(text)`
```python
import re

def clean_html(text: str) -> str:
    # Replace common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

#### Stage 3: `normalize_unicode(text)`
```python
import unicodedata

def normalize_unicode(text: str) -> str:
    # NFC = composed form. Turns tổ (t + o + combining hook + dot) → tổ (single char)
    return unicodedata.normalize("NFC", text)
```

**Why this matters:** "tổ" in NFC is 3 characters. In NFD it's 5. The tokenizer sees them as DIFFERENT words. Normalize or your vocabulary doubles for no reason.

#### Stage 4: `validate_genre(df)`

Each genre has expected line counts:

| Genre | Lines | Syllables per line |
|-------|-------|-------------------|
| lục bát | 4, 6, 8, 10, ... (even) | 6→8→6→8... |
| thất ngôn tứ tuyệt | 4 | 7 each |
| thất ngôn bát cú | 8 | 7 each |
| ngũ ngôn tứ tuyệt | 4 | 5 each |
| song thất lục bát | alternating 7-7-6-8 | pattern |

```python
GENRE_RULES = {
    "lục bát":            {"lines_divisible_by": 2},
    "thất ngôn tứ tuyệt":  {"lines_exact": 4},
    "thất ngôn bát cú":    {"lines_exact": 8},
    "ngũ ngôn tứ tuyệt":   {"lines_exact": 4},
    "song thất lục bát":   {"lines_gte": 4},
}

def validate_genre(df):
    bad = []
    for idx, row in df.iterrows():
        genre = row["genre"]
        n_lines = str(row["content"]).count(" <\\n> ") + 1
        rule = GENRE_RULES.get(genre)
        if not rule:
            continue  # unknown genre, keep it
        if "lines_exact" in rule and n_lines != rule["lines_exact"]:
            bad.append(idx)
        elif "lines_divisible_by" in rule and n_lines % rule["lines_divisible_by"] != 0:
            bad.append(idx)
    print(f"Wrong genre tag: {len(bad)} poems → dropped")
    return df.drop(bad)
```

#### Stage 5: `remove_duplicates(df)`
```python
import hashlib

def content_hash(text: str) -> str:
    """Hash the normalized content so duplicates share the same hash."""
    normalized = normalize_unicode(clean_html(text.lower()))
    return hashlib.md5(normalized.encode()).hexdigest()

def remove_duplicates(df):
    df["_hash"] = df["content"].apply(content_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_hash", keep="first")
    df = df.drop(columns=["_hash"])
    print(f"Duplicates removed: {before - len(df)}")
    return df
```

#### Stage 6: `filter_min_lines(df, min_lines=4)`
```python
def filter_min_lines(df, min_lines=4):
    n_lines = df["content"].apply(lambda t: str(t).count(" <\\n> ") + 1)
    mask = n_lines >= min_lines
    print(f"Too short (< {min_lines} lines): {(~mask).sum()} poems → dropped")
    return df[mask]
```

### 11.4 Main pipeline

```python
def clean_dataset(csv_path="data/poems_dataset.csv", output="data/poems_dataset_clean.csv"):
    df = pd.read_csv(csv_path)
    print(f"Original: {len(df):,} poems")

    df = remove_empty(df)
    df["content"] = df["content"].apply(clean_html)
    df["content"] = df["content"].apply(normalize_unicode)
    df = validate_genre(df)
    df = remove_duplicates(df)
    df = filter_min_lines(df)

    df.to_csv(output, index=False)
    print(f"Clean: {len(df):,} poems → {output}")
    return df
```

### 11.5 Expected output

```
Original:          198,000 poems
After empty:       ~197,000
After genre check: ~190,000  (wrong labels dropped)
After duplicates:  ~178,000  (12K duplicates!)
After min lines:   ~175,000
Clean:             ~175,000 poems → data/poems_dataset_clean.csv
```

### 11.6 Verification checklist

After running, verify:
- [ ] `poems_dataset_clean.csv` exists and is smaller than original
- [ ] No HTML tags in content (grep for `<br>`, `<p>`)
- [ ] Unicode is normalized (spot-check: `"tổ"` should be `"tổ"`)
- [ ] All Lục Bát poems have even line counts
- [ ] No duplicate content hashes
- [ ] Minimum 4 lines per poem

---

## 12. Multi-Genre Support

### 12.1 Current state

```
src/train_bpe.py:  SPECIAL_TOKENS = [
    "<|pad|>",              # 0
    "<|start|>",            # 1
    "<|reply|>",            # 2
    "<|end|>",              # 3
    "[LUC_BAT]",            # 4  ← only one used
    "[TU_TUYET]",           # 5  ← reserved, unused
    "[THAT_NGON_BAT_CU]",   # 6  ← reserved, unused
]

src/preprocess.py:  only filters df[df["genre"] == "lục bát"]
```

### 12.2 What to change

You need **one new genre token per genre** in the CSV. First, map CSV genres to tokens:

```python
# Map from CSV's "genre" column to control tokens
GENRE_TO_TOKEN = {
    "lục bát":              "[LUC_BAT]",
    "thất ngôn tứ tuyệt":      "[TU_TUYET]",
    "thất ngôn bát cú":        "[THAT_NGON_BAT_CU]",
    "ngũ ngôn tứ tuyệt":       "[NGU_NGON]",        # new token
    "song thất lục bát":       "[SONG_THAT]",        # new token
    "thơ tự do":              "[TU_DO]",            # new token
    "thơ 5 chữ":              "[NGU_NGON]",         # reuse
    "thơ 7 chữ":              "[TU_TUYET]",         # reuse
}
```

### 12.3 Step 1: Add new special tokens to `train_bpe.py`

```python
# src/train_bpe.py — update SPECIAL_TOKENS list:
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|start|>",
    "<|reply|>",
    "<|end|>",
    "[LUC_BAT]",
    "[TU_TUYET]",
    "[THAT_NGON_BAT_CU]",
    "[NGU_NGON]",        # new: ngũ ngôn / thơ 5 chữ
    "[SONG_THAT]",        # new: song thất lục bát
    "[TU_DO]",            # new: thơ tự do
]
```

**Important:** Adding tokens changes indices. After re-training the tokenizer:
- `[LUC_BAT]` must still be index 4
- New tokens append at the end (indices 7, 8, 9)

Verify with: `assert tok.token_to_id("[NGU_NGON]") == 7`

### 12.4 Step 2: Update `preprocess.py` — multi-genre pairs

```python
# src/preprocess.py — replace the Lục Bát-only filter

GENRE_PAIR_RULES = {
    "lục bát": {
        "tag": "[LUC_BAT]",
        "step": 2,               # pair lines 0-1, 2-3, 4-5 (6-8 couplets)
        "prompt_range": (5, 7),  # ±1 tolerance for noisy data
        "reply_range": (7, 9),
    },
    "thất ngôn tứ tuyệt": {
        "tag": "[TU_TUYET]",
        "step": 2,               # pair lines 0-1, 2-3 (two 7-7 couplets)
        "prompt_range": (6, 8),  # 7-syllable ± 1
        "reply_range": (6, 8),
    },
    "thất ngôn bát cú": {
        "tag": "[THAT_NGON_BAT_CU]",
        "step": 2,               # pair lines 0-1, 2-3, 4-5, 6-7
        "prompt_range": (6, 8),
        "reply_range": (6, 8),
    },
    "song thất lục bát": {
        "tag": "[SONG_THAT]",
        "step": 4,               # pair the full 4-line stanza
        "prompt_range": (12, 30),  # 2×7 + 1×6 = ~20 syllables
        "reply_range": (7, 9),     # 8-syllable closing
    },
    # ... add more genres
}

def make_pairs(lines, genre_tag, rule):
    """Create training pairs for any genre."""
    pairs = []
    step = rule["step"]
    p_range = rule["prompt_range"]
    r_range = rule["reply_range"]

    for i in range(0, len(lines) - 1, step):
        if i + 1 >= len(lines):
            break
        prompt = lines[i]
        reply = lines[i + 1]

        p_syl = count_syllables(prompt)
        r_syl = count_syllables(reply)

        if p_range[0] <= p_syl <= p_range[1] and r_range[0] <= r_syl <= r_range[1]:
            pairs.append(f"{START} {genre_tag} {prompt} {REPLY} {reply} {END}")
    return pairs
```

### 12.5 Step 3: Updated `preprocess()` main

```python
def preprocess(csv_path=None, output_path=None, max_poems=None):
    csv_path = Path(csv_path or CSV_PATH)
    df = pd.read_csv(csv_path)          # ← NO MORE genre filter!

    all_pairs = []
    for _, row in df.iterrows():
        genre = row["genre"]
        rule = GENRE_PAIR_RULES.get(genre)
        if not rule:
            continue                     # skip unknown genres

        lines = parse_poem(row["content"])
        if len(lines) < 2:
            continue

        pairs = make_pairs(lines, rule["tag"], rule)
        all_pairs.extend(pairs)

    # Save (mixed genres, order matters for training!)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(pair + "\n")
```

### 12.6 Expected training data

```
<|start|> [LUC_BAT] prompt_6_syl reply_8_syl <|end|>
<|start|> [TU_TUYET] prompt_7_syl reply_7_syl <|end|>
<|start|> [SONG_THAT] prompt_20_syl reply_8_syl <|end|>
...
```

The model sees the genre tag → learns to generate in that form.

### 12.7 Verification checklist

- [ ] All genre tokens are in `SPECIAL_TOKENS`
- [ ] Token indices are correct (0-6 unchanged, 7+ are new)
- [ ] `corpus.txt` contains mixed genre pairs
- [ ] Each genre has correct tag in output
- [ ] Run `sample.py` with different tags:
  ```bash
  python src/sample.py --prompt "[TU_TUYET] ..."   # should give 7-syl
  python src/sample.py --prompt "[LUC_BAT] ..."    # should give 8-syl
  ```

---

## 9. Rhyme Conditioning

### 9.1 What is rhyme in Lục Bát?

```
Line 1 (6 syl): Trăm năm  trong  cõi   người  ta
                [1]     [2]   [3]     [4]     [5]   [6]
                 B       B     T       B       B     rhyme here ↑
                                                        ↓ must match
Line 2 (8 syl): Chữ tài  chữ   mệnh   khéo   là    ghét  nhau
                [1]     [2]   [3]     [4]     [5]   [6]   [7]  [8]
                 B       T     B       B       rhyme at pos 6

Rule: syllable 6 of line 1 rhymes with syllable 6 of line 2 (vần lưng)
```

### 9.2 Vietnamese rhyme groups

Vietnamese rhymes are based on the **final sound** (vần). A syllable ending is determined by its final consonant + tone:

```
Group "a":   ta, ba, xa, nhà, gà, la, đà...
Group "an":  tan, ban, lan, ngàn, đàn, tàn...
Group "ang": trang, làng, đàng, vàng, sáng...
Group "ong": trong, lòng, sông, đồng, chồng...
Group "ơ":   thơ, chờ, bơ, ngơ, mơ...
Group "iêu": nhiêu, phiêu, tiêu, siêu...
Group "inh": tình, mình, hình, đình, linh...
```

### 9.3 Rhyme extraction heuristic

Since we don't have a labeled dictionary, use a **suffix heuristic**:

```python
def get_rhyme_group(word: str) -> str:
    """Extract rhyme group from a syllable using suffix matching."""
    # Strip tone mark for group matching
    # "tà" → "a", "sông" → "ông", "tiếng" → "iêng"
    word = word.lower()
    # Remove tone diacritics (approximate)
    tone_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ằ': 'ă', 'ắ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
        'ầ': 'â', 'ấ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ề': 'ê', 'ế': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ồ': 'ô', 'ố': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
        'ờ': 'ơ', 'ớ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ừ': 'ư', 'ứ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
    }
    base = ''.join(tone_map.get(c, c) for c in word)

    # Find the vowel nucleus → take everything from last vowel
    vowels = set('aăâeêioôơuưy')
    last_vowel_idx = -1
    for i, c in enumerate(base):
        if c in vowels:
            last_vowel_idx = i

    if last_vowel_idx >= 0:
        return base[last_vowel_idx:]  # everything from last vowel onward
    return base
```

```
Examples:
  "ta"     → "a"
  "sông"   → "ông"    (ô + ng)
  "tiếng"  → "iêng"   (iê + ng)
  "thơ"    → "ơ"
  "tình"   → "inh"    (i + nh)
  "nhiêu"  → "iêu"    (iê + u)
```

### 9.4 Inject rhyme tags into training data

**File:** `src/preprocess.py`, in `make_pairs()`

For Lục Bát only — extract the rhyme of the prompt's 6th syllable:

```python
def make_pairs(lines, genre_tag, rule):
    pairs = []
    for i in range(0, len(lines), rule["step"]):
        if i + 1 >= len(lines):
            break
        prompt = lines[i]
        reply = lines[i + 1]

        p_syl = count_syllables(prompt)
        r_syl = count_syllables(reply)

        if not (p_range[0] <= p_syl <= p_range[1] and r_range[0] <= r_syl <= r_range[1]):
            continue

        # ── NEW: rhyme conditioning for Lục Bát ──
        rhyme_part = ""
        if genre_tag == "[LUC_BAT]" and p_syl >= 6:
            syls = prompt.split()
            if len(syls) >= 6:
                rhyme_6th = syls[5]  # 0-indexed: syllable at position 6
                rhyme_group = get_rhyme_group(rhyme_6th)  # e.g., "ông"
                rhyme_part = f" [VAN:{rhyme_group}]"  # inject "[VAN:ông]"
        # ──────────────────────────────────────────

        pairs.append(f"{START} {genre_tag}{rhyme_part} {prompt} {REPLY} {reply} {END}")
    return pairs
```

Training data becomes:
```
<|start|> [LUC_BAT] [VAN:ong] Trăm năm trong cõi người ta <|reply|> Chữ tài chữ mệnh khéo là ghét nhau <|end|>
```

### 9.5 Handle rhyme tokens

Since rhyme groups are dynamic (`[VAN:ong]`, `[VAN:ơ]`, `[VAN:a]`, ...), you can't add them all to `SPECIAL_TOKENS`. Options:

**Option A (simpler):** Let BPE learn them as subwords. `[VAN:ong]` becomes tokens like `[`, `VAN`, `:`, `ong`, `]`. The pattern is learnable.

**Option B (cleaner):** Pre-compute all unique rhyme groups in the corpus, add the top 50 as special tokens. 50 new tokens costs nothing.

Option A is recommended for Phase 4. If rhyme quality is poor, switch to B later.

### 9.6 During generation

The `/chat` endpoint already auto-wraps `[LUC_BAT]`. Add rhyme extraction there too:

```python
# server.py — in generate():
if not prompt.startswith("[") and "[LUC_BAT]" not in prompt:
    prompt = f"[LUC_BAT] {prompt}"

# NEW: extract rhyme from user's prompt
if "[LUC_BAT]" in prompt and "[VAN:" not in prompt:
    syls = prompt.replace("[LUC_BAT]", "").strip().split()
    if len(syls) >= 6:
        rhyme_6th = syls[5]
        rhyme_group = get_rhyme_group(rhyme_6th)
        prompt = prompt.replace("[LUC_BAT]", f"[LUC_BAT] [VAN:{rhyme_group}]")
```

Now when the user types "Trăm năm trong cõi người ta", it becomes `[LUC_BAT] [VAN:a] Trăm năm trong cõi người ta` before encoding. The model sees the rhyme tag and is conditioned to rhyme with "a".

### 9.7 Verification checklist

- [ ] `get_rhyme_group()` produces reasonable groups (spot check 20 words)
- [ ] Training data contains `[VAN:...]` tags for Lục Bát pairs only
- [ ] Other genres don't get rhyme tags (they have different rules)
- [ ] During generation, rhyme tag is auto-injected
- [ ] Generated Lục Bát responses actually rhyme more often:
  ```bash
  # Before: maybe 30% of responses rhyme correctly
  # After:  targeting 60-70%
  ```

---

## 10. Two-Stage Training

### 10.1 What it is

```
Stage 1: Train on ALL poems (all genres)
         → Model learns Vietnamese grammar, poetic vocabulary, general patterns
         → 15K-20K steps, full LR schedule

Stage 2: Fine-tune on Lục Bát ONLY
         → Model specializes in 6→8 format
         → 5K steps, lower LR (1e-4 → 1e-5)
```

This is exactly how GPT-3 → ChatGPT works: big pretraining, then specialized fine-tuning.

### 10.2 Why this order in Phase 4

By the time you reach this step, you've:
- Clean data (Step 11) → no garbage in training
- Multi-genre corpus (Step 12) → Stage 1 has diverse poetry
- Rhyme conditioning (Step 9) → enhanced data

Stage 1 benefits from all three. Stage 2 is Lục Bát only (with rhyme tags).

### 10.3 Implementation plan

#### Step 1: Generate TWO corpora

```bash
# Stage 1 corpus: ALL genres
python src/preprocess.py --output data/corpus_all_genres.txt --csv data/poems_dataset_clean.csv

# Stage 2 corpus: Lục Bát only (with rhyme tags)
python src/preprocess.py --output data/corpus_luc_bat.txt --csv data/poems_dataset_clean.csv --genre "lục bát"
```

Or modify `preprocess.py` to output both in one run.

#### Step 2: Train Stage 1

```python
# CONFIG for Stage 1
STAGE1 = {
    "max_steps": 15000,
    "batch_size": 192,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "min_lr": 1e-5,
    "corpus_path": "data/corpus_all_genres.txt",
    "checkpoint_name": "stage1_all_genres.pt",
}
```

Training time: ~3 hours on L4 for 175K poems.

#### Step 3: Train Stage 2 (fine-tune)

```python
# CONFIG for Stage 2
STAGE2 = {
    "max_steps": 5000,
    "batch_size": 192,
    "learning_rate": 1e-4,          # lower LR for fine-tuning
    "warmup_steps": 100,
    "min_lr": 1e-5,
    "corpus_path": "data/corpus_luc_bat.txt",
    "checkpoint_name": "stage2_luc_bat_finetune.pt",
    "resume_from": "checkpoints/stage1_all_genres.pt",  # ← KEY: load Stage 1
}
```

Training time: ~1 hour on L4.

### 10.4 Code changes needed in `train.py`

Add `--resume` flag (we skipped this earlier but now it's essential):

```python
# train.py — add to argparse:
p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

# In train(), after model creation:
if args.resume:
    ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    step = ckpt["step"]
    print(f"📂  Resumed from step {step} ({args.resume})")
```

### 10.5 Colab notebook for two-stage

```python
# Cell: Stage 1 — Pretrain on all genres
!python src/preprocess.py --output data/corpus_all_genres.txt
!python src/train.py --corpus data/corpus_all_genres.txt --checkpoint_name stage1.pt --max_steps 15000

# Cell: Stage 2 — Fine-tune on Lục Bát
!python src/preprocess.py --output data/corpus_luc_bat.txt --genre lục_bát
!python src/train.py --corpus data/corpus_luc_bat.txt --resume checkpoints/stage1.pt --max_steps 5000 --lr 1e-4
```

### 10.6 Expected results

| Metric | Before (Lục Bát only) | After (two-stage) |
|--------|-----------------------|-------------------|
| Loss | ~2.65 (val) | ~2.3-2.4 (val) |
| Rhyme accuracy | ~30% | ~50-60% |
| 8-syl correctness | ~90% | ~95% |
| Vocabulary richness | Limited to Lục Bát | Broader poetic vocabulary |
| Multi-genre support | None | All tagged genres work |

### 10.7 Verification checklist

- [ ] Stage 1 trains without errors on all-genre corpus
- [ ] Stage 2 resumes from Stage 1 checkpoint (step counter continues)
- [ ] Stage 2 has lower final loss than Stage 1
- [ ] Generated Lục Bát after Stage 2 is better than Stage 1 alone
- [ ] Multi-genre generation still works:
  ```bash
  python src/sample.py --prompt "[TU_TUYET] ..."         # 7-syllable
  python src/sample.py --prompt "[LUC_BAT] [VAN:ơ] ..."  # 8-syllable with rhyme
  ```

---

## 📊 Files affected by Phase 4

```
New files:
  src/clean_data.py                (Step 11)

Modified files:
  src/preprocess.py                (Steps 11, 12, 9 — genre tags, rhyme tags, clean input)
  src/train_bpe.py                 (Step 12 — new special tokens)
  src/train.py                     (Step 10 — --resume flag)
  client/server.py                 (Step 9 — auto rhyme injection)
  colab/colab_train.ipynb          (Step 10 — two-stage cells)

Regenerated files:
  data/poems_dataset_clean.csv     (Step 11 — output of clean_data.py)
  data/poetry_corpus.txt           (Steps 12, 9 — new format with multi-genre + rhyme)
  tokenizer/poetry_bpe.model       (Step 12 — retrain with new special tokens)
  checkpoints/                     (Step 10 — new stage1 + stage2 checkpoints)
```

---

## 🎯 Success criteria for Phase 4

After completing all four steps, the model should:

1. Accept any genre tag: `[LUC_BAT]`, `[TU_TUYET]`, `[SONG_THAT]`, `[TU_DO]`
2. Auto-inject rhyme tags when given `[LUC_BAT]` with a 6th syllable
3. Generate responses that match the requested genre's syllable pattern
4. Rhyme correctly for Lục Bát in 50%+ of generations
5. Use richer vocabulary (from all-genre pretraining)
6. Run on clean, deduplicated data with no HTML artifacts
