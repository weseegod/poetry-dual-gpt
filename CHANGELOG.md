# 📋 Changelog

All notable changes to PoetryDuel-GPT.

## [v4.2.3] — 2026-05-27

### Added
- Content-weighted training loss: prompt-only control tokens receive 0.3× weight
- N-gram diversity loss (weight=0.03) — eliminates adjacent word repeats
- Linebreak position reinforcement (+0.2 bonus)
- Unified `src/generation.py` — single canonical generator for CLI, API, and eval
- `evaluate/eval_quality.py` — 9 semantic quality metrics
- Soft rhyme constraint (logit boost +2.0 instead of hard masking, with safety fallback)

### Changed
- Rhyme: 84% → **94%**
- Tone: 90% → **100%**
- Trầm-Bổng: 90% → **98%**
- All-5-pass: 76% → **92%**
- Lexical diversity: 0.89 → **0.936**
- Adjacent repeats: ~8% → **0.0%**
- BPE artifacts: ~15% → **0.4%**

### Fixed
- Server format mismatch — `<|start|>` and `[TRAMBONG:NH/HN]` now included in API prompts
- Three divergent generation paths unified into `src/generation.py`
- BPE artifact false negatives — dictionary-based syllable validity check

## [v4.1] — 2026-05-25

### Added
- Trầm-Bổng rule (`[TRAMBONG:NH/HN]` control token) — diacritic contrast enforcement
- Rhyme constraint (beam masking at output pos6)
- Scheduled sampling with control token protection (IDs 0-214)

### Changed
- Removed Thất Ngôn genre (moved to v5) to focus on Lục Bát quality
- Removed post-processing syllable truncation hack — raw metrics now
- Pure Lục Bát corpus (540K pairs, window=1)

### Results
- All-5-pass: 76%
- Rhyme: 84%, Tone: 92%, Trầm-Bổng: 90%
- Syllable: 100%, Rhythm: 100%

## [v4.0] — 2026-05-23

### Added
- Thất Ngôn (7-7 syllable) data pipeline — 41K poems, 748K training pairs
- Beam rhyme constraint at inference
- Overgeneration fix (post-cleanup for >2 line outputs)

### Issues Identified
- Thất Ngôn at 28% data ratio degraded Lục Bát quality
- Post-processing hacks masked real syllable accuracy
- Format mismatch between training and inference

## [v3.0] — 2026-05-22

### Added
- Example-aligned batching — zero cross-poem noise
- Window-1 only corpus (removed window=2 dead format data)
- Repetition penalty (-1.2 for recent 16 tokens)
- Syllable enforcement (post-process truncation to 6/8)
- Independent duel parsing (C1→C3, C2→C4)

### Results
- Stress test: **100%** (0 BPE collapse, 0 control tokens in output)
- Batch size increased to 192
- Corpus refined from 998K to 541K pairs

## [v2.0] — 2026-05-21

### Added
- Multi-couplet `[DOI_THO]` format with sliding window pairs
- Two-stage training strategy (all-genre pretrain → Lục Bát fine-tune)
- `data_service/` — scraping pipeline for canonical poets

## [v1.0] — 2026-05-20

### Initial Release
- 31M-param decoder-only Transformer from scratch
- 11,392 BPE tokens, two-stage training (10K + 5K steps)
- 4 rules: rhyme (58%), tone (88%), syllable (78%), đối âm (69%)
- FastAPI + React chat UI
- Colab one-click training notebook
