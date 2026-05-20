"""
Tests for src/clean_data.py — one test per pipeline stage.

Run:  python -m pytest src/test_clean_data.py -v
  or:  python src/test_clean_data.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clean_data import (
    filter_genres, remove_song_that, clean_content, clean_text,
    filter_short, remove_duplicates, content_hash,
)


# ═══════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════

_passed = 0
_failed = 0

def check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        print(f"  ❌ {name}  {detail}")


# ═══════════════════════════════════════════════════════════════
# Stage 1: filter_genres
# ═══════════════════════════════════════════════════════════════

def test_filter_genres():
    print("\n── filter_genres ──")
    df = pd.DataFrame({
        "genre": ["lục bát", "bảy chữ", "thơ tự do", "tám chữ", "năm chữ", "lục bát"],
        "content": [f"poem {i}" for i in range(6)],
    })
    result = filter_genres(df)
    check("keeps lục bát",  "lục bát" in result["genre"].values)
    check("keeps bảy chữ",  "bảy chữ" in result["genre"].values)
    check("drops thơ tự do", "thơ tự do" not in result["genre"].values)
    check("drops tám chữ",  "tám chữ" not in result["genre"].values)
    check("correct count",  len(result) == 3, f"got {len(result)} expected 3")


# ═══════════════════════════════════════════════════════════════
# Stage 2: remove_song_that
# ═══════════════════════════════════════════════════════════════

def test_remove_song_that():
    print("\n── remove_song_that ──")
    df = pd.DataFrame({
        "genre": ["bảy chữ", "bảy chữ", "bảy chữ", "bảy chữ"],
        "specific_genre": [
            "bảy chữ",
            "song thất lục bát",
            "SONG THẤT LỤC BÁT",
            "thất ngôn bát cú",
        ],
        "content": ["a", "b", "c", "d"],
    })
    result = remove_song_that(df)
    check("drops song thất lục bát (lowercase)",
          "song thất lục bát" not in result["specific_genre"].values)
    check("drops SONG THẤT (uppercase)",
          "SONG THẤT LỤC BÁT" not in result["specific_genre"].values)
    check("keeps thất ngôn bát cú",
          "thất ngôn bát cú" in result["specific_genre"].values)
    check("keeps bảy chữ",
          "bảy chữ" in result["specific_genre"].values)
    check("correct count", len(result) == 2, f"got {len(result)} expected 2")


# ═══════════════════════════════════════════════════════════════
# Stage 3: clean_content / clean_text
# ═══════════════════════════════════════════════════════════════

def test_clean_text():
    print("\n── clean_text ──")

    sep = "<" + chr(10) + ">"  # actual newline separator

    # HTML entities
    result = clean_text("hello &amp; world &nbsp; test &lt;tag&gt;")
    check("strips &amp;", "&amp;" not in result)
    check("strips &nbsp;", "&nbsp;" not in result)
    check("strips &lt;", "<" not in result or "tag" in result)

    # HTML tags (per line — after splitting)
    result = clean_text(f"line one<br>still line one{sep}line two<p>still two")
    check("strips <br>", "<br>" not in result)
    check("strips <p>", "<p>" not in result)
    check("preserves separator", sep in result, f"separator missing: {repr(result[:80])}")
    check("multi-line", result.count(sep) == 1, f"got {result.count(sep)} separators")

    # Unicode normalization (NFC)
    nfd = "t" + chr(111) + chr(770) + chr(777)  # tổ in NFD: t + o + circumflex + hook (U+0309)
    nfc_correct = "tổ"          # tổ in NFC (3 chars)
    result = clean_text(nfd)
    check("NFC normalization", result == nfc_correct, f"got {repr(result)} expected {repr(nfc_correct)}")

    # Space collapsing per line
    result = clean_text(f"hello     world{sep}foo\t\tbar")
    lines = result.split(sep)
    check("collapse multi-space", "    " not in lines[0])
    check("collapse tabs", "\t" not in lines[1])

    # Empty lines dropped
    result = clean_text(f"  {sep}  {sep}valid line")
    check("drops empty lines", result.count(sep) == 0, f"expected 1 line, got {result.count(sep)+1}")

    # NaN / None
    result = clean_text(None)
    check("handles None", result == "")

    result = clean_text(float("nan"))
    check("handles NaN", result == "")


def test_clean_content():
    print("\n── clean_content ──")

    sep = "<" + chr(10) + ">"

    df = pd.DataFrame({
        "genre": ["lục bát", "lục bát", "lục bát"],
        "content": [
            f"line one{sep}line two",
            None,
            "   ",  # whitespace only
        ],
    })
    result = clean_content(df)
    check("keeps valid poem", len(result) >= 1, f"got {len(result)}")
    check("drops None", len(result) < 3, f"expected <3, got {len(result)}")
    check("drops whitespace-only", len(result) == 1, f"expected 1, got {len(result)}")


# ═══════════════════════════════════════════════════════════════
# Stage 4: filter_short
# ═══════════════════════════════════════════════════════════════

def test_filter_short():
    print("\n── filter_short ──")

    sep = "<" + chr(10) + ">"

    df = pd.DataFrame({
        "genre": ["lục bát"] * 4,
        "content": [
            f"only one line",                                    # 1 line → drop
            f"line 1{sep}line 2",                                # 2 lines → keep
            f"line 1{sep}line 2{sep}line 3{sep}line 4",          # 4 lines → keep
            f"line 1{sep}line 2{sep}line 3",                     # 3 lines → keep
        ],
    })
    result = filter_short(df, min_lines=2)
    check("drops 1-line poem", len(result[result["content"] == "only one line"]) == 0)
    check("keeps 2-line poem",  len(result) == 3, f"got {len(result)} expected 3")


# ═══════════════════════════════════════════════════════════════
# Stage 5: remove_duplicates
# ═══════════════════════════════════════════════════════════════

def test_remove_duplicates():
    print("\n── remove_duplicates ──")

    sep = "<" + chr(10) + ">"

    df = pd.DataFrame({
        "genre": ["lục bát", "lục bát", "lục bát", "lục bát"],
        "content": [
            f"poem a{sep}line 2",     # unique
            f"Poem A{sep}line 2",     # DUPLICATE (case-insensitive after cleaning)
            f"poem b{sep}line 2",     # unique
            f"poem a{sep}line 2",     # DUPLICATE (exact match)
        ],
    })
    result = remove_duplicates(df)
    check("drops exact duplicate", len(result) == 2, f"got {len(result)} expected 2")
    check("drops case-insensitive duplicate", len(result) == 2)

    # Verify no _hash column leaked
    check("no _hash column", "_hash" not in result.columns)

    # Verify content_hash is stable
    h1 = content_hash(f"test poem{sep}line two")
    h2 = content_hash(f"test poem{sep}line two")
    check("content_hash is deterministic", h1 == h2, f"{h1} != {h2}")


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🧪  Testing src/clean_data.py")
    print("=" * 60)

    test_filter_genres()
    test_remove_song_that()
    test_clean_text()
    test_clean_content()
    test_filter_short()
    test_remove_duplicates()

    print(f"\n{'='*60}")
    total = _passed + _failed
    print(f"Results: {_passed}/{total} passed", end="")
    if _failed:
        print(f"  ({_failed} FAILED)")
        sys.exit(1)
    else:
        print("  ✅ All good!")
