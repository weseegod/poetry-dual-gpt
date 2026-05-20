"""Tests for src/preprocess.py — pair generation + syllable counting."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocess import (
    count_syllables, clean_line, parse_poem,
    make_pairs, make_pairs_song_that, is_song_that,
)


class TestCountSyllables(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(count_syllables("thân em như chẽn lúa đòng"), 6)
        self.assertEqual(count_syllables("chữ tài chữ mệnh khéo là ghét nhau"), 8)
        self.assertEqual(count_syllables("ta"), 1)

    def test_whitespace(self):
        self.assertEqual(count_syllables("  a   b c  "), 3)
        self.assertEqual(count_syllables(""), 0)


class TestCleanLine(unittest.TestCase):
    def test_strips_punctuation(self):
        self.assertEqual(clean_line("  hello world,  "), "hello world")
        self.assertEqual(clean_line("thơ..."), "thơ")

    def test_collapse_spaces(self):
        self.assertEqual(clean_line("a    b   c"), "a b c")


class TestParsePoem(unittest.TestCase):
    def setUp(self):
        self.sep = "<" + chr(10) + ">"

    def test_splits_lines(self):
        content = f"line one{self.sep}line two{self.sep}line three"
        result = parse_poem(content)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "line one")

    def test_filters_single_word_lines(self):
        # "a" has 1 word → dropped; "x y" has 2 words → kept
        content = f"a{self.sep}hello world"
        result = parse_poem(content)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "hello world")

    def test_cleans_each_line(self):
        # Trailing punctuation stripped, spaces collapsed
        content = f"  hello world ,  {self.sep}  foo bar .  "
        result = parse_poem(content)
        self.assertEqual(result[0], "hello world")
        self.assertEqual(result[1], "foo bar")


class TestMakePairs(unittest.TestCase):
    def test_luc_bat_strict(self):
        lines = ["một hai ba bốn năm sáu",    # 6 syl
                  "một hai ba bốn năm sáu bảy tám"]  # 8 syl
        pairs = make_pairs(lines, "lục bát")
        self.assertEqual(len(pairs), 1)
        self.assertIn("[LUC_BAT]", pairs[0])
        self.assertIn("<|start|>", pairs[0])
        self.assertIn("<|reply|>", pairs[0])
        self.assertIn("<|end|>", pairs[0])

    def test_luc_bat_rejects_wrong_syllables(self):
        lines = ["a b c d e",       # 5 syl — too short
                  "a b c d e f g"]   # 7 syl — too short for 8
        pairs = make_pairs(lines, "lục bát")
        self.assertEqual(len(pairs), 0)

    def test_that_ngon_strict(self):
        lines = ["một hai ba bốn năm sáu bảy",  # 7 syl
                  "bảy sáu năm bốn ba hai một"]   # 7 syl
        pairs = make_pairs(lines, "bảy chữ")
        self.assertEqual(len(pairs), 1)
        self.assertIn("[THAT_NGON]", pairs[0])

    def test_that_ngon_rejects_wrong(self):
        lines = ["a b c d e f g",   # 7 syl ✅
                  "a b c d e f"]     # 6 syl ❌
        pairs = make_pairs(lines, "bảy chữ")
        self.assertEqual(len(pairs), 0)

    def test_step_by_two(self):
        """Only pairs 0→1, 2→3, 4→5... not 1→2, 3→4."""
        lines = [
            "một hai ba bốn năm sáu", "một hai ba bốn năm sáu bảy tám",  # pair 1
            "một hai ba bốn năm sáu", "một hai ba bốn năm sáu bảy tám",  # pair 2
            "một hai ba bốn năm sáu", "một hai ba bốn năm sáu bảy tám",  # pair 3
        ]
        pairs = make_pairs(lines, "lục bát")
        self.assertEqual(len(pairs), 3)

    def test_odd_lines_dropped(self):
        lines = [
            "một hai ba bốn năm sáu", "một hai ba bốn năm sáu bảy tám",  # pair
            "một hai ba bốn năm sáu",  # orphan — no reply
        ]
        pairs = make_pairs(lines, "lục bát")
        self.assertEqual(len(pairs), 1)


class TestMakePairsSongThat(unittest.TestCase):
    def test_basic_stanza(self):
        """Single song thất stanza: 7-7-6-8 → one [THAT_NGON] + one [LUC_BAT]."""
        lines = [
            "một hai ba bốn năm sáu bảy",   # 7
            "bảy sáu năm bốn ba hai một",   # 7
            "một hai ba bốn năm sáu",        # 6
            "một hai ba bốn năm sáu bảy tám",  # 8
        ]
        pairs = make_pairs_song_that(lines)
        self.assertEqual(len(pairs), 2)
        self.assertIn("[THAT_NGON]", pairs[0])
        self.assertIn("[LUC_BAT]", pairs[1])

    def test_two_stanzas(self):
        """Two stanzas → 4 pairs."""
        lines = [
            "a b c d e f g", "a b c d e f g", "a b c d e f", "a b c d e f g h",  # stanza 1
            "a b c d e f g", "a b c d e f g", "a b c d e f", "a b c d e f g h",  # stanza 2
        ]
        pairs = make_pairs_song_that(lines)
        self.assertEqual(len(pairs), 4)
        # Alternating: THAT_NGON, LUC_BAT, THAT_NGON, LUC_BAT
        self.assertIn("[THAT_NGON]", pairs[0])
        self.assertIn("[LUC_BAT]", pairs[1])
        self.assertIn("[THAT_NGON]", pairs[2])
        self.assertIn("[LUC_BAT]", pairs[3])

    def test_rejects_bad_syllables(self):
        lines = [
            "a b c d e",        # 5 — bad
            "a b c d e f g",    # 7
            "a b c d e f",      # 6
            "a b c d e f g h",  # 8
        ]
        pairs = make_pairs_song_that(lines)
        self.assertEqual(len(pairs), 1)  # only 6→8 pair, 7→7 rejected
        self.assertIn("[LUC_BAT]", pairs[0])

    def test_partial_final_stanza(self):
        """Only 2 lines left at end (should skip)."""
        lines = [
            "a b c d e f g", "a b c d e f g", "a b c d e f", "a b c d e f g h",
            "a b c d e f g", "a b c d e f g",  # only 2 lines, no 6-8
        ]
        pairs = make_pairs_song_that(lines)
        self.assertEqual(len(pairs), 2)  # just the first stanza


class TestIsSongThat(unittest.TestCase):
    def test_detects(self):
        self.assertTrue(is_song_that({"specific_genre": "song thất lục bát"}))
        self.assertTrue(is_song_that({"specific_genre": "SONG THẤT LỤC BÁT"}))

    def test_rejects_others(self):
        self.assertFalse(is_song_that({"specific_genre": "lục bát"}))
        self.assertFalse(is_song_that({"specific_genre": "thất ngôn tứ tuyệt"}))
        self.assertFalse(is_song_that({"genre": "bảy chữ"}))  # no specific_genre key


if __name__ == "__main__":
    unittest.main()
