"""Tests for src/clean_data.py — one TestCase per pipeline stage."""

import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clean_data import (
    filter_genres, clean_content, clean_text, fix_spacing,
    filter_short, remove_duplicates, content_hash,
)

SEP = "<" + chr(10) + ">"


class TestFilterGenres(unittest.TestCase):
    def test_keeps_wanted_genres(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "bảy chữ", "thơ tự do", "tám chữ", "năm chữ", "lục bát"],
            "content": [f"poem {i}" for i in range(6)],
        })
        result = filter_genres(df)
        self.assertIn("lục bát", result["genre"].values)
        self.assertIn("bảy chữ", result["genre"].values)
        self.assertNotIn("thơ tự do", result["genre"].values)
        self.assertNotIn("tám chữ", result["genre"].values)
        self.assertEqual(len(result), 3)


class TestFixSpacing(unittest.TestCase):
    def test_space_before_punctuation(self):
        self.assertEqual(fix_spacing("hello , world . test ; x : y ? z ! a %"),
                         "hello, world. test; x: y? z! a%")

    def test_space_after_opening_paren(self):
        self.assertEqual(fix_spacing("( hello world"), "(hello world")

    def test_space_before_closing_paren(self):
        self.assertEqual(fix_spacing("hello world )"), "hello world)")

    def test_parentheses_with_content(self):
        self.assertEqual(fix_spacing("( câu 39-170 )"), "(câu 39-170)")

    def test_no_change_on_correct_text(self):
        text = "trăm năm trong cõi người ta"
        self.assertEqual(fix_spacing(text), text)

    def test_vietnamese_with_spaces(self):
        result = fix_spacing("sắm sanh nếp tử , xe châu ,")
        self.assertEqual(result, "sắm sanh nếp tử, xe châu,")


class TestCleanText(unittest.TestCase):
    def test_html_entities(self):
        result = clean_text("hello &amp; world &nbsp; test &lt;tag&gt;")
        self.assertNotIn("&amp;", result)
        self.assertNotIn("&nbsp;", result)

    def test_html_tags_per_line(self):
        result = clean_text(f"line one<br>still one{SEP}line two<p>still two")
        self.assertNotIn("<br>", result)
        self.assertNotIn("<p>", result)
        self.assertIn(SEP, result)
        self.assertEqual(result.count(SEP), 1)

    def test_unicode_normalization(self):
        nfd = "t" + chr(111) + chr(770) + chr(777)
        result = clean_text(nfd)
        self.assertEqual(result, "tổ")

    def test_collapse_spaces(self):
        result = clean_text(f"hello     world{SEP}foo\t\tbar")
        lines = result.split(SEP)
        self.assertNotIn("    ", lines[0])
        self.assertNotIn("\t", lines[1])

    def test_drops_empty_lines(self):
        result = clean_text(f"  {SEP}  {SEP}valid line")
        self.assertEqual(result.count(SEP), 0)

    def test_handles_none(self):
        self.assertEqual(clean_text(None), "")

    def test_handles_nan(self):
        self.assertEqual(clean_text(float("nan")), "")

    def test_strips_line_numbers(self):
        """( câu 39-170 ) and similar should be dropped."""
        result = clean_text(f"poem line{SEP}( câu 39-170 ){SEP}another line")
        self.assertNotIn("câu", result)
        self.assertEqual(result.count(SEP), 1)

    def test_strips_page_numbers(self):
        result = clean_text(f"verse one{SEP}( 5 ){SEP}verse two")
        self.assertNotIn("( 5 )", result)

    def test_fixes_vietnamese_spacing(self):
        """Real-world example: space before punctuation."""
        result = clean_text(f"sắm sanh nếp tử , xe châu ,{SEP}vì chưng chút phận hẩm hiu")
        self.assertIn("nếp tử, xe châu,", result)
        self.assertNotIn(" ,", result)

    def test_fixes_paren_spacing(self):
        result = clean_text(f"lời ( thơ ) nhỏ{SEP}người ( xa ) rồi")
        self.assertIn("lời (thơ) nhỏ", result)
        self.assertIn("người (xa) rồi", result)

    def test_preserves_valid_parentheses(self):
        """Parenthetical content without spacing issues should stay."""
        result = clean_text("ngày xuân (con én) đưa thoi")
        self.assertIn("(con én)", result)


class TestCleanContent(unittest.TestCase):
    def test_drops_empties(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát", "lục bát"],
            "content": [f"line one{SEP}line two", None, "   "],
        })
        result = clean_content(df)
        self.assertEqual(len(result), 1)
        self.assertIn("line one", result["content"].iloc[0])


class TestFilterShort(unittest.TestCase):
    def test_drops_short_poems(self):
        df = pd.DataFrame({
            "genre": ["lục bát"] * 3,
            "content": [
                "only one line",
                f"line 1{SEP}line 2",
                f"a{SEP}b{SEP}c{SEP}d",
            ],
        })
        result = filter_short(df, min_lines=2)
        self.assertEqual(len(result), 2)
        self.assertNotIn("only one line", result["content"].values)


class TestRemoveDuplicates(unittest.TestCase):
    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát", "lục bát"],
            "content": [
                f"poem a{SEP}line 2",
                f"poem b{SEP}unique",
                f"poem a{SEP}line 2",
            ],
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 2)
        self.assertNotIn("_hash", result.columns)

    def test_drops_case_insensitive(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát"],
            "content": [
                f"Poem A{SEP}line 2",
                f"poem a{SEP}line 2",
            ],
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 1)

    def test_drops_spacing_duplicates(self):
        """Poems that differ only by spacing should be duplicates."""
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát"],
            "content": [
                "hello , world",
                "hello, world",
            ],
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 1)

    def test_hash_deterministic(self):
        h1 = content_hash(f"test{SEP}poem")
        h2 = content_hash(f"test{SEP}poem")
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
