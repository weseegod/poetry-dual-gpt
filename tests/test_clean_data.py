"""Tests for src/clean_data.py — one TestCase per pipeline stage."""

import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from clean_data import (
    filter_genres, clean_content, clean_text,
    filter_short, remove_duplicates, content_hash,
)


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


class TestCleanText(unittest.TestCase):
    def setUp(self):
        self.sep = "<" + chr(10) + ">"  # actual newline separator in CSV

    def test_html_entities(self):
        result = clean_text("hello &amp; world &nbsp; test &lt;tag&gt;")
        self.assertNotIn("&amp;", result)
        self.assertNotIn("&nbsp;", result)

    def test_html_tags_per_line(self):
        result = clean_text(f"line one<br>still one{self.sep}line two<p>still two")
        self.assertNotIn("<br>", result)
        self.assertNotIn("<p>", result)
        self.assertIn(self.sep, result)
        self.assertEqual(result.count(self.sep), 1)

    def test_unicode_normalization(self):
        # tổ in NFD: t + o + combining circumflex (U+0302) + hook above (U+0309)
        nfd = "t" + chr(111) + chr(770) + chr(777)
        result = clean_text(nfd)
        self.assertEqual(result, "tổ")

    def test_collapse_spaces(self):
        result = clean_text(f"hello     world{self.sep}foo\t\tbar")
        lines = result.split(self.sep)
        self.assertNotIn("    ", lines[0])
        self.assertNotIn("\t", lines[1])

    def test_drops_empty_lines(self):
        result = clean_text(f"  {self.sep}  {self.sep}valid line")
        self.assertEqual(result.count(self.sep), 0)

    def test_handles_none(self):
        self.assertEqual(clean_text(None), "")

    def test_handles_nan(self):
        self.assertEqual(clean_text(float("nan")), "")


class TestCleanContent(unittest.TestCase):
    def setUp(self):
        self.sep = "<" + chr(10) + ">"

    def test_drops_empties(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát", "lục bát"],
            "content": [
                f"line one{self.sep}line two",
                None,
                "   ",
            ],
        })
        result = clean_content(df)
        self.assertEqual(len(result), 1)
        self.assertIn("line one", result["content"].iloc[0])


class TestFilterShort(unittest.TestCase):
    def setUp(self):
        self.sep = "<" + chr(10) + ">"

    def test_drops_short_poems(self):
        df = pd.DataFrame({
            "genre": ["lục bát"] * 3,
            "content": [
                "only one line",
                f"line 1{self.sep}line 2",
                f"a{self.sep}b{self.sep}c{self.sep}d",
            ],
        })
        result = filter_short(df, min_lines=2)
        self.assertEqual(len(result), 2)
        self.assertNotIn("only one line", result["content"].values)


class TestRemoveDuplicates(unittest.TestCase):
    def setUp(self):
        self.sep = "<" + chr(10) + ">"

    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát", "lục bát"],
            "content": [
                f"poem a{self.sep}line 2",
                f"poem b{self.sep}unique",
                f"poem a{self.sep}line 2",  # exact dupe
            ],
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 2)
        self.assertNotIn("_hash", result.columns)

    def test_drops_case_insensitive(self):
        df = pd.DataFrame({
            "genre": ["lục bát", "lục bát"],
            "content": [
                f"Poem A{self.sep}line 2",
                f"poem a{self.sep}line 2",  # case-insensitive dupe
            ],
        })
        result = remove_duplicates(df)
        self.assertEqual(len(result), 1)

    def test_hash_deterministic(self):
        h1 = content_hash(f"test{self.sep}poem")
        h2 = content_hash(f"test{self.sep}poem")
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
