"""
Extract and clean poems from Facebook groups scraper JSON.
Outputs: resources/facebook_poems.csv

Usage:
  python src/extract_facebook.py
  python src/extract_facebook.py --input resources/dataset_facebook.json --output data/out.csv
  python src/extract_facebook.py --to-raw        # also save to data_service/raw/ for merge pipeline

Merge into main dataset:
  # 1. Extract Facebook poems to raw format:
  python src/extract_facebook.py --to-raw
  # 2. Merge with existing dataset:
  python data_service/merge_dataset.py            # dry-run stats
  python data_service/merge_dataset.py --extract-new  # extract deduped new poems
  python data_service/merge_dataset.py --commit       # commit to merged dataset
  # 3. Re-run cleaning pipeline:
  python src/clean_data.py --csv data/poems_dataset_merged.csv --output data/clean_data_v2.csv
"""

import json, re, argparse, unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = ROOT / "resources" / "dataset_facebook-groups-scraper_2026-05-22_10-03-20-361.json"
DEFAULT_OUTPUT = ROOT / "resources" / "facebook_poems.csv"

SEP = "<" + chr(10) + ">"


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def strip_emojis(text: str) -> str:
    """Remove emojis and other non-text symbols."""
    # Emoji ranges (broad coverage)
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    text = re.sub(r'[\U0001FA00-\U0001FAFF]', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)
    text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)
    text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)
    text = re.sub(r'[\u2600-\u27BF]', '', text)   # misc symbols
    text = re.sub(r'[\u2700-\u27BF]', '', text)
    text = re.sub(r'[\u2B50-\u2B55]', '', text)
    text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # flags
    # Other decorative chars
    text = re.sub(r'[☆★✿❀❁💮🌸💐🏵🌹🌺🌻🌼🌷⚘❦❧]', '', text)
    # Variation selectors & ZWJ (leftover from stripped emoji sequences)
    text = re.sub(r'[\uFE00-\uFE0F\u200D]', '', text)
    return text


def clean_line(line: str) -> str:
    """Clean a single line of poetry."""
    line = strip_emojis(line)
    # Remove hashtags
    line = re.sub(r'#\S+', '', line)
    # Remove Facebook artifacts
    line = line.replace("&nbsp;", " ").replace("&amp;", "&")
    line = re.sub(r'\(?https?://\S+\)?', '', line)
    # Remove "(Xem thêm...)", "...Xem thêm"
    line = re.sub(r'\(?[Xx]em th[eêêm]+.*?\)?$', '', line)
    # Remove marker bullets like •, -, *, ▶, ▸, ▪, etc.
    line = re.sub(r'^[\s•\-\*▶▸▪▹►▻○●◉◎◌◦⦁➤➜➢➣➔➔]+', '', line)
    # Fix spacing
    line = re.sub(r'\s+([,.;:?!%])', r'\1', line)
    line = re.sub(r'\(\s+', '(', line)
    line = re.sub(r'\s+\)', ')', line)
    # Collapse whitespace
    line = re.sub(r'[ \t]+', ' ', line)
    # Unicode normalize
    line = unicodedata.normalize("NFC", line)
    return line.strip()


def is_metadata_line(line: str) -> bool:
    """Check if a line is metadata (author credit, genre tag, etc.)."""
    stripped = line.strip().lower()
    patterns = [
        r'^tác giả\s*:',
        r'^thể loại\s*:',
        r'^tg\s+', r'^tg[:\s]',
        r'^thơ\s+tg', r'^thơ\s+tác giả',
        r'^tuyển tập\s*:',
        r'^đôi lời từ tác giả',
        r'^thất ngôn', r'^tứ tuyệt', r'^ngũ ngôn',
        r'^lục bát$', r'^bảy chữ$', r'^năm chữ$', r'^tám chữ$',
        r'^\(.*?(?:tác giả|sưu tầm|st|nguồn|trích|kể chuyện).*?\)$',
        r'^[Cc]h[uư]o[ơ]n[ng]\s+\d+.*?:',
        r'^[-–—]{3,}$',  # separators
        r'^…+$', r'^\.{3,}$',
    ]
    return any(re.match(p, stripped) for p in patterns)


def is_poem(text: str) -> bool:
    """Heuristic: detect if text is a poem vs prose/other."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 4:
        return False

    # Exclude spam/AI-generated: more than 3 emojis
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U0001F600-\U0001F64F'
                                  r'\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u27BF'
                                  r'\U0001F1E0-\U0001F1FF☆★✿❀💮🌸🏵🌹🌺🌻🌼⚘❦❧]',
                                  text))
    if emoji_count > 3:
        return False

    # Remove title-like first line for analysis
    body_lines = lines[1:] if len(lines) > 1 else lines

    # Check for prose: long paragraphs (narrative text)
    long_lines = sum(1 for l in body_lines if len(l) > 120)
    if long_lines > len(body_lines) * 0.3:
        return False

    # Check for very long single lines (paragraphs disguised as lines)
    very_long = sum(1 for l in body_lines if len(l) > 200)
    if very_long > 0 and very_long > len(body_lines) * 0.15:
        return False

    # Check average words per line (Vietnamese poetry: ~3-12 words)
    word_counts = [len(l.split()) for l in body_lines]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    if avg_words < 2 or avg_words > 18:
        return False

    # Check line uniformity (poems have relatively uniform line lengths)
    if len(word_counts) >= 4:
        std = (sum((w - avg_words) ** 2 for w in word_counts) / len(word_counts)) ** 0.5
        # Allow more variation for free verse
        if avg_words > 3 and std > avg_words * 1.2:
            return False

    # Check: if >40% of lines end with period (prose marker)
    period_lines = sum(1 for l in body_lines if l.rstrip().endswith('.'))
    if period_lines > len(body_lines) * 0.5 and len(body_lines) > 5:
        return False

    return True


def extract_title(text: str) -> tuple[str, str]:
    """Extract title from first line(s). Returns (title, remaining_content)."""
    text = text.strip()
    lines = text.split('\n')

    # Patterns for title detection:
    # 1. ALL CAPS line
    # 2. Line wrapped in === ... ===
    # 3. Line ending with colon (Tên bài:)
    # 4. Line in brackets [Tên bài]

    title = ""
    content_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            content_start = i + 1
            continue

        # === Title ===
        m = re.match(r'^={1,3}\s*(.+?)\s*={1,3}$', stripped)
        if m:
            title = m.group(1).strip()
            content_start = i + 1
            break

        # [Title] or (Title)
        m = re.match(r'^[\[\(](.+?)[\]\)]$', stripped)
        if m and len(stripped) < 60:
            title = m.group(1).strip()
            content_start = i + 1
            break

        # ALL CAPS (at least 4 chars, not the whole poem)
        if stripped.isupper() and len(stripped) >= 4 and len(stripped) < 80:
            words = stripped.split()
            if len(words) <= 8:
                title = stripped.title()
                content_start = i + 1
                break

        # "Tên: nội dung" pattern
        m = re.match(r'^(.{3,60}):\s*$', stripped)
        if m and i < 3:
            title = m.group(1).strip()
            content_start = i + 1
            break

        # If first non-empty line looks like a title (short, no ending punctuation)
        if i == 0 and len(stripped) < 60 and not re.search(r'[,.!?;]$', stripped):
            body = [l.strip() for l in lines[1:] if l.strip()]
            if body:
                body_avg = sum(len(l) for l in body) / len(body)
                if len(stripped) < body_avg * 0.7:
                    title = stripped.title() if stripped.isupper() else stripped
                    content_start = 1
                    break

        # First line is the title by default
        if i == 0:
            title = stripped.title() if stripped.isupper() else stripped
            content_start = 1
            break

        break

    # Clean title
    title = strip_emojis(title)
    title = re.sub(r'[.。…]{2,}$', '', title)
    # Remove leading/trailing pipe separators and emoji leftovers
    title = re.sub(r'^\s*[|]\s*', '', title)
    title = re.sub(r'\s*[|]\s*$', '', title)
    title = title.strip()

    remaining = '\n'.join(lines[content_start:]).strip()
    return title, remaining


def classify_genre(text: str) -> str:
    """Try to classify poem genre."""
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 3]
    if len(lines) < 2:
        return "thơ tự do"

    # Count 6-8 syllable patterns
    six_count = sum(1 for l in lines if 5 <= len(l.split()) <= 7)
    eight_count = sum(1 for l in lines if 7 <= len(l.split()) <= 9)

    if six_count + eight_count > len(lines) * 0.6:
        # Check for alternating 6-8 (lục bát)
        alt_lb = 0
        for i in range(0, len(lines) - 1, 2):
            w1 = len(lines[i].split())
            w2 = len(lines[i + 1].split()) if i + 1 < len(lines) else 0
            if 5 <= w1 <= 7 and 7 <= w2 <= 9:
                alt_lb += 1
        if alt_lb >= len(lines) * 0.25:
            return "lục bát"

    # 7-syllable detection
    seven_count = sum(1 for l in lines if 6 <= len(l.split()) <= 8)
    if seven_count > len(lines) * 0.5:
        return "bảy chữ"

    # 5-syllable detection
    five_count = sum(1 for l in lines if 4 <= len(l.split()) <= 6)
    if five_count > len(lines) * 0.5:
        return "năm chữ"

    # 8-syllable detection
    strict_eight = sum(1 for l in lines if 7 <= len(l.split()) <= 9)
    if strict_eight > len(lines) * 0.5:
        return "tám chữ"

    return "thơ tự do"


def clean_content(text: str) -> str:
    """Clean poem content: strip metadata notes, format."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        stripped = line.strip()
        if is_metadata_line(stripped):
            continue
        # Skip lines starting with Ảnh (photo credits)
        if re.match(r'^[Ảả]nh\b', stripped, re.IGNORECASE):
            continue
        # Skip lines containing date patterns (dd.mm.yyyy, dd/mm/yyyy, etc.)
        if re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', stripped):
            continue
        # Skip lines with no alphabet characters (pure symbol dividers)
        if not re.search(r'[a-zA-Zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', stripped, re.IGNORECASE):
            continue
        # Skip standalone "..."
        if stripped in ('...', '…', '..', '....'):
            continue
        cleaned.append(line.strip())
    return SEP.join(cleaned) if cleaned else ""


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def extract(input_path=None, output_path=None):
    input_path = Path(input_path or DEFAULT_INPUT)
    output_path = Path(output_path or DEFAULT_OUTPUT)

    print(f"📖  Loading: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    print(f"    Total posts: {len(data)}")

    # Filter non-empty
    posts = [item for item in data if item.get('text', '').strip()]
    print(f"    Non-empty: {len(posts)}")

    poems = []
    skipped = {"empty": 0, "not_poem": 0, "too_short": 0}

    for item in posts:
        raw_text = item['text'].strip()
        if not raw_text:
            skipped["empty"] += 1
            continue

        if not is_poem(raw_text):
            skipped["not_poem"] += 1
            continue

        title, body = extract_title(raw_text)
        content = clean_content(body)
        if not content or content.count(SEP) < 3:
            skipped["too_short"] += 1
            continue

        genre = classify_genre(body)
        author = item.get('user', {}).get('name', '')
        source_url = item.get('facebookUrl', '')
        likes = item.get('likesCount', 0)
        comments = item.get('commentsCount', 0)

        poems.append({
            'title': title,
            'author': author,
            'genre': genre,
            'specific_genre': '',
            'content': content,
            'source': source_url,
            'likes': likes,
            'comments': comments,
        })

    df = pd.DataFrame(poems)
    print(f"\n📊  Poems extracted: {len(df)}")
    print(f"    Skipped: empty={skipped['empty']}, not_poem={skipped['not_poem']}, "
          f"too_short={skipped['too_short']}")

    # Genre distribution
    if len(df) > 0:
        print(f"\n    Genre breakdown:")
        for g, cnt in df['genre'].value_counts().items():
            print(f"      {g:<15s}: {cnt:>5d}")

        # Author distribution
        known = df[df['author'] != '']
        print(f"\n    Authors: {len(known['author'].unique())} unique "
              f"({len(known)} poems with author)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅  Saved → {output_path}")
    return df


def export_to_raw(df, raw_dir=None):
    """Export to data_service/raw/ format for merge pipeline."""
    raw_dir = Path(raw_dir or ROOT / "data_service" / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Map columns to merge pipeline format
    raw = df.copy()
    raw["url"] = raw["source"]
    raw["period"] = ""
    raw = raw[["content", "title", "url", "genre", "period", "specific_genre", "author"]]

    out = raw_dir / "facebook_poems.csv"
    raw.to_csv(out, index=False)
    print(f"✅  Raw export → {out} ({len(raw)} poems)")
    return raw


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract poems from Facebook JSON")
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--to-raw", action="store_true",
                   help="Also export to data_service/raw/ for merge pipeline")
    args = p.parse_args()
    df = extract(args.input, args.output)
    if args.to_raw and len(df) > 0:
        export_to_raw(df)
