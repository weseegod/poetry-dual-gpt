"""
Scrape Vietnamese poems from isach.info using Playwright (JS rendering).

Target authors: Nguyễn Du, Hồ Xuân Hương, Hàn Mặc Tử, Nguyễn Khuyến,
                Tố Hữu, Xuân Diệu, Huy Cận, Nguyễn Bính

Usage:
  python data_service/scraper.py --author xuan_dieu
  python data_service/scraper.py --all --browser
  python data_service/scraper.py --dry-run
"""

import argparse, hashlib, re, sys, time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "poems_dataset_clean.csv"
RAW_DIR = ROOT / "data_service" / "raw"

AUTHORS = {
    "xuan_dieu":     {"name": "Xuân Diệu",       "period": "hiện đại", "pages": 6},
    "huy_can":       {"name": "Huy Cận",          "period": "hiện đại", "pages": 3},
    "han_mac_tu":    {"name": "Hàn Mặc Tử",       "period": "hiện đại", "pages": 3},
    "to_huu":        {"name": "Tố Hữu",           "period": "hiện đại", "pages": 2},
    "nguyen_binh":   {"name": "Nguyễn Bính",      "period": "hiện đại", "pages": 3},
    "ho_xuan_huong": {"name": "Hồ Xuân Hương",    "period": "trung đại", "pages": 2},
    "nguyen_khuyen": {"name": "Nguyễn Khuyến",    "period": "trung đại", "pages": 2},
    "nguyen_du":     {"name": "Nguyễn Du",         "period": "trung đại", "pages": 1},
}

LIST_URL = "https://isach.info/poem.php?list=poem&author={author}&order=poem_id&page={page}"


def get_poem_urls(author_key):
    """Extract poem URLs from paginated author listing pages (static HTML)."""
    cfg = AUTHORS[author_key]
    poems = []
    for page in range(1, cfg["pages"] + 1):
        url = LIST_URL.format(author=author_key, page=page)
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        links = re.findall(
            r'href="(/poem\.php\?poem=[a-z0-9_]+__' + author_key + r')"',
            resp.text
        )
        for link in links:
            poem_url = urljoin(url, link)
            title_match = re.search(
                r'href="' + re.escape(link) + r'"[^>]*>([^<]+)<', resp.text
            )
            title = title_match.group(1).strip() if title_match else link
            poems.append((poem_url, title))
        print(f"  Page {page}/{cfg['pages']}: {len(links)} poems")
        if not links:
            break

    seen = set()
    return [(u, t) for u, t in poems if not (u in seen or seen.add(u))]


def scrape_poem(url, author_name):
    """Scrape poem content using Playwright for JS rendering."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)  # Wait for AJAX poem load
        html = page.content()
        browser.close()

    # Extract title
    title_match = re.search(r'<title>(.*?)</title>', html)
    title = title_match.group(1) if title_match else ""
    title = re.sub(r'^Thơ\s*[-–]\s*', '', title)
    title = re.sub(r'\s*[-–]\s*' + re.escape(author_name) + r'.*$', '', title).strip()

    # Extract content: find poem body in the page
    content = _extract_poem_body(html)

    return {"title": title, "content": content, "url": url}


def _extract_poem_body(html):
    """Extract poem text from JS-rendered page.
    
    Strategy: find Vietnamese text blocks near the poem title, 
    skip navigation/menu text by looking for contiguous poetry lines.
    """
    # Remove scripts, styles
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
    # Convert block elements and <br> to newlines
    text = re.sub(r'<(br|p|div|li|h\d)[^>]*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Filter: keep only lines with Vietnamese characters, skip nav/menu
    noise = {
        'đăng nhập', 'đăng ký', 'quên mật khẩu', 'trang chủ', 'truyện',
        'kiếm hiệp', 'tiên hiệp', 'tiểu thuyết', 'ngôn tình', 'trinh thám',
        'cổ tích', 'tuổi học trò', 'kinh dị', 'tùy bút', 'khoa học',
        'facebook', 'google', 'download', 'tải về', 'menu', 'banner',
        'isach.info', 'ebook', 'sách nói', 'lời nhạc', 'sưu tầm',
        'tiện ích', 'trợ giúp', 'liên hệ', 'điều lệ', 'diễn đàn',
        'liên kết nhanh', 'hướng dẫn', 'biên tập', 'phần mềm',
        'upload', 'nạo gạo', 'tùy chỉnh', 'phong tục', 'kỹ năng',
        'nghệ thuật sống', 'việc làm', 'giáo dục', 'lịch sử',
        'danh mục', 'ngẫu nhiên', 'thêm bài', 'xem hướng dẫn',
        'ai đang online', 'nạp gạo', 'phú ông', 'ca sĩ', 'tác giả',
        'chọn ngày giờ', 'cưới hỏi', 'đạo hiếu', 'giỗ tết',
        'khấn nôm', 'phong thủy', 'sinh dưỡng', 'tang lễ',
        'cẩm nang', 'chuyện công sở', 'thăng tiến', 'phỏng vấn',
        'tìm việc', 'thị trường', 'nhà tuyển dụng', 'đổi nghề',
        'lương bổng', 'sức khỏe', 'thưởng thức', 'công sở',
        'học đường', 'gia đình', 'nấu ăn', 'tình yêu', 'đắc nhân tâm',
    }

    vn_lines = []
    for line in lines:
        if not any(ord(c) > 127 for c in line):
            continue
        lower = line.lower()
        if any(n in lower for n in noise):
            continue
        if len(line) < 5:
            continue
        vn_lines.append(line)

    # Find contiguous blocks of poem lines (skip isolated lines)
    # Poem lines typically have 5-10 words each and appear in blocks of 4+
    blocks = []
    current = []
    for line in vn_lines:
        words = line.split()
        if 3 <= len(words) <= 15:
            current.append(line)
        else:
            if len(current) >= 4:
                blocks.append('\n'.join(current))
            current = []
    if len(current) >= 4:
        blocks.append('\n'.join(current))

    # Return the largest block (likely the poem)
    if blocks:
        result = max(blocks, key=len)
        # Remove header line like "Tập thơ Xuân Diệu:" or "Thơ Xuân Diệu:"
        lines = result.split('\n')
        if lines and re.match(r'(Tập thơ|Thơ)\s+[\wĐĂÂÊÔƠƯ]+\s*:', lines[0]):
            lines = lines[1:]
        return '\n'.join(lines)

    # Fallback: return all VN lines
    return '\n'.join(vn_lines)


def detect_genre(content):
    """Detect genre from syllable patterns."""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    if len(lines) < 2:
        return "thơ tự do"
    syl_counts = [len(re.findall(r'\b\w+\b', l)) for l in lines]
    lb = sum(1 for i in range(0, len(syl_counts)-1, 2)
             if syl_counts[i] in (5,6,7) and syl_counts[i+1] in (7,8,9))
    tn = sum(1 for s in syl_counts if s in (6,7,8))
    if lb >= len(lines) * 0.3:
        return "lục bát"
    if tn >= len(lines) * 0.5:
        return "bảy chữ"
    return "thơ tự do"


def clean_content(content):
    """Normalize content: join lines with '<\n>' separator."""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    return '<\n>'.join(lines)


def compute_hash(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:16]


def merge_into_csv(poems, csv_path):
    """Merge new poems into CSV, dedup by content hash."""
    csv_path = Path(csv_path)
    existing = pd.read_csv(csv_path)
    existing['_h'] = existing['content'].apply(compute_hash)
    seen = set(existing['_h'])

    new = []
    for p in poems:
        h = compute_hash(p['content'])
        if h not in seen:
            seen.add(h)
            new.append(p)

    if not new:
        print(f"  0 new (all {len(poems)} duplicates of {len(existing):,} existing)")
        return 0

    new_df = pd.DataFrame(new)
    cols = ['content', 'title', 'url', 'genre', 'period', 'specific_genre', 'author']
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = ""
    new_df = new_df[cols]

    merged = pd.concat([existing.drop(columns=['_h']), new_df], ignore_index=True)
    merged.to_csv(csv_path, index=False)
    print(f"  +{len(new_df)} → {len(merged):,} total")
    return len(new_df)


def download_author(key):
    """Full pipeline for one author: URLs → scrape → clean → merge."""
    cfg = AUTHORS[key]
    print(f"\n{'='*60}\n  {cfg['name']} ({key})  |  {cfg['period']}\n{'='*60}")

    urls = get_poem_urls(key)
    print(f"  {len(urls)} unique poems")

    poems = []
    for i, (url, title) in enumerate(urls):
        print(f"  [{i+1}/{len(urls)}] {title[:60]}")
        try:
            p = scrape_poem(url, cfg['name'])
            if p['content']:
                p['author'] = cfg['name']
                p['period'] = cfg['period']
                p['genre'] = detect_genre(p['content'])
                p['content'] = clean_content(p['content'])
                poems.append(p)
            else:
                print(f"    ⚠️  empty")
        except Exception as e:
            print(f"    ❌ {e}")
        time.sleep(0.3)

    # Save raw
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"{key}_{int(time.time())}.csv"
    pd.DataFrame(poems).to_csv(raw, index=False)
    print(f"  💾  {raw} ({len(poems)} poems)")

    # Merge
    added = merge_into_csv(poems, CSV_PATH)
    return added


def main():
    p = argparse.ArgumentParser(description="Scrape Vietnamese poems from isach.info")
    p.add_argument("--author", type=str, help=f"Author key: {', '.join(AUTHORS)}")
    p.add_argument("--all", action="store_true", help="Download all configured authors")
    p.add_argument("--dry-run", action="store_true", help="List poems without downloading")
    args = p.parse_args()

    # Check playwright
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ playwright not installed. Run:\n"
              "   pip install playwright && playwright install chromium")
        sys.exit(1)

    keys = list(AUTHORS) if args.all else ([args.author] if args.author else [])
    if not keys:
        p.print_help()
        return

    if args.dry_run:
        for key in keys:
            cfg = AUTHORS[key]
            urls = get_poem_urls(key)
            print(f"{cfg['name']}: {len(urls)} poems")
            for url, title in urls[:3]:
                print(f"  {title}")
            if len(urls) > 3:
                print(f"  ... +{len(urls)-3}")
    else:
        total = 0
        for key in keys:
            total += download_author(key)
        print(f"\n✅ Done. Added {total} new poems total.")


if __name__ == "__main__":
    main()
