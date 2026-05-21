"""
Scrape Vietnamese poems from isach.info using Playwright.
Saves each poem immediately to data_service/raw/{author}.csv.

Target: Nguyễn Du, Hồ Xuân Hương, Hàn Mặc Tử, Nguyễn Khuyến,
        Tố Hữu, Xuân Diệu, Huy Cận, Nguyễn Bính

Usage:
  python data_service/scraper.py --all
  python data_service/scraper.py --author xuan_dieu
  python data_service/scraper.py --dry-run
"""

import argparse, csv, os, re, sys, time
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).parent.parent
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


# ═══════════════════════════════════════════════════════
#  URL EXTRACTION (static HTML, no playwright needed)
# ═══════════════════════════════════════════════════════

def get_poem_urls(author_key):
    """Extract poem URLs from paginated author listing pages."""
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


# ═══════════════════════════════════════════════════════
#  POEM SCRAPING (playwright for JS rendering)
# ═══════════════════════════════════════════════════════

def scrape_poem(url, author_name):
    """Scrape a single poem page. Returns dict with title, content."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()

    title = _extract_title(html, author_name)
    content = _extract_body(html)
    genre = _detect_genre(content)
    content_clean = _clean_content(content)

    return {
        "title": title,
        "content": content_clean,
        "genre": genre,
        "url": url,
    }


def _extract_title(html, author_name):
    m = re.search(r'<title>(.*?)</title>', html)
    if not m:
        return ""
    title = m.group(1)
    title = re.sub(r'^Thơ\s*[-–]\s*', '', title)
    title = re.sub(r'\s*[-–]\s*' + re.escape(author_name) + r'.*$', '', title).strip()
    return title


def _extract_body(html):
    """Extract poem text from JS-rendered HTML."""
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
    text = re.sub(r'<(br|p|div|li|h\d)[^>]*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    NOISE = {
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
        'cách sử dụng', 'qr code', 'chuỗi tìm kiếm', 'tập thơ',
        'chọn ngày giờ', 'cưới hỏi', 'đạo hiếu', 'giỗ tết',
        'khấn nôm', 'phong thủy', 'sinh dưỡng', 'tang lễ',
        'cẩm nang', 'chuyện công sở', 'thăng tiến', 'phỏng vấn',
        'tìm việc', 'thị trường', 'nhà tuyển dụng', 'đổi nghề',
        'lương bổng', 'sức khỏe', 'thưởng thức', 'công sở',
        'học đường', 'gia đình', 'nấu ăn', 'tình yêu', 'đắc nhân tâm',
        'bạn không đủ sức', 'nếu bạn', 'mae west', 'they say love',
        'an institution', 'mọt sách', 'thư viện trực tuyến',
    }
    vn_lines = []
    for line in lines:
        if not any(ord(c) > 127 for c in line):
            continue
        lower = line.lower()
        if any(n in lower for n in NOISE):
            continue
        if len(line) < 5:
            continue
        vn_lines.append(line)

    # Group into contiguous blocks of poem-like lines (5-12 words)
    blocks, current = [], []
    for line in vn_lines:
        words = line.split()
        if 4 <= len(words) <= 15:
            current.append(line)
        else:
            if len(current) >= 3:
                blocks.append('\n'.join(current))
            current = []
    if len(current) >= 3:
        blocks.append('\n'.join(current))

    if not blocks:
        return '\n'.join(vn_lines)

    result = max(blocks, key=len)
    # Strip any remaining header lines at the start
    lines = result.split('\n')
    # Remove leading lines that look like headers (short, no sentence structure)
    while lines and (len(lines[0].split()) <= 3 or 'tập thơ' in lines[0].lower()):
        lines = lines[1:]
    return '\n'.join(lines) if lines else result


def _detect_genre(content):
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    if len(lines) < 2:
        return "thơ tự do"
    syl = [len(re.findall(r'\b\w+\b', l)) for l in lines]
    lb = sum(1 for i in range(0, len(syl)-1, 2) if syl[i] in (5,6,7) and syl[i+1] in (7,8,9))
    tn = sum(1 for s in syl if s in (6,7,8))
    if lb >= len(lines) * 0.3:
        return "lục bát"
    if tn >= len(lines) * 0.5:
        return "bảy chữ"
    return "thơ tự do"


def _clean_content(content):
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    return '<\n>'.join(lines)


# ═══════════════════════════════════════════════════════
#  PROGRESS TRACKING & SAVE
# ═══════════════════════════════════════════════════════

def save_poem(poem, csv_path):
    """Append a single poem to CSV immediately."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=poem.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(poem)


def load_scraped_urls(csv_path):
    """Load already-scraped URLs from existing CSV to resume."""
    if not os.path.exists(csv_path):
        return set()
    urls = set()
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            urls.add(row.get('url', ''))
    return urls


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def download_author(key):
    """Download all poems for one author, saving each poem immediately."""
    cfg = AUTHORS[key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f"{key}.csv"

    print(f"\n{'='*60}")
    print(f"  {cfg['name']} ({key})  |  {cfg['period']}")
    print(f"{'='*60}")

    urls = get_poem_urls(key)
    already = load_scraped_urls(csv_path)
    new_urls = [(u, t) for u, t in urls if u not in already]
    skip = len(urls) - len(new_urls)

    print(f"  {len(urls)} poems total ({skip} already scraped, {len(new_urls)} new)")

    if not new_urls:
        print("  ✅ All done.")
        return 0

    new = 0
    for i, (url, title) in enumerate(new_urls):
        print(f"  [{i+1}/{len(new_urls)}] {title[:70]}")
        try:
            poem = scrape_poem(url, cfg['name'])
            if not poem['content']:
                print(f"    ⚠️  empty content")
                continue
            poem['author'] = cfg['name']
            poem['period'] = cfg['period']
            if 'specific_genre' not in poem:
                poem['specific_genre'] = ''
            save_poem(poem, csv_path)
            new += 1
        except Exception as e:
            print(f"    ❌ {e}")
        time.sleep(0.3)  # polite

    print(f"  ✅ {new} new poems saved → {csv_path}")
    return new


def main():
    p = argparse.ArgumentParser(description="Scrape Vietnamese poems from isach.info")
    p.add_argument("--author", type=str, help=f"Author key: {', '.join(AUTHORS)}")
    p.add_argument("--all", action="store_true", help="Download all configured authors")
    p.add_argument("--dry-run", action="store_true", help="List poems without downloading")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-scraped poems (default: on)")
    args = p.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ playwright not installed.\n"
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
            print(f"\n{cfg['name']}: {len(urls)} poems")
            for url, title in urls[:3]:
                print(f"  {title}")
            if len(urls) > 3:
                print(f"  ... +{len(urls)-3}")
        return

    total = 0
    for key in keys:
        total += download_author(key)
    print(f"\n{'='*60}")
    print(f"✅ Done. {total} new poems scraped.")
    print(f"   Raw files: data_service/raw/")
    print(f"   Next: python data_service/merge_dataset.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
