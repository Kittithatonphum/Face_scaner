
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

import httpx
from bs4 import BeautifulSoup


# ============================================================
# 1. ตั้งค่าพาธและแหล่งข้อมูล
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"
PAGES_FILE = DATA_DIR / "source_pages.txt"
SOURCES_FILE = DATA_DIR / "sources.json"

ALLOWED_HOSTS = {
    "dusit.ac.th",
    "www.dusit.ac.th",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_PAGES = 50
MAX_IMAGES = 1000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
REQUEST_TIMEOUT = 20.0
REQUEST_DELAY = 0.3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


# ============================================================
# 2. ตรวจสอบ URL
# ============================================================

def normalize_url(url: str) -> str:
    """ลบ fragment และ query ที่ซ้ำ โดยคง query ที่จำเป็นไว้"""
    url = url.strip()

    parsed = urlparse(url)
    query = urlencode(parse_qsl(parsed.query, keep_blank_values=True))

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            query,
            "",
        )
    )


def allowed_url(url: str) -> bool:
    """อนุญาตเฉพาะ HTTPS และโดเมนสวนดุสิตที่กำหนดไว้"""
    try:
        parsed = urlparse(url)

        if parsed.scheme.lower() != "https":
            return False

        if (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
            return False

        if parsed.username or parsed.password:
            return False

        if parsed.port not in (None, 443):
            return False

        return True

    except (ValueError, TypeError):
        return False


def is_image_url(url: str) -> bool:
    """ตรวจนามสกุลรูปภาพจาก path ของ URL"""
    path = urlparse(url).path.lower()
    return Path(path).suffix in IMAGE_EXTENSIONS


# ============================================================
# 3. อ่านรายการหน้าเว็บจาก source_pages.txt
# ============================================================

def load_source_pages() -> list[str]:
    """อ่าน URL ต้นทาง และเลือกเฉพาะ URL ที่ได้รับอนุญาต"""
    if not PAGES_FILE.exists():
        print(f"ไม่พบไฟล์รายการหน้าเว็บ: {PAGES_FILE}")
        return []

    urls = []

    for line in PAGES_FILE.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        url = line.strip()

        if not url or url.startswith("#"):
            continue

        url = normalize_url(url)

        if not allowed_url(url):
            print(f"ข้าม URL ที่ไม่อนุญาต: {url}")
            continue

        if url not in urls:
            urls.append(url)

    return urls[:MAX_PAGES]


# ============================================================
# 4. ค้นหา URL รูปภาพในหน้าเว็บ
# ============================================================

def extract_image_urls(
    html: str,
    page_url: str,
) -> list[str]:
    """
    ค้นหา URL รูปภาพจาก:
    - img src
    - data-src และ lazy loading
    - srcset / data-srcset
    - รูปที่เชื่อมผ่าน a href
    - รูปจาก meta og:image และ twitter:image

    จำกัดรูปไว้ในโดเมนสวนดุสิตที่อนุญาต
    """
    soup = BeautifulSoup(html, "html.parser")
    found: set[str] = set()

    def add_image(raw_url: str | None) -> None:
        if not raw_url:
            return

        raw_url = raw_url.strip()

        if not raw_url:
            return

        if raw_url.startswith(("data:", "blob:", "javascript:")):
            return

        url = normalize_url(urljoin(page_url, raw_url))

        if not allowed_url(url):
            return

        if not is_image_url(url):
            return

        path_lower = urlparse(url).path.lower()

        # ข้ามรูปประกอบที่ไม่ต้องการ
        excluded_names = (
            "new_head",
            "ธงชาติไทย",
            "36x24",
        )

        if any(name.lower() in path_lower for name in excluded_names):
            return

        found.add(url)

    # 4.1 รูปภาพในแท็ก img
    img_attributes = (
        "src",
        "data-src",
        "data-lazy-src",
        "data-original",
        "data-large_image",
        "data-flickity-lazyload",
    )

    for img in soup.find_all("img"):
        for attr in img_attributes:
            add_image(img.get(attr))

        # 4.2 รูปภาพแบบ srcset
        for attr in ("srcset", "data-srcset"):
            srcset = img.get(attr, "")

            for item in srcset.split(","):
                parts = item.strip().split()

                if parts:
                    add_image(parts[0])

    # 4.3 รูปที่เชื่อมผ่านลิงก์
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        absolute_url = urljoin(page_url, href)

        if is_image_url(absolute_url):
            add_image(absolute_url)

    # 4.4 รูปจาก meta tags
    for meta in soup.find_all("meta"):
        property_name = (
            meta.get("property")
            or meta.get("name")
            or ""
        ).lower()

        if property_name in {
            "og:image",
            "og:image:url",
            "twitter:image",
            "twitter:image:src",
        }:
            add_image(meta.get("content"))

    return sorted(found)


# ============================================================
# 5. ดาวน์โหลดภาพอย่างปลอดภัย
# ============================================================

def download_image(
    client: httpx.Client,
    image_url: str,
) -> tuple[bytes | None, str | None]:
    """ดาวน์โหลดภาพเมื่อ URL และขนาดไฟล์ผ่านเงื่อนไข"""
    if not allowed_url(image_url):
        return None, "URL ไม่ได้รับอนุญาต"

    try:
        with client.stream(
            "GET",
            image_url,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                return None, f"HTTP {response.status_code}"

            content_type = response.headers.get(
                "content-type", ""
            ).lower()

            if not content_type.startswith("image/"):
                return None, f"ไม่ใช่ไฟล์ภาพ: {content_type}"

            content_length = response.headers.get(
                "content-length"
            )

            if content_length:
                try:
                    if int(content_length) > MAX_IMAGE_BYTES:
                        return None, "ไฟล์ใหญ่เกินกำหนด"
                except ValueError:
                    pass

            chunks = []
            total_size = 0

            for chunk in response.iter_bytes():
                total_size += len(chunk)

                if total_size > MAX_IMAGE_BYTES:
                    return None, "ไฟล์ใหญ่เกินกำหนด"

                chunks.append(chunk)

            data = b"".join(chunks)

            if not data:
                return None, "ไฟล์ว่าง"

            return data, None

    except httpx.HTTPError as exc:
        return None, f"ดาวน์โหลดไม่สำเร็จ: {exc}"


def make_image_filename(
    image_url: str,
) -> str:
    """สร้างชื่อไฟล์ที่ไม่ซ้ำกันจาก URL"""
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"

    digest = hashlib.sha256(
        image_url.encode("utf-8")
    ).hexdigest()[:16]

    return f"{digest}{suffix}"


# ============================================================
# 6. บันทึกข้อมูลต้นทาง
# ============================================================

def load_existing_sources() -> dict[str, dict]:
    if not SOURCES_FILE.exists():
        return {}

    try:
        data = json.loads(
            SOURCES_FILE.read_text(encoding="utf-8")
        )

        if isinstance(data, list):
            result = {}

            for item in data:
                if isinstance(item, dict) and item.get("url"):
                    result[item["url"]] = item

            return result

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        print("คำเตือน: อ่าน sources.json เดิมไม่ได้")

    return {}


def save_sources(sources: dict[str, dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    SOURCES_FILE.write_text(
        json.dumps(
            list(sources.values()),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# 7. รวบรวมภาพจากหน้าเว็บ
# ============================================================

def collect_images() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    pages = load_source_pages()

    if not pages:
        print("ไม่มี URL ที่ใช้ได้ใน data/source_pages.txt")
        return

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,*/*",
    }

    sources = load_existing_sources()
    seen_images: set[str] = set()

    new_downloads = 0
    pages_read = 0
    failed_downloads = 0

    with httpx.Client(
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    ) as client:

        for page_url in pages:
            print(f"\nกำลังอ่านหน้าเว็บ: {page_url}")

            if not allowed_url(page_url):
                print("ข้าม: URL ไม่ได้รับอนุญาต")
                continue

            try:
                response = client.get(
                    page_url,
                    follow_redirects=True,
                )

                # ตรวจ URL ปลายทางหลัง redirect ด้วย
                if not allowed_url(str(response.url)):
                    print("ข้าม: เปลี่ยนเส้นทางไปโดเมนอื่น")
                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type", ""
                ).lower()

                if "html" not in content_type:
                    print("ข้าม: เนื้อหาที่ได้รับไม่ใช่ HTML")
                    continue

                html = response.text
                pages_read += 1

            except httpx.HTTPError as exc:
                print(f"อ่านหน้าเว็บไม่สำเร็จ: {exc}")
                continue

            image_urls = extract_image_urls(
                html,
                str(response.url),
            )

            print(
                "พบ URL รูปภาพที่ผ่านตัวกรอง:",
                len(image_urls),
            )

            for image_url in image_urls:
                if image_url in seen_images:
                    continue

                seen_images.add(image_url)

                if len(seen_images) > MAX_IMAGES:
                    print("ถึงจำนวนภาพสูงสุดที่กำหนด")
                    break

                filename = make_image_filename(image_url)
                image_path = GALLERY_DIR / filename

                # ถ้ามีไฟล์อยู่แล้ว ให้คงไฟล์เดิมไว้
                if image_path.exists() and image_path.stat().st_size > 0:
                    sources[image_url] = {
                        "url": image_url,
                        "filename": filename,
                        "page_url": page_url,
                        "local_path": str(
                            image_path.relative_to(ROOT)
                        ).replace("\\", "/"),
                    }
                    continue

                image_data, error = download_image(
                    client,
                    image_url,
                )

                if error:
                    failed_downloads += 1
                    print(f"ดาวน์โหลดไม่ได้: {image_url}")
                    print(f"  สาเหตุ: {error}")
                    continue

                try:
                    image_path.write_bytes(image_data)

                except OSError as exc:
                    failed_downloads += 1
                    print(f"บันทึกไฟล์ไม่ได้: {exc}")
                    continue

                sources[image_url] = {
                    "url": image_url,
                    "filename": filename,
                    "page_url": page_url,
                    "local_path": str(
                        image_path.relative_to(ROOT)
                    ).replace("\\", "/"),
                }

                new_downloads += 1
                print(f"บันทึกภาพแล้ว: {filename}")

                time.sleep(REQUEST_DELAY)

            time.sleep(REQUEST_DELAY)

    save_sources(sources)

    gallery_files = [
        p for p in GALLERY_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("\n========== ผลการรวบรวมภาพ ==========")
    print(f"URL หน้าเว็บที่กำหนด: {len(pages)}")
    print(f"อ่านหน้าเว็บสำเร็จ: {pages_read}")
    print(f"URL รูปภาพไม่ซ้ำที่พบ: {len(seen_images)}")
    print(f"ดาวน์โหลดภาพใหม่: {new_downloads}")
    print(f"ดาวน์โหลด/บันทึกไม่สำเร็จ: {failed_downloads}")
    print(f"รายการภาพใน sources.json: {len(sources)}")
    print(f"ไฟล์ภาพใน data/gallery: {len(gallery_files)}")
    print(f"โฟลเดอร์ภาพ: {GALLERY_DIR}")


if __name__ == "__main__":
    collect_images()