"""
Multi-Category Unsplash Image Scraper
=====================================
Searches Unsplash for specific visual categories (trap objects, true negatives)
and saves images into organised subdirectories for YOLO training.

Usage:
    pip install selenium requests
    python scrape_images.py
"""

import os
import time
import requests
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


# ─── Configuration ───────────────────────────────────────────────────────────
BASE_URL = "https://unsplash.com/s/photos/"
SAVE_ROOT = "data/raw_images/new"
SCROLL_PAUSE = 2.0
NUM_SCROLLS = 3
REQUEST_TIMEOUT = 15

# Each category: (folder_name, [search_terms])
CATEGORIES = {
    # ── Category 1: Trap Objects ──
    "red_postboxes": [
        "Pillar box",
        "Royal Mail postbox",
        "red street mailbox",
        "freestanding postbox",
    ],
    "water_drums": [
        "Blue plastic water drum",
        "rain barrel",
        "industrial liquid container",
        "55-gallon drum",
    ],
    "poster_bins_2d": [
        "Recycling campaign poster",
        "anti-littering sign",
        "Keep Britain Tidy poster",
        "trash can icon sign",
    ],
    # ── Category 2: Pure Backgrounds (True Negatives) ──
    "empty_ground": [
        "Empty concrete sidewalk",
        "asphalt texture",
        "brick pavement street",
        "dirt path in park",
        "cobblestone background",
    ],
    "scattered_garbage": [
        "Litter on sidewalk",
        "trash in grass",
        "street debris",
        "crushed cans on floor",
        "discarded wrappers",
    ],
    "natural_leaves": [
        "Autumn leaves on ground",
        "forest floor texture",
        "fallen leaves background",
        "pile of dead leaves",
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


def create_driver():
    """Spin up a Chrome instance with anti-detection flags."""
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # Uncomment for headless mode:
    # options.add_argument("--headless=new")

    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver


def smooth_scroll(driver, scrolls=NUM_SCROLLS, pause=SCROLL_PAUSE):
    """Scroll down to trigger lazy-loaded images."""
    for i in range(1, scrolls + 1):
        driver.execute_script(
            "window.scrollBy({top: window.innerHeight, behavior: 'smooth'});"
        )
        time.sleep(pause)


def extract_image_urls(driver):
    """Extract real image URLs from <img> tags (skip base64, SVG, tiny)."""
    imgs = driver.find_elements(By.TAG_NAME, "img")
    urls = set()
    for img in imgs:
        src = img.get_attribute("src") or ""
        if not src or src.startswith("data:") or src.endswith(".svg") or len(src) < 30:
            continue
        urls.add(src)
    return list(urls)


def download_images(urls, save_dir, start_idx=1):
    """Download images to save_dir, returns count of successes."""
    os.makedirs(save_dir, exist_ok=True)
    success = 0

    for idx, url in enumerate(urls, start=start_idx):
        ext = ".jpg"
        for candidate in [".png", ".webp", ".gif", ".jpeg"]:
            if candidate in url.lower():
                ext = candidate
                break

        filename = f"image_{idx}{ext}"
        filepath = os.path.join(save_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            continue

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            total_bytes = 0
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_bytes += len(chunk)

            success += 1
            print(f"      ✅ {filename} ({total_bytes // 1024} KB)")

        except requests.RequestException as e:
            print(f"      ❌ {filename}: {e}")

    return success


def main():
    driver = create_driver()
    grand_total = 0

    try:
        for category, search_terms in CATEGORIES.items():
            save_dir = os.path.join(SAVE_ROOT, category)
            category_total = 0
            file_counter = 1

            print(f"\n{'='*60}")
            print(f"📂 Category: {category}")
            print(f"{'='*60}")

            for term in search_terms:
                url = BASE_URL + quote(term)
                print(f"\n   🔍 Searching: \"{term}\"")
                print(f"      URL: {url}")

                driver.get(url)
                time.sleep(3)  # Wait for initial load

                smooth_scroll(driver)

                image_urls = extract_image_urls(driver)
                print(f"      🔗 Found {len(image_urls)} images")

                if image_urls:
                    downloaded = download_images(image_urls, save_dir, start_idx=file_counter)
                    file_counter += len(image_urls)
                    category_total += downloaded

            print(f"\n   📊 {category}: {category_total} images saved to {save_dir}/")
            grand_total += category_total

    finally:
        driver.quit()

    print(f"\n{'='*60}")
    print(f"🏁 DONE — {grand_total} total images downloaded to {SAVE_ROOT}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
