import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import json

def get_main_blog_and_links():
    base_url = "https://joinditto.in"
    blog_url = f"{base_url}/health-insurance/best-health-plans-in-india/"
    response = requests.get(blog_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try multiple selectors to find main content
    content = (
        soup.find("article") or
        soup.find("div", class_="content") or
        soup.find("div", class_="post-body") or
        soup.find("main") or
        soup.body
    )

    if not content:
        print("⚠️ Could not find main content.")
        return [], []

    main_text = content.get_text(separator="\n", strip=True)
    links = [urljoin(base_url, a['href']) for a in content.find_all("a", href=True)]
    links = list(set(links))

    return [{"url": blog_url, "title": "Best Health Plans in India", "text": main_text}], links


def scrape_linked_pages(links):
    pages = []
    for url in links:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            # Extract <article> or large <div>
            main_section = soup.find("article") or soup.find("div", class_="main") or soup.find("body")
            text = main_section.get_text(separator="\n", strip=True) if main_section else ""

            if len(text) > 300:  # Filter low-content pages
                title = soup.title.string.strip() if soup.title else "Untitled"
                pages.append({"url": url, "title": title, "text": text})
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
    return pages

def save_to_json(main_blog, linked_pages):
    os.makedirs("blog_data", exist_ok=True)
    all_data = main_blog + linked_pages
    with open("blog_data/insurance_pages.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print("✅ Saved structured JSON to blog_data/insurance_pages.json")


if __name__ == "__main__":
    main_blog, links = get_main_blog_and_links()
    linked_pages = scrape_linked_pages(links)
    save_to_json(main_blog, linked_pages)
