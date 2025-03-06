import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://researchmap.jp/ezaki" 
URLS = {
    "research_interests": BASE_URL + "/research_interests",
    "research_areas": BASE_URL + "/research_areas",
    "education": BASE_URL + "/education?limit=100",
    "research_experience": BASE_URL + "/research_experience?limit=100",
    "association_memberships": BASE_URL + "/association_memberships?limit=100",
    "awards": BASE_URL + "/awards?limit=100",
    "misc": BASE_URL + "/misc?limit=100",
    "books_etc": BASE_URL + "/books_etc?limit=100",
    "presentations": BASE_URL + "/presentations?limit=100",
    "research_projects": BASE_URL + "/research_projects?limit=100"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def get_soup(url):
    """指定URLのHTMLを取得し、BeautifulSoupオブジェクトを返す"""
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"Failed to fetch {url}, Status Code: {response.status_code}")
        return None

def extract_text(soup, selector, multiple=False):
    """CSSセレクタを使ってデータを取得"""
    if multiple:
        elements = soup.select(selector)
        return [e.text.strip() for e in elements]
    else:
        element = soup.select_one(selector)
        return element.text.strip() if element else "N/A"

def extract_items_from_url(url, selector):
    """URLから指定されたセレクタでデータを取得（ページネーション対応）"""
    items = set()  # 重複を避けるためにセットを使用
    page_start = 0  # ページネーション対応
    while True:
        page_url = f"{url}&start={page_start}" if page_start > 0 else url
        soup = get_soup(page_url)
        if not soup:
            break
        
        elements = soup.select(selector)
        items.update([element.text.strip() for element in elements if element.text.strip()])
        
        if len(elements) < 100:
            break

        page_start += 100
        time.sleep(2)  # サーバー負荷軽減
    
    return list(items)

def scrape_researchmap():
    """ResearchMapから研究者情報をスクレイピング"""
    soup = get_soup(BASE_URL)
    if not soup:
        return

    # 基本情報
    name = extract_text(soup, "header h1")
    
    # 各種情報をスクレイピング
    selectors = {
        "research_interests": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div a",
        "research_areas": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div a",
        "education": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div div a",
        "research_experience": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div div a",
        "association_memberships": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div div a",
        "awards": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div:nth-of-type(1) div a",
        "misc": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div:nth-of-type(1) a",
        "books_etc": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div:nth-of-type(1) a",
        "presentations": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div:nth-of-type(1) a",
        "research_projects": "body main div div:nth-of-type(1) section div div:nth-of-type(2) ul li div div:nth-of-type(1) a"
    }

    researcher_data = {"name": name}
    for key, url in URLS.items():
        researcher_data[key] = extract_items_from_url(url, selectors[key])

    return researcher_data

if __name__ == "__main__":
    researcher_info = scrape_researchmap()
    if researcher_info:
        for key, value in researcher_info.items():
            print(f"\n{key.capitalize()}: ")
            if isinstance(value, list):
                for i, item in enumerate(value, 1):
                    print(f"  {i}. {item}")
            else:
                print(f"  {value}")