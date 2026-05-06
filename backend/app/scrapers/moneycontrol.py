"""
MoneyControl scraper — news listing, article metadata, article content, and indices.
"""
import logging
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.config import BASE_URL, NEWS_URL
from app.utils.sentiment import (
    extract_stock_names,
    extract_stock_names_with_sentiment,
    analyze_sentiment,
    simple_summarize,
)

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/90.0.4430.93 Safari/537.36"
    )
}


def get_mc_article_metadata(url):
    """Fetch publish date, author, and first-paragraph description from a MoneyControl article."""
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return None, None, None

        soup = BeautifulSoup(response.text, 'html.parser')

        published_at = None
        date_div = soup.find("div", class_="article_schedule")
        if date_div:
            span = date_div.find("span")
            if span:
                published_at = span.get_text(strip=True)

        author = None
        author_tag = soup.find("div", class_="article_author")
        if author_tag:
            author = author_tag.get_text(strip=True).replace("By", "").strip()

        description = None
        for selector in [".content_wrapper p", ".article-content p", ".story-content p"]:
            paragraphs = soup.select(selector)
            if paragraphs:
                first_para = paragraphs[0].get_text(strip=True)
                if len(first_para) > 50:
                    description = first_para[:200] + "..." if len(first_para) > 200 else first_para
                    break

        return published_at, author, description

    except Exception as e:
        logger.error(f"Error fetching MC metadata: {e}")
        return None, None, None


def get_moneycontrol_news():
    """Scrape top news items from MoneyControl."""
    try:
        response = requests.get(NEWS_URL, timeout=10)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []

        for a_tag in soup.select("h2 a")[:10]:
            title = a_tag.get('title') or a_tag.text.strip()
            href = a_tag.get('href')
            full_link = urljoin(BASE_URL, href)

            published_at, author, description = get_mc_article_metadata(full_link)
            combined_text = title + " " + (description or "")
            stocks_with_sentiment = extract_stock_names_with_sentiment(combined_text)

            news_items.append({
                'title': title,
                'link': full_link,
                'publishedAt': published_at,
                'author': author,
                'description': description,
                'source': 'MoneyControl',
                'stocks': extract_stock_names(combined_text),
                'stocks_with_sentiment': stocks_with_sentiment,
            })

        return news_items

    except Exception as e:
        logger.error(f"Error fetching MoneyControl news: {e}")
        return []


def extract_mc_article_content(soup, url):
    """Extract full article content from a MoneyControl page (BeautifulSoup object)."""
    from flask import jsonify  # local import to avoid circular dependency at module level

    try:
        content = ""
        for selector in [".article_body p", ".content_wrapper p", ".clearfix p", "article p", ".content p"]:
            paragraphs = soup.select(selector)
            if paragraphs:
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                break

        if not content:
            content = soup.get_text(strip=True)

        summary = simple_summarize(content, max_sentences=3)

        author = None
        for selector in [".article_author", ".author", ".byline", "[class*='author']"]:
            tag = soup.select_one(selector)
            if tag:
                author = tag.get_text(strip=True).replace("By", "").strip()
                break

        published_at = None
        for selector in [".article_schedule", ".schedule", "time", "[class*='date']"]:
            tag = soup.select_one(selector)
            if tag:
                published_at = tag.get_text(strip=True)
                break

        all_text = content or soup.get_text()
        stocks_with_sentiment = extract_stock_names_with_sentiment(all_text)
        article_sentiment = analyze_sentiment(content)

        return jsonify({
            "content": content,
            "summary": summary,
            "author": author,
            "publishedAt": published_at,
            "stocks": extract_stock_names(all_text),
            "stocks_with_sentiment": stocks_with_sentiment,
            "article_sentiment": {
                "score": round(article_sentiment, 3),
                "label": "positive" if article_sentiment > 0.1 else "negative" if article_sentiment < -0.1 else "neutral",
            },
            "source": "MoneyControl",
            "url": url,
        })

    except Exception as e:
        from flask import jsonify
        return jsonify({"error": f"Failed to extract MC article: {str(e)}"}), 500


def fetch_indices_from_moneycontrol():
    """Scrape Nifty 50, Sensex, and Bank Nifty from MoneyControl."""
    results = {}

    try:
        url = "https://www.moneycontrol.com/stocksmarketsindia/"
        r = requests.get(url, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        index_rows = [
            ("NIFTY50",   "#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(1)"),
            ("SENSEX",    "#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(2)"),
            ("BANKNIFTY", "#maindindi > div:nth-of-type(1) table tbody tr:nth-of-type(3)"),
        ]

        for key, selector in index_rows:
            tr = soup.select_one(selector)
            cells = tr.find_all("td")
            results[key] = {
                "name": cells[0].get_text(strip=True),
                "current_price": float(cells[1].get_text(strip=True).replace(",", "")),
                "change": float(cells[2].get_text(strip=True).replace(",", "")),
                "change_percent": float(cells[3].get_text(strip=True).replace("%", "")),
                "last_updated": datetime.now().isoformat(),
                "market_status": "open",
                "currency": "INR",
            }

    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        for key in ("NIFTY50", "SENSEX", "BANKNIFTY"):
            results[key] = None

    return results
