"""
Economic Times scraper — news listing and article content extraction.
"""
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.config import ET_BASE_URL, ET_NEWS_URL
from app.utils.sentiment import (
    extract_stock_names,
    extract_stock_names_with_sentiment,
    analyze_sentiment,
    simple_summarize,
)

logger = logging.getLogger(__name__)


def get_economictimes_news():
    """Scrape top news items from Economic Times."""
    try:
        response = requests.get(ET_NEWS_URL, timeout=10)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []

        stories = []
        for selector in [".eachStory", ".story-box", ".story", "article"]:
            stories = soup.select(selector)
            if stories:
                break

        for story in stories[:10]:
            title_tag = (
                story.select_one("h3 a")
                or story.select_one("h2 a")
                or story.select_one("a")
            )
            desc_tag = story.select_one("p")
            time_tag = story.select_one("time") or story.select_one(".date")

            if title_tag:
                title = title_tag.get_text(strip=True)
                relative_url = title_tag.get('href')
                if relative_url:
                    full_link = urljoin(ET_BASE_URL, relative_url)
                    description = desc_tag.get_text(strip=True) if desc_tag else None
                    published_at = time_tag.get_text(strip=True) if time_tag else None

                    combined_text = title + " " + (description or "")
                    stocks_with_sentiment = extract_stock_names_with_sentiment(combined_text)

                    news_items.append({
                        "title": title,
                        "link": full_link,
                        "description": description,
                        "publishedAt": published_at,
                        "source": "Economic Times",
                        "stocks": extract_stock_names(combined_text),
                        "stocks_with_sentiment": stocks_with_sentiment,
                    })

        return news_items

    except Exception as e:
        logger.error(f"Error fetching Economic Times news: {e}")
        return []


def extract_et_article_content(soup, url):
    """Extract full article content from an Economic Times page (BeautifulSoup object)."""
    from flask import jsonify  # local import to avoid circular dependency at module level

    try:
        content = ""
        for selector in [".artText p", ".Normal p", ".story-content p", "article p", ".content p"]:
            paragraphs = soup.select(selector)
            if paragraphs:
                content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                break

        summary = simple_summarize(content, max_sentences=3)

        author = None
        for selector in [".byline", ".author", ".writer", "[class*='author']"]:
            tag = soup.select_one(selector)
            if tag:
                author = tag.get_text(strip=True).replace("By", "").strip()
                break

        published_at = None
        for selector in [".publish_on", ".date", "time", "[class*='date']"]:
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
            "source": "Economic Times",
            "url": url,
        })

    except Exception as e:
        from flask import jsonify
        return jsonify({"error": f"Failed to extract ET article: {str(e)}"}), 500
