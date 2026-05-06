"""
TradingView scraper — ideas and enhanced ideas for Indian stocks.
"""
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.config import TRADINGVIEW_URL

logger = logging.getLogger(__name__)


def _classify_condition(condition_text):
    """Return (signal_label, signal_color) for a TradingView condition string."""
    if condition_text == "Long":
        return "BUY", "green"
    if condition_text == "Short":
        return "SELL", "red"
    return "EDUCATIONAL", "blue"


def get_tradingview_ideas():
    """Scrape TradingView ideas for Indian stocks (up to 50)."""
    try:
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        all_ideas = soup.find_all(
            "a",
            class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl",
        )
        all_conditions = soup.find_all("span", class_="visuallyHiddenLabel-cYxls04V")

        ideas_list = []
        for i in range(min(len(all_ideas), len(all_conditions))):
            try:
                idea_tag = all_ideas[i]
                condition_tag = all_conditions[i]

                idea_href = idea_tag.get("href", "")
                full_link = urljoin("https://in.tradingview.com", idea_href)
                stock_split = idea_href.split("/")
                stock_symbol = stock_split[4] if len(stock_split) > 4 else "Unknown"
                title = idea_tag.get_text(strip=True) or stock_symbol
                condition_text = condition_tag.get_text(strip=True)

                signal_label, signal_color = _classify_condition(condition_text)

                ideas_list.append({
                    "stock_symbol": stock_symbol,
                    "title": title,
                    "link": full_link,
                    "condition": condition_text or "Educational",
                    "signal_label": signal_label,
                    "signal_color": signal_color,
                })
            except Exception as e:
                logger.warning(f"Skipped idea: {e}")
                continue

        return ideas_list[:50]

    except Exception as e:
        logger.error(f"Failed to fetch TradingView ideas: {e}")
        return []


def get_enhanced_tradingview_ideas():
    """Scrape TradingView ideas with enhanced signal labeling (up to 30)."""
    try:
        response = requests.get(TRADINGVIEW_URL, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        all_ideas = soup.find_all(class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl")
        all_conditions = soup.find_all(class_="visually-hidden-label-cbI7LT3N", name="span")

        ideas_list = []
        for idea in all_ideas:
            try:
                idea_href = idea.get("href")
                if idea_href:
                    stock_split = idea_href.split("/")
                    if len(stock_split) >= 5:
                        stock_symbol = stock_split[4]
                        title = idea.get_text(strip=True) or stock_symbol
                        ideas_list.append({
                            'stock_symbol': stock_symbol,
                            'title': title,
                            'link': idea_href,
                            'condition': 'Educational',
                        })
            except Exception as e:
                logger.warning(f"Error processing idea: {e}")
                continue

        conditions = []
        for span in all_conditions:
            try:
                condition_text = span.get_text(strip=True) if span.get_text() else ""
                if condition_text in ("Long", "Short"):
                    conditions.append(condition_text)
                else:
                    conditions.append("Educational")
            except Exception as e:
                logger.warning(f"Error processing condition: {e}")
                conditions.append("Educational")

        enhanced_ideas = []
        for i, idea in enumerate(ideas_list):
            if i < len(conditions):
                idea['condition'] = conditions[i]

            signal_label, signal_color = _classify_condition(idea['condition'])
            idea['signal_label'] = signal_label
            idea['signal_color'] = signal_color
            enhanced_ideas.append(idea)

        return enhanced_ideas[:30]

    except Exception as e:
        logger.error(f"Error fetching enhanced TradingView ideas: {e}")
        return []
