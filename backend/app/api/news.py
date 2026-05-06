"""
/api/news/* and /api/news-with-ideas routes.
"""
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from flask import Blueprint, jsonify, request

from app.scrapers.moneycontrol import get_moneycontrol_news, extract_mc_article_content
from app.scrapers.economictimes import get_economictimes_news, extract_et_article_content
from app.scrapers.tradingview import get_enhanced_tradingview_ideas
from app.utils.sentiment import analyze_sentiment, simple_summarize
from app.utils.cache import cache_response, rate_limit
from app.config import CACHE_DURATION

logger = logging.getLogger(__name__)

news_bp = Blueprint('news', __name__)


def get_all_news():
    """Return combined news list from MoneyControl and Economic Times (up to 20 items)."""
    try:
        mc_news = get_moneycontrol_news()
        et_news = get_economictimes_news()
        return (mc_news + et_news)[:20]
    except Exception as e:
        logger.error(f"Error fetching all news: {e}")
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@news_bp.route("/api/news")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_news():
    """Get news from both MoneyControl and Economic Times."""
    return get_all_news()


@news_bp.route("/api/news/moneycontrol")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_moneycontrol_only():
    """Get news from MoneyControl only."""
    return get_moneycontrol_news()


@news_bp.route("/api/news/economic-times")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=CACHE_DURATION)
def get_et_news():
    """Get news from Economic Times only."""
    return get_economictimes_news()


@news_bp.route("/api/article")
@rate_limit(max_requests=30, window=60)
def get_article_content():
    """Get full article content with summary and sentiment analysis."""
    url = request.args.get('url')
    if not url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch article"}), 500

        soup = BeautifulSoup(response.text, 'html.parser')

        if "economictimes.indiatimes.com" in url:
            return extract_et_article_content(soup, url)
        elif "moneycontrol.com" in url:
            return extract_mc_article_content(soup, url)
        else:
            return jsonify({"error": "Unsupported source"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@news_bp.route("/api/news/enhanced-summary")
@rate_limit(max_requests=30, window=60)
def get_enhanced_news_summary():
    """Get news with enhanced summaries and sentiment analysis (up to 10 articles)."""
    try:
        all_news = get_all_news()
        enhanced_news = []

        for news_item in all_news[:10]:
            enhanced_item = news_item.copy()
            desc = news_item.get('description', '')

            if desc and len(desc) > 50:
                enhanced_item['summary'] = simple_summarize(desc, max_sentences=2)
                enhanced_item['has_summary'] = True
            elif news_item.get('title'):
                enhanced_item['summary'] = news_item['title']
                enhanced_item['has_summary'] = False
            else:
                enhanced_item['summary'] = "Summary not available"
                enhanced_item['has_summary'] = False

            full_text = (news_item.get('title', '') + " " + (desc or ''))
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
            }
            enhanced_item['debug_info'] = {
                'title_length': len(news_item.get('title', '')),
                'description_length': len(desc),
                'summary_length': len(enhanced_item['summary']),
            }
            enhanced_news.append(enhanced_item)

        return jsonify({
            "news": enhanced_news,
            "total_count": len(enhanced_news),
            "sources": ["MoneyControl", "Economic Times"],
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@news_bp.route("/api/news/summary")
@rate_limit(max_requests=30, window=60)
def get_news_with_summary():
    """Get news with summaries and sentiment analysis (up to 15 articles)."""
    try:
        all_news = get_all_news()
        enhanced_news = []

        for news_item in all_news[:15]:
            enhanced_item = news_item.copy()
            desc = news_item.get('description') or ''

            enhanced_item['summary'] = (
                simple_summarize(desc, max_sentences=2) if desc
                else "Summary not available"
            )

            full_text = (news_item.get('title', '') + " " + desc)
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
            }
            enhanced_news.append(enhanced_item)

        return jsonify({
            "news": enhanced_news,
            "total_count": len(enhanced_news),
            "sources": ["MoneyControl", "Economic Times"],
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@news_bp.route("/api/news/by-stock")
@rate_limit(max_requests=30, window=60)
def get_news_by_stock():
    """Get news filtered by a specific stock symbol (?stock=SYMBOL)."""
    stock_symbol = request.args.get('stock', '').upper()

    if not stock_symbol:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    try:
        all_news = get_all_news()
        filtered_news = []

        for news_item in all_news:
            for stock in news_item.get('stocks_with_sentiment', []):
                if stock['symbol'].upper() == stock_symbol:
                    filtered_news.append(news_item)
                    break

        return jsonify({
            "news": filtered_news,
            "stock_symbol": stock_symbol,
            "count": len(filtered_news),
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@news_bp.route("/api/news/for-tickers")
@rate_limit(max_requests=30, window=60)
def get_news_for_tickers():
    """Get news filtered by one or more tickers (?tickers=SYM1,SYM2)."""
    tickers_param = request.args.get('tickers', '')
    symbols = {t.strip().upper() for t in tickers_param.split(',') if t.strip()}

    try:
        all_news = get_all_news()
        enhanced = []

        for item in all_news[:20]:
            enhanced_item = item.copy()
            desc = item.get('description') or ''
            enhanced_item['summary'] = (
                simple_summarize(desc, max_sentences=2) if desc
                else "Summary not available"
            )
            title = item.get('title') or ''
            score = analyze_sentiment(title + " " + desc)
            enhanced_item['sentiment'] = {
                'score': round(score, 3),
                'label': 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral',
            }
            enhanced_item['article_id'] = str(hash(title + item.get('link', '')))
            enhanced.append(enhanced_item)

        if symbols:
            filtered = []
            for item in enhanced:
                item_symbols = {
                    (s.get('symbol') or '').upper()
                    for s in (item.get('stocks_with_sentiment') or item.get('stocks') or [])
                }
                if item_symbols & symbols:
                    filtered.append(item)
        else:
            filtered = enhanced

        return jsonify({
            "news": filtered,
            "tickers": list(symbols),
            "total_count": len(filtered),
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Error in get_news_for_tickers: {e}")
        return jsonify({"error": str(e)}), 500


@news_bp.route("/api/news-with-ideas")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=300)
def get_news_with_tradingview_ideas():
    """Get news with summaries/sentiment combined with TradingView trading ideas."""
    try:
        all_news = get_all_news()
        enhanced_news = []

        for news_item in all_news[:15]:
            enhanced_item = {
                'title': news_item.get('title', ''),
                'description': news_item.get('description', ''),
                'link': news_item.get('link', ''),
                'source': news_item.get('source', ''),
                'publishedAt': news_item.get('publishedAt', ''),
                'author': news_item.get('author', ''),
                'stocks': news_item.get('stocks', []),
                'stocks_with_sentiment': news_item.get('stocks_with_sentiment', []),
            }
            title = news_item.get('title') or ''
            link = news_item.get('link') or ''
            enhanced_item['article_id'] = str(hash(title + link))

            desc = news_item.get('description') or ''
            enhanced_item['summary'] = (
                simple_summarize(desc, max_sentences=2) if desc
                else "Summary not available"
            )

            full_text = title + " " + desc
            sentiment_score = analyze_sentiment(full_text)
            enhanced_item['sentiment'] = {
                'score': round(sentiment_score, 3),
                'label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
            }
            enhanced_news.append(enhanced_item)

        tradingview_ideas = get_enhanced_tradingview_ideas()

        return jsonify({
            "news": enhanced_news,
            "tradingview_ideas": tradingview_ideas,
            "metadata": {
                "news_count": len(enhanced_news),
                "ideas_count": len(tradingview_ideas),
                "last_updated": datetime.now().isoformat(),
                "sources": ["MoneyControl", "Economic Times", "TradingView"],
            },
        })

    except Exception as e:
        logger.error(f"Error in get_news_with_tradingview_ideas: {e}")
        return jsonify({"error": str(e)}), 500
