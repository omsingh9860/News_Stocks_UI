"""
/api/stocks/* and /api/trending routes.
"""
import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from app.config import NSE_STOCKS_LIST, US_STOCKS
from app.api.news import get_all_news
from app.utils.sentiment import analyze_sentiment
from app.utils.cache import cache_response, rate_limit

logger = logging.getLogger(__name__)

stocks_bp = Blueprint('stocks', __name__)


@stocks_bp.route("/api/stocks")
@rate_limit(max_requests=60, window=60)
@cache_response(timeout=3600)
def list_stocks():
    """List all available stocks: NSE blue-chips + select US indices/equities."""
    exchange_filter = request.args.get('exchange', '').upper()
    all_stocks = NSE_STOCKS_LIST + US_STOCKS

    if exchange_filter:
        all_stocks = [s for s in all_stocks if s['exchange'] == exchange_filter]

    return jsonify({
        "stocks": all_stocks,
        "total": len(all_stocks),
        "exchanges": list({s['exchange'] for s in all_stocks}),
        "note": "NSE stocks + select US indices/equities. More coming in future phases.",
    })


@stocks_bp.route("/api/stocks/trending")
@rate_limit(max_requests=30, window=60)
def get_trending_stocks():
    """Get trending stocks based on news mentions and sentiment."""
    try:
        all_news = get_all_news()

        if not all_news:
            return jsonify({"error": "No news available"}), 404

        stock_data = {}
        for news_item in all_news:
            for stock in news_item.get('stocks_with_sentiment', []):
                symbol = stock['symbol']
                if symbol not in stock_data:
                    stock_data[symbol] = {
                        'name': stock['name'],
                        'symbol': symbol,
                        'mention_count': 0,
                        'sentiments': [],
                        'news_items': [],
                    }
                stock_data[symbol]['mention_count'] += 1
                stock_data[symbol]['sentiments'].append(stock['sentiment'])
                stock_data[symbol]['news_items'].append({
                    'title': news_item.get('title', ''),
                    'source': news_item.get('source', ''),
                    'sentiment': stock['sentiment_label'],
                })

        trending_stocks = []
        for symbol, data in stock_data.items():
            sentiments = data['sentiments']
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            trending_score = data['mention_count'] * (1 + abs(avg_sentiment))

            trending_stocks.append({
                'name': data['name'],
                'symbol': symbol,
                'mention_count': data['mention_count'],
                'average_sentiment': round(avg_sentiment, 3),
                'sentiment_label': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                'trending_score': round(trending_score, 2),
                'recent_news': data['news_items'][:3],
            })

        trending_stocks.sort(key=lambda x: x['trending_score'], reverse=True)

        return jsonify({
            "trending_stocks": trending_stocks[:10],
            "analysis_date": datetime.now().isoformat(),
            "total_stocks_analyzed": len(trending_stocks),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stocks_bp.route("/api/sentiment/analysis")
@rate_limit(max_requests=20, window=60)
def analyze_market_sentiment():
    """Analyze overall market sentiment from recent news."""
    try:
        all_news = get_all_news()

        if not all_news:
            return jsonify({"error": "No news available"}), 404

        sentiment_scores = []
        stock_sentiments = {}

        for news_item in all_news:
            full_text = (news_item.get('title', '') + " " + (news_item.get('description') or ''))
            sentiment_score = analyze_sentiment(full_text)
            sentiment_scores.append(sentiment_score)

            for stock in news_item.get('stocks_with_sentiment', []):
                symbol = stock['symbol']
                stock_sentiments.setdefault(symbol, []).append(stock['sentiment'])

        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        stock_avg_sentiments = {}
        for symbol, sentiments in stock_sentiments.items():
            avg = sum(sentiments) / len(sentiments)
            stock_avg_sentiments[symbol] = {
                'average_sentiment': round(avg, 3),
                'sentiment_count': len(sentiments),
                'label': 'positive' if avg > 0.1 else 'negative' if avg < -0.1 else 'neutral',
            }

        return jsonify({
            "overall_market_sentiment": {
                "score": round(overall_sentiment, 3),
                "label": "positive" if overall_sentiment > 0.1 else "negative" if overall_sentiment < -0.1 else "neutral",
            },
            "stock_sentiments": stock_avg_sentiments,
            "analysis_date": datetime.now().isoformat(),
            "news_analyzed": len(all_news),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@stocks_bp.route("/api/trending")
@rate_limit(max_requests=30, window=60)
def get_trending():
    """Trending stocks/news — alias for /api/stocks/trending with extra metadata."""
    try:
        all_news = get_all_news()
        stock_data = {}

        for item in all_news:
            for stock in (item.get('stocks_with_sentiment') or item.get('stocks') or []):
                sym = stock.get('symbol')
                if not sym:
                    continue
                if sym not in stock_data:
                    stock_data[sym] = {
                        'name': stock.get('name', sym),
                        'symbol': sym,
                        'mention_count': 0,
                        'sentiments': [],
                        'recent_news': [],
                    }
                stock_data[sym]['mention_count'] += 1
                if isinstance(stock.get('sentiment'), (int, float)):
                    stock_data[sym]['sentiments'].append(stock['sentiment'])
                if len(stock_data[sym]['recent_news']) < 3:
                    stock_data[sym]['recent_news'].append({
                        'title': item.get('title', ''),
                        'source': item.get('source', ''),
                    })

        trending = []
        for sym, data in stock_data.items():
            sentiments = data['sentiments']
            avg = (sum(sentiments) / len(sentiments)) if sentiments else 0
            trending.append({
                'symbol': sym,
                'name': data['name'],
                'mention_count': data['mention_count'],
                'average_sentiment': round(avg, 3),
                'sentiment_label': 'positive' if avg > 0.1 else 'negative' if avg < -0.1 else 'neutral',
                'trending_score': round(data['mention_count'] * (1 + abs(avg)), 2),
                'recent_news': data['recent_news'],
            })

        trending.sort(key=lambda x: x['trending_score'], reverse=True)

        return jsonify({
            "trending_stocks": trending[:10],
            "total_analyzed": len(trending),
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Error in get_trending: {e}")
        return jsonify({"error": str(e)}), 500
