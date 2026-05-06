"""
Miscellaneous routes: /api/health, /api/feedback, auth stubs, and TradingView endpoints.
"""
import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from app.scrapers.tradingview import get_tradingview_ideas, get_enhanced_tradingview_ideas
from app.utils.cache import cache_response, rate_limit

logger = logging.getLogger(__name__)

misc_bp = Blueprint('misc', __name__)


@misc_bp.route("/api/health")
def health_check():
    """Enhanced health check endpoint."""
    from app.api.indices import live_index_data  # local import to avoid circular at module level

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_available": True,
        "live_indices_loaded": len(live_index_data),
        "sources": ["MoneyControl", "Economic Times", "TradingView", "Yahoo Finance"],
        "features": [
            "summarization",
            "sentiment_analysis",
            "stock_extraction",
            "tradingview_ideas",
            "live_index_data",
            "historical_data",
            "market_comparison",
        ],
    })


@misc_bp.route("/api/tradingview/ideas")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=600)
def get_tradingview_ideas_api():
    """Get TradingView ideas for Indian stocks."""
    try:
        ideas = get_tradingview_ideas()
        return jsonify({
            "ideas": ideas,
            "total_count": len(ideas),
            "source": "TradingView",
            "last_updated": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@misc_bp.route("/api/tradingview/ideas/by-condition")
@rate_limit(max_requests=30, window=60)
def get_ideas_by_condition():
    """Get TradingView ideas filtered by condition (?condition=Long|Short|Educational)."""
    condition = request.args.get('condition', '').title()

    if condition not in ('Long', 'Short', 'Educational'):
        return jsonify({"error": "Invalid condition. Use 'Long', 'Short', or 'Educational'"}), 400

    try:
        all_ideas = get_tradingview_ideas()
        filtered_ideas = [idea for idea in all_ideas if idea['condition'] == condition]

        return jsonify({
            "ideas": filtered_ideas,
            "condition": condition,
            "count": len(filtered_ideas),
            "source": "TradingView",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@misc_bp.route("/api/tradingview/ideas/enhanced")
@rate_limit(max_requests=30, window=60)
@cache_response(timeout=600)
def get_enhanced_tradingview_ideas_api():
    """Get enhanced TradingView ideas with signal labeling and grouping."""
    try:
        ideas = get_enhanced_tradingview_ideas()

        buy_signals = [i for i in ideas if i['signal_label'] == 'BUY']
        sell_signals = [i for i in ideas if i['signal_label'] == 'SELL']
        educational = [i for i in ideas if i['signal_label'] == 'EDUCATIONAL']

        return jsonify({
            "ideas": ideas,
            "grouped_ideas": {
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "educational": educational,
            },
            "summary": {
                "total_ideas": len(ideas),
                "buy_signals_count": len(buy_signals),
                "sell_signals_count": len(sell_signals),
                "educational_count": len(educational),
            },
            "source": "TradingView",
            "last_updated": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@misc_bp.route("/api/tradingview/test")
@rate_limit(max_requests=10, window=60)
def test_tradingview_scraping():
    """Test TradingView scraping functionality."""
    try:
        ideas = get_enhanced_tradingview_ideas()

        return jsonify({
            "status": "success",
            "ideas_found": len(ideas),
            "sample_ideas": ideas[:5],
            "signal_distribution": {
                "buy": len([i for i in ideas if i['signal_label'] == 'BUY']),
                "sell": len([i for i in ideas if i['signal_label'] == 'SELL']),
                "educational": len([i for i in ideas if i['signal_label'] == 'EDUCATIONAL']),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@misc_bp.route("/api/feedback", methods=["POST"])
@rate_limit(max_requests=10, window=60)
def submit_feedback():
    """Accept user feedback. Phase 1: logs to server console."""
    try:
        data = request.get_json(silent=True) or {}
        logger.info(
            f"[Feedback] category={data.get('category')} rating={data.get('rating')} "
            f"subject={data.get('subject')!r} message={data.get('message', '')[:200]!r}"
        )
        return jsonify({
            "status": "received",
            "message": "Thank you for your feedback! It has been logged.",
        })
    except Exception as e:
        logger.error(f"Error accepting feedback: {e}")
        return jsonify({"error": str(e)}), 500


@misc_bp.route("/api/auth/register", methods=["POST"])
def auth_register_stub():
    """[STUB] User registration — not implemented in Phase 1."""
    return jsonify({
        "status": "not_implemented",
        "message": "User authentication is planned for Phase 2. Watchlists are currently stored locally in the browser.",
        "phase": 2,
    }), 501


@misc_bp.route("/api/auth/login", methods=["POST"])
def auth_login_stub():
    """[STUB] User login — not implemented in Phase 1."""
    return jsonify({
        "status": "not_implemented",
        "message": "User authentication is planned for Phase 2.",
        "phase": 2,
    }), 501
