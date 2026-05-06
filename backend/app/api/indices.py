"""
/api/indices/* routes — live data, comparison, and summary.
Also owns the live_index_data store, data_lock, and the background scheduler.
"""
import logging
import threading
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Blueprint, jsonify

from app.config import LIVE_DATA_CACHE_DURATION
from app.scrapers.moneycontrol import fetch_indices_from_moneycontrol
from app.utils.cache import rate_limit

logger = logging.getLogger(__name__)

indices_bp = Blueprint('indices', __name__)

# Shared live-data store
live_index_data = {}
data_lock = threading.Lock()

scheduler = BackgroundScheduler()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_market_status(timezone_str):
    """Return 'open', 'closed', or 'unknown' for the given timezone."""
    try:
        tz = pytz.timezone(timezone_str)
        local_time = datetime.now(tz)
        current_hour = local_time.hour
        current_minute = local_time.minute
        weekday = local_time.weekday()

        if weekday >= 5:
            return "closed"

        if timezone_str == 'Asia/Kolkata':
            if 9 <= current_hour < 15 or (current_hour == 15 and current_minute <= 30):
                return "open"

        return "closed"

    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return "unknown"


def update_all_indices():
    """Fetch fresh index data from MoneyControl and store it."""
    global live_index_data
    try:
        logger.info("Updating indices data from MoneyControl...")
        new_data = fetch_indices_from_moneycontrol()
        with data_lock:
            live_index_data = new_data
        logger.info("Indices updated successfully")
    except Exception as e:
        logger.error(f"Error updating indices: {e}")


def start_scheduler():
    """Perform initial data fetch and schedule periodic updates."""
    try:
        update_all_indices()
        scheduler.add_job(
            func=update_all_indices,
            trigger="interval",
            minutes=60,
            id='update_indices',
            name='Update live indices data',
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Background scheduler started for live data updates")
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@indices_bp.route("/api/indices/live")
@rate_limit(max_requests=120, window=60)
def get_live_indices():
    """Get live data for all major indices."""
    global live_index_data

    try:
        with data_lock:
            if not live_index_data:
                update_all_indices()

            current_time = datetime.now()
            for data in live_index_data.values():
                if data and 'last_updated' in data:
                    last_updated = datetime.fromisoformat(
                        data['last_updated'].replace('Z', '+00:00')
                    )
                    if (current_time - last_updated.replace(tzinfo=None)).seconds > 120:
                        threading.Thread(target=update_all_indices).start()
                        break

        return jsonify({
            "indices": live_index_data,
            "total_count": len(live_index_data),
            "last_updated": datetime.now().isoformat(),
            "cache_duration": f"{LIVE_DATA_CACHE_DURATION} seconds",
        })

    except Exception as e:
        logger.error(f"Error in get_live_indices: {e}")
        return jsonify({"error": str(e)}), 500


@indices_bp.route("/api/indices/comparison")
@rate_limit(max_requests=30, window=60)
def get_indices_comparison():
    """Get side-by-side comparison data for all major indices."""
    try:
        global live_index_data

        with data_lock:
            if not live_index_data:
                update_all_indices()

        comparison_data = [
            {
                'index': key,
                'name': data['name'],
                'current_price': data['current_price'],
                'change': data['change'],
                'change_percent': data['change_percent'],
                'market_status': data['market_status'],
                'currency': data['currency'],
            }
            for key, data in live_index_data.items()
            if data
        ]

        comparison_data.sort(key=lambda x: x['change_percent'], reverse=True)

        return jsonify({
            "comparison": comparison_data,
            "best_performer": comparison_data[0] if comparison_data else None,
            "worst_performer": comparison_data[-1] if comparison_data else None,
            "last_updated": datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Error in indices comparison: {e}")
        return jsonify({"error": str(e)}), 500


@indices_bp.route("/api/indices/summary")
@rate_limit(max_requests=60, window=60)
def get_market_summary():
    """Get overall market summary across all tracked indices."""
    try:
        global live_index_data

        with data_lock:
            if not live_index_data:
                update_all_indices()

        summary = {
            "total_indices": len(live_index_data),
            "indices_up": 0,
            "indices_down": 0,
            "indices_unchanged": 0,
            "markets_open": 0,
            "markets_closed": 0,
            "average_change": 0,
            "indices_data": [],
        }

        total_change = 0.0
        for key, data in live_index_data.items():
            if not data:
                continue
            change_percent = data['change_percent']

            if change_percent > 0:
                summary["indices_up"] += 1
            elif change_percent < 0:
                summary["indices_down"] += 1
            else:
                summary["indices_unchanged"] += 1

            if data['market_status'] == 'open':
                summary["markets_open"] += 1
            else:
                summary["markets_closed"] += 1

            total_change += change_percent
            summary["indices_data"].append({
                'index': key,
                'name': data['name'],
                'change_percent': change_percent,
                'market_status': data['market_status'],
            })

        if live_index_data:
            summary["average_change"] = round(total_change / len(live_index_data), 2)

        summary["market_sentiment"] = (
            "positive" if summary["average_change"] > 0
            else "negative" if summary["average_change"] < 0
            else "neutral"
        )
        summary["last_updated"] = datetime.now().isoformat()

        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error in market summary: {e}")
        return jsonify({"error": str(e)}), 500
