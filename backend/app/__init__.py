"""
Market Insights Dashboard — Flask Application Package.
"""
import logging

from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def create_app():
    """Application factory pattern."""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from app.api.indices import indices_bp
    from app.api.news import news_bp
    from app.api.stocks import stocks_bp
    from app.api.misc import misc_bp

    app.register_blueprint(indices_bp)
    app.register_blueprint(news_bp)
    app.register_blueprint(stocks_bp)
    app.register_blueprint(misc_bp)

    # Start background scheduler for live index data
    from app.api.indices import start_scheduler
    start_scheduler()

    return app
