"""Entry point for the Market Insights Dashboard backend."""
import logging

from app import create_app

logger = logging.getLogger(__name__)

app = create_app()

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        from app.api.indices import scheduler
        if scheduler.running:
            scheduler.shutdown()
        logger.info("Scheduler stopped")
