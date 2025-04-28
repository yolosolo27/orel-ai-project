import os
import logging
import signal
from app import app, db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}")
    if signum in (signal.SIGTERM, signal.SIGINT):
        logger.info("Shutting down gracefully...")
        exit(0)

# Initialize critical systems only once
_initialized = False

def initialize_systems():
    global _initialized
    if _initialized:
        return

    # Initialize database with proper schema
    with app.app_context():
        logger.info("Beginning experimental model table recreation...")
        from models import db, ExperimentalModel, ExperimentalModelTemp

        # Phase 1: Create temporary tables
        db.session.execute(db.text("CREATE TABLE IF NOT EXISTS experimental_model_backup AS SELECT * FROM experimental_model"))
        db.session.commit()

        # Phase 2: Drop existing tables
        db.session.execute(db.text("DROP TABLE IF EXISTS experimental_model"))
        db.session.commit()

        # Phase 3: Create new schema
        db.create_all()
        db.session.commit()

        # Phase 4: Migrate data with active column
        db.session.execute(db.text("""
            INSERT INTO experimental_model (
                name, description, base_model, active, architecture,
                training_progress, parameters, evaluation_metrics,
                created_at, last_trained
            )
            SELECT 
                name, description, base_model, true as active, architecture,
                training_progress, parameters, evaluation_metrics,
                created_at, last_trained
            FROM experimental_model_backup
        """))
        db.session.commit()

        # Phase 5: Cleanup
        db.session.execute(db.text("DROP TABLE IF EXISTS experimental_model_backup"))
        db.session.commit()

        logger.info("Experimental model table recreation complete.")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    _initialized = True

initialize_systems()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True,
        threaded=True
    )