import sqlite3
import json
from datetime import datetime
from pathlib import Path

# The database will be created inside the platform's persistent storage
DATABASE_NAME = Path("data/corrections.db")

def init_db():
    """Initializes the database and creates the 'corrections' table if it doesn't exist."""
    # Ensure the 'data' directory exists
    DATABASE_NAME.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                ai_prediction TEXT,
                user_correction TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        conn.commit()
    print("Database initialized successfully.")

def log_correction(image_id: str, ai_prediction: dict, user_correction: dict):
    """Logs a user's correction to the database."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO corrections (image_id, ai_prediction, user_correction, timestamp) VALUES (?, ?, ?, ?)",
            (
                image_id,
                json.dumps(ai_prediction),
                json.dumps(user_correction),
                datetime.now()
            )
        )
        conn.commit()
    print(f"Correction logged for image_id: {image_id}")