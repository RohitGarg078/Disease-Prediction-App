import sqlite3
from datetime import datetime

DB_NAME = "prediction.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# Create table (run once)
conn = get_connection()
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        gender TEXT,
        symptoms TEXT,
        disease TEXT,
        timestamp TEXT
    )
""")
conn.commit()
conn.close()

# Save prediction
def save_prediction(age, gender, symptoms, disease):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO predictions (age, gender, symptoms, disease, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            age,
            gender,
            " | ".join(symptoms),
            disease,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()

# Fetch all predictions
def get_all_predictions():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# Delete all history
def delete_all_predictions():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
