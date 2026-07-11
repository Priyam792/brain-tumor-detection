import sqlite3
import json
from pathlib import Path
from datetime import datetime

DATABASE_PATH = Path(__file__).resolve().parent / "brain_tumor.db"

def get_db_connection():
    """Establish database connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create predictions table if it does not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            processing_time REAL NOT NULL,
            original_image_path TEXT NOT NULL,
            gradcam_image_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DATABASE_PATH.resolve())

def insert_prediction(image_name, predicted_class, confidence, probabilities, processing_time, original_image_path, gradcam_image_path):
    """Insert a new prediction record into the database."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Store probabilities as a JSON string
    probabilities_json = json.dumps(probabilities)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            image_name, predicted_class, confidence, probabilities, date, time, processing_time, original_image_path, gradcam_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_name, predicted_class, confidence, probabilities_json, date_str, time_str, processing_time, original_image_path, gradcam_image_path))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def get_prediction_by_id(prediction_id):
    """Retrieve a single prediction by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        record = dict(row)
        # Decode probabilities from JSON
        record['probabilities'] = json.loads(record['probabilities'])
        return record
    return None

def get_all_predictions(search_query=None, limit=10, offset=0):
    """
    Retrieve predictions, with optional search filter on predicted_class or image_name,
    supporting pagination.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if search_query:
        # Search matching class name or image name
        like_query = f"%{search_query}%"
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE predicted_class LIKE ? OR image_name LIKE ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (like_query, like_query, limit, offset))
        rows = cursor.fetchall()
        
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE predicted_class LIKE ? OR image_name LIKE ?
        """, (like_query, like_query))
        total_count = cursor.fetchone()[0]
    else:
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_count = cursor.fetchone()[0]
        
    conn.close()
    
    records = []
    for row in rows:
        record = dict(row)
        record['probabilities'] = json.loads(record['probabilities'])
        records.append(record)
        
    return records, total_count

def delete_prediction(prediction_id):
    """Delete a prediction record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # First get image paths to delete actual files if needed (done in app.py)
    cursor.execute("SELECT original_image_path, gradcam_image_path FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    conn.close()
    
    return row

def get_analytics_summary():
    """Retrieve analytics stats for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    if total_predictions == 0:
        conn.close()
        return {
            'total_predictions': 0,
            'class_distribution': {},
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0,
            'recent_predictions': []
        }
    
    # Class distribution
    cursor.execute("SELECT predicted_class, COUNT(*) as count FROM predictions GROUP BY predicted_class")
    class_distribution = {row['predicted_class']: row['count'] for row in cursor.fetchall()}
    
    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_confidence = cursor.fetchone()[0] or 0.0
    
    # Average processing time
    cursor.execute("SELECT AVG(processing_time) FROM predictions")
    avg_processing_time = cursor.fetchone()[0] or 0.0
    
    # Recent predictions (last 5)
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5")
    recent_rows = cursor.fetchall()
    conn.close()
    
    recent_predictions = []
    for row in recent_rows:
        record = dict(row)
        record['probabilities'] = json.loads(record['probabilities'])
        recent_predictions.append(record)
        
    return {
        'total_predictions': total_predictions,
        'class_distribution': class_distribution,
        'avg_confidence': round(avg_confidence * 100, 2),
        'avg_processing_time': round(avg_processing_time, 4),
        'recent_predictions': recent_predictions
    }

if __name__ == "__main__":
    init_db()
