#!/usr/bin/env python3
# Save this as: backend/scripts/create_accuracy_migration.py

import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from flask import Flask
from models.database import db
from config.setting import Config
import sqlalchemy as sa

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

def add_feedback_columns():
    """Add new columns to feedback table for accuracy tracking"""
    with app.app_context():
        try:
            with db.engine.connect() as conn:
                # Check if columns already exist
                inspector = sa.inspect(db.engine)
                
                # Check if feedback table exists
                if 'feedback' not in inspector.get_table_names():
                    print("Feedback table doesn't exist. Creating all tables...")
                    db.create_all()
                    print("All tables created successfully!")
                    return
                
                columns = [col['name'] for col in inspector.get_columns('feedback')]
                
                # Add new columns if they don't exist
                if 'paper_id' not in columns:
                    conn.execute(sa.text(
                        "ALTER TABLE feedback ADD COLUMN paper_id VARCHAR(100)"
                    ))
                    conn.commit()
                    print("Added paper_id column")
                
                if 'risk_category' not in columns:
                    conn.execute(sa.text(
                        "ALTER TABLE feedback ADD COLUMN risk_category VARCHAR(50)"
                    ))
                    conn.commit()
                    print("Added risk_category column")
                
                if 'reported_risk_level' not in columns:
                    conn.execute(sa.text(
                        "ALTER TABLE feedback ADD COLUMN reported_risk_level VARCHAR(20)"
                    ))
                    conn.commit()
                    print("Added reported_risk_level column")
                
                if 'actual_risk_level' not in columns:
                    conn.execute(sa.text(
                        "ALTER TABLE feedback ADD COLUMN actual_risk_level VARCHAR(20)"
                    ))
                    conn.commit()
                    print("Added actual_risk_level column")
                
                print("Database migration completed successfully!")
                
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            print("Attempting to create all tables...")
            db.create_all()
            print("Tables created/verified successfully!")

if __name__ == '__main__':
    print("Starting database migration...")
    add_feedback_columns()