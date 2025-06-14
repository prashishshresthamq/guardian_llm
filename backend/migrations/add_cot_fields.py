# migrations/add_cot_fields.py
"""
Add Chain of Thought fields to Analysis table
"""
from app import app, db
from sqlalchemy import text

def upgrade_database():
    with app.app_context():
        try:
            # Check if columns already exist
            result = db.session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='analyses' 
                AND column_name IN ('cot_reasoning', 'cot_confidence', 'cot_risk_scores', 'reasoning_quality_score')
            """))
            existing_columns = [row[0] for row in result]
            
            # Add missing columns
            if 'cot_reasoning' not in existing_columns:
                db.session.execute(text('ALTER TABLE analyses ADD COLUMN cot_reasoning TEXT'))
            if 'cot_confidence' not in existing_columns:
                db.session.execute(text('ALTER TABLE analyses ADD COLUMN cot_confidence FLOAT'))
            if 'cot_risk_scores' not in existing_columns:
                db.session.execute(text('ALTER TABLE analyses ADD COLUMN cot_risk_scores TEXT'))
            if 'reasoning_quality_score' not in existing_columns:
                db.session.execute(text('ALTER TABLE analyses ADD COLUMN reasoning_quality_score FLOAT'))
            
            db.session.commit()
            print("✅ CoT fields added successfully!")
        except Exception as e:
            print(f"❌ Error adding CoT fields: {e}")
            db.session.rollback()

if __name__ == '__main__':
    upgrade_database()