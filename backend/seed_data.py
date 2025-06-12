from app import app, db
from models.database import Paper, RiskResult
from datetime import datetime
import json

def seed_database():
    with app.app_context():
        # Clear existing data (optional)
        # Paper.query.delete()
        # RiskResult.query.delete()
        # db.session.commit()
        
        # Add test papers
        test_papers = [
            {
                'paper_id': 'paper_test_001',
                'title': 'AI Ethics in Healthcare: A Comprehensive Study',
                'authors': json.dumps(['Dr. Smith', 'Dr. Johnson']),
                'abstract': 'This paper examines ethical considerations in healthcare AI...',
                'content_preview': 'Sample content preview...',
                'overall_risk_score': 7.5,
                'processing_time': 12.5,
                'status': 'completed'
            },
            {
                'paper_id': 'paper_test_002',
                'title': 'Privacy Concerns in Machine Learning Applications',
                'authors': json.dumps(['Dr. Brown', 'Dr. Davis']),
                'abstract': 'An analysis of privacy risks in ML systems...',
                'content_preview': 'Sample content preview...',
                'overall_risk_score': 8.2,
                'processing_time': 15.3,
                'status': 'completed'
            },
            {
                'paper_id': 'paper_test_003',
                'title': 'Bias Detection in Natural Language Processing',
                'authors': json.dumps(['Dr. Wilson', 'Dr. Garcia']),
                'abstract': 'Exploring bias in NLP models...',
                'content_preview': 'Sample content preview...',
                'overall_risk_score': 3.5,
                'processing_time': 10.2,
                'status': 'completed'
            }
        ]
        
        for paper_data in test_papers:
            paper = Paper(**paper_data)
            db.session.add(paper)
            db.session.commit()
            
            # Add risk results for each paper
            risk_categories = [
                'bias_fairness', 'privacy_data', 'safety_security',
                'dual_use', 'societal_impact', 'transparency'
            ]
            
            for i, category in enumerate(risk_categories):
                risk_result = RiskResult(
                    paper_id=paper.paper_id,
                    category=category,
                    score=(i + 1) * 0.1,  # Sample scores
                    confidence=0.85,
                    level='medium' if i < 3 else 'low',
                    explanation=f'Analysis for {category}',
                    evidence=json.dumps([f'Evidence item {j+1}' for j in range(3)]),
                    recommendations=json.dumps([f'Recommendation {j+1}' for j in range(2)])
                )
                db.session.add(risk_result)
            
        db.session.commit()
        print("Test data added successfully!")

if __name__ == '__main__':
    seed_database()