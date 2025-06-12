from app import app, db
from models.database import Paper, Analysis, User

with app.app_context():
    # Check papers
    papers_count = Paper.query.count()
    print(f"Total papers in database: {papers_count}")
    
    # Check analyses
    analyses_count = Analysis.query.count()
    print(f"Total analyses in database: {analyses_count}")
    
    # List all papers
    papers = Paper.query.all()
    for paper in papers:
        print(f"Paper: {paper.title} - Score: {paper.overall_risk_score}")