from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Paper(db.Model):
    """Paper model for storing analyzed research papers"""
    __tablename__ = 'papers'
    
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    title = db.Column(db.String(500), nullable=False)
    authors = db.Column(db.Text)  # JSON string
    abstract = db.Column(db.Text)
    content_preview = db.Column(db.Text)  # First 1000 chars for privacy
    upload_time = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    overall_risk_score = db.Column(db.Float, index=True)
    processing_time = db.Column(db.Float)
    status = db.Column(db.String(50), default='processing', index=True)
    
    # Relationships
    risk_results = db.relationship('RiskResult', backref='paper', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert paper to dictionary"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': json.loads(self.authors) if self.authors else [],
            'abstract': self.abstract,
            'upload_time': self.upload_time.isoformat(),
            'overall_risk_score': self.overall_risk_score,
            'processing_time': self.processing_time,
            'status': self.status
        }

class RiskResult(db.Model):
    """Risk assessment results for each paper"""
    __tablename__ = 'risk_results'
    
    id = db.Column(db.Integer, primary_key=True)
    paper_id = db.Column(db.String(100), db.ForeignKey('papers.paper_id'), nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False, index=True)
    score = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    level = db.Column(db.String(20), nullable=False, index=True)
    explanation = db.Column(db.Text)
    evidence = db.Column(db.Text)  # JSON string
    recommendations = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert risk result to dictionary"""
        return {
            'category': self.category,
            'score': self.score,
            'confidence': self.confidence,
            'level': self.level,
            'explanation': self.explanation,
            'evidence': json.loads(self.evidence) if self.evidence else [],
            'recommendations': json.loads(self.recommendations) if self.recommendations else []
        }

class SystemStats(db.Model):
    """System statistics and metrics"""
    __tablename__ = 'system_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(100), nullable=False, index=True)
    metric_value = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
