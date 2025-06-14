"""
Guardian LLM - Database Models
SQLAlchemy database models for Guardian LLM
"""

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
        try:
            authors_list = json.loads(self.authors) if self.authors else []
        except json.JSONDecodeError:
            authors_list = self.authors if isinstance(self.authors, list) else []
        
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': authors_list,
            'abstract': self.abstract,
            'overall_risk_score': self.overall_risk_score,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'status': self.status,
            'processing_time': self.processing_time
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


class User(db.Model):
    """User model for authentication and tracking"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary (safe for API)"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'is_admin': self.is_admin
        }


class Analysis(db.Model):
    """Analysis model for storing text analysis results"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    text = db.Column(db.Text, nullable=False)
    text_preview = db.Column(db.String(200))  # First 200 chars
    
    # Risk scores
    critical_risk_score = db.Column(db.Float, index=True)
    overall_risk_score = db.Column(db.Float, index=True)
    risk_level = db.Column(db.String(20), index=True)
    
    # Sentiment analysis
    sentiment_score = db.Column(db.Float)
    sentiment_type = db.Column(db.String(20))
    sentiment_positive = db.Column(db.Float)
    sentiment_negative = db.Column(db.Float)
    sentiment_neutral = db.Column(db.Float)
    
    # Statistics
    word_count = db.Column(db.Integer)
    character_count = db.Column(db.Integer)
    high_risk_keywords = db.Column(db.Integer)
    medium_risk_keywords = db.Column(db.Integer)
    
    # Metadata
    processing_time = db.Column(db.Float)
    status = db.Column(db.String(50), default='completed', index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Add CoT-specific fields
    cot_reasoning = db.Column(db.Text)  # JSON string for CoT reasoning chain
    cot_confidence = db.Column(db.Float)
    cot_risk_scores = db.Column(db.Text)  # JSON string for CoT risk scores
    reasoning_quality_score = db.Column(db.Float)
    
    # JSON fields for complex data
    risk_assessments = db.Column(db.Text)  # JSON string
    recommendations = db.Column(db.Text)  # JSON string
    analysis_metadata = db.Column(db.Text)  # JSON string (renamed from metadata)
    
    # Relationships
    risk_detections = db.relationship('RiskDetection', backref='analysis', lazy='dynamic', cascade='all, delete-orphan')
    
    feedback = db.relationship('Feedback', backref='analysis_record', lazy='dynamic', foreign_keys='Feedback.analysis_id')

    def __init__(self, **kwargs):
        super(Analysis, self).__init__(**kwargs)
        if 'analysis_id' not in kwargs:
            self.analysis_id = self.generate_analysis_id()
        if self.text and not self.text_preview:
            self.text_preview = self.text[:200] + '...' if len(self.text) > 200 else self.text
    
    @staticmethod
    def generate_analysis_id():
        """Generate unique analysis ID"""
        import uuid
        return f"analysis_{uuid.uuid4().hex[:12]}"
    
    def to_dict(self):
        """Convert analysis to dictionary"""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'user_id': self.user_id,
            'text_preview': self.text_preview,
            'critical_risk_score': self.critical_risk_score,
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level,
            'sentiment_score': self.sentiment_score,
            'sentiment_type': self.sentiment_type,
            'word_count': self.word_count,
            'processing_time': self.processing_time,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'risk_assessments': json.loads(self.risk_assessments) if self.risk_assessments else [],
            'recommendations': json.loads(self.recommendations) if self.recommendations else [],
            'cot_reasoning': json.loads(self.cot_reasoning) if self.cot_reasoning else None,
            'cot_confidence': self.cot_confidence,
            'cot_risk_scores': json.loads(self.cot_risk_scores) if self.cot_risk_scores else {},
            'reasoning_quality_score': self.reasoning_quality_score
        }


class RiskDetection(db.Model):
    """Individual risk detections within an analysis"""
    __tablename__ = 'risk_detections'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False, index=True)
    level = db.Column(db.String(20), nullable=False, index=True)
    score = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    keywords = db.Column(db.Text)  # JSON string
    context = db.Column(db.Text)
    position_start = db.Column(db.Integer)
    position_end = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert risk detection to dictionary"""
        return {
            'category': self.category,
            'level': self.level,
            'score': self.score,
            'confidence': self.confidence,
            'keywords': json.loads(self.keywords) if self.keywords else [],
            'context': self.context,
            'position': {
                'start': self.position_start,
                'end': self.position_end
            }
        }


class Feedback(db.Model):
    """User feedback on analysis accuracy"""
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analyses.id'), nullable=True)  # Made nullable
    paper_id = db.Column(db.String(100), db.ForeignKey('papers.paper_id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    feedback_type = db.Column(db.String(50), default='accuracy')
    rating = db.Column(db.Integer)  # 1-5 rating
    comment = db.Column(db.Text)
    is_accurate = db.Column(db.Boolean)  # True if user confirms accuracy
    risk_category = db.Column(db.String(50))  # Which risk category feedback is for
    reported_risk_level = db.Column(db.String(20))  # What we predicted
    actual_risk_level = db.Column(db.String(20))  # What user says it should be
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Remove the duplicate relationship or rename it
    # analysis = db.relationship('Analysis', backref='feedbacks')  # REMOVE THIS LINE
    user = db.relationship('User', backref='feedbacks')
    
    def to_dict(self):
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'paper_id': self.paper_id,
            'user_id': self.user_id,
            'feedback_type': self.feedback_type,
            'rating': self.rating,
            'comment': self.comment,
            'is_accurate': self.is_accurate,
            'risk_category': self.risk_category,
            'reported_risk_level': self.reported_risk_level,
            'actual_risk_level': self.actual_risk_level,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Database utilities
def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create default admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@guardianlm.com',
                is_admin=True
            )
            admin.set_password('admin123')  # Change in production!
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created")
        
        print("Database initialized successfully!")


def reset_db(app):
    """Reset database - WARNING: This will delete all data!"""
    with app.app_context():
        # Drop all tables
        db.drop_all()
        
        # Recreate all tables
        db.create_all()
        
        print("Database reset successfully!")


def get_or_create(session, model, defaults=None, **kwargs):
    """
    Get or create a database record
    
    Args:
        session: Database session
        model: Model class
        defaults: Default values for creation
        **kwargs: Lookup parameters
        
    Returns:
        tuple: (instance, created)
    """
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance, False
    else:
        params = dict((k, v) for k, v in kwargs.items())
        params.update(defaults or {})
        instance = model(**params)
        session.add(instance)
        return instance, True