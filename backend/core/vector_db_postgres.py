# core/vector_db_postgres.py
"""
Guardian LLM - PostgreSQL pgvector Integration
Production-ready vector database using PostgreSQL with pgvector extension
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

Base = declarative_base()


class RiskPatternVector(Base):
    """Risk pattern with vector embedding stored in PostgreSQL"""
    __tablename__ = 'risk_pattern_vectors'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(String, unique=True, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)
    pattern_text = Column(String, nullable=False)
    embedding = Column(Vector(384))  # Dimension for all-MiniLM-L6-v2
    severity = Column(Float, nullable=False)
    pattern_metadata = Column(JSON, name='pattern_metadata')  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'pattern_id': self.pattern_id,
            'category': self.category,
            'pattern_text': self.pattern_text,
            'severity': self.severity,
            'metadata': self.pattern_metadata,  # Keep the same key in the dict for API compatibility
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AnalyzedTextVector(Base):
    """Store analyzed texts with their embeddings for similarity search"""
    __tablename__ = 'analyzed_text_vectors'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(String, index=True)
    text_segment = Column(String, nullable=False)
    embedding = Column(Vector(384))
    risk_scores = Column(JSON)  # Store risk scores for this segment
    analysis_metadata = Column(JSON, name='analysis_metadata')  # Renamed from metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'paper_id': self.paper_id,
            'text_segment': self.text_segment[:200] + '...' if len(self.text_segment) > 200 else self.text_segment,
            'risk_scores': self.risk_scores,
            'metadata': self.analysis_metadata,  # Keep the same key in the dict for API compatibility
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class PgVectorDatabase:
    """PostgreSQL with pgvector for production vector storage"""
    
    def __init__(self, connection_string: str = None, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize PostgreSQL vector database
        
        Args:
            connection_string: PostgreSQL connection string
            model_name: Sentence transformer model name
        """
        # Use environment variable or default connection string
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL',
            'postgresql://guardian:guardian@localhost:5432/guardian_vectors'
        )
        
        # Initialize sentence transformer
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Create engine and session
        self.engine = create_engine(self.connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with pgvector extension"""
        try:
            # Create pgvector extension
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # Create tables
            Base.metadata.create_all(self.engine)
            
            logger.info("PostgreSQL vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def add_risk_patterns(self, patterns: List[Dict[str, Any]]):
        """
        Add risk patterns to the database
        
        Args:
            patterns: List of pattern dictionaries with 'category', 'text', 'severity'
        """
        with self.get_session() as session:
            for pattern in patterns:
                # Generate embedding
                embedding = self.model.encode(pattern['text'], convert_to_numpy=True)
                
                # Check if pattern already exists
                existing = session.query(RiskPatternVector).filter_by(
                    pattern_id=pattern.get('id', f"{pattern['category']}_{hash(pattern['text'])}")
                ).first()
                
                if existing:
                    # Update existing pattern
                    existing.pattern_text = pattern['text']
                    existing.embedding = embedding.tolist()
                    existing.severity = pattern.get('severity', 0.7)
                    existing.pattern_metadata = pattern.get('metadata', {})
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new pattern
                    new_pattern = RiskPatternVector(
                        pattern_id=pattern.get('id', f"{pattern['category']}_{hash(pattern['text'])}"),
                        category=pattern['category'],
                        pattern_text=pattern['text'],
                        embedding=embedding.tolist(),
                        severity=pattern.get('severity', 0.7),
                        pattern_metadata=pattern.get('metadata', {})
                    )
                    session.add(new_pattern)
            
            session.commit()
            logger.info(f"Added/updated {len(patterns)} risk patterns")
    
    def search_similar_patterns(self, query_text: str, category: str = None, 
                              top_k: int = 5, threshold: float = None) -> List[Dict]:
        """
        Search for similar risk patterns using vector similarity
        
        Args:
            query_text: Text to search for
            category: Optional category filter
            top_k: Number of results to return
            threshold: Optional similarity threshold (0-1)
            
        Returns:
            List of similar patterns with scores
        """
        # Generate query embedding
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        with self.get_session() as session:
            # Build base query using pgvector's <-> operator for L2 distance
            query = session.query(
                RiskPatternVector,
                RiskPatternVector.embedding.l2_distance(query_embedding).label('distance')
            )
            
            # Apply category filter if specified
            if category:
                query = query.filter(RiskPatternVector.category == category)
            
            # Order by distance and limit results
            query = query.order_by('distance').limit(top_k)
            
            results = []
            for pattern, distance in query:
                # Convert L2 distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + distance)
                
                # Apply threshold if specified
                if threshold and similarity < threshold:
                    continue
                
                results.append({
                    'pattern_id': pattern.pattern_id,
                    'category': pattern.category,
                    'text': pattern.pattern_text,
                    'severity': pattern.severity,
                    'similarity': similarity,
                    'distance': distance,
                    'metadata': pattern.pattern_metadata
                })
            
            return results
    
    def analyze_text_semantic_risk(self, text: str, store_results: bool = True) -> Dict[str, float]:
        """
        Analyze semantic risk of text using vector similarity
        
        Args:
            text: Text to analyze
            store_results: Whether to store the analysis results
            
        Returns:
            Dictionary of risk scores by category
        """
        # Split text into segments for better analysis
        segments = self._split_text_into_segments(text)
        
        category_scores = {}
        all_categories = ['bias_fairness', 'privacy_data', 'safety_security', 
                         'dual_use', 'societal_impact', 'transparency']
        
        for category in all_categories:
            segment_scores = []
            
            for segment in segments:
                # Search for similar patterns in this category
                similar_patterns = self.search_similar_patterns(
                    segment, 
                    category=category, 
                    top_k=3,
                    threshold=0.3
                )
                
                if similar_patterns:
                    # Calculate weighted score based on similarity and severity
                    weighted_scores = [
                        p['similarity'] * p['severity'] 
                        for p in similar_patterns
                    ]
                    segment_score = np.mean(weighted_scores)
                    segment_scores.append(segment_score)
            
            # Average scores across segments
            category_scores[category] = float(np.mean(segment_scores)) if segment_scores else 0.0
        
        # Store analyzed text if requested
        if store_results:
            self._store_analyzed_text(text, segments, category_scores)
        
        return category_scores
    
    def find_evidence_sentences(self, text: str, category: str, num_sentences: int = 5) -> List[Dict]:
        """
        Find evidence sentences for a specific risk category
        
        Args:
            text: Full text to analyze
            category: Risk category
            num_sentences: Number of evidence sentences to return
            
        Returns:
            List of evidence dictionaries
        """
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        evidence_list = []
        
        with self.get_session() as session:
            for i, sentence in enumerate(sentences):
                # Search for similar risk patterns
                similar_patterns = self.search_similar_patterns(
                    sentence,
                    category=category,
                    top_k=1,
                    threshold=0.4
                )
                
                if similar_patterns:
                    best_match = similar_patterns[0]
                    evidence_list.append({
                        'sentence': sentence,
                        'score': best_match['similarity'],
                        'matched_pattern': best_match['text'],
                        'position': i / len(sentences),
                        'index': i
                    })
        
        # Sort by score and return top evidence
        evidence_list.sort(key=lambda x: x['score'], reverse=True)
        
        # Format results with context
        results = []
        for item in evidence_list[:num_sentences]:
            results.append({
                'text': item['sentence'],
                'relevance': item['score'],
                'matched_pattern': item['matched_pattern'],
                'context': self._get_sentence_context(sentences, item['index'])
            })
        
        return results
    
    def find_similar_analyzed_texts(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Find previously analyzed texts similar to the query
        
        Args:
            query_text: Text to search for
            top_k: Number of results
            
        Returns:
            List of similar analyzed texts
        """
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        with self.get_session() as session:
            results = session.query(
                AnalyzedTextVector,
                AnalyzedTextVector.embedding.l2_distance(query_embedding).label('distance')
            ).order_by('distance').limit(top_k).all()
            
            similar_texts = []
            for text_vector, distance in results:
                similarity = 1.0 / (1.0 + distance)
                similar_texts.append({
                    'paper_id': text_vector.paper_id,
                    'text_preview': text_vector.text_segment[:200] + '...',
                    'similarity': similarity,
                    'risk_scores': text_vector.risk_scores,
                    'created_at': text_vector.created_at.isoformat()
                })
            
            return similar_texts
    
    def update_pattern_severity(self, pattern_id: str, new_severity: float):
        """Update severity score for a pattern based on feedback"""
        with self.get_session() as session:
            pattern = session.query(RiskPatternVector).filter_by(
                pattern_id=pattern_id
            ).first()
            
            if pattern:
                pattern.severity = new_severity
                pattern.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Updated severity for pattern {pattern_id} to {new_severity}")
    
    def add_custom_patterns_from_feedback(self, category: str, text: str, severity: float):
        """Add new risk pattern based on user feedback"""
        pattern = {
            'category': category,
            'text': text,
            'severity': severity,
            'metadata': {
                'source': 'user_feedback',
                'added_at': datetime.utcnow().isoformat()
            }
        }
        self.add_risk_patterns([pattern])
    
    def get_category_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each risk category"""
        with self.get_session() as session:
            stats = {}
            
            categories = session.query(RiskPatternVector.category).distinct().all()
            
            for (category,) in categories:
                count = session.query(RiskPatternVector).filter_by(
                    category=category
                ).count()
                
                avg_severity = session.query(
                    func.avg(RiskPatternVector.severity)
                ).filter_by(category=category).scalar()
                
                stats[category] = {
                    'pattern_count': count,
                    'average_severity': float(avg_severity) if avg_severity else 0.0
                }
            
            return stats
    
    def _split_text_into_segments(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into segments for analysis"""
        words = text.split()
        segments = []
        current_segment = []
        current_length = 0
        
        for word in words:
            current_segment.append(word)
            current_length += len(word) + 1
            
            if current_length >= max_length:
                segments.append(' '.join(current_segment))
                current_segment = []
                current_length = 0
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    def _get_sentence_context(self, sentences: List[str], index: int, window: int = 1) -> str:
        """Get context around a sentence"""
        start = max(0, index - window)
        end = min(len(sentences), index + window + 1)
        return ' '.join(sentences[start:end])
    
    def _store_analyzed_text(self, full_text: str, segments: List[str], risk_scores: Dict[str, float]):
        """Store analyzed text segments for future similarity search"""
        with self.get_session() as session:
            # Store each segment
            for segment in segments[:5]:  # Limit to first 5 segments
                embedding = self.model.encode(segment, convert_to_numpy=True)
                
                analyzed_text = AnalyzedTextVector(
                    text_segment=segment,
                    embedding=embedding.tolist(),
                    risk_scores=risk_scores,
                    analysis_metadata={
                        'full_text_length': len(full_text),
                        'segment_count': len(segments)
                    }
                )
                session.add(analyzed_text)
            
            session.commit()
    
    def initialize_default_patterns(self):
        """Initialize database with default risk patterns"""
        default_patterns = {
            'bias_fairness': [
                {
                    'text': "The algorithm shows systematic bias against certain demographic groups",
                    'severity': 0.9
                },
                {
                    'text': "Model performance varies significantly across different racial populations",
                    'severity': 0.9
                },
                {
                    'text': "Training data underrepresents minority communities",
                    'severity': 0.85
                }
            ],
            'privacy_data': [
                {
                    'text': "Personal data collected without explicit user consent",
                    'severity': 0.85
                },
                {
                    'text': "System enables re-identification of anonymized individuals",
                    'severity': 0.95
                },
                {
                    'text': "Biometric data stored indefinitely without justification",
                    'severity': 0.9
                }
            ],
            'safety_security': [
                {
                    'text': "Critical safety systems vulnerable to adversarial attacks",
                    'severity': 0.95
                },
                {
                    'text': "Lack of fail-safe mechanisms in life-critical applications",
                    'severity': 0.9
                },
                {
                    'text': "System manipulation could cause physical harm",
                    'severity': 0.95
                }
            ],
            'dual_use': [
                {
                    'text': "Technology easily adaptable for military weapons systems",
                    'severity': 0.95
                },
                {
                    'text': "Surveillance capabilities suitable for authoritarian control",
                    'severity': 0.9
                },
                {
                    'text': "Research has clear offensive military applications",
                    'severity': 0.9
                }
            ],
            'societal_impact': [
                {
                    'text': "Automation will displace millions of workers",
                    'severity': 0.85
                },
                {
                    'text': "Technology exacerbates existing social inequalities",
                    'severity': 0.8
                },
                {
                    'text': "Economic benefits concentrated among wealthy elites",
                    'severity': 0.85
                }
            ],
            'transparency': [
                {
                    'text': "Black box algorithm makes unexplainable decisions",
                    'severity': 0.9
                },
                {
                    'text': "No mechanism for users to understand decisions",
                    'severity': 0.85
                },
                {
                    'text': "Proprietary system prevents independent auditing",
                    'severity': 0.8
                }
            ]
        }
        
        # Add all patterns
        all_patterns = []
        for category, patterns in default_patterns.items():
            for i, pattern in enumerate(patterns):
                all_patterns.append({
                    'id': f"{category}_default_{i}",
                    'category': category,
                    'text': pattern['text'],
                    'severity': pattern['severity'],
                    'metadata': {'default': True}
                })
        
        self.add_risk_patterns(all_patterns)
        logger.info(f"Initialized database with {len(all_patterns)} default patterns")


# Factory function for Guardian Engine integration
def create_pgvector_analyzer(connection_string: str = None, 
                           model_name: str = 'all-MiniLM-L6-v2') -> PgVectorDatabase:
    """
    Create PostgreSQL vector database analyzer
    
    Args:
        connection_string: PostgreSQL connection string
        model_name: Sentence transformer model
        
    Returns:
        Initialized PgVectorDatabase
    """
    analyzer = PgVectorDatabase(connection_string, model_name)
    
    # Initialize with default patterns if empty
    with analyzer.get_session() as session:
        pattern_count = session.query(RiskPatternVector).count()
        if pattern_count == 0:
            analyzer.initialize_default_patterns()
    
    return analyzer