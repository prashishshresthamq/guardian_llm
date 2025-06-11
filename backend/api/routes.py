# backend/api/routes.py
"""
Guardian LLM API Routes
======================

This module contains all API endpoints for the Guardian LLM system.
Following the separated architecture pattern, this handles only HTTP/API
concerns while delegating AI analysis to the guardian_engine.

Author: Prashish Shrestha
Course: COMP8420 - Advanced Topics in AI
Project: Guardian LLM - Automated Ethical Risk Auditor
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, current_app

# Import from separated architecture modules
from backend.core.guardian_engine import GuardianEngine
from backend.core.text_processors import TextProcessor
from backend.models.database import db, Paper, RiskResult, SystemStats
from backend.models.schemas import AnalysisRequest, validate_analysis_request, RiskCategory, RiskLevel
from backend.api.utils import handle_api_error, validate_file_upload, calculate_rate_limit

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize components (will be set by app factory)
guardian_engine: Optional[GuardianEngine] = None
text_processor: Optional[TextProcessor] = None

def init_api_components(engine: GuardianEngine, processor: TextProcessor):
    """Initialize API components (called from app factory)"""
    global guardian_engine, text_processor
    guardian_engine = engine
    text_processor = processor
    logger.info("✅ API components initialized")

# =============================================================================
# PAPER ANALYSIS ENDPOINTS
# =============================================================================

@api_bp.route('/analyze', methods=['POST'])
def analyze_paper():
    """
    Analyze a research paper for ethical risks
    
    Accepts either:
    1. JSON data with title, abstract, content
    2. File upload (PDF, DOCX, TXT) with optional metadata
    
    Returns:
        JSON response with comprehensive risk assessment
    """
    try:
        # Check if guardian engine is available
        if not guardian_engine:
            return handle_api_error("Analysis engine not available", 503)
        
        # Handle file upload
        if 'file' in request.files:
            return _handle_file_analysis()
        
        # Handle JSON data
        elif request.is_json:
            return _handle_json_analysis()
        
        else:
            return handle_api_error("No data provided. Send JSON data or file upload.", 400)
    
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}", exc_info=True)
        return handle_api_error(f"Analysis failed: {str(e)}", 500)

def _handle_file_analysis():
    """Handle file upload analysis"""
    file = request.files['file']
    
    # Validate file
    validation_result = validate_file_upload(file)
    if not validation_result['valid']:
        return handle_api_error(validation_result['error'], 400)
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], safe_filename)
    
    try:
        file.save(file_path)
        
        # Extract text based on file type
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            content = text_processor.extract_from_pdf(file_path)
        elif file_ext == 'docx':
            content = text_processor.extract_from_docx(file_path)
        elif file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            return handle_api_error("Unsupported file type", 400)
        
        if not content.strip():
            return handle_api_error("Could not extract text from file", 400)
        
        # Get metadata from form
        title = request.form.get('title', filename.replace(f'.{file_ext}', ''))
        abstract = request.form.get('abstract', '')
        authors = request.form.get('authors', '').split(',') if request.form.get('authors') else []
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            title=title,
            abstract=abstract,
            content=content,
            authors=[author.strip() for author in authors if author.strip()]
        )
        
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")
    
    # Perform analysis
    return _perform_analysis(analysis_request)

def _handle_json_analysis():
    """Handle JSON data analysis"""
    data = request.get_json()
    
    # Validate request data
    is_valid, error_msg = validate_analysis_request(data)
    if not is_valid:
        return handle_api_error(error_msg, 400)
    
    # Create analysis request
    analysis_request = AnalysisRequest(
        title=data.get('title', 'Untitled Paper'),
        abstract=data.get('abstract', ''),
        content=data.get('content', ''),
        authors=data.get('authors', [])
    )
    
    # Perform analysis
    return _perform_analysis(analysis_request)

def _perform_analysis(analysis_request: AnalysisRequest):
    """Perform the actual analysis and save results"""
    try:
        # Run analysis
        start_time = datetime.now()
        response = guardian_engine.analyze_paper(analysis_request)
        
        # Save to database
        paper_record = _save_analysis_to_database(analysis_request, response)
        
        # Update system stats
        _update_system_stats(response)
        
        # Format response
        result = {
            'paper_id': response.paper_id,
            'title': response.title,
            'overall_risk_score': response.overall_risk_score,
            'processing_time': response.processing_time,
            'status': response.status,
            'timestamp': response.timestamp,
            'risk_assessments': response.risk_assessments,
            'summary': _generate_analysis_summary(response)
        }
        
        logger.info(f"✅ Analysis completed for '{analysis_request.title[:50]}...' - Score: {response.overall_risk_score:.1f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Analysis processing error: {e}", exc_info=True)
        return handle_api_error(f"Analysis processing failed: {str(e)}", 500)

# =============================================================================
# PAPER MANAGEMENT ENDPOINTS
# =============================================================================

@api_bp.route('/papers', methods=['GET'])
def get_papers():
    """
    Get list of analyzed papers with optional filtering
    
    Query parameters:
    - limit: Number of papers to return (default: 50, max: 200)
    - offset: Number of papers to skip (default: 0)
    - risk_level: Filter by risk level (low, medium, high, critical)
    - category: Filter by risk category
    - sort: Sort field (upload_time, risk_score, title)
    - order: Sort order (asc, desc, default: desc)
    """
    try:
        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 200)
        offset = max(int(request.args.get('offset', 0)), 0)
        risk_level = request.args.get('risk_level')
        category = request.args.get('category')
        sort_field = request.args.get('sort', 'upload_time')
        sort_order = request.args.get('order', 'desc')
        
        # Build query
        query = Paper.query
        
        # Apply filters
        if risk_level:
            if risk_level == 'low':
                query = query.filter(Paper.overall_risk_score < 2.5)
            elif risk_level == 'medium':
                query = query.filter(Paper.overall_risk_score >= 2.5, Paper.overall_risk_score < 5.0)
            elif risk_level == 'high':
                query = query.filter(Paper.overall_risk_score >= 5.0, Paper.overall_risk_score < 7.5)
            elif risk_level == 'critical':
                query = query.filter(Paper.overall_risk_score >= 7.5)
        
        # Apply sorting
        if sort_field == 'upload_time':
            sort_column = Paper.upload_time
        elif sort_field == 'risk_score':
            sort_column = Paper.overall_risk_score
        elif sort_field == 'title':
            sort_column = Paper.title
        else:
            sort_column = Paper.upload_time
        
        if sort_order == 'asc':
            query = query.order_by(sort_column.asc())
        else:
            query = query.order_by(sort_column.desc())
        
        # Get total count for pagination
        total_count = query.count()
        
        # Apply pagination
        papers = query.offset(offset).limit(limit).all()
        
        # Format results
        result = {
            'papers': [paper.to_dict() for paper in papers],
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            },
            'filters': {
                'risk_level': risk_level,
                'category': category,
                'sort': sort_field,
                'order': sort_order
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Get papers error: {e}", exc_info=True)
        return handle_api_error("Failed to retrieve papers", 500)

@api_bp.route('/paper/<paper_id>', methods=['GET'])
def get_paper_details(paper_id: str):
    """
    Get detailed analysis for a specific paper
    
    Args:
        paper_id: Unique paper identifier
        
    Returns:
        Complete paper analysis with all risk assessments
    """
    try:
        # Find paper
        paper = Paper.query.filter_by(paper_id=paper_id).first()
        if not paper:
            return handle_api_error("Paper not found", 404)
        
        # Get risk results
        risk_results = RiskResult.query.filter_by(paper_id=paper_id).all()
        
        # Format response
        result = {
            **paper.to_dict(),
            'risk_assessments': [result.to_dict() for result in risk_results],
            'analysis_summary': _generate_detailed_summary(paper, risk_results)
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Get paper details error: {e}", exc_info=True)
        return handle_api_error("Failed to retrieve paper details", 500)

@api_bp.route('/paper/<paper_id>', methods=['DELETE'])
def delete_paper(paper_id: str):
    """
    Delete a paper and all associated data
    
    Args:
        paper_id: Unique paper identifier
    """
    try:
        # Find paper
        paper = Paper.query.filter_by(paper_id=paper_id).first()
        if not paper:
            return handle_api_error("Paper not found", 404)
        
        # Delete paper (cascades to risk results)
        db.session.delete(paper)
        db.session.commit()
        
        logger.info(f"🗑️ Deleted paper {paper_id}")
        
        return jsonify({'message': 'Paper deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete paper error: {e}", exc_info=True)
        db.session.rollback()
        return handle_api_error("Failed to delete paper", 500)

# =============================================================================
# SYSTEM INFORMATION ENDPOINTS
# =============================================================================

@api_bp.route('/stats', methods=['GET'])
def get_system_stats():
    """
    Get comprehensive system statistics and metrics
    
    Returns:
        System performance metrics, analysis statistics, and health info
    """
    try:
        # Basic paper statistics
        total_papers = Paper.query.count()
        recent_papers = Paper.query.filter(
            Paper.upload_time >= datetime.now().replace(hour=0, minute=0, second=0)
        ).count()
        
        # Risk level distribution
        risk_distribution = {
            'low': Paper.query.filter(Paper.overall_risk_score < 2.5).count(),
            'medium': Paper.query.filter(
                Paper.overall_risk_score >= 2.5, 
                Paper.overall_risk_score < 5.0
            ).count(),
            'high': Paper.query.filter(
                Paper.overall_risk_score >= 5.0, 
                Paper.overall_risk_score < 7.5
            ).count(),
            'critical': Paper.query.filter(Paper.overall_risk_score >= 7.5).count()
        }
        
        # Performance metrics
        avg_processing_time = db.session.query(
            db.func.avg(Paper.processing_time)
        ).scalar() or 0
        
        avg_risk_score = db.session.query(
            db.func.avg(Paper.overall_risk_score)
        ).scalar() or 0
        
        # Category-specific statistics
        category_stats = {}
        for category in RiskCategory:
            avg_score = db.session.query(
                db.func.avg(RiskResult.score)
            ).filter_by(category=category.value).scalar() or 0
            
            category_stats[category.value] = {
                'average_score': round(avg_score, 2),
                'high_risk_count': RiskResult.query.filter_by(
                    category=category.value
                ).filter(RiskResult.score >= 5.0).count()
            }
        
        # System health
        system_health = _get_system_health()
        
        result = {
            'overview': {
                'total_papers_analyzed': total_papers,
                'papers_today': recent_papers,
                'average_risk_score': round(avg_risk_score, 2),
                'average_processing_time': round(avg_processing_time, 2)
            },
            'risk_distribution': risk_distribution,
            'category_statistics': category_stats,
            'system_health': system_health,
            'performance': {
                'uptime_percentage': 99.9,  # Mock - implement real uptime tracking
                'accuracy_rate': 99.7,     # Mock - implement real accuracy tracking
                'requests_today': recent_papers,
                'average_response_time': round(avg_processing_time, 2)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"System stats error: {e}", exc_info=True)
        return handle_api_error("Failed to retrieve system statistics", 500)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    System health check endpoint
    
    Returns:
        System health status and component availability
    """
    try:
        health_status = _get_system_health()
        
        # Overall health determination
        all_healthy = all(
            status for status in health_status['components'].values()
        )
        
        status_code = 200 if all_healthy else 503
        
        result = {
            'status': 'healthy' if all_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            **health_status
        }
        
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@api_bp.route('/info', methods=['GET'])
def get_system_info():
    """
    Get Guardian LLM system information
    
    Returns:
        System version, capabilities, and configuration
    """
    try:
        if guardian_engine:
            engine_info = guardian_engine.get_system_info()
        else:
            engine_info = {'error': 'Guardian engine not available'}
        
        result = {
            'name': 'Guardian LLM',
            'version': '1.0.0',
            'description': 'Automated Ethical Risk Auditor for AI Research Papers',
            'author': 'Prashish Shrestha',
            'course': 'COMP8420 - Advanced Topics in AI',
            'capabilities': {
                'supported_formats': ['PDF', 'DOCX', 'TXT', 'JSON'],
                'risk_categories': [category.value for category in RiskCategory],
                'risk_levels': [level.value for level in RiskLevel],
                'max_file_size': '16MB',
                'average_processing_time': '10-30 seconds'
            },
            'engine_info': engine_info,
            'api_version': 'v1',
            'endpoints': {
                'analysis': '/api/analyze',
                'papers': '/api/papers',
                'stats': '/api/stats',
                'health': '/api/health'
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"System info error: {e}", exc_info=True)
        return handle_api_error("Failed to retrieve system information", 500)

# =============================================================================
# RISK ANALYSIS ENDPOINTS
# =============================================================================

@api_bp.route('/categories', methods=['GET'])
def get_risk_categories():
    """
    Get information about all risk categories
    
    Returns:
        Detailed information about each risk category
    """
    try:
        categories_info = {
            'bias_fairness': {
                'name': 'Bias and Fairness',
                'description': 'Algorithmic bias, demographic discrimination, unfair treatment',
                'examples': ['Gender bias in hiring AI', 'Racial bias in criminal justice algorithms'],
                'keywords': ['bias', 'fairness', 'discrimination', 'demographic']
            },
            'privacy_data': {
                'name': 'Privacy and Data Protection',
                'description': 'Personal data handling, consent, surveillance, privacy violations',
                'examples': ['Facial recognition without consent', 'Personal data collection'],
                'keywords': ['privacy', 'personal data', 'surveillance', 'biometric']
            },
            'safety_security': {
                'name': 'Safety and Security',
                'description': 'System failures, security vulnerabilities, adversarial attacks',
                'examples': ['Autonomous vehicle safety', 'AI system security breaches'],
                'keywords': ['safety', 'security', 'vulnerability', 'attack']
            },
            'dual_use': {
                'name': 'Dual-Use Potential',
                'description': 'Military applications, weaponization, surveillance potential',
                'examples': ['Autonomous weapons', 'Mass surveillance systems'],
                'keywords': ['military', 'weapon', 'surveillance', 'dual-use']
            },
            'societal_impact': {
                'name': 'Societal Impact',
                'description': 'Economic displacement, democratic effects, social consequences',
                'examples': ['Job automation impact', 'Democratic manipulation'],
                'keywords': ['employment', 'democracy', 'social', 'economic']
            },
            'transparency': {
                'name': 'Transparency and Explainability',
                'description': 'Black box systems, algorithmic accountability, interpretability',
                'examples': ['Unexplainable AI decisions', 'Lack of algorithmic transparency'],
                'keywords': ['explainable', 'interpretable', 'transparent', 'accountable']
            }
        }
        
        return jsonify(categories_info), 200
        
    except Exception as e:
        logger.error(f"Risk categories error: {e}", exc_info=True)
        return handle_api_error("Failed to retrieve risk categories", 500)

@api_bp.route('/analyze/quick', methods=['POST'])
def quick_analyze():
    """
    Quick analysis endpoint for simple text analysis
    
    Expects:
        {"text": "text to analyze", "title": "optional title"}
        
    Returns:
        Simplified analysis results
    """
    try:
        if not guardian_engine:
            return handle_api_error("Analysis engine not available", 503)
        
        data = request.get_json()
        if not data or not data.get('text'):
            return handle_api_error("No text provided", 400)
        
        text = data.get('text', '')
        title = data.get('title', 'Quick Analysis')
        
        if len(text.strip()) < 50:
            return handle_api_error("Text too short for analysis (minimum 50 characters)", 400)
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            title=title,
            content=text
        )
        
        # Perform analysis
        response = guardian_engine.analyze_paper(analysis_request)
        
        # Simplified response
        high_risk_categories = [
            assessment['category'] for assessment in response.risk_assessments
            if assessment['level'] in ['high', 'critical']
        ]
        
        top_recommendations = []
        for assessment in response.risk_assessments:
            if assessment['level'] in ['high', 'critical']:
                top_recommendations.extend(assessment['recommendations'][:2])
        
        result = {
            'overall_score': response.overall_risk_score,
            'risk_level': _get_overall_risk_level(response.overall_risk_score),
            'processing_time': response.processing_time,
            'high_risk_categories': high_risk_categories,
            'critical_issues': len([a for a in response.risk_assessments if a['level'] == 'critical']),
            'top_recommendations': top_recommendations[:5],
            'summary': f"Analysis detected {len(high_risk_categories)} high-risk categories with an overall score of {response.overall_risk_score}/10"
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Quick analysis error: {e}", exc_info=True)
        return handle_api_error("Quick analysis failed", 500)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _save_analysis_to_database(request: AnalysisRequest, response) -> Paper:
    """Save analysis results to database"""
    try:
        # Create paper record
        paper = Paper(
            paper_id=response.paper_id,
            title=response.title,
            authors=json.dumps(request.authors or []),
            abstract=request.abstract or '',
            content_preview=(request.content or '')[:1000],  # Store preview only
            overall_risk_score=response.overall_risk_score,
            processing_time=response.processing_time,
            status=response.status
        )
        
        db.session.add(paper)
        
        # Create risk result records
        for assessment in response.risk_assessments:
            risk_result = RiskResult(
                paper_id=response.paper_id,
                category=assessment['category'],
                score=assessment['score'],
                confidence=assessment['confidence'],
                level=assessment['level'],
                explanation=assessment['explanation'],
                evidence=json.dumps(assessment['evidence']),
                recommendations=json.dumps(assessment['recommendations'])
            )
            db.session.add(risk_result)
        
        db.session.commit()
        logger.info(f"💾 Saved analysis for paper {response.paper_id}")
        
        return paper
        
    except Exception as e:
        logger.error(f"Database save error: {e}", exc_info=True)
        db.session.rollback()
        raise

def _update_system_stats(response):
    """Update system statistics"""
    try:
        # Record processing time
        stat = SystemStats(
            metric_name='processing_time',
            metric_value=response.processing_time
        )
        db.session.add(stat)
        
        # Record overall risk score
        stat = SystemStats(
            metric_name='risk_score',
            metric_value=response.overall_risk_score
        )
        db.session.add(stat)
        
        db.session.commit()
        
    except Exception as e:
        logger.warning(f"Failed to update system stats: {e}")

def _generate_analysis_summary(response) -> Dict:
    """Generate analysis summary"""
    high_risk_count = len([a for a in response.risk_assessments if a['level'] in ['high', 'critical']])
    critical_count = len([a for a in response.risk_assessments if a['level'] == 'critical'])
    
    return {
        'risk_level': _get_overall_risk_level(response.overall_risk_score),
        'high_risk_categories': high_risk_count,
        'critical_categories': critical_count,
        'total_recommendations': sum(len(a['recommendations']) for a in response.risk_assessments),
        'analysis_quality': 'comprehensive' if response.processing_time > 5 else 'fast'
    }

def _generate_detailed_summary(paper: Paper, risk_results: List[RiskResult]) -> Dict:
    """Generate detailed analysis summary"""
    category_scores = {result.category: result.score for result in risk_results}
    high_risk_categories = [r.category for r in risk_results if r.level in ['high', 'critical']]
    
    return {
        'overall_assessment': _get_overall_risk_level(paper.overall_risk_score),
        'highest_risk_category': max(category_scores, key=category_scores.get) if category_scores else None,
        'lowest_risk_category': min(category_scores, key=category_scores.get) if category_scores else None,
        'high_risk_categories': high_risk_categories,
        'total_evidence_items': sum(len(json.loads(r.evidence or '[]')) for r in risk_results),
        'total_recommendations': sum(len(json.loads(r.recommendations or '[]')) for r in risk_results),
        'analysis_date': paper.upload_time.isoformat(),
        'processing_performance': 'fast' if paper.processing_time < 10 else 'standard'
    }

def _get_overall_risk_level(score: float) -> str:
    """Get overall risk level from score"""
    if score < 2.5:
        return 'low'
    elif score < 5.0:
        return 'medium'
    elif score < 7.5:
        return 'high'
    else:
        return 'critical'

def _get_system_health() -> Dict:
    """Get system health status"""
    health_status = {
        'overall': 'healthy',
        'components': {
            'database': True,
            'guardian_engine': guardian_engine is not None,
            'text_processor': text_processor is not None
        },
        'metrics': {
            'database_connections': 'available',
            'memory_usage': 'normal',
            'response_time': 'optimal'
        }
    }
    
    # Test database connection
    try:
        db.session.execute('SELECT 1')
        health_status['components']['database'] = True
    except Exception:
        health_status['components']['database'] = False
        health_status['overall'] = 'degraded'
    
    # Test guardian engine
    if guardian_engine:
        try:
            validation = guardian_engine.validate_system()
            if not all(validation.values()):
                health_status['components']['guardian_engine'] = False
                health_status['overall'] = 'degraded'
        except Exception:
            health_status['components']['guardian_engine'] = False
            health_status['overall'] = 'degraded'
    
    return health_status

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@api_bp.errorhandler(400)
def bad_request(error):
    return handle_api_error("Bad request", 400)

@api_bp.errorhandler(404)
def not_found(error):
    return handle_api_error("Resource not found", 404)

@api_bp.errorhandler(405)
def method_not_allowed(error):
    return handle_api_error("Method not allowed", 405)

@api_bp.errorhandler(413)
def payload_too_large(error):
    return handle_api_error("File too large. Maximum size is 16MB.", 413)

@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}", exc_info=True)
    return handle_api_error("Internal server error", 500)

# =============================================================================
# RATE LIMITING (Optional - implement if needed)
# =============================================================================

# @api_bp.before_request
# def before_request():
#     """Rate limiting logic (implement if needed)"""
#     # Check rate limits
#     if not calculate_rate_limit(request.remote_addr):
#         return handle_api_error("Rate limit exceeded", 429)

# =============================================================================
# MAIN BLUEPRINT REGISTRATION
# =============================================================================

# The blueprint is registered in the main app factory
# This allows for proper initialization of components