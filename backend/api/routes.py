"""
Guardian LLM - API Routes
Defines all API endpoints for the Guardian LLM service
"""

from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
from datetime import datetime
import json
import traceback
from sqlalchemy.exc import SQLAlchemyError

# Import models and schemas
from backend.models.database import db, Analysis, User, RiskDetection, Feedback
from backend.models.schemas import (
    AnalysisRequest, 
    BatchAnalysisRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    RiskCategory, 
    RiskLevel,
    RiskAssessment,
    SentimentAnalysis,
    TextStatistics,
    Recommendation,
    ErrorResponse
)
from backend.core.guardian_engine import GuardianEngine

# Create blueprint
api_blueprint = Blueprint('api', __name__)

# Initialize Guardian Engine
guardian_engine = GuardianEngine()


# Error handlers
@api_blueprint.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error)
    }), 400


@api_blueprint.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404


@api_blueprint.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    current_app.logger.error(f'Internal error: {error}')
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


# Main routes
@api_blueprint.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_text():
    """
    Analyze text for risks and sentiment
    
    Supports both JSON and multipart/form-data (file upload)
    """
    try:
        # Check if this is a file upload
        if 'file' in request.files:
            return analyze_file()
        
        # Otherwise, expect JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request using Pydantic
        try:
            analysis_request = AnalysisRequest(**data)
        except Exception as e:
            return jsonify({
                'error': 'Validation error',
                'message': str(e)
            }), 400
        
        # Perform analysis
        start_time = datetime.utcnow()
        
        # Create analysis record
        analysis = Analysis(
            text=analysis_request.text,
            user_id=analysis_request.user_id,
            status='processing'
        )
        db.session.add(analysis)
        db.session.commit()
        
        try:
            # Run analysis through Guardian Engine
            results = guardian_engine.analyze_text(
                analysis_request.text,
                options=analysis_request.options
            )
            
            # Update analysis record with results
            analysis.critical_risk_score = results['risk_analysis']['critical_risk']['score']
            analysis.overall_risk_score = results['risk_analysis']['overall_risk']['score']
            analysis.risk_level = results['risk_analysis']['critical_risk']['level']
            
            analysis.sentiment_score = results['sentiment']['score']
            analysis.sentiment_type = results['sentiment']['type']
            analysis.sentiment_positive = results['sentiment']['positive_score']
            analysis.sentiment_negative = results['sentiment']['negative_score']
            analysis.sentiment_neutral = results['sentiment']['neutral_score']
            
            analysis.word_count = results['statistics']['word_count']
            analysis.character_count = results['statistics']['character_count']
            analysis.high_risk_keywords = results['statistics']['high_risk_keywords']
            analysis.medium_risk_keywords = results['statistics']['medium_risk_keywords']
            
            analysis.risk_assessments = json.dumps(results['risk_assessments'])
            analysis.recommendations = json.dumps(results['recommendations'])
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            analysis.processing_time = processing_time
            analysis.status = 'completed'
            
            db.session.commit()
            
            # Store risk detections
            for risk in results['risk_assessments']:
                risk_detection = RiskDetection(
                    analysis_id=analysis.id,
                    category=risk['category'],
                    level=risk['level'],
                    score=risk['score'],
                    confidence=risk['confidence'],
                    keywords=json.dumps(risk.get('keywords', [])),
                    context=risk.get('context', '')
                )
                db.session.add(risk_detection)
            
            db.session.commit()
            
            # Prepare response
            response = {
                'id': analysis.analysis_id,
                'text': analysis.text,
                'timestamp': datetime.utcnow().isoformat(),
                'analysis': {
                    'critical_risk': {
                        'score': analysis.critical_risk_score,
                        'level': analysis.risk_level,
                        'percentage': int(analysis.critical_risk_score * 100)
                    },
                    'overall_risk': {
                        'score': analysis.overall_risk_score,
                        'level': results['risk_analysis']['overall_risk']['level'],
                        'percentage': int(analysis.overall_risk_score * 100)
                    },
                    'sentiment': {
                        'type': analysis.sentiment_type,
                        'score': analysis.sentiment_score,
                        'positive': analysis.sentiment_positive,
                        'negative': analysis.sentiment_negative,
                        'neutral': analysis.sentiment_neutral
                    },
                    'statistics': {
                        'wordCount': analysis.word_count,
                        'characterCount': analysis.character_count,
                        'highRiskKeywords': analysis.high_risk_keywords,
                        'mediumRiskKeywords': analysis.medium_risk_keywords
                    }
                },
                'risk_assessments': results['risk_assessments'],
                'recommendations': results['recommendations'],
                'processing_time': processing_time
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            # Update analysis status to failed
            analysis.status = 'failed'
            db.session.commit()
            
            current_app.logger.error(f'Analysis error: {str(e)}\n{traceback.format_exc()}')
            return jsonify({
                'error': 'Analysis failed',
                'message': str(e)
            }), 500
            
    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f'Database error: {str(e)}')
        return jsonify({
            'error': 'Database error',
            'message': 'Failed to save analysis'
        }), 500
    except Exception as e:
        current_app.logger.error(f'Unexpected error: {str(e)}\n{traceback.format_exc()}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


def analyze_file():
    """
    Analyze uploaded file (PDF, DOCX, TXT)
    """
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'pdf', 'docx', 'txt', 'doc'}
        filename = file.filename.lower()
        file_ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'}), 400
        
        # Get metadata from form data
        title = request.form.get('title', file.filename)
        abstract = request.form.get('abstract', '')
        authors = request.form.get('authors', '')
        
        # Read file content based on type
        content = ""
        
        if file_ext == 'txt':
            content = file.read().decode('utf-8')
        elif file_ext == 'pdf':
            # You'll need to install PyPDF2: pip install PyPDF2
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            except ImportError:
                return jsonify({'error': 'PDF support not installed. Please install PyPDF2'}), 500
            except Exception as e:
                return jsonify({'error': f'Error reading PDF: {str(e)}'}), 500
        elif file_ext in ['doc', 'docx']:
            # You'll need to install python-docx: pip install python-docx
            try:
                from docx import Document
                doc = Document(file)
                for paragraph in doc.paragraphs:
                    content += paragraph.text + '\n'
            except ImportError:
                return jsonify({'error': 'DOCX support not installed. Please install python-docx'}), 500
            except Exception as e:
                return jsonify({'error': f'Error reading document: {str(e)}'}), 500
        
        if not content.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        # Limit content length
        if len(content) > 50000:
            content = content[:50000]
        
        # Now analyze the extracted text
        start_time = datetime.utcnow()
        
        # Create analysis record
        analysis = Analysis(
            text=content,
            user_id=request.form.get('user_id'),
            status='processing'
        )
        
        # Store title in text_preview if provided
        if title and title != file.filename:
            analysis.text_preview = f"Title: {title}"
        
        db.session.add(analysis)
        db.session.commit()
        
        try:
            # Run analysis
            results = guardian_engine.analyze_text(content)
            
            # Update analysis record with results
            analysis.critical_risk_score = results['risk_analysis']['critical_risk']['score']
            analysis.overall_risk_score = results['risk_analysis']['overall_risk']['score']
            analysis.risk_level = results['risk_analysis']['critical_risk']['level']
            
            analysis.sentiment_score = results['sentiment']['score']
            analysis.sentiment_type = results['sentiment']['type']
            analysis.sentiment_positive = results['sentiment']['positive_score']
            analysis.sentiment_negative = results['sentiment']['negative_score']
            analysis.sentiment_neutral = results['sentiment']['neutral_score']
            
            analysis.word_count = results['statistics']['word_count']
            analysis.character_count = results['statistics']['character_count']
            analysis.high_risk_keywords = results['statistics']['high_risk_keywords']
            analysis.medium_risk_keywords = results['statistics']['medium_risk_keywords']
            
            analysis.risk_assessments = json.dumps(results['risk_assessments'])
            analysis.recommendations = json.dumps(results['recommendations'])
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            analysis.processing_time = processing_time
            analysis.status = 'completed'
            
            db.session.commit()
            
            # Store risk detections
            for risk in results['risk_assessments']:
                risk_detection = RiskDetection(
                    analysis_id=analysis.id,
                    category=risk['category'],
                    level=risk['level'],
                    score=risk['score'],
                    confidence=risk['confidence'],
                    keywords=json.dumps(risk.get('keywords', [])),
                    context=risk.get('context', '')
                )
                db.session.add(risk_detection)
            
            db.session.commit()
            
            # Prepare response matching the expected format
            response = {
                'paper_id': analysis.analysis_id,
                'title': title,
                'upload_time': datetime.utcnow().isoformat(),
                'overall_risk_score': round(analysis.overall_risk_score * 10, 1),  # Convert to 0-10 scale
                'risk_assessments': [
                    {
                        'category': risk['category'],
                        'score': round(risk['score'] * 10, 1),  # Convert to 0-10 scale
                        'level': risk['level'],
                        'confidence': risk['confidence'],
                        'explanation': f"Risk assessment for {risk['category']}",
                        'evidence': risk.get('keywords', []),
                        'recommendations': [
                            f"Review {risk['category']} concerns",
                            "Consider mitigation strategies"
                        ]
                    }
                    for risk in results['risk_assessments']
                ],
                'processing_time': round(processing_time, 2)
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            # Update analysis status to failed
            analysis.status = 'failed'
            db.session.commit()
            
            current_app.logger.error(f'Analysis error: {str(e)}\n{traceback.format_exc()}')
            return jsonify({
                'error': 'Analysis failed',
                'message': str(e)
            }), 500
        
    except Exception as e:
        current_app.logger.error(f'File analysis error: {str(e)}')
        return jsonify({'error': 'File analysis failed', 'message': str(e)}), 500


@api_blueprint.route('/analyze/batch', methods=['POST'])
@cross_origin()
def analyze_batch():
    """
    Analyze multiple texts in batch
    
    Request body:
    {
        "texts": ["text1", "text2", ...],
        "user_id": 1 (optional),
        "options": {} (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request
        try:
            batch_request = BatchAnalysisRequest(**data)
        except Exception as e:
            return jsonify({
                'error': 'Validation error',
                'message': str(e)
            }), 400
        
        results = []
        total_critical_risk = 0
        total_overall_risk = 0
        high_risk_count = 0
        
        for text in batch_request.texts:
            try:
                # Analyze each text
                analysis_results = guardian_engine.analyze_text(text, batch_request.options)
                
                # Create simple result
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'criticalRisk': analysis_results['risk_analysis']['critical_risk']['score'],
                    'overallRisk': analysis_results['risk_analysis']['overall_risk']['score'],
                    'sentiment': analysis_results['sentiment']['score']
                }
                
                results.append(result)
                
                # Update statistics
                total_critical_risk += result['criticalRisk']
                total_overall_risk += result['overallRisk']
                if result['criticalRisk'] > 0.7:
                    high_risk_count += 1
                    
            except Exception as e:
                current_app.logger.error(f'Error analyzing text: {str(e)}')
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'error': str(e)
                })
        
        # Calculate summary
        count = len(batch_request.texts)
        summary = {
            'averageCriticalRisk': total_critical_risk / count if count > 0 else 0,
            'averageOverallRisk': total_overall_risk / count if count > 0 else 0,
            'highRiskCount': high_risk_count
        }
        
        response = {
            'count': count,
            'results': results,
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f'Batch analysis error: {str(e)}\n{traceback.format_exc()}')
        return jsonify({
            'error': 'Batch analysis failed',
            'message': str(e)
        }), 500


@api_blueprint.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except:
        db_status = 'unhealthy'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': db_status,
        'engine': guardian_engine.get_status()
    }), 200


@api_blueprint.route('/stats', methods=['GET'])
@cross_origin()
def get_stats():
    """Get system statistics"""
    try:
        # Get analysis statistics
        total_analyses = Analysis.query.count()
        completed_analyses = Analysis.query.filter_by(status='completed').count()
        failed_analyses = Analysis.query.filter_by(status='failed').count()
        
        # Get risk statistics
        high_risk_count = Analysis.query.filter(Analysis.risk_level == 'high').count()
        critical_risk_count = Analysis.query.filter(Analysis.risk_level == 'critical').count()
        
        # Calculate average processing time
        avg_processing_time = db.session.query(
            db.func.avg(Analysis.processing_time)
        ).filter(Analysis.processing_time.isnot(None)).scalar() or 0
        
        stats = {
            'totalAnalyses': total_analyses,
            'completedAnalyses': completed_analyses,
            'failedAnalyses': failed_analyses,
            'highRiskDetected': high_risk_count + critical_risk_count,
            'criticalRiskDetected': critical_risk_count,
            'averageProcessingTime': round(avg_processing_time, 3),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        current_app.logger.error(f'Stats error: {str(e)}')
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': str(e)
        }), 500


@api_blueprint.route('/analysis/<analysis_id>', methods=['GET'])
@cross_origin()
def get_analysis(analysis_id):
    """Get specific analysis by ID"""
    try:
        analysis = Analysis.query.filter_by(analysis_id=analysis_id).first()
        
        if not analysis:
            return jsonify({
                'error': 'Analysis not found',
                'message': f'No analysis found with ID: {analysis_id}'
            }), 404
        
        # Get risk detections
        risk_detections = RiskDetection.query.filter_by(analysis_id=analysis.id).all()
        
        response = analysis.to_dict()
        response['risk_detections'] = [rd.to_dict() for rd in risk_detections]
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f'Get analysis error: {str(e)}')
        return jsonify({
            'error': 'Failed to retrieve analysis',
            'message': str(e)
        }), 500


@api_blueprint.route('/feedback', methods=['POST'])
@cross_origin()
def submit_feedback():
    """Submit feedback for an analysis"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'analysis_id' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'analysis_id is required'
            }), 400
        
        # Find analysis
        analysis = Analysis.query.filter_by(analysis_id=data['analysis_id']).first()
        if not analysis:
            return jsonify({
                'error': 'Analysis not found',
                'message': f'No analysis found with ID: {data["analysis_id"]}'
            }), 404
        
        # Create feedback
        feedback = Feedback(
            analysis_id=analysis.id,
            user_id=data.get('user_id'),
            feedback_type=data.get('feedback_type', 'general'),
            rating=data.get('rating'),
            comment=data.get('comment'),
            is_accurate=data.get('is_accurate')
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        return jsonify({
            'message': 'Feedback submitted successfully',
            'feedback_id': feedback.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'Feedback error: {str(e)}')
        return jsonify({
            'error': 'Failed to submit feedback',
            'message': str(e)
        }), 500


# Export the blueprint
__all__ = ['api_blueprint']