#!/usr/bin/env python3
"""
Guardian LLM - Main Flask Application
AI-powered text analysis for risk detection and content moderation
"""

import os
import sys
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from backend.config.setting import Config
from backend.models.database import db
from backend.api.routes import api_blueprint
from backend.core.guardian_engine import GuardianEngine

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Load configuration
app.config.from_object(Config)

# Initialize extensions
CORS(app, resources={r"/api/*": {"origins": "*"}})
db.init_app(app)

# Initialize Guardian Engine
guardian_engine = GuardianEngine()

# Register blueprints
app.register_blueprint(api_blueprint, url_prefix='/api')

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/guardian_llm.log', maxBytes=10240000, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Guardian LLM startup')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    app.logger.error(f'Internal error: {error}')
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden_error(error):
    """Handle 403 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Forbidden'}), 403
    return render_template('403.html'), 403

# Main routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html', 
                         title='Guardian LLM - AI Text Analysis',
                         active_page='home')

@app.route('/dashboard')
def dashboard():
    """Main dashboard"""
    stats = {
        'total_analyses': 1247,
        'high_risk_detected': 89,
        'avg_response_time': '234ms',
        'accuracy_rate': '94.3%'
    }
    return render_template('dashboard.html', 
                         title='Dashboard - Guardian LLM',
                         active_page='dashboard',
                         stats=stats)

@app.route('/analysis')
def analysis():
    """Analysis page"""
    return render_template('analysis.html', 
                         title='Text Analysis - Guardian LLM',
                         active_page='analysis')

@app.route('/reports')
def reports():
    """Reports page"""
    # Get recent analyses from database
    recent_analyses = []  # TODO: Fetch from database
    return render_template('reports.html', 
                         title='Reports - Guardian LLM',
                         active_page='reports',
                         analyses=recent_analyses)

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html', 
                         title='Settings - Guardian LLM',
                         active_page='settings')

# API Health check
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'engine_status': guardian_engine.get_status()
    })

# Static file serving for development
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# Context processors
@app.context_processor
def inject_globals():
    """Inject global variables into templates"""
    return {
        'app_name': 'Guardian LLM',
        'current_year': datetime.now().year,
        'version': '1.0.0'
    }

# CLI commands
@app.cli.command()
def init_db():
    """Initialize the database"""
    db.create_all()
    print("Database initialized successfully!")

@app.cli.command()
def seed_db():
    """Seed the database with sample data"""
    from backend.models import Analysis, User
    
    # Create sample user
    user = User(
        username='admin',
        email='admin@guardianlm.com'
    )
    user.set_password('admin123')
    db.session.add(user)
    
    # Create sample analyses
    sample_texts = [
        "This is a normal text without any risks.",
        "This text contains some concerning content that needs review.",
        "Emergency situation detected with high risk factors."
    ]
    
    for i, text in enumerate(sample_texts):
        analysis = Analysis(
            text=text,
            critical_risk_score=0.2 * (i + 1),
            overall_risk_score=0.15 * (i + 1),
            sentiment_score=-0.5 * i,
            risk_level=['low', 'medium', 'high'][i],
            user_id=1
        )
        db.session.add(analysis)
    
    db.session.commit()
    print("Database seeded successfully!")

@app.cli.command()
def test():
    """Run the test suite"""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)

# Development server configuration
if __name__ == '__main__':
    # Ensure all required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        app.logger.info('Database tables created/verified')
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=True
    )
else:
    # Production logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)