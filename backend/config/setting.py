import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'guardian-llm-dev-key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # AI Model Configuration
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'bert-base-uncased')
    GPU_ENABLED = os.environ.get('GPU_ENABLED', 'false').lower() == 'true'
    
    # Risk Assessment Thresholds
    RISK_THRESHOLDS = {
        'low': (0.0, 2.5),
        'medium': (2.5, 5.0),
        'high': (5.0, 7.5),
        'critical': (7.5, 10.0)
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///guardian_llm_dev.db'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://user:pass@localhost/guardian_llm'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}