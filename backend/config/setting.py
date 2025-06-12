"""
Guardian LLM - Configuration Settings
Application configuration for different environments
"""

import os
from datetime import timedelta


class Config:
    """Base configuration"""
    
    # Application Settings
    APP_NAME = "Guardian LLM"
    VERSION = "1.0.0"
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-please-change-in-production'
    
    # Flask Settings
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    WTF_CSRF_ENABLED = True
    
    # Database Settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(__file__), '../../guardian_llm.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # Security Settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # API Settings
    API_RATE_LIMIT = "100 per hour"
    API_BURST_LIMIT = "10 per minute"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Text Analysis Settings
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 10
    BATCH_SIZE_LIMIT = 100
    ANALYSIS_TIMEOUT = 30  # seconds
    
    # Model Settings
    MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), '../../models')
    USE_GPU = False
    MODEL_BATCH_SIZE = 32
    
    # Model paths
    LORA_ADAPTER_PATH = os.path.join(MODEL_CACHE_DIR, 'lora_adapters')
    
    # Risk Detection Settings
    RISK_KEYWORDS = {
        'high_risk': [
            'suicide', 'kill', 'die', 'death', 'harm', 'hurt', 'attack',
            'violence', 'abuse', 'threat', 'weapon', 'bomb', 'terror',
            'hate', 'racist', 'discrimination'
        ],
        'medium_risk': [
            'angry', 'hate', 'fight', 'destroy', 'revenge', 'punish',
            'suffer', 'pain', 'cry', 'depressed', 'anxiety', 'stress',
            'worried', 'scared', 'fear'
        ],
        'low_risk': [
            'sad', 'unhappy', 'disappointed', 'frustrated', 'annoyed',
            'upset', 'concern', 'worry', 'doubt'
        ]
    }
    
    # Sentiment Analysis Settings
    SENTIMENT_THRESHOLD = {
        'positive': 0.6,
        'negative': -0.6,
        'neutral_range': (-0.2, 0.2)
    }
    
    # Logging Settings
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT', 'False').lower() == 'true'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 10
    
    # Cache Settings
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300
    
     # LoRA Settings
    LORA_ENABLED = True
    LORA_RANK = 16
    LORA_ALPHA = 32.0
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    LORA_DOMAINS = ['biomedical', 'legal', 'financial', 'technical', 'social']
    
    
    # Email Settings (for notifications)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@guardianlm.com')
    
    # External API Keys (if needed)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
    
    # File Upload Settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../../uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv'}
    
    # CORS Settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Redis Settings (for caching/queuing if used)
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Feature Flags
    ENABLE_REAL_TIME_ANALYSIS = True
    ENABLE_BATCH_PROCESSING = True
    ENABLE_EXPORT_FEATURE = True
    ENABLE_USER_FEEDBACK = True
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    SESSION_COOKIE_SECURE = False
    
    # Development-specific settings
    SEND_FILE_MAX_AGE_DEFAULT = 0
    TEMPLATES_AUTO_RELOAD = True
    
    @staticmethod
    def init_app(app):
        """Initialize development environment"""
        Config.init_app(app)
        
        # Log to stdout in development
        import logging
        from logging import StreamHandler
        stream_handler = StreamHandler()
        stream_handler.setLevel(logging.INFO)
        app.logger.addHandler(stream_handler)


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable rate limiting in tests
    API_RATE_LIMIT = None
    
    # Use smaller limits for testing
    MAX_TEXT_LENGTH = 1000
    BATCH_SIZE_LIMIT = 10


class ProductionConfig(Config):
    """Production configuration"""
    
    # Override with production values
    SESSION_COOKIE_SECURE = True
    
    # Production database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:pass@localhost/guardian_llm'
    
    # Stricter limits
    API_RATE_LIMIT = "1000 per hour"
    API_BURST_LIMIT = "20 per minute"
    
    @staticmethod
    def init_app(app):
        """Initialize production environment"""
        Config.init_app(app)
        
        # Email errors to admins
        import logging
        from logging.handlers import SMTPHandler
        
        credentials = None
        secure = None
        
        if getattr(Config, 'MAIL_USERNAME', None) is not None:
            credentials = (Config.MAIL_USERNAME, Config.MAIL_PASSWORD)
            if getattr(Config, 'MAIL_USE_TLS', None):
                secure = ()
        
        if Config.MAIL_SERVER:
            mail_handler = SMTPHandler(
                mailhost=(Config.MAIL_SERVER, Config.MAIL_PORT),
                fromaddr=Config.MAIL_DEFAULT_SENDER,
                toaddrs=Config.ADMIN_EMAILS,
                subject='Guardian LLM Application Error',
                credentials=credentials,
                secure=secure
            )
            mail_handler.setLevel(logging.ERROR)
            app.logger.addHandler(mail_handler)


class DockerConfig(ProductionConfig):
    """Docker configuration"""
    
    # Docker-specific overrides
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://guardian:guardian@db:5432/guardian_llm'
    
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration object"""
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, DevelopmentConfig)