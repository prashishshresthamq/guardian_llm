# backend/api/utils.py
"""
Guardian LLM API Utilities
==========================

This module contains utility functions for the Guardian LLM API layer.
Provides common functionality for request validation, error handling,
file processing, and response formatting.

Author: Prashish Shrestha
Course: COMP8420 - Advanced Topics in AI
Project: Guardian LLM - Automated Ethical Risk Auditor
"""

import os
import re
import json
import hashlib
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
import logging

from flask import jsonify, request, current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def handle_api_error(message: str, status_code: int = 500, details: Optional[Dict] = None) -> Tuple[Any, int]:
    """
    Standardized API error response handler
    
    Args:
        message: Error message to display to user
        status_code: HTTP status code
        details: Optional additional error details
        
    Returns:
        Tuple of (JSON response, status code)
    """
    error_response = {
        'error': True,
        'message': message,
        'status_code': status_code,
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        error_response['details'] = details
    
    # Log error for debugging (but don't expose sensitive info)
    if status_code >= 500:
        logger.error(f"API Error {status_code}: {message}", extra={'details': details})
    elif status_code >= 400:
        logger.warning(f"API Warning {status_code}: {message}")
    
    return jsonify(error_response), status_code

def handle_validation_error(field: str, message: str, value: Any = None) -> Tuple[Any, int]:
    """
    Handle validation errors with field-specific information
    
    Args:
        field: Name of the field that failed validation
        message: Validation error message
        value: Optional value that failed validation
        
    Returns:
        Standardized validation error response
    """
    details = {
        'field': field,
        'message': message,
        'validation_type': 'field_validation'
    }
    
    if value is not None:
        details['received_value'] = str(value)[:100]  # Limit length for security
    
    return handle_api_error(f"Validation failed for field '{field}': {message}", 400, details)

def create_success_response(data: Any, message: str = "Success", status_code: int = 200) -> Tuple[Any, int]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Standardized success response
    """
    response = {
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(response), status_code

# =============================================================================
# FILE UPLOAD UTILITIES
# =============================================================================

def validate_file_upload(file: FileStorage) -> Dict[str, Any]:
    """
    Comprehensive file upload validation
    
    Args:
        file: Flask FileStorage object
        
    Returns:
        Dictionary with validation results
    """
    if not file or not file.filename:
        return {'valid': False, 'error': 'No file provided'}
    
    filename = file.filename.lower()
    
    # Check file extension
    allowed_extensions = {'.pdf', '.docx', '.txt', '.doc'}
    file_ext = os.path.splitext(filename)[1]
    
    if file_ext not in allowed_extensions:
        return {
            'valid': False, 
            'error': f'Unsupported file type: {file_ext}. Allowed: {", ".join(allowed_extensions)}'
        }
    
    # Check file size (16MB limit)
    max_size = 16 * 1024 * 1024  # 16MB
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > max_size:
        return {
            'valid': False,
            'error': f'File too large: {file_size / (1024*1024):.1f}MB. Maximum: {max_size / (1024*1024)}MB'
        }
    
    if file_size == 0:
        return {'valid': False, 'error': 'Empty file'}
    
    # Check filename security
    secure_name = secure_filename(file.filename)
    if not secure_name:
        return {'valid': False, 'error': 'Invalid filename'}
    
    # Additional MIME type validation
    if hasattr(file, 'content_type'):
        allowed_mimes = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain'
        }
        
        if file.content_type and file.content_type not in allowed_mimes:
            logger.warning(f"Suspicious MIME type: {file.content_type} for file {filename}")
    
    return {
        'valid': True,
        'filename': secure_name,
        'size': file_size,
        'extension': file_ext
    }

def generate_safe_filename(original_filename: str, prefix: str = None) -> str:
    """
    Generate a safe filename with timestamp
    
    Args:
        original_filename: Original file name
        prefix: Optional prefix for the filename
        
    Returns:
        Safe filename with timestamp
    """
    secure_name = secure_filename(original_filename)
    name, ext = os.path.splitext(secure_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if prefix:
        return f"{prefix}_{timestamp}_{name}{ext}"
    else:
        return f"{timestamp}_{name}{ext}"

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""

# =============================================================================
# REQUEST VALIDATION UTILITIES
# =============================================================================

def validate_json_request(required_fields: List[str] = None, optional_fields: List[str] = None) -> Dict[str, Any]:
    """
    Validate JSON request data
    
    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names
        
    Returns:
        Validation result dictionary
    """
    if not request.is_json:
        return {'valid': False, 'error': 'Request must be JSON'}
    
    try:
        data = request.get_json()
    except Exception as e:
        return {'valid': False, 'error': f'Invalid JSON: {str(e)}'}
    
    if not data:
        return {'valid': False, 'error': 'Empty JSON data'}
    
    # Check required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {
                'valid': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }
    
    # Validate field types and values
    validation_errors = []
    
    for field, value in data.items():
        if field in ['title', 'abstract', 'content']:
            if not isinstance(value, str):
                validation_errors.append(f'{field} must be a string')
            elif field == 'title' and len(value.strip()) == 0:
                validation_errors.append('Title cannot be empty')
            elif field == 'content' and len(value.strip()) < 10:
                validation_errors.append('Content too short (minimum 10 characters)')
        
        elif field == 'authors':
            if not isinstance(value, list):
                validation_errors.append('Authors must be a list')
            elif not all(isinstance(author, str) for author in value):
                validation_errors.append('All authors must be strings')
    
    if validation_errors:
        return {'valid': False, 'error': '; '.join(validation_errors)}
    
    return {'valid': True, 'data': data}

def validate_query_parameters(allowed_params: Dict[str, type]) -> Dict[str, Any]:
    """
    Validate query parameters
    
    Args:
        allowed_params: Dictionary of parameter names and their expected types
        
    Returns:
        Validation result with parsed parameters
    """
    validated_params = {}
    errors = []
    
    for param_name, param_type in allowed_params.items():
        value = request.args.get(param_name)
        
        if value is not None:
            try:
                if param_type == int:
                    validated_params[param_name] = int(value)
                elif param_type == float:
                    validated_params[param_name] = float(value)
                elif param_type == bool:
                    validated_params[param_name] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    validated_params[param_name] = str(value)
            except ValueError:
                errors.append(f"Invalid {param_name}: expected {param_type.__name__}")
    
    if errors:
        return {'valid': False, 'error': '; '.join(errors)}
    
    return {'valid': True, 'params': validated_params}

def sanitize_input(text: str, max_length: int = 10000, remove_html: bool = True) -> str:
    """
    Sanitize text input for security
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        remove_html: Whether to remove HTML tags
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    # Remove HTML tags if requested
    if remove_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# =============================================================================
# RATE LIMITING UTILITIES
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}  # IP -> list of timestamps
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = datetime.now()
    
    def is_allowed(self, ip_address: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            ip_address: Client IP address
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        # Cleanup old entries periodically
        if (now - self.last_cleanup).seconds > self.cleanup_interval:
            self._cleanup_old_requests(window_start)
            self.last_cleanup = now
        
        # Get requests for this IP
        if ip_address not in self.requests:
            self.requests[ip_address] = []
        
        # Filter recent requests
        recent_requests = [
            req_time for req_time in self.requests[ip_address]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(recent_requests) >= max_requests:
            return False
        
        # Add current request
        recent_requests.append(now)
        self.requests[ip_address] = recent_requests
        
        return True
    
    def _cleanup_old_requests(self, cutoff_time: datetime):
        """Remove old request records"""
        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                req_time for req_time in self.requests[ip]
                if req_time > cutoff_time
            ]
            if not self.requests[ip]:
                del self.requests[ip]

# Global rate limiter instance
rate_limiter = RateLimiter()

def calculate_rate_limit(ip_address: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
    """
    Check rate limit for IP address
    
    Args:
        ip_address: Client IP address
        max_requests: Maximum requests allowed
        window_minutes: Time window in minutes
        
    Returns:
        True if request is allowed, False if rate limited
    """
    return rate_limiter.is_allowed(ip_address, max_requests, window_minutes)

def rate_limit_decorator(max_requests: int = 100, window_minutes: int = 60):
    """
    Decorator for rate limiting API endpoints
    
    Args:
        max_requests: Maximum requests allowed
        window_minutes: Time window in minutes
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
            
            if not calculate_rate_limit(ip_address, max_requests, window_minutes):
                return handle_api_error(
                    f"Rate limit exceeded: {max_requests} requests per {window_minutes} minutes",
                    429
                )
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# =============================================================================
# RESPONSE FORMATTING UTILITIES
# =============================================================================

def format_analysis_response(analysis_data: Dict, include_details: bool = True) -> Dict[str, Any]:
    """
    Format analysis response for API consumption
    
    Args:
        analysis_data: Raw analysis data
        include_details: Whether to include detailed information
        
    Returns:
        Formatted response dictionary
    """
    # Basic response structure
    response = {
        'paper_id': analysis_data.get('paper_id'),
        'title': analysis_data.get('title'),
        'overall_risk_score': analysis_data.get('overall_risk_score'),
        'processing_time': analysis_data.get('processing_time'),
        'timestamp': analysis_data.get('timestamp'),
        'status': analysis_data.get('status', 'completed')
    }
    
    if include_details:
        response.update({
            'risk_assessments': analysis_data.get('risk_assessments', []),
            'summary': analysis_data.get('summary', {}),
            'metadata': analysis_data.get('metadata', {})
        })
    else:
        # Simplified response
        high_risk_categories = [
            assessment['category'] for assessment in analysis_data.get('risk_assessments', [])
            if assessment['level'] in ['high', 'critical']
        ]
        response['high_risk_categories'] = high_risk_categories
        response['risk_level'] = get_overall_risk_level(analysis_data.get('overall_risk_score', 0))
    
    return response

def format_paper_list_response(papers: List[Dict], pagination: Dict = None, filters: Dict = None) -> Dict[str, Any]:
    """
    Format paper list response
    
    Args:
        papers: List of paper dictionaries
        pagination: Pagination information
        filters: Applied filters
        
    Returns:
        Formatted list response
    """
    response = {
        'papers': papers,
        'count': len(papers)
    }
    
    if pagination:
        response['pagination'] = pagination
    
    if filters:
        response['filters'] = filters
    
    # Add summary statistics
    if papers:
        scores = [p.get('overall_risk_score', 0) for p in papers]
        response['statistics'] = {
            'average_risk_score': round(sum(scores) / len(scores), 2),
            'highest_risk_score': max(scores),
            'lowest_risk_score': min(scores),
            'high_risk_papers': len([s for s in scores if s >= 5.0])
        }
    
    return response

def get_overall_risk_level(score: float) -> str:
    """
    Convert risk score to risk level
    
    Args:
        score: Risk score (0-10)
        
    Returns:
        Risk level string
    """
    if score < 2.5:
        return 'low'
    elif score < 5.0:
        return 'medium'
    elif score < 7.5:
        return 'high'
    else:
        return 'critical'

def format_risk_level_color(level: str) -> str:
    """
    Get color code for risk level
    
    Args:
        level: Risk level string
        
    Returns:
        CSS color class or hex color
    """
    colors = {
        'low': '#28a745',      # Green
        'medium': '#ffc107',   # Yellow
        'high': '#fd7e14',     # Orange
        'critical': '#dc3545'  # Red
    }
    return colors.get(level, '#6c757d')  # Default gray

# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def paginate_results(query_results: List[Any], page: int = 1, per_page: int = 50) -> Dict[str, Any]:
    """
    Paginate query results
    
    Args:
        query_results: List of results to paginate
        page: Page number (1-based)
        per_page: Results per page
        
    Returns:
        Paginated results with metadata
    """
    total_count = len(query_results)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    paginated_data = query_results[start_idx:end_idx]
    
    return {
        'data': paginated_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total_count,
            'total_pages': (total_count + per_page - 1) // per_page,
            'has_next': end_idx < total_count,
            'has_prev': page > 1,
            'next_page': page + 1 if end_idx < total_count else None,
            'prev_page': page - 1 if page > 1 else None
        }
    }

def filter_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str] = None) -> Dict[str, Any]:
    """
    Remove sensitive fields from response data
    
    Args:
        data: Data dictionary
        sensitive_fields: List of field names to remove
        
    Returns:
        Filtered data dictionary
    """
    if sensitive_fields is None:
        sensitive_fields = ['password', 'secret', 'key', 'token', 'private']
    
    filtered_data = {}
    
    for key, value in data.items():
        if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
            filtered_data[key] = '[REDACTED]'
        elif isinstance(value, dict):
            filtered_data[key] = filter_sensitive_data(value, sensitive_fields)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            filtered_data[key] = [filter_sensitive_data(item, sensitive_fields) for item in value]
        else:
            filtered_data[key] = value
    
    return filtered_data

def convert_datetime_to_iso(data: Any) -> Any:
    """
    Convert datetime objects to ISO format strings
    
    Args:
        data: Data that may contain datetime objects
        
    Returns:
        Data with datetime objects converted to ISO strings
    """
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {key: convert_datetime_to_iso(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_datetime_to_iso(item) for item in data]
    else:
        return data

# =============================================================================
# CACHING UTILITIES
# =============================================================================

class SimpleCache:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cached values"""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (value, expiry) in self.cache.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self.cache[key]

# Global cache instance
api_cache = SimpleCache()

def cache_response(key: str, ttl: int = 300):
    """
    Decorator for caching API responses
    
    Args:
        key: Cache key template (can use {param} placeholders)
        ttl: Time to live in seconds
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            cache_key = key.format(**kwargs, **request.args.to_dict())
            
            # Try to get from cache
            cached_result = api_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function
            result = f(*args, **kwargs)
            
            # Cache successful results
            if isinstance(result, tuple) and len(result) == 2:
                response, status_code = result
                if 200 <= status_code < 300:
                    api_cache.set(cache_key, result, ttl)
                    logger.debug(f"Cached result for key: {cache_key}")
            
            return result
        return decorated_function
    return decorator

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_api_request(endpoint: str, method: str, ip_address: str, user_agent: str = None):
    """
    Log API request for monitoring
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        ip_address: Client IP address
        user_agent: User agent string
    """
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'method': method,
        'ip_address': ip_address,
        'user_agent': user_agent[:200] if user_agent else None  # Truncate for safety
    }
    
    logger.info(f"API Request: {method} {endpoint}", extra=log_data)

def log_analysis_performance(paper_id: str, processing_time: float, overall_score: float):
    """
    Log analysis performance metrics
    
    Args:
        paper_id: Paper identifier
        processing_time: Time taken for analysis
        overall_score: Overall risk score
    """
    logger.info(
        f"Analysis Performance - Paper: {paper_id}, "
        f"Time: {processing_time:.2f}s, Score: {overall_score:.1f}",
        extra={
            'paper_id': paper_id,
            'processing_time': processing_time,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
    )

# =============================================================================
# HEALTH CHECK UTILITIES
# =============================================================================

def check_system_resources() -> Dict[str, Any]:
    """
    Check system resource usage
    
    Returns:
        Dictionary with resource usage information
    """
    import psutil
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'disk_usage': disk_percent,
            'status': 'healthy' if all(x < 90 for x in [cpu_percent, memory_percent, disk_percent]) else 'warning'
        }
    
    except ImportError:
        return {
            'cpu_usage': 'unknown',
            'memory_usage': 'unknown',
            'disk_usage': 'unknown',
            'status': 'psutil not available'
        }
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def get_api_config() -> Dict[str, Any]:
    """
    Get API configuration settings
    
    Returns:
        Dictionary with API configuration
    """
    return {
        'version': '1.0.0',
        'max_file_size': '16MB',
        'supported_formats': ['PDF', 'DOCX', 'TXT'],
        'rate_limits': {
            'requests_per_hour': 100,
            'analysis_per_day': 50
        },
        'cache_ttl': 300,  # 5 minutes
        'features': {
            'file_upload': True,
            'batch_analysis': False,
            'real_time_analysis': True,
            'caching': True,
            'rate_limiting': True
        }
    }

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key (placeholder for future implementation)
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Placeholder implementation
    # In production, implement proper API key validation
    return len(api_key) >= 32 if api_key else False

# =============================================================================
# TESTING UTILITIES
# =============================================================================

def create_test_response(data: Any, status_code: int = 200) -> Tuple[Any, int]:
    """
    Create test response for unit testing
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Test response tuple
    """
    response = {
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'test_mode': True
    }
    
    return jsonify(response), status_code

# =============================================================================
# MAIN UTILITY EXPORTS
# =============================================================================

__all__ = [
    # Error handling
    'handle_api_error',
    'handle_validation_error', 
    'create_success_response',
    
    # File utilities
    'validate_file_upload',
    'generate_safe_filename',
    'calculate_file_hash',
    
    # Request validation
    'validate_json_request',
    'validate_query_parameters',
    'sanitize_input',
    
    # Rate limiting
    'calculate_rate_limit',
    'rate_limit_decorator',
    
    # Response formatting
    'format_analysis_response',
    'format_paper_list_response',
    'get_overall_risk_level',
    
    # Data processing
    'paginate_results',
    'filter_sensitive_data',
    'convert_datetime_to_iso',
    
    # Caching
    'api_cache',
    'cache_response',
    
    # Logging
    'log_api_request',
    'log_analysis_performance',
    
    # Health checks
    'check_system_resources',
    
    # Configuration
    'get_api_config',
    'validate_api_key'
]