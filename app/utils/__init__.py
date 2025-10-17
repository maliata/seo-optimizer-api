"""
Utilities package for SEO optimization API.

This package contains utility functions, logging configuration,
custom exceptions, and helper classes.
"""

from .logging import setup_logging, get_logger
from .exceptions import (
    SEOOptimizationError,
    ValidationError,
    ProductProcessingError,
    AIServiceError,
    RateLimitError,
    ConfigurationError,
    ExternalServiceError,
    ContentGenerationError,
    SEOAnalysisError,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    
    # Exceptions
    "SEOOptimizationError",
    "ValidationError",
    "ProductProcessingError",
    "AIServiceError",
    "RateLimitError",
    "ConfigurationError",
    "ExternalServiceError",
    "ContentGenerationError",
    "SEOAnalysisError",
]