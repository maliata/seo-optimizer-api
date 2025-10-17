"""
Custom exception classes for SEO optimization API.

This module defines custom exceptions for better error handling and
more specific error responses.
"""

from typing import Optional, Dict, Any


class SEOOptimizationError(Exception):
    """Base exception for SEO optimization API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        """
        Initialize SEO optimization error.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            status_code: HTTP status code
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code


class ValidationError(SEOOptimizationError):
    """Exception for input validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            details: Additional validation details
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["invalid_value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=error_details,
            status_code=400
        )
        self.field = field
        self.value = value


class ProductProcessingError(SEOOptimizationError):
    """Exception for product data processing errors."""
    
    def __init__(
        self,
        message: str,
        product_title: Optional[str] = None,
        processing_stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize product processing error.
        
        Args:
            message: Error message
            product_title: Title of product being processed
            processing_stage: Stage where error occurred
            details: Additional processing details
        """
        error_details = details or {}
        if product_title:
            error_details["product_title"] = product_title
        if processing_stage:
            error_details["processing_stage"] = processing_stage
        
        super().__init__(
            message=message,
            error_code="PRODUCT_PROCESSING_ERROR",
            details=error_details,
            status_code=422
        )
        self.product_title = product_title
        self.processing_stage = processing_stage


class AIServiceError(SEOOptimizationError):
    """Exception for AI service integration errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retry_count: int = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize AI service error.
        
        Args:
            message: Error message
            provider: AI provider name
            model: AI model name
            retry_count: Number of retries attempted
            details: Additional service details
        """
        error_details = details or {}
        if provider:
            error_details["provider"] = provider
        if model:
            error_details["model"] = model
        if retry_count > 0:
            error_details["retry_count"] = retry_count
        
        super().__init__(
            message=message,
            error_code="AI_SERVICE_ERROR",
            details=error_details,
            status_code=503
        )
        self.provider = provider
        self.model = model
        self.retry_count = retry_count


class RateLimitError(SEOOptimizationError):
    """Exception for rate limiting errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            limit: Rate limit threshold
            window: Rate limit window in seconds
            retry_after: Seconds to wait before retry
            details: Additional rate limit details
        """
        error_details = details or {}
        if limit:
            error_details["rate_limit"] = limit
        if window:
            error_details["window_seconds"] = window
        if retry_after:
            error_details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=error_details,
            status_code=429
        )
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class ConfigurationError(SEOOptimizationError):
    """Exception for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that's invalid
            expected_type: Expected configuration type
            details: Additional configuration details
        """
        error_details = details or {}
        if config_key:
            error_details["config_key"] = config_key
        if expected_type:
            error_details["expected_type"] = expected_type
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=error_details,
            status_code=500
        )
        self.config_key = config_key
        self.expected_type = expected_type


class ExternalServiceError(SEOOptimizationError):
    """Exception for external service integration errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_url: Optional[str] = None,
        response_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize external service error.
        
        Args:
            message: Error message
            service_name: Name of external service
            service_url: URL of external service
            response_code: HTTP response code from service
            details: Additional service details
        """
        error_details = details or {}
        if service_name:
            error_details["service_name"] = service_name
        if service_url:
            error_details["service_url"] = service_url
        if response_code:
            error_details["response_code"] = response_code
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=error_details,
            status_code=502
        )
        self.service_name = service_name
        self.service_url = service_url
        self.response_code = response_code


class ContentGenerationError(SEOOptimizationError):
    """Exception for content generation errors."""
    
    def __init__(
        self,
        message: str,
        strategy: Optional[str] = None,
        attempt_count: int = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize content generation error.
        
        Args:
            message: Error message
            strategy: Optimization strategy being used
            attempt_count: Number of generation attempts
            details: Additional generation details
        """
        error_details = details or {}
        if strategy:
            error_details["strategy"] = strategy
        if attempt_count > 0:
            error_details["attempt_count"] = attempt_count
        
        super().__init__(
            message=message,
            error_code="CONTENT_GENERATION_ERROR",
            details=error_details,
            status_code=422
        )
        self.strategy = strategy
        self.attempt_count = attempt_count


class SEOAnalysisError(SEOOptimizationError):
    """Exception for SEO analysis errors."""
    
    def __init__(
        self,
        message: str,
        analysis_type: Optional[str] = None,
        content_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SEO analysis error.
        
        Args:
            message: Error message
            analysis_type: Type of analysis that failed
            content_length: Length of content being analyzed
            details: Additional analysis details
        """
        error_details = details or {}
        if analysis_type:
            error_details["analysis_type"] = analysis_type
        if content_length:
            error_details["content_length"] = content_length
        
        super().__init__(
            message=message,
            error_code="SEO_ANALYSIS_ERROR",
            details=error_details,
            status_code=422
        )
        self.analysis_type = analysis_type
        self.content_length = content_length