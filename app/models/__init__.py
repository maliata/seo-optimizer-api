"""
Data models package for SEO optimization API.

This package contains all Pydantic models used for request/response validation,
internal data processing, and API documentation.
"""

from .schemas import (
    # Enums
    OptimizationStrategy,
    
    # Input models
    ProductInput,
    OptimizationConfig,
    OptimizationRequest,
    
    # Output models
    TitleSuggestion,
    CompetitorInsight,
    SEOAnalysis,
    OptimizationResponse,
    CharacterCounts,
    
    # Error models
    ErrorResponse,
    
    # Internal processing models
    ProcessingContext,
    AIProviderResponse,
    SEOMetrics,
)

__all__ = [
    # Enums
    "OptimizationStrategy",
    
    # Input models
    "ProductInput",
    "OptimizationConfig", 
    "OptimizationRequest",
    
    # Output models
    "TitleSuggestion",
    "CompetitorInsight",
    "SEOAnalysis",
    "OptimizationResponse",
    "CharacterCounts",
    
    # Error models
    "ErrorResponse",
    
    # Internal processing models
    "ProcessingContext",
    "AIProviderResponse",
    "SEOMetrics",
]