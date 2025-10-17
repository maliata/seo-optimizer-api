"""
Pydantic data models for SEO optimization API.

This module defines all the input and output schemas for the API endpoints,
including validation rules and documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    KEYWORD_FOCUSED = "keyword_focused"
    BENEFIT_DRIVEN = "benefit_driven"
    EMOTIONAL_APPEAL = "emotional_appeal"
    TECHNICAL_SPECS = "technical_specs"
    BRAND_CENTRIC = "brand_centric"


class ProductInput(BaseModel):
    """Product information input schema."""
    
    current_title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Current product title",
        example="Wireless Bluetooth Headphones"
    )
    
    features: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of product features",
        example=["Noise cancellation", "30-hour battery", "Quick charge"]
    )
    
    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Product category",
        example="Electronics"
    )
    
    brand: Optional[str] = Field(
        None,
        max_length=100,
        description="Product brand name",
        example="TechBrand"
    )
    
    price_range: Optional[str] = Field(
        None,
        max_length=50,
        description="Price range or specific price",
        example="$99-$149"
    )
    
    target_audience: Optional[str] = Field(
        None,
        max_length=200,
        description="Target audience description",
        example="Tech-savvy professionals and music enthusiasts"
    )
    
    keywords: Optional[List[str]] = Field(
        None,
        max_items=15,
        description="Target keywords for SEO",
        example=["wireless headphones", "bluetooth", "noise cancelling"]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features list."""
        if not v:
            raise ValueError("At least one feature is required")
        
        # Remove empty strings and duplicates
        features = list(set([f.strip() for f in v if f.strip()]))
        if not features:
            raise ValueError("At least one non-empty feature is required")
        
        return features
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate keywords list."""
        if v is None:
            return v
        
        # Remove empty strings and duplicates, convert to lowercase
        keywords = list(set([k.strip().lower() for k in v if k.strip()]))
        return keywords if keywords else None


class OptimizationConfig(BaseModel):
    """Configuration for optimization process."""
    
    strategies: List[OptimizationStrategy] = Field(
        default=[OptimizationStrategy.KEYWORD_FOCUSED],
        min_items=1,
        max_items=5,
        description="Optimization strategies to apply",
        example=["keyword_focused", "benefit_driven"]
    )
    
    competitor_analysis: bool = Field(
        default=True,
        description="Enable competitor analysis",
        example=True
    )
    
    max_suggestions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of suggestions to return",
        example=5
    )
    
    include_descriptions: bool = Field(
        default=True,
        description="Include meta descriptions in suggestions",
        example=True
    )
    
    target_title_length: Optional[int] = Field(
        default=60,
        ge=30,
        le=100,
        description="Target title length in characters",
        example=60
    )
    
    target_description_length: Optional[int] = Field(
        default=160,
        ge=120,
        le=200,
        description="Target description length in characters",
        example=160
    )


class CharacterCounts(BaseModel):
    """Character count information."""
    
    title: int = Field(
        ...,
        ge=0,
        description="Title character count",
        example=45
    )
    
    description: int = Field(
        ...,
        ge=0,
        description="Description character count",
        example=155
    )


class TitleSuggestion(BaseModel):
    """Individual title and description suggestion."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Optimized title suggestion",
        example="Premium Wireless Bluetooth Headphones with Active Noise Cancellation"
    )
    
    description: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Optimized meta description",
        example="Experience superior sound quality with our premium wireless Bluetooth headphones featuring active noise cancellation, 30-hour battery life, and quick charge technology. Perfect for professionals and music lovers."
    )
    
    strategy: OptimizationStrategy = Field(
        ...,
        description="Strategy used for this suggestion",
        example="keyword_focused"
    )
    
    seo_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="SEO optimization score (0-100)",
        example=85.5
    )
    
    character_counts: CharacterCounts = Field(
        ...,
        description="Character count information"
    )
    
    keywords_used: List[str] = Field(
        ...,
        description="Keywords incorporated in this suggestion",
        example=["wireless headphones", "bluetooth", "noise cancellation"]
    )
    
    optimization_notes: str = Field(
        ...,
        description="Notes about the optimization approach",
        example="Focused on primary keywords while maintaining readability and including key benefits"
    )


class CompetitorInsight(BaseModel):
    """Competitor analysis insights."""
    
    common_keywords: List[str] = Field(
        ...,
        description="Most common keywords found in competitor titles",
        example=["wireless", "bluetooth", "premium", "noise cancelling"]
    )
    
    title_patterns: List[str] = Field(
        ...,
        description="Common title patterns and structures",
        example=["Brand + Product Type + Key Feature", "Feature + Product + Benefit"]
    )
    
    recommendations: List[str] = Field(
        ...,
        description="Strategic recommendations based on competitor analysis",
        example=["Include 'premium' keyword for positioning", "Emphasize battery life as differentiator"]
    )
    
    market_positioning: Optional[str] = Field(
        None,
        description="Suggested market positioning",
        example="Position as premium alternative with superior battery life"
    )


class SEOAnalysis(BaseModel):
    """SEO analysis and scoring information."""
    
    original_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="SEO score of original title",
        example=65.2
    )
    
    best_suggestion_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="SEO score of best suggestion",
        example=85.5
    )
    
    improvement_percentage: float = Field(
        ...,
        description="Percentage improvement over original",
        example=31.2
    )
    
    keyword_density_analysis: Dict[str, float] = Field(
        ...,
        description="Keyword density analysis",
        example={"wireless": 0.15, "bluetooth": 0.10, "headphones": 0.20}
    )
    
    readability_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Content readability score",
        example=78.5
    )


class OptimizationRequest(BaseModel):
    """Complete optimization request schema."""
    
    product: ProductInput = Field(
        ...,
        description="Product information"
    )
    
    optimization_config: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )


class OptimizationResponse(BaseModel):
    """Complete optimization response schema."""
    
    suggestions: List[TitleSuggestion] = Field(
        ...,
        min_items=1,
        description="List of optimization suggestions"
    )
    
    competitor_insights: Optional[CompetitorInsight] = Field(
        None,
        description="Competitor analysis insights"
    )
    
    seo_analysis: SEOAnalysis = Field(
        ...,
        description="SEO analysis and scoring"
    )
    
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Processing time in seconds",
        example=2.45
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier",
        example="req_123e4567-e89b-12d3-a456-426614174000"
    )
    
    timestamp: str = Field(
        ...,
        description="Response timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(
        ...,
        description="Error type identifier",
        example="VALIDATION_ERROR"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid product information provided"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details",
        example={"field": "features", "issue": "At least one feature is required"}
    )
    
    request_id: str = Field(
        ...,
        description="Request identifier for tracking",
        example="req_123e4567-e89b-12d3-a456-426614174000"
    )
    
    timestamp: str = Field(
        ...,
        description="Error timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )


# Additional utility models for internal processing

class ProcessingContext(BaseModel):
    """Internal processing context."""
    
    request_id: str
    start_time: float
    product_data: ProductInput
    config: OptimizationConfig
    user_ip: Optional[str] = None
    user_agent: Optional[str] = None


class AIProviderResponse(BaseModel):
    """Response from AI provider."""
    
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    processing_time: float
    provider: str


class SEOMetrics(BaseModel):
    """Internal SEO metrics calculation."""
    
    title_length_score: float
    description_length_score: float
    keyword_density_score: float
    readability_score: float
    uniqueness_score: float
    overall_score: float