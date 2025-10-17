"""
SEO optimization endpoint router.

This module contains the main API endpoint for SEO title and description optimization.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import time
from datetime import datetime
import uuid

from app.models import (
    OptimizationRequest,
    OptimizationResponse,
    ErrorResponse,
    TitleSuggestion,
    SEOAnalysis,
    CharacterCounts,
    OptimizationStrategy
)
from app.services import get_product_processor
from app.config import get_settings
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/optimize-title",
    response_model=OptimizationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Optimize product title and description",
    description="""
    Generate SEO-optimized title and description suggestions for a product.
    
    This endpoint analyzes product information and generates multiple optimization
    suggestions using different strategies such as keyword-focused, benefit-driven,
    and emotional appeal approaches.
    
    **Features:**
    - Multiple optimization strategies
    - SEO scoring and analysis
    - Character count optimization
    - Keyword integration
    - Competitor analysis (when enabled)
    
    **Processing Time:** Typically 2-5 seconds depending on configuration
    """
)
async def optimize_title(
    request: OptimizationRequest,
    http_request: Request,
    settings = Depends(get_settings),
    product_processor = Depends(get_product_processor)
) -> OptimizationResponse:
    """
    Optimize product title and description for SEO.
    
    Args:
        request: Optimization request with product data and configuration
        http_request: FastAPI request object for metadata
        settings: Application settings
        product_processor: Product processing service
        
    Returns:
        OptimizationResponse: Optimized suggestions with analysis
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', str(uuid.uuid4()))
    
    logger.info(
        "Starting SEO optimization request",
        request_id=request_id,
        product_title=request.product.current_title,
        strategies=request.optimization_config.strategies,
        max_suggestions=request.optimization_config.max_suggestions
    )
    
    try:
        # Process and validate product data
        processing_context = product_processor.process_product_data(
            product=request.product,
            config=request.optimization_config,
            request_id=request_id
        )
        
        # For Phase 2, we'll create mock suggestions since AI integration comes in Phase 3
        suggestions = await _generate_mock_suggestions(
            processing_context,
            request.optimization_config,
            request_id
        )
        
        # Calculate SEO analysis
        seo_analysis = _calculate_seo_analysis(
            original_title=request.product.current_title,
            suggestions=suggestions,
            keywords=request.product.keywords or []
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = OptimizationResponse(
            suggestions=suggestions,
            competitor_insights=None,  # Will be implemented in Phase 5
            seo_analysis=seo_analysis,
            processing_time=processing_time,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(
            "SEO optimization completed successfully",
            request_id=request_id,
            suggestions_count=len(suggestions),
            processing_time=processing_time,
            best_score=max(s.seo_score for s in suggestions) if suggestions else 0
        )
        
        return response
        
    except ValueError as e:
        logger.warning(
            "Validation error in optimization request",
            request_id=request_id,
            error=str(e),
            processing_time=time.time() - start_time
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
        
    except Exception as e:
        logger.error(
            "Unexpected error in optimization request",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            processing_time=time.time() - start_time
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PROCESSING_ERROR",
                "message": "An error occurred while processing your optimization request",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


async def _generate_mock_suggestions(
    context,
    config,
    request_id: str
) -> list[TitleSuggestion]:
    """
    Generate mock suggestions for Phase 2 testing.
    
    In Phase 3, this will be replaced with actual AI-powered generation.
    """
    logger.info(
        "Generating mock suggestions for testing",
        request_id=request_id,
        strategies=config.strategies
    )
    
    product = context.product_data
    suggestions = []
    
    # Base templates for different strategies
    strategy_templates = {
        OptimizationStrategy.KEYWORD_FOCUSED: [
            "{keywords} - {title}",
            "{title} | {keywords}",
            "Best {keywords} - {title}"
        ],
        OptimizationStrategy.BENEFIT_DRIVEN: [
            "{title} - {benefit}",
            "Get {benefit} with {title}",
            "{title}: {benefit}"
        ],
        OptimizationStrategy.EMOTIONAL_APPEAL: [
            "Amazing {title} You'll Love",
            "Transform Your Life with {title}",
            "Discover the Perfect {title}"
        ],
        OptimizationStrategy.TECHNICAL_SPECS: [
            "{title} - {specs}",
            "Professional {title} with {specs}",
            "{specs} {title}"
        ],
        OptimizationStrategy.BRAND_CENTRIC: [
            "{brand} {title} - Premium Quality",
            "Trusted {brand} {title}",
            "{brand}: {title}"
        ]
    }
    
    # Generate suggestions for each strategy
    for i, strategy in enumerate(config.strategies[:config.max_suggestions]):
        templates = strategy_templates.get(strategy, strategy_templates[OptimizationStrategy.KEYWORD_FOCUSED])
        template = templates[i % len(templates)]
        
        # Create mock optimized title
        optimized_title = _create_mock_title(template, product, strategy)
        
        # Create mock description
        optimized_description = _create_mock_description(product, strategy)
        
        # Calculate character counts
        char_counts = CharacterCounts(
            title=len(optimized_title),
            description=len(optimized_description)
        )
        
        # Mock SEO score (will be calculated properly in Phase 4)
        seo_score = 75.0 + (i * 2.5)  # Incrementally better scores
        
        # Extract keywords used
        keywords_used = product.keywords[:3] if product.keywords else ["quality", "premium"]
        
        suggestion = TitleSuggestion(
            title=optimized_title,
            description=optimized_description,
            strategy=strategy,
            seo_score=min(seo_score, 100.0),
            character_counts=char_counts,
            keywords_used=keywords_used,
            optimization_notes=f"Applied {strategy.value} strategy with focus on {', '.join(keywords_used)}"
        )
        
        suggestions.append(suggestion)
    
    return suggestions


def _create_mock_title(template: str, product, strategy: OptimizationStrategy) -> str:
    """Create a mock optimized title based on strategy."""
    replacements = {
        'title': product.current_title,
        'keywords': ', '.join(product.keywords[:2]) if product.keywords else 'Premium Quality',
        'benefit': product.features[0] if product.features else 'Superior Performance',
        'specs': product.features[0] if product.features else 'Advanced Technology',
        'brand': product.brand or 'Premium'
    }
    
    # Apply template
    mock_title = template
    for key, value in replacements.items():
        mock_title = mock_title.replace(f'{{{key}}}', value)
    
    # Ensure reasonable length
    if len(mock_title) > 60:
        mock_title = mock_title[:57] + "..."
    
    return mock_title


def _create_mock_description(product, strategy: OptimizationStrategy) -> str:
    """Create a mock optimized description based on strategy."""
    base_desc = f"Discover our {product.current_title.lower()} featuring {', '.join(product.features[:3])}."
    
    strategy_additions = {
        OptimizationStrategy.KEYWORD_FOCUSED: f" Perfect for {', '.join(product.keywords[:2])} enthusiasts." if product.keywords else " Ideal for quality seekers.",
        OptimizationStrategy.BENEFIT_DRIVEN: f" Experience the benefits of {product.features[0].lower()} technology." if product.features else " Experience superior performance.",
        OptimizationStrategy.EMOTIONAL_APPEAL: " Transform your experience with this amazing product that exceeds expectations.",
        OptimizationStrategy.TECHNICAL_SPECS: f" Advanced specifications include {product.features[0].lower()} for optimal performance." if product.features else " Advanced technology for optimal performance.",
        OptimizationStrategy.BRAND_CENTRIC: f" From {product.brand}, a trusted name in quality." if product.brand else " From a trusted brand in quality."
    }
    
    addition = strategy_additions.get(strategy, strategy_additions[OptimizationStrategy.KEYWORD_FOCUSED])
    mock_description = base_desc + addition
    
    # Ensure reasonable length
    if len(mock_description) > 160:
        mock_description = mock_description[:157] + "..."
    
    return mock_description


def _calculate_seo_analysis(
    original_title: str,
    suggestions: list[TitleSuggestion],
    keywords: list[str]
) -> SEOAnalysis:
    """Calculate SEO analysis for the optimization results."""
    
    # Mock original score calculation (will be properly implemented in Phase 4)
    original_score = 45.0 + (len(original_title) * 0.5)  # Simple mock calculation
    original_score = min(original_score, 100.0)
    
    # Get best suggestion score
    best_score = max(s.seo_score for s in suggestions) if suggestions else original_score
    
    # Calculate improvement
    improvement = ((best_score - original_score) / original_score) * 100 if original_score > 0 else 0
    
    # Mock keyword density analysis
    keyword_density = {}
    if keywords:
        for keyword in keywords[:5]:  # Limit to top 5 keywords
            keyword_density[keyword] = 0.15 + (hash(keyword) % 10) * 0.01  # Mock density
    
    return SEOAnalysis(
        original_score=original_score,
        best_suggestion_score=best_score,
        improvement_percentage=improvement,
        keyword_density_analysis=keyword_density,
        readability_score=78.5  # Mock readability score
    )


@router.get(
    "/strategies",
    summary="Get available optimization strategies",
    description="Returns a list of all available optimization strategies with descriptions."
)
async def get_optimization_strategies() -> Dict[str, Any]:
    """Get available optimization strategies."""
    
    strategies = {
        "keyword_focused": {
            "name": "Keyword Focused",
            "description": "Emphasizes primary keywords and search terms for better search visibility",
            "best_for": "Products with clear target keywords and search terms"
        },
        "benefit_driven": {
            "name": "Benefit Driven", 
            "description": "Highlights product benefits and value propositions",
            "best_for": "Products with clear customer benefits and value propositions"
        },
        "emotional_appeal": {
            "name": "Emotional Appeal",
            "description": "Uses persuasive language and emotional triggers",
            "best_for": "Consumer products where emotional connection matters"
        },
        "technical_specs": {
            "name": "Technical Specifications",
            "description": "Focuses on technical features and specifications",
            "best_for": "Technical products where specifications are important"
        },
        "brand_centric": {
            "name": "Brand Centric",
            "description": "Emphasizes brand authority and trust signals",
            "best_for": "Established brands with strong market recognition"
        }
    }
    
    return {
        "strategies": strategies,
        "default_strategy": "keyword_focused",
        "max_strategies_per_request": 5
    }