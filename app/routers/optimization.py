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
import re
import textstat

from app.models import (
    OptimizationRequest,
    OptimizationResponse,
    ErrorResponse,
    TitleSuggestion,
    SEOAnalysis,
    CharacterCounts,
    OptimizationStrategy
)
from app.services import get_product_processor, get_ai_service
from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.exceptions import AIServiceError, ContentGenerationError

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
    product_processor = Depends(get_product_processor),
    ai_service = Depends(get_ai_service)
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
        
        # Generate AI-powered suggestions
        try:
            suggestions = await ai_service.generate_suggestions(
                product=processing_context.product_data,
                config=request.optimization_config,
                request_id=request_id
            )
        except (AIServiceError, ContentGenerationError) as e:
            logger.warning(
                "AI generation failed, falling back to mock suggestions",
                request_id=request_id,
                error=str(e)
            )
            # Fallback to mock suggestions if AI fails
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
    
    except (AIServiceError, ContentGenerationError) as e:
        logger.error(
            "AI service error in optimization request",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            processing_time=time.time() - start_time
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI_SERVICE_ERROR",
                "message": "AI content generation is temporarily unavailable. Please try again later.",
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
    """Calculate SEO analysis for the optimization results (Phase 4)."""

    # Helper scoring components aligned with AIService advanced scoring
    def _title_length_score(n: int) -> float:
        if 55 <= n <= 60:
            return 25.0
        elif 50 <= n < 55 or 60 < n <= 65:
            return 20.0
        elif 40 <= n < 50 or 65 < n <= 70:
            return 12.0
        return 6.0

    def _description_length_score(n: int) -> float:
        if 155 <= n <= 160:
            return 20.0
        elif 150 <= n < 155 or 160 < n <= 170:
            return 16.0
        elif 140 <= n < 150 or 170 < n <= 180:
            return 10.0
        return 5.0

    def _keyword_usage_score(used: list[str]) -> float:
        uniq = set([k.lower() for k in used if k])
        return min(len(uniq) * 8.5, 25.0)

    def _readability_component(text: str) -> tuple[float, float]:
        """
        Returns (component_score_0_to_10, readability_0_to_100)
        """
        try:
            flesch = textstat.flesch_reading_ease(text or "")
            return max(0.0, min(10.0, (flesch / 10.0))), max(0.0, min(100.0, flesch))
        except Exception:
            return 5.0, 50.0

    def _keyword_position_score(title: str, kw_list: list[str]) -> float:
        pos_score = 0.0
        if not title or not kw_list:
            return pos_score
        normalized = title.lower()
        first_pos = None
        for kw in kw_list:
            idx = normalized.find(kw.lower())
            if idx != -1:
                first_pos = idx if first_pos is None else min(first_pos, idx)
        if first_pos is not None:
            if first_pos <= 15:
                pos_score = 10.0
            elif first_pos <= 25:
                pos_score = 6.0
            elif first_pos <= 35:
                pos_score = 3.0
            else:
                pos_score = 1.0
        return pos_score

    def _penalties(title: str, description: str, kw_list: list[str]) -> tuple[float, Dict[str, float]]:
        total = 0.0
        details: Dict[str, float] = {}
        tl = len(title or "")
        # All caps penalty
        if title and title.upper() == title and tl >= 8:
            details["all_caps_title"] = -8.0
            total += 8.0
        # Spammy patterns
        spam_patterns = [
            r"!{2,}", r"\bfree\b", r"buy now", r"\b100%\b", r"cheap", r"limited time"
        ]
        combined = f"{title or ''} {description or ''}".lower()
        for pat in spam_patterns:
            if re.search(pat, combined):
                details[f"spam:{pat}"] = -3.0
                total += 3.0
        # Missing keyword presence
        if kw_list:
            if not any(kw.lower() in combined for kw in kw_list):
                details["missing_keywords"] = -9.0
                total += 9.0
        return total, details

    # Choose best suggestion by provided seo_score
    best: TitleSuggestion | None = None
    if suggestions:
        best = max(suggestions, key=lambda s: s.seo_score)

    # Compute component scores for best suggestion
    if best:
        title_len = best.character_counts.title
        desc_len = best.character_counts.description
        title_len_score = _title_length_score(title_len)
        desc_len_score = _description_length_score(desc_len)
        kw_usage_score = _keyword_usage_score(best.keywords_used)
        read_comp, read_100 = _readability_component(best.description or best.title)
        kw_pos_score = _keyword_position_score(best.title, keywords or best.keywords_used)
        penalties_total, penalties_map = _penalties(best.title, best.description, keywords or best.keywords_used)

        # Aggregate aligned with AIService logic (max 100 then minus penalties)
        component_total = title_len_score + desc_len_score + kw_usage_score + read_comp + kw_pos_score
        aggregated_score = max(0.0, min(100.0, component_total - penalties_total))
        best_score = max(best.seo_score, aggregated_score)

        component_scores = {
            "title_length": round(title_len_score, 2),
            "description_length": round(desc_len_score, 2),
            "keyword_usage": round(kw_usage_score, 2),
            "readability": round(read_comp, 2),
            "keyword_position": round(kw_pos_score, 2)
        }
        readability_score = round(read_100, 2)
        keyword_position_score = round(kw_pos_score, 2)
        penalties_dict = {k: round(v, 2) for k, v in penalties_map.items()}
        notes = "Scores aggregated with Phase 4 model; penalties applied where applicable."
    else:
        # No suggestions; fall back gracefully
        best_score = 0.0
        component_scores = {}
        readability_score = 50.0
        keyword_position_score = 0.0
        penalties_dict = {}
        notes = "No suggestions available; analysis limited."

    # Original title baseline score (length + readability only)
    orig_title_len_score = _title_length_score(len(original_title or ""))
    _, orig_read_100 = _readability_component(original_title or "")
    original_score = max(0.0, min(100.0, orig_title_len_score + (orig_read_100 / 10.0)))

    # Improvement
    improvement = ((best_score - original_score) / original_score) * 100 if original_score > 0 else 0.0

    # Keyword density analysis (simple proportion in best suggestion text)
    keyword_density: Dict[str, float] = {}
    if best and keywords:
        text = f"{best.title} {best.description}".lower()
        for keyword in keywords[:5]:
            if keyword:
                occurrences = text.count(keyword.lower())
                # naive density: occurrence per 100 words
                words = max(1, len(text.split()))
                density = min(1.0, (occurrences / words) * 100.0)
                keyword_density[keyword] = round(density, 4)

    return SEOAnalysis(
        original_score=round(original_score, 2),
        best_suggestion_score=round(best_score, 2),
        improvement_percentage=round(improvement, 4),
        keyword_density_analysis=keyword_density,
        readability_score=readability_score,
        component_scores=component_scores if component_scores else None,
        keyword_position_score=keyword_position_score if keyword_position_score else None,
        penalties=penalties_dict if penalties_dict else None,
        notes=notes
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