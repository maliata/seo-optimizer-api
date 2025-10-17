"""
AI service for content generation using LiteLLM.

This module provides AI-powered content generation with support for multiple
LLM providers through LiteLLM, including OpenAI, Anthropic, and others.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import textstat

import litellm
from litellm import completion, acompletion

from app.models import (
    ProductInput, 
    OptimizationConfig, 
    OptimizationStrategy,
    TitleSuggestion,
    CharacterCounts,
    AIProviderResponse
)
from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.exceptions import AIServiceError, ContentGenerationError
from .cost_tracker import get_cost_tracker

logger = get_logger(__name__)


class AIService:
    """Service for AI-powered content generation using LiteLLM."""
    
    def __init__(self):
        """Initialize the AI service."""
        self.settings = get_settings()
        self._configure_litellm()
        self.prompt_templates = self._load_prompt_templates()
        self.fallback_providers = self._get_fallback_providers()
        self.cost_tracker = get_cost_tracker()
    
    def _configure_litellm(self):
        """Configure LiteLLM settings."""
        # Set API keys from environment
        if self.settings.ai_api_key:
            litellm.api_key = self.settings.ai_api_key
        
        # Configure timeout and retries
        litellm.request_timeout = self.settings.ai_timeout
        litellm.num_retries = self.settings.ai_max_retries
        
        # Enable logging for debugging
        litellm.set_verbose = self.settings.debug
        
        logger.info(
            "LiteLLM configured",
            provider=self.settings.ai_provider,
            model=self.settings.ai_model,
            timeout=self.settings.ai_timeout
        )
    
    def _get_fallback_providers(self) -> List[Dict[str, str]]:
        """Get list of fallback providers in order of preference."""
        fallbacks = []
        
        # Primary provider
        fallbacks.append({
            "provider": self.settings.ai_provider,
            "model": self.settings.ai_model
        })
        
        # Fallback providers based on primary
        if self.settings.ai_provider == "openai":
            fallbacks.extend([
                {"provider": "openai", "model": "gpt-3.5-turbo"},
                {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            ])
        elif self.settings.ai_provider == "anthropic":
            fallbacks.extend([
                {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
                {"provider": "openai", "model": "gpt-3.5-turbo"},
            ])
        else:
            # Default fallbacks
            fallbacks.extend([
                {"provider": "openai", "model": "gpt-3.5-turbo"},
                {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            ])
        
        return fallbacks
    
    async def generate_suggestions(
        self,
        product: ProductInput,
        config: OptimizationConfig,
        request_id: str
    ) -> List[TitleSuggestion]:
        """
        Generate AI-powered title and description suggestions.
        
        Args:
            product: Product information
            config: Optimization configuration
            request_id: Request identifier for tracking
            
        Returns:
            List of AI-generated suggestions
            
        Raises:
            AIServiceError: If AI generation fails
            ContentGenerationError: If content validation fails
        """
        logger.info(
            "Starting AI content generation",
            request_id=request_id,
            strategies=config.strategies,
            max_suggestions=config.max_suggestions
        )
        
        suggestions = []
        
        try:
            # Generate suggestions for each strategy
            for i, strategy in enumerate(config.strategies[:config.max_suggestions]):
                suggestion = await self._generate_single_suggestion(
                    product=product,
                    strategy=strategy,
                    config=config,
                    request_id=request_id,
                    suggestion_index=i
                )
                
                if suggestion:
                    suggestions.append(suggestion)
            
            if not suggestions:
                raise ContentGenerationError(
                    "No valid suggestions generated",
                    attempt_count=len(config.strategies)
                )
            
            logger.info(
                "AI content generation completed",
                request_id=request_id,
                suggestions_count=len(suggestions)
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(
                "AI content generation failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            if isinstance(e, (AIServiceError, ContentGenerationError)):
                raise
            else:
                raise AIServiceError(
                    f"Unexpected error during content generation: {str(e)}",
                    provider=self.settings.ai_provider,
                    model=self.settings.ai_model
                )
    
    async def _generate_single_suggestion(
        self,
        product: ProductInput,
        strategy: OptimizationStrategy,
        config: OptimizationConfig,
        request_id: str,
        suggestion_index: int
    ) -> Optional[TitleSuggestion]:
        """Generate a single suggestion using AI."""
        
        try:
            # Create prompt for the strategy
            prompt = self._create_prompt(product, strategy, config)
            
            # Generate content with fallback
            ai_response = await self._generate_with_fallback(
                prompt=prompt,
                request_id=request_id,
                strategy=strategy.value
            )
            
            # Parse and validate response
            suggestion = self._parse_ai_response(
                ai_response=ai_response,
                strategy=strategy,
                product=product,
                suggestion_index=suggestion_index
            )
            
            return suggestion
            
        except Exception as e:
            logger.warning(
                f"Failed to generate suggestion for strategy {strategy.value}",
                request_id=request_id,
                strategy=strategy.value,
                error=str(e)
            )
            return None
    
    async def _generate_with_fallback(
        self,
        prompt: str,
        request_id: str,
        strategy: str,
        max_attempts: int = 3
    ) -> AIProviderResponse:
        """Generate content with fallback providers."""
        
        last_error = None
        
        for attempt, provider_config in enumerate(self.fallback_providers[:max_attempts]):
            try:
                start_time = time.time()
                
                # Prepare model identifier for LiteLLM
                model = f"{provider_config['provider']}/{provider_config['model']}"
                
                logger.debug(
                    f"Attempting AI generation (attempt {attempt + 1})",
                    request_id=request_id,
                    provider=provider_config['provider'],
                    model=provider_config['model']
                )
                
                # Make async call to LiteLLM
                response = await acompletion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert SEO copywriter specializing in product title and description optimization."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=500,
                    timeout=self.settings.ai_timeout
                )
                
                processing_time = time.time() - start_time
                
                # Extract content from response
                content = response.choices[0].message.content.strip()
                
                # Create AI response object
                ai_response = AIProviderResponse(
                    content=content,
                    model_used=provider_config['model'],
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    processing_time=processing_time,
                    provider=provider_config['provider']
                )
                
                # Record the successful API call
                cost_estimate = self.cost_tracker.record_api_call(
                    provider=provider_config['provider'],
                    model=provider_config['model'],
                    tokens_used=ai_response.tokens_used,
                    processing_time=processing_time,
                    success=True,
                    request_id=request_id
                )
                
                logger.info(
                    "AI generation successful",
                    request_id=request_id,
                    provider=provider_config['provider'],
                    model=provider_config['model'],
                    processing_time=processing_time,
                    tokens_used=ai_response.tokens_used,
                    cost_estimate=cost_estimate
                )
                
                return ai_response
                
            except Exception as e:
                last_error = e
                
                # Record the failed API call
                self.cost_tracker.record_api_call(
                    provider=provider_config['provider'],
                    model=provider_config['model'],
                    tokens_used=0,
                    processing_time=time.time() - start_time,
                    success=False,
                    request_id=request_id
                )
                
                logger.warning(
                    f"AI generation attempt {attempt + 1} failed",
                    request_id=request_id,
                    provider=provider_config['provider'],
                    model=provider_config['model'],
                    error=str(e)
                )
                
                # Wait before retry (exponential backoff)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All attempts failed
        raise AIServiceError(
            f"All AI generation attempts failed. Last error: {str(last_error)}",
            provider=self.fallback_providers[0]['provider'],
            model=self.fallback_providers[0]['model'],
            retry_count=max_attempts
        )
    
    def _create_prompt(
        self,
        product: ProductInput,
        strategy: OptimizationStrategy,
        config: OptimizationConfig
    ) -> str:
        """Create optimized prompt for the given strategy."""
        
        template = self.prompt_templates.get(strategy, self.prompt_templates[OptimizationStrategy.KEYWORD_FOCUSED])
        
        # Prepare context variables
        context = {
            "current_title": product.current_title,
            "features": ", ".join(product.features),
            "category": product.category,
            "brand": product.brand or "the brand",
            "keywords": ", ".join(product.keywords) if product.keywords else "relevant keywords",
            "target_audience": product.target_audience or "general consumers",
            "price_range": product.price_range or "competitive pricing",
            "target_title_length": config.target_title_length or 60,
            "target_description_length": config.target_description_length or 160,
            "strategy_name": strategy.value.replace("_", " ").title()
        }
        
        # Format template with context
        prompt = template.format(**context)
        
        return prompt
    
    def _parse_ai_response(
        self,
        ai_response: AIProviderResponse,
        strategy: OptimizationStrategy,
        product: ProductInput,
        suggestion_index: int
    ) -> TitleSuggestion:
        """Parse AI response into a TitleSuggestion object."""
        
        try:
            # Try to parse as JSON first
            if ai_response.content.strip().startswith('{'):
                parsed = json.loads(ai_response.content)
                title = parsed.get('title', '').strip()
                description = parsed.get('description', '').strip()
                optimization_notes = parsed.get('notes', '')
            else:
                # Parse structured text response
                lines = [line.strip() for line in ai_response.content.split('\n') if line.strip()]
                
                title = ""
                description = ""
                optimization_notes = ""
                
                for line in lines:
                    if line.lower().startswith('title:'):
                        title = line[6:].strip()
                    elif line.lower().startswith('description:'):
                        description = line[12:].strip()
                    elif line.lower().startswith('notes:'):
                        optimization_notes = line[6:].strip()
                
                # If no structured format, try to extract from plain text
                if not title and not description:
                    parts = ai_response.content.split('\n\n')
                    if len(parts) >= 2:
                        title = parts[0].strip()
                        description = parts[1].strip()
                    else:
                        # Fallback: use first line as title
                        first_line = lines[0] if lines else ""
                        title = first_line[:60] if len(first_line) > 60 else first_line
                        description = ai_response.content.replace(title, "").strip()[:160]
            
            # Validate and clean
            if not title:
                raise ContentGenerationError(
                    "No title found in AI response",
                    strategy=strategy.value
                )
            
            if not description:
                # Fallback minimal description if AI omitted it
                base_desc_parts = []
                if product.brand:
                    base_desc_parts.append(product.brand)
                base_desc_parts.append(product.category)
                if product.features:
                    base_desc_parts.append(", ".join(product.features[:2]))
                description = " - ".join([p for p in base_desc_parts if p])[:200]
                if not description:
                    # last resort fallback
                    description = product.current_title[:200]
            
            # Ensure length limits (Phase 4 strict limits)
            if len(title) > 60:
                title = title[:57] + "..."
            
            if len(description) > 160:
                description = description[:157] + "..."
            
            # Calculate character counts
            char_counts = CharacterCounts(
                title=len(title),
                description=len(description)
            )
            
            # Extract keywords used (simple approach)
            keywords_used = []
            if product.keywords:
                for keyword in product.keywords:
                    if keyword.lower() in title.lower() or keyword.lower() in description.lower():
                        keywords_used.append(keyword)
            
            # Calculate basic SEO score (will be enhanced in Phase 4)
            seo_score = self._calculate_basic_seo_score(title, description, keywords_used)
            
            # Create suggestion
            suggestion = TitleSuggestion(
                title=title,
                description=description,
                strategy=strategy,
                seo_score=seo_score,
                character_counts=char_counts,
                keywords_used=keywords_used,
                optimization_notes=optimization_notes or f"AI-generated using {strategy.value} strategy with {ai_response.provider} {ai_response.model_used}"
            )
            
            return suggestion
            
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse AI response as JSON, trying text parsing",
                error=str(e),
                content_preview=ai_response.content[:100]
            )
            # Continue with text parsing above
            
        except Exception as e:
            raise ContentGenerationError(
                f"Failed to parse AI response: {str(e)}",
                strategy=strategy.value,
                attempt_count=1
            )
    
    def _calculate_basic_seo_score(
        self,
        title: str,
        description: str,
        keywords_used: List[str]
    ) -> float:
        """Calculate basic SEO score (enhanced version will be in Phase 4)."""
        
        # Advanced Phase 4 scoring (max 100)
        total = 0.0

        # 1) Title length (max 25)
        title_len = len(title or "")
        if 55 <= title_len <= 60:
            title_len_score = 25.0
        elif 50 <= title_len < 55 or 60 < title_len <= 65:
            title_len_score = 20.0
        elif 40 <= title_len < 50 or 65 < title_len <= 70:
            title_len_score = 12.0
        else:
            title_len_score = 6.0
        total += title_len_score

        # 2) Description length (max 20)
        desc_len = len(description or "")
        if 155 <= desc_len <= 160:
            desc_len_score = 20.0
        elif 150 <= desc_len < 155 or 160 < desc_len <= 170:
            desc_len_score = 16.0
        elif 140 <= desc_len < 150 or 170 < desc_len <= 180:
            desc_len_score = 10.0
        else:
            desc_len_score = 5.0
        total += desc_len_score

        # 3) Keyword usage coverage (max 25)
        kw_score = 0.0
        if keywords_used:
            unique_kw = set([k.lower() for k in keywords_used if k])
            # Reward up to 3 unique keywords
            kw_score = min(len(unique_kw) * 8.5, 25.0)
        total += kw_score

        # 4) Readability using Flesch Reading Ease (max 10)
        readability_component = 0.0
        try:
            # Use description for readability if available, otherwise title
            text_for_readability = description if description else title
            flesch = textstat.flesch_reading_ease(text_for_readability or "")
            # Map 0..100 to 0..10 band (clamped)
            readability_component = max(0.0, min(10.0, (flesch / 10.0)))
        except Exception:
            readability_component = 5.0  # neutral fallback
        total += readability_component

        # 5) Primary keyword early placement (max 10)
        # If any keyword provided, check position of the first one found in title
        keyword_pos_score = 0.0
        normalized_title = (title or "").lower()
        if keywords_used:
            first_pos = None
            for kw in keywords_used:
                idx = normalized_title.find(kw.lower())
                if idx != -1:
                    first_pos = idx if first_pos is None else min(first_pos, idx)
            if first_pos is not None:
                if first_pos <= 15:
                    keyword_pos_score = 10.0
                elif first_pos <= 25:
                    keyword_pos_score = 6.0
                elif first_pos <= 35:
                    keyword_pos_score = 3.0
                else:
                    keyword_pos_score = 1.0
        total += keyword_pos_score

        # 6) Penalties (caps, spammy tokens, missing keywords) - up to -20
        penalties = 0.0
        # All caps (excluding short words and brand acronyms is complex; simple heuristic)
        if title and title.upper() == title and title_len >= 8:
            penalties += 8.0
        # Spammy punctuation or phrases
        spam_patterns = [
            r"!{2,}", r"\bfree\b", r"buy now", r"\b100%\b", r"cheap", r"limited time"
        ]
        combined_text = f"{title or ''} {description or ''}".lower()
        for pat in spam_patterns:
            if re.search(pat, combined_text):
                penalties += 3.0
        # Missing any keyword presence
        if keywords_used:
            if not any(kw.lower() in combined_text for kw in keywords_used):
                penalties += 9.0

        final_score = max(0.0, min(100.0, total - penalties))
        return final_score
    
    def _load_prompt_templates(self) -> Dict[OptimizationStrategy, str]:
        """Load prompt templates for different optimization strategies."""
        
        return {
            OptimizationStrategy.KEYWORD_FOCUSED: """
Create an SEO-optimized product title and description using a {strategy_name} strategy.

Product Information:
- Current Title: {current_title}
- Features: {features}
- Category: {category}
- Brand: {brand}
- Target Keywords: {keywords}
- Target Audience: {target_audience}

Requirements:
- Title: Maximum {target_title_length} characters, include primary keywords early
- Description: Maximum {target_description_length} characters, naturally incorporate keywords
- Focus on search visibility and keyword density
- Maintain readability and natural flow

Please respond in this format:
Title: [Your optimized title here]
Description: [Your optimized description here]
Notes: [Brief explanation of keyword strategy used]
""",

            OptimizationStrategy.BENEFIT_DRIVEN: """
Create an SEO-optimized product title and description using a {strategy_name} strategy.

Product Information:
- Current Title: {current_title}
- Features: {features}
- Category: {category}
- Brand: {brand}
- Target Audience: {target_audience}
- Price Range: {price_range}

Requirements:
- Title: Maximum {target_title_length} characters, highlight key benefits
- Description: Maximum {target_description_length} characters, focus on value proposition
- Emphasize what the customer gains from this product
- Use action-oriented and benefit-focused language

Please respond in this format:
Title: [Your benefit-focused title here]
Description: [Your benefit-focused description here]
Notes: [Brief explanation of benefits highlighted]
""",

            OptimizationStrategy.EMOTIONAL_APPEAL: """
Create an SEO-optimized product title and description using a {strategy_name} strategy.

Product Information:
- Current Title: {current_title}
- Features: {features}
- Category: {category}
- Brand: {brand}
- Target Audience: {target_audience}

Requirements:
- Title: Maximum {target_title_length} characters, use emotional triggers
- Description: Maximum {target_description_length} characters, create emotional connection
- Use persuasive language that evokes feelings
- Include elements of desire, aspiration, or problem-solving

Please respond in this format:
Title: [Your emotionally appealing title here]
Description: [Your emotionally appealing description here]
Notes: [Brief explanation of emotional elements used]
""",

            OptimizationStrategy.TECHNICAL_SPECS: """
Create an SEO-optimized product title and description using a {strategy_name} strategy.

Product Information:
- Current Title: {current_title}
- Features: {features}
- Category: {category}
- Brand: {brand}
- Target Audience: {target_audience}

Requirements:
- Title: Maximum {target_title_length} characters, highlight key technical features
- Description: Maximum {target_description_length} characters, detail specifications
- Use precise technical terminology
- Appeal to technically-minded customers

Please respond in this format:
Title: [Your technical specification-focused title here]
Description: [Your technical specification-focused description here]
Notes: [Brief explanation of technical aspects emphasized]
""",

            OptimizationStrategy.BRAND_CENTRIC: """
Create an SEO-optimized product title and description using a {strategy_name} strategy.

Product Information:
- Current Title: {current_title}
- Features: {features}
- Category: {category}
- Brand: {brand}
- Target Audience: {target_audience}

Requirements:
- Title: Maximum {target_title_length} characters, prominently feature brand
- Description: Maximum {target_description_length} characters, emphasize brand trust and quality
- Highlight brand reputation and authority
- Use language that builds confidence in the brand

Please respond in this format:
Title: [Your brand-centric title here]
Description: [Your brand-centric description here]
Notes: [Brief explanation of brand elements emphasized]
"""
        }


# Global instance
ai_service = AIService()


def get_ai_service() -> AIService:
    """Get the global AI service instance."""
    return ai_service