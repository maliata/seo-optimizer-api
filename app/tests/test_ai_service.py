"""
Tests for AI service integration.

This module contains tests for the AI service functionality,
including mock tests and integration tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from app.services.ai_service import AIService, get_ai_service
from app.models import (
    ProductInput,
    OptimizationConfig,
    OptimizationStrategy,
    TitleSuggestion,
    AIProviderResponse
)
from app.utils.exceptions import AIServiceError, ContentGenerationError


class TestAIService:
    """Test cases for AI service."""
    
    @pytest.fixture
    def ai_service(self):
        """Create AI service instance for testing."""
        return AIService()
    
    @pytest.fixture
    def sample_product(self):
        """Create sample product data for testing."""
        return ProductInput(
            current_title="Wireless Bluetooth Headphones",
            features=["Noise cancellation", "30-hour battery", "Quick charge"],
            category="Electronics",
            brand="TechBrand",
            keywords=["wireless headphones", "bluetooth", "noise cancelling"]
        )
    
    @pytest.fixture
    def sample_config(self):
        """Create sample optimization config for testing."""
        return OptimizationConfig(
            strategies=[OptimizationStrategy.KEYWORD_FOCUSED, OptimizationStrategy.BENEFIT_DRIVEN],
            max_suggestions=2
        )
    
    def test_ai_service_initialization(self, ai_service):
        """Test AI service initializes correctly."""
        assert ai_service is not None
        assert ai_service.prompt_templates is not None
        assert ai_service.fallback_providers is not None
        assert len(ai_service.fallback_providers) > 0
    
    def test_prompt_template_creation(self, ai_service, sample_product, sample_config):
        """Test prompt template creation for different strategies."""
        for strategy in OptimizationStrategy:
            prompt = ai_service._create_prompt(sample_product, strategy, sample_config)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert sample_product.current_title in prompt
            assert sample_product.category in prompt
            assert strategy.value.replace("_", " ").title() in prompt
    
    def test_basic_seo_score_calculation(self, ai_service):
        """Test basic SEO score calculation."""
        title = "Premium Wireless Bluetooth Headphones with Noise Cancellation"
        description = "Experience superior sound quality with our premium wireless Bluetooth headphones featuring active noise cancellation and 30-hour battery life."
        keywords = ["wireless headphones", "bluetooth", "noise cancellation"]
        
        score = ai_service._calculate_basic_seo_score(title, description, keywords)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 0  # Should have some score for valid input
    
    @patch('app.services.ai_service.acompletion')
    @pytest.mark.asyncio
    async def test_generate_with_fallback_success(self, mock_acompletion, ai_service):
        """Test successful AI generation with fallback."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
Title: Premium Wireless Bluetooth Headphones with Active Noise Cancellation
Description: Experience superior sound quality with our premium wireless Bluetooth headphones featuring active noise cancellation, 30-hour battery life, and quick charge technology.
Notes: Focused on primary keywords while highlighting key benefits.
"""
        mock_response.usage.total_tokens = 150
        mock_acompletion.return_value = mock_response
        
        prompt = "Test prompt"
        response = await ai_service._generate_with_fallback(
            prompt=prompt,
            request_id="test-123",
            strategy="keyword_focused"
        )
        
        assert isinstance(response, AIProviderResponse)
        assert response.content is not None
        assert response.tokens_used == 150
        assert response.processing_time > 0
    
    @patch('app.services.ai_service.acompletion')
    @pytest.mark.asyncio
    async def test_generate_with_fallback_failure(self, mock_acompletion, ai_service):
        """Test AI generation failure and fallback behavior."""
        # Mock all attempts failing
        mock_acompletion.side_effect = Exception("API Error")
        
        with pytest.raises(AIServiceError):
            await ai_service._generate_with_fallback(
                prompt="Test prompt",
                request_id="test-123",
                strategy="keyword_focused",
                max_attempts=2
            )
    
    def test_parse_ai_response_json_format(self, ai_service, sample_product, sample_config):
        """Test parsing AI response in JSON format."""
        ai_response = AIProviderResponse(
            content='{"title": "Test Title", "description": "Test Description", "notes": "Test notes"}',
            model_used="gpt-3.5-turbo",
            processing_time=1.5,
            provider="openai"
        )
        
        suggestion = ai_service._parse_ai_response(
            ai_response=ai_response,
            strategy=OptimizationStrategy.KEYWORD_FOCUSED,
            product=sample_product,
            suggestion_index=0
        )
        
        assert isinstance(suggestion, TitleSuggestion)
        assert suggestion.title == "Test Title"
        assert suggestion.description == "Test Description"
        assert suggestion.strategy == OptimizationStrategy.KEYWORD_FOCUSED
    
    def test_parse_ai_response_text_format(self, ai_service, sample_product, sample_config):
        """Test parsing AI response in structured text format."""
        ai_response = AIProviderResponse(
            content="""Title: Premium Wireless Bluetooth Headphones
Description: Experience superior sound quality with advanced features.
Notes: Optimized for keyword focus and readability.""",
            model_used="gpt-3.5-turbo",
            processing_time=1.5,
            provider="openai"
        )
        
        suggestion = ai_service._parse_ai_response(
            ai_response=ai_response,
            strategy=OptimizationStrategy.KEYWORD_FOCUSED,
            product=sample_product,
            suggestion_index=0
        )
        
        assert isinstance(suggestion, TitleSuggestion)
        assert suggestion.title == "Premium Wireless Bluetooth Headphones"
        assert "Experience superior sound quality" in suggestion.description
        assert suggestion.strategy == OptimizationStrategy.KEYWORD_FOCUSED
    
    def test_parse_ai_response_invalid_format(self, ai_service, sample_product, sample_config):
        """Test parsing invalid AI response format."""
        ai_response = AIProviderResponse(
            content="Invalid response format",
            model_used="gpt-3.5-turbo",
            processing_time=1.5,
            provider="openai"
        )
        
        # Should still create a suggestion with fallback parsing
        suggestion = ai_service._parse_ai_response(
            ai_response=ai_response,
            strategy=OptimizationStrategy.KEYWORD_FOCUSED,
            product=sample_product,
            suggestion_index=0
        )
        
        assert isinstance(suggestion, TitleSuggestion)
        assert len(suggestion.title) > 0
    
    @patch('app.services.ai_service.acompletion')
    @pytest.mark.asyncio
    async def test_generate_suggestions_success(self, mock_acompletion, ai_service, sample_product, sample_config):
        """Test successful suggestion generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
Title: Premium Wireless Bluetooth Headphones with Active Noise Cancellation
Description: Experience superior sound quality with our premium wireless Bluetooth headphones featuring active noise cancellation and 30-hour battery life.
Notes: Focused on primary keywords while highlighting key benefits.
"""
        mock_response.usage.total_tokens = 150
        mock_acompletion.return_value = mock_response
        
        suggestions = await ai_service.generate_suggestions(
            product=sample_product,
            config=sample_config,
            request_id="test-123"
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) == len(sample_config.strategies)
        
        for suggestion in suggestions:
            assert isinstance(suggestion, TitleSuggestion)
            assert len(suggestion.title) > 0
            assert len(suggestion.description) > 0
            assert suggestion.seo_score > 0
    
    @patch('app.services.ai_service.acompletion')
    @pytest.mark.asyncio
    async def test_generate_suggestions_partial_failure(self, mock_acompletion, ai_service, sample_product, sample_config):
        """Test suggestion generation with partial failures."""
        # Mock first call succeeding, second failing
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
Title: Premium Wireless Bluetooth Headphones
Description: Experience superior sound quality with advanced features.
Notes: Optimized for keyword focus.
"""
        mock_response.usage.total_tokens = 150
        
        mock_acompletion.side_effect = [mock_response, Exception("API Error")]
        
        suggestions = await ai_service.generate_suggestions(
            product=sample_product,
            config=sample_config,
            request_id="test-123"
        )
        
        # Should get at least one suggestion
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1
    
    def test_get_ai_service_singleton(self):
        """Test that get_ai_service returns the same instance."""
        service1 = get_ai_service()
        service2 = get_ai_service()
        
        assert service1 is service2


# Integration test (requires actual API key)
@pytest.mark.integration
class TestAIServiceIntegration:
    """Integration tests for AI service (requires API keys)."""
    
    @pytest.mark.asyncio
    async def test_real_ai_generation(self):
        """Test real AI generation (requires valid API key)."""
        # Skip if no API key is configured
        import os
        if not os.getenv("AI_API_KEY") or os.getenv("AI_API_KEY") == "your-openai-api-key-here":
            pytest.skip("No valid AI API key configured")
        
        ai_service = AIService()
        
        product = ProductInput(
            current_title="Basic Headphones",
            features=["Good sound", "Comfortable"],
            category="Electronics",
            brand="TestBrand",
            keywords=["headphones", "audio"]
        )
        
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.KEYWORD_FOCUSED],
            max_suggestions=1
        )
        
        suggestions = await ai_service.generate_suggestions(
            product=product,
            config=config,
            request_id="integration-test"
        )
        
        assert len(suggestions) == 1
        suggestion = suggestions[0]
        
        # Verify the suggestion is better than the original
        assert len(suggestion.title) > len(product.current_title)
        assert suggestion.seo_score > 50  # Should be reasonably optimized
        assert any(keyword in suggestion.title.lower() for keyword in product.keywords)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])