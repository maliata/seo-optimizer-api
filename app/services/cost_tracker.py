"""
Cost tracking service for AI API usage.

This module tracks AI API usage, costs, and provides analytics
for monitoring and optimization.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class APICall:
    """Represents a single API call for cost tracking."""
    
    timestamp: float
    provider: str
    model: str
    tokens_used: Optional[int]
    processing_time: float
    success: bool
    cost_estimate: float = 0.0
    request_id: str = ""


@dataclass
class CostMetrics:
    """Cost and usage metrics."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_processing_time: float = 0.0
    calls_by_provider: Dict[str, int] = field(default_factory=dict)
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    cost_by_provider: Dict[str, float] = field(default_factory=dict)


class CostTracker:
    """Service for tracking AI API costs and usage."""
    
    def __init__(self):
        """Initialize the cost tracker."""
        self.api_calls: List[APICall] = []
        self.model_costs = self._load_model_costs()
        self.start_time = time.time()
    
    def record_api_call(
        self,
        provider: str,
        model: str,
        tokens_used: Optional[int],
        processing_time: float,
        success: bool,
        request_id: str = ""
    ) -> float:
        """
        Record an API call and return estimated cost.
        
        Args:
            provider: AI provider name
            model: Model name used
            tokens_used: Number of tokens consumed
            processing_time: Time taken for the call
            success: Whether the call was successful
            request_id: Request identifier
            
        Returns:
            Estimated cost for this call
        """
        timestamp = time.time()
        
        # Calculate cost estimate
        cost_estimate = self._calculate_cost(provider, model, tokens_used or 0)
        
        # Create API call record
        api_call = APICall(
            timestamp=timestamp,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            processing_time=processing_time,
            success=success,
            cost_estimate=cost_estimate,
            request_id=request_id
        )
        
        # Store the call
        self.api_calls.append(api_call)
        
        # Log the call
        logger.info(
            "API call recorded",
            request_id=request_id,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            processing_time=processing_time,
            success=success,
            cost_estimate=cost_estimate
        )
        
        # Clean old records (keep last 1000 calls)
        if len(self.api_calls) > 1000:
            self.api_calls = self.api_calls[-1000:]
        
        return cost_estimate
    
    def get_metrics(self, hours: int = 24) -> CostMetrics:
        """
        Get cost and usage metrics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            CostMetrics with aggregated data
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_calls = [call for call in self.api_calls if call.timestamp >= cutoff_time]
        
        if not recent_calls:
            return CostMetrics()
        
        # Calculate metrics
        total_calls = len(recent_calls)
        successful_calls = sum(1 for call in recent_calls if call.success)
        failed_calls = total_calls - successful_calls
        total_tokens = sum(call.tokens_used or 0 for call in recent_calls)
        total_cost = sum(call.cost_estimate for call in recent_calls)
        
        # Calculate average processing time
        processing_times = [call.processing_time for call in recent_calls if call.processing_time > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Group by provider and model
        calls_by_provider = {}
        calls_by_model = {}
        cost_by_provider = {}
        
        for call in recent_calls:
            # Provider counts
            calls_by_provider[call.provider] = calls_by_provider.get(call.provider, 0) + 1
            cost_by_provider[call.provider] = cost_by_provider.get(call.provider, 0.0) + call.cost_estimate
            
            # Model counts
            model_key = f"{call.provider}/{call.model}"
            calls_by_model[model_key] = calls_by_model.get(model_key, 0) + 1
        
        return CostMetrics(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_tokens=total_tokens,
            total_cost=total_cost,
            average_processing_time=avg_processing_time,
            calls_by_provider=calls_by_provider,
            calls_by_model=calls_by_model,
            cost_by_provider=cost_by_provider
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs and usage."""
        metrics_24h = self.get_metrics(24)
        metrics_1h = self.get_metrics(1)
        
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            "uptime_hours": round(uptime_hours, 2),
            "last_24_hours": {
                "total_calls": metrics_24h.total_calls,
                "successful_calls": metrics_24h.successful_calls,
                "failed_calls": metrics_24h.failed_calls,
                "success_rate": round(metrics_24h.successful_calls / metrics_24h.total_calls * 100, 2) if metrics_24h.total_calls > 0 else 0,
                "total_tokens": metrics_24h.total_tokens,
                "total_cost_usd": round(metrics_24h.total_cost, 4),
                "average_processing_time": round(metrics_24h.average_processing_time, 3),
                "calls_by_provider": metrics_24h.calls_by_provider,
                "cost_by_provider": {k: round(v, 4) for k, v in metrics_24h.cost_by_provider.items()}
            },
            "last_hour": {
                "total_calls": metrics_1h.total_calls,
                "total_cost_usd": round(metrics_1h.total_cost, 4),
                "calls_per_hour": metrics_1h.total_calls,
                "cost_per_hour": round(metrics_1h.total_cost, 4)
            }
        }
    
    def _calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """
        Calculate estimated cost for an API call.
        
        Args:
            provider: AI provider name
            model: Model name
            tokens: Number of tokens used
            
        Returns:
            Estimated cost in USD
        """
        if tokens <= 0:
            return 0.0
        
        # Get cost per 1K tokens
        model_key = f"{provider}/{model}"
        cost_per_1k = self.model_costs.get(model_key, self.model_costs.get(model, 0.002))
        
        # Calculate cost (tokens / 1000 * cost_per_1k)
        estimated_cost = (tokens / 1000.0) * cost_per_1k
        
        return estimated_cost
    
    def _load_model_costs(self) -> Dict[str, float]:
        """
        Load model cost estimates (cost per 1K tokens).
        
        Returns:
            Dictionary mapping model names to costs per 1K tokens
        """
        return {
            # OpenAI models (approximate costs per 1K tokens)
            "openai/gpt-4": 0.03,
            "openai/gpt-4-turbo": 0.01,
            "openai/gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            
            # Anthropic models
            "anthropic/claude-3-opus-20240229": 0.015,
            "anthropic/claude-3-sonnet-20240229": 0.003,
            "anthropic/claude-3-haiku-20240307": 0.00025,
            "claude-3-opus-20240229": 0.015,
            "claude-3-sonnet-20240229": 0.003,
            "claude-3-haiku-20240307": 0.00025,
            
            # Default fallback cost
            "default": 0.002
        }
    
    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent API calls for debugging.
        
        Args:
            limit: Maximum number of calls to return
            
        Returns:
            List of recent API calls
        """
        recent_calls = sorted(self.api_calls, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "timestamp": datetime.fromtimestamp(call.timestamp).isoformat(),
                "provider": call.provider,
                "model": call.model,
                "tokens_used": call.tokens_used,
                "processing_time": round(call.processing_time, 3),
                "success": call.success,
                "cost_estimate": round(call.cost_estimate, 6),
                "request_id": call.request_id
            }
            for call in recent_calls
        ]


# Global instance
cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return cost_tracker