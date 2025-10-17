from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import psutil
import os
from datetime import datetime

from app.config import get_settings
from app.utils.logging import get_logger
from app.services import get_cost_tracker

router = APIRouter()
logger = get_logger(__name__)

# Simple metrics storage (in production, use proper metrics system)
class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.ai_api_calls = 0
    
    def record_request(self, response_time: float, success: bool = True):
        """Record request metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        
        if not success:
            self.error_count += 1
    
    def record_ai_call(self):
        """Record AI API call."""
        self.ai_api_calls += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.request_count,
            'average_response_time_seconds': round(avg_response_time, 4),
            'error_rate': round(error_rate, 4),
            'ai_api_calls': self.ai_api_calls,
            'requests_per_second': round(self.request_count / uptime, 2) if uptime > 0 else 0
        }

# Global metrics instance
metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector."""
    return metrics


@router.get("/health")
async def health_check(settings = Depends(get_settings)):
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        - status: Application health status
        - timestamp: Current timestamp
        - version: Application version
        - environment: Current environment info
    """
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "environment": {
                "debug": settings.debug,
                "log_level": settings.log_level
            }
        }
        
        logger.debug("Health check performed", status="healthy")
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/detailed")
async def detailed_health_check(settings = Depends(get_settings)):
    """
    Detailed health check with system information.
    
    Returns comprehensive health and system metrics.
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "process_id": os.getpid()
            },
            "configuration": {
                "ai_provider": settings.ai_provider,
                "ai_model": settings.ai_model,
                "competitor_analysis_enabled": settings.enable_competitor_analysis,
                "rate_limit": {
                    "requests": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window
                }
            }
        }
        
        # Add warnings for resource usage
        warnings = []
        if cpu_percent > 80:
            warnings.append("High CPU usage detected")
        if memory.percent > 85:
            warnings.append("High memory usage detected")
        if (disk.used / disk.total) * 100 > 90:
            warnings.append("Low disk space")
        
        if warnings:
            health_info["warnings"] = warnings
        
        return health_info
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/metrics")
async def get_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
    settings = Depends(get_settings)
):
    """
    Get application metrics.
    
    Returns performance and usage metrics for monitoring.
    """
    if not settings.enable_metrics:
        return {"error": "Metrics collection is disabled"}
    
    try:
        app_metrics = metrics_collector.get_metrics()
        
        # Add system metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_connections": len(psutil.net_connections())
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "application": app_metrics,
            "system": system_metrics
        }
        
    except Exception as e:
        logger.error("Failed to collect metrics", error=str(e))
        return {
            "error": "Failed to collect metrics",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/readiness")
async def readiness_check(settings = Depends(get_settings)):
    """
    Readiness check for Kubernetes deployments.
    
    Checks if the application is ready to serve requests.
    """
    try:
        # Check if required configuration is present
        checks = {
            "ai_configuration": bool(settings.ai_provider and settings.ai_model),
            "logging_configured": True,  # Always true if we reach this point
        }
        
        # Check if AI API key is configured (if required)
        if settings.ai_provider in ['openai', 'anthropic']:
            checks["ai_api_key"] = bool(settings.ai_api_key)
        else:
            checks["ai_api_key"] = True  # Not required for mock provider
        
        all_ready = all(checks.values())
        
        return {
            "ready": all_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/liveness")
async def liveness_check():
    """
    Liveness check for Kubernetes deployments.
    
    Simple check to verify the application is alive.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/costs")
async def get_cost_summary(
    cost_tracker = Depends(get_cost_tracker),
    settings = Depends(get_settings)
):
    """
    Get AI API cost summary and usage statistics.
    
    Returns cost and usage metrics for monitoring AI API consumption.
    """
    if not settings.enable_metrics:
        return {"error": "Cost tracking is disabled"}
    
    try:
        cost_summary = cost_tracker.get_cost_summary()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cost_summary": cost_summary
        }
        
    except Exception as e:
        logger.error("Failed to get cost summary", error=str(e))
        return {
            "error": "Failed to retrieve cost summary",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/costs/recent-calls")
async def get_recent_api_calls(
    limit: int = 10,
    cost_tracker = Depends(get_cost_tracker),
    settings = Depends(get_settings)
):
    """
    Get recent AI API calls for debugging and monitoring.
    
    Args:
        limit: Maximum number of recent calls to return (default: 10, max: 50)
    """
    if not settings.enable_metrics:
        return {"error": "Cost tracking is disabled"}
    
    # Limit the maximum number of calls that can be requested
    limit = min(max(1, limit), 50)
    
    try:
        recent_calls = cost_tracker.get_recent_calls(limit=limit)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "recent_calls": recent_calls,
            "total_calls_returned": len(recent_calls)
        }
        
    except Exception as e:
        logger.error("Failed to get recent API calls", error=str(e))
        return {
            "error": "Failed to retrieve recent API calls",
            "timestamp": datetime.utcnow().isoformat()
        }