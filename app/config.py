from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Application settings
    app_name: str = "SEO Optimization API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # AI Service settings
    ai_provider: str = "openai"
    ai_model: str = "gpt-3.5-turbo"
    ai_api_key: Optional[str] = None
    ai_base_url: Optional[str] = None
    ai_timeout: int = 30
    ai_max_retries: int = 3
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds
    
    # Security settings
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = ["*"]
    
    # External services
    enable_competitor_analysis: bool = True
    scraping_delay: float = 1.0
    max_concurrent_requests: int = 10
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Environment-specific configurations
class DevelopmentConfig(Settings):
    debug: bool = True
    log_level: str = "DEBUG"
    ai_model: str = "gpt-3.5-turbo"


class ProductionConfig(Settings):
    debug: bool = False
    log_level: str = "INFO"
    workers: int = 4
    ai_model: str = "gpt-4"


class TestingConfig(Settings):
    debug: bool = True
    log_level: str = "DEBUG"
    ai_provider: str = "mock"
    enable_competitor_analysis: bool = False


def get_config_by_environment(env: str = None) -> Settings:
    """Get configuration based on environment."""
    import os
    
    env = env or os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(env, DevelopmentConfig)
    return config_class()