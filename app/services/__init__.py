"""
Services package for SEO optimization API.

This package contains business logic services for processing requests,
integrating with external APIs, and generating optimized content.
"""

from .product_processor import ProductProcessor, get_product_processor

__all__ = [
    "ProductProcessor",
    "get_product_processor",
]