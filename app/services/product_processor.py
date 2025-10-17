"""
Product data validation and processing service.

This module handles product information validation, normalization,
and preprocessing for SEO optimization.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.models import ProductInput, OptimizationConfig, ProcessingContext
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ProductProcessor:
    """Service for processing and validating product data."""
    
    def __init__(self):
        """Initialize the product processor."""
        self.category_mappings = self._load_category_mappings()
        self.stop_words = self._load_stop_words()
        self.feature_patterns = self._compile_feature_patterns()
    
    def process_product_data(
        self, 
        product: ProductInput, 
        config: OptimizationConfig,
        request_id: str
    ) -> ProcessingContext:
        """
        Process and validate product data for optimization.
        
        Args:
            product: Raw product input data
            config: Optimization configuration
            request_id: Unique request identifier
            
        Returns:
            ProcessingContext: Processed and validated data
        """
        start_time = datetime.utcnow().timestamp()
        
        logger.info(
            "Starting product data processing",
            request_id=request_id,
            product_title=product.current_title,
            category=product.category
        )
        
        try:
            # Normalize and validate product data
            normalized_product = self._normalize_product_data(product)
            
            # Create processing context
            context = ProcessingContext(
                request_id=request_id,
                start_time=start_time,
                product_data=normalized_product,
                config=config
            )
            
            logger.info(
                "Product data processing completed",
                request_id=request_id,
                processing_time=datetime.utcnow().timestamp() - start_time
            )
            
            return context
            
        except Exception as e:
            logger.error(
                "Product data processing failed",
                request_id=request_id,
                error=str(e),
                processing_time=datetime.utcnow().timestamp() - start_time
            )
            raise
    
    def _normalize_product_data(self, product: ProductInput) -> ProductInput:
        """
        Normalize and clean product data.
        
        Args:
            product: Raw product input
            
        Returns:
            ProductInput: Normalized product data
        """
        # Clean and normalize title
        normalized_title = self._clean_text(product.current_title)
        
        # Normalize features
        normalized_features = [
            self._clean_text(feature) for feature in product.features
        ]
        normalized_features = [f for f in normalized_features if f]  # Remove empty
        
        # Normalize category
        normalized_category = self._normalize_category(product.category)
        
        # Clean brand name
        normalized_brand = None
        if product.brand:
            normalized_brand = self._clean_text(product.brand)
        
        # Normalize keywords
        normalized_keywords = None
        if product.keywords:
            normalized_keywords = self._normalize_keywords(product.keywords)
        
        # Clean target audience
        normalized_audience = None
        if product.target_audience:
            normalized_audience = self._clean_text(product.target_audience)
        
        return ProductInput(
            current_title=normalized_title,
            features=normalized_features,
            category=normalized_category,
            brand=normalized_brand,
            price_range=product.price_range,
            target_audience=normalized_audience,
            keywords=normalized_keywords
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-\.\,\!\?\(\)\&]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        return text.strip()
    
    def _normalize_category(self, category: str) -> str:
        """
        Normalize product category using mappings.
        
        Args:
            category: Raw category string
            
        Returns:
            str: Normalized category
        """
        cleaned_category = self._clean_text(category).lower()
        
        # Check for category mappings
        for standard_category, variations in self.category_mappings.items():
            if cleaned_category in variations or cleaned_category == standard_category:
                return standard_category.title()
        
        # Return cleaned version if no mapping found
        return cleaned_category.title()
    
    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """
        Normalize and filter keywords.
        
        Args:
            keywords: List of raw keywords
            
        Returns:
            List[str]: Normalized keywords
        """
        normalized = []
        
        for keyword in keywords:
            # Clean the keyword
            clean_keyword = self._clean_text(keyword).lower()
            
            # Skip if empty or too short
            if not clean_keyword or len(clean_keyword) < 2:
                continue
            
            # Skip stop words
            if clean_keyword in self.stop_words:
                continue
            
            # Add if not already present
            if clean_keyword not in normalized:
                normalized.append(clean_keyword)
        
        return normalized
    
    def extract_key_features(self, product: ProductInput) -> Dict[str, Any]:
        """
        Extract and categorize key product features.
        
        Args:
            product: Product input data
            
        Returns:
            Dict: Categorized features
        """
        features_analysis = {
            'technical_specs': [],
            'benefits': [],
            'materials': [],
            'dimensions': [],
            'performance': [],
            'other': []
        }
        
        for feature in product.features:
            feature_lower = feature.lower()
            categorized = False
            
            # Check against patterns
            for category, patterns in self.feature_patterns.items():
                for pattern in patterns:
                    if pattern.search(feature_lower):
                        features_analysis[category].append(feature)
                        categorized = True
                        break
                if categorized:
                    break
            
            # Add to 'other' if not categorized
            if not categorized:
                features_analysis['other'].append(feature)
        
        return features_analysis
    
    def calculate_content_quality_score(self, product: ProductInput) -> float:
        """
        Calculate a quality score for the input content.
        
        Args:
            product: Product input data
            
        Returns:
            float: Quality score (0-100)
        """
        score = 0.0
        max_score = 100.0
        
        # Title quality (30 points)
        title_score = self._score_title_quality(product.current_title)
        score += title_score * 0.3
        
        # Features quality (25 points)
        features_score = self._score_features_quality(product.features)
        score += features_score * 0.25
        
        # Category specificity (15 points)
        category_score = self._score_category_quality(product.category)
        score += category_score * 0.15
        
        # Brand presence (10 points)
        brand_score = 100.0 if product.brand else 0.0
        score += brand_score * 0.1
        
        # Keywords presence (10 points)
        keywords_score = 100.0 if product.keywords else 0.0
        score += keywords_score * 0.1
        
        # Target audience specificity (10 points)
        audience_score = 100.0 if product.target_audience else 0.0
        score += audience_score * 0.1
        
        return min(score, max_score)
    
    def _score_title_quality(self, title: str) -> float:
        """Score title quality based on various factors."""
        if not title:
            return 0.0
        
        score = 0.0
        
        # Length score (optimal 30-60 characters)
        length = len(title)
        if 30 <= length <= 60:
            score += 40.0
        elif 20 <= length < 30 or 60 < length <= 80:
            score += 25.0
        elif length < 20 or length > 80:
            score += 10.0
        
        # Word count (optimal 4-8 words)
        word_count = len(title.split())
        if 4 <= word_count <= 8:
            score += 30.0
        elif 3 <= word_count < 4 or 8 < word_count <= 10:
            score += 20.0
        else:
            score += 10.0
        
        # Capitalization check
        if title.istitle() or title.isupper():
            score += 15.0
        else:
            score += 5.0
        
        # Special characters penalty
        special_chars = len(re.findall(r'[^\w\s\-\.]', title))
        score -= min(special_chars * 5, 15)
        
        return max(score, 0.0)
    
    def _score_features_quality(self, features: List[str]) -> float:
        """Score features quality."""
        if not features:
            return 0.0
        
        score = 0.0
        
        # Number of features
        feature_count = len(features)
        if 3 <= feature_count <= 8:
            score += 40.0
        elif 2 <= feature_count < 3 or 8 < feature_count <= 12:
            score += 25.0
        else:
            score += 10.0
        
        # Feature specificity
        specific_features = 0
        for feature in features:
            if len(feature.split()) >= 2:  # Multi-word features are more specific
                specific_features += 1
        
        specificity_ratio = specific_features / feature_count
        score += specificity_ratio * 30.0
        
        # Feature diversity (different starting words)
        starting_words = set(feature.split()[0].lower() for feature in features if feature.split())
        diversity_ratio = len(starting_words) / feature_count
        score += diversity_ratio * 30.0
        
        return min(score, 100.0)
    
    def _score_category_quality(self, category: str) -> float:
        """Score category specificity."""
        if not category:
            return 0.0
        
        # More specific categories get higher scores
        word_count = len(category.split())
        if word_count >= 2:
            return 100.0
        elif word_count == 1:
            # Check if it's a known specific category
            if category.lower() in self.category_mappings:
                return 80.0
            else:
                return 60.0
        
        return 0.0
    
    def _load_category_mappings(self) -> Dict[str, List[str]]:
        """Load category normalization mappings."""
        return {
            'electronics': ['electronic', 'tech', 'technology', 'gadget', 'gadgets'],
            'clothing': ['apparel', 'fashion', 'wear', 'garment', 'garments'],
            'home': ['household', 'domestic', 'house', 'living'],
            'sports': ['sport', 'fitness', 'athletic', 'exercise'],
            'beauty': ['cosmetic', 'cosmetics', 'skincare', 'makeup'],
            'automotive': ['auto', 'car', 'vehicle', 'automotive'],
            'books': ['book', 'literature', 'reading'],
            'toys': ['toy', 'games', 'kids', 'children'],
            'health': ['medical', 'healthcare', 'wellness', 'medicine'],
            'food': ['grocery', 'nutrition', 'edible', 'consumable']
        }
    
    def _load_stop_words(self) -> set:
        """Load stop words for keyword filtering."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'these',
            'they', 'we', 'you', 'your', 'our', 'their', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall'
        }
    
    def _compile_feature_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for feature categorization."""
        return {
            'technical_specs': [
                re.compile(r'\d+\s*(gb|mb|tb|ghz|mhz|mp|inch|"|cm|mm)'),
                re.compile(r'(processor|cpu|ram|memory|storage|resolution|dpi)'),
                re.compile(r'(bluetooth|wifi|usb|hdmi|port|connector)'),
            ],
            'benefits': [
                re.compile(r'(easy|simple|convenient|comfortable|efficient)'),
                re.compile(r'(save|saves|saving|reduce|reduces|improve|improves)'),
                re.compile(r'(fast|quick|instant|immediate|rapid)'),
            ],
            'materials': [
                re.compile(r'(cotton|leather|plastic|metal|wood|glass|ceramic)'),
                re.compile(r'(stainless|aluminum|steel|fabric|textile)'),
            ],
            'dimensions': [
                re.compile(r'\d+\s*(x|×)\s*\d+'),
                re.compile(r'(size|dimension|weight|height|width|length)'),
            ],
            'performance': [
                re.compile(r'(battery|power|energy|performance|speed|capacity)'),
                re.compile(r'(hour|hours|day|days|usage|runtime)'),
            ]
        }


# Global instance
product_processor = ProductProcessor()


def get_product_processor() -> ProductProcessor:
    """Get the global product processor instance."""
    return product_processor