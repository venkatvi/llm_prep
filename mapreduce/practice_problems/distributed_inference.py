"""
Distributed Model Inference Pipeline using MapReduce

Business Context:
E-commerce company needs to score all products for recommendation engine using a trained ML model.

Problem: Run inference on 1B+ products distributed across HDFS
- Input: Product features stored in CSV files across multiple machines
- Output: Product scores and rankings for recommendation system
- Scale: 1B+ products, ~10ms inference time per product
- Challenge: Distribute model and minimize data movement

MapReduce Strategy:
1. Map Phase: Load model on each mapper, run inference on product batches
2. Shuffle Phase: Group scores by category/business logic
3. Reduce Phase: Rank products within groups, output top recommendations

Key Considerations:
- Model distribution: Broadcast model to all mappers vs model serving
- Batch processing: Process multiple products per map task for efficiency
- Memory management: Handle large feature vectors and model size
- Fault tolerance: Handle mapper failures with model reloading
- Output format: Structured rankings for downstream recommendation system
"""

import pickle
import csv
import heapq
import numpy as np
import time
from typing import Generator, Tuple, List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ProductFeatures:
    """
    Product feature representation with validation.

    Represents product data from CSV files with business logic validation
    to ensure data quality for ML inference.
    """
    product_id: str
    category: str
    price: float
    rating: float
    review_count: int
    brand: str
    features: List[float]  # Numerical feature vector

    def __post_init__(self):
        """Validate product data after initialization."""
        if not self.product_id or not self.product_id.strip():
            raise ValueError("Product ID cannot be empty")

        if self.price < 0:
            raise ValueError(f"Price cannot be negative: {self.price}")

        if not (0 <= self.rating <= 5):
            raise ValueError(f"Rating must be between 0-5: {self.rating}")

        if self.review_count < 0:
            raise ValueError(f"Review count cannot be negative: {self.review_count}")

        if not self.features or len(self.features) == 0:
            raise ValueError("Features vector cannot be empty")

        if any(not isinstance(f, (int, float)) for f in self.features):
            raise ValueError("All features must be numeric")

    @classmethod
    def from_csv_row(cls, row: List[str]) -> 'ProductFeatures':
        """
        Create ProductFeatures from CSV row.

        Args:
            row: CSV row as list of strings

        Returns:
            ProductFeatures instance

        Raises:
            ValueError: If row format is invalid
        """
        if len(row) < 7:
            raise ValueError(f"CSV row must have at least 7 columns, got {len(row)}")

        try:
            # Parse features from string representation
            import ast
            features_str = row[6]
            features = ast.literal_eval(features_str) if features_str.startswith('[') else [float(features_str)]

            return cls(
                product_id=row[0].strip(),
                category=row[1].strip(),
                price=float(row[2]),
                rating=float(row[3]),
                review_count=int(row[4]),
                brand=row[5].strip(),
                features=features
            )
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse CSV row: {e}")

    def normalize_features(self) -> List[float]:
        """
        Normalize features for ML model input.

        Returns:
            Normalized feature vector
        """
        # Simple min-max normalization for demo
        if not self.features:
            return []

        min_val = min(self.features)
        max_val = max(self.features)

        if max_val == min_val:
            return [0.5] * len(self.features)  # All same values

        return [(f - min_val) / (max_val - min_val) for f in self.features]


@dataclass
class ProductScore:
    """
    Product inference result with business metadata.

    Contains ML model output plus business context for recommendation ranking.
    """
    product_id: str
    category: str
    score: float
    confidence: float
    inference_timestamp: float = None
    model_version: str = "v1.0"

    def __post_init__(self):
        """Validate inference results."""
        import time

        if not self.product_id or not self.product_id.strip():
            raise ValueError("Product ID cannot be empty")

        if not isinstance(self.score, (int, float)):
            raise ValueError("Score must be numeric")

        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be between 0-1: {self.confidence}")

        if self.inference_timestamp is None:
            self.inference_timestamp = time.time()

    def __lt__(self, other: 'ProductScore') -> bool:
        """Compare scores for ranking (higher score is better)."""
        return self.score < other.score

    def __gt__(self, other: 'ProductScore') -> bool:
        """Compare scores for ranking (higher score is better)."""
        return self.score > other.score

    def to_recommendation_format(self) -> Dict[str, Any]:
        """
        Convert to format expected by recommendation system.

        Returns:
            Dictionary with recommendation metadata
        """
        return {
            'product_id': self.product_id,
            'category': self.category,
            'recommendation_score': round(self.score, 4),
            'confidence': round(self.confidence, 4),
            'model_version': self.model_version,
            'timestamp': self.inference_timestamp
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high confidence prediction."""
        return self.confidence >= 0.8

    @property
    def recommendation_tier(self) -> str:
        """Categorize recommendation strength."""
        if self.confidence >= 0.8 and self.score >= 0.7:
            return "HIGHLY_RECOMMENDED"
        elif self.confidence >= 0.6 and self.score >= 0.5:
            return "RECOMMENDED"
        elif self.confidence >= 0.4:
            return "MAYBE_RECOMMENDED"
        else:
            return "NOT_RECOMMENDED"


class MockMLModel:
    """
    Mock ML model for product scoring with realistic business logic.

    Simulates a trained recommendation model that considers:
    - Product features (price, rating, reviews)
    - Category-specific scoring
    - Realistic confidence estimation

    In production, this would be replaced with actual ML models
    (sklearn, tensorflow, pytorch, etc.) loaded from model registry.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize model with realistic parameters.

        Args:
            model_path: Path to model file (unused in mock)
        """
        # Category-specific scoring weights
        self.category_weights = {
            'Electronics': {'price_weight': 0.3, 'rating_weight': 0.4, 'review_weight': 0.3},
            'Books': {'price_weight': 0.2, 'rating_weight': 0.5, 'review_weight': 0.3},
            'Clothing': {'price_weight': 0.4, 'rating_weight': 0.3, 'review_weight': 0.3},
            'Home': {'price_weight': 0.3, 'rating_weight': 0.4, 'review_weight': 0.3},
            'Sports': {'price_weight': 0.25, 'rating_weight': 0.35, 'review_weight': 0.4}
        }

        # Mock neural network weights for feature vector
        self.feature_weights = np.random.normal(0, 0.1, 10)  # More realistic weights
        self.bias = 0.05
        self.model_version = "recommendation_model_v2.1"

        # Model loading simulation
        if model_path:
            print(f"Loading model from {model_path}")
            # In reality: self.model = joblib.load(model_path)

    def predict(self, product: ProductFeatures) -> Tuple[float, float]:
        """
        Run inference with business logic.

        Combines traditional ML features with business rules for
        realistic e-commerce recommendation scoring.

        Args:
            product: ProductFeatures object

        Returns:
            Tuple of (recommendation_score, confidence)
        """
        # Business logic scoring
        business_score = self._calculate_business_score(product)

        # Feature-based ML scoring
        ml_score = self._calculate_ml_score(product.features)

        # Combine scores (60% business logic, 40% ML features)
        final_score = 0.6 * business_score + 0.4 * ml_score

        # Realistic confidence based on data quality
        confidence = self._calculate_confidence(product)

        # Normalize to [0, 1] range
        normalized_score = max(0, min(1, final_score))

        return float(normalized_score), float(confidence)

    def _calculate_business_score(self, product: ProductFeatures) -> float:
        """Calculate score based on business rules."""
        # Get category-specific weights
        weights = self.category_weights.get(product.category,
                                          self.category_weights['Electronics'])

        # Price score (inverse - lower price is better up to a point)
        price_score = 1.0 / (1 + np.log(max(1, product.price / 100)))

        # Rating score (linear with rating)
        rating_score = product.rating / 5.0

        # Review count score (logarithmic - diminishing returns)
        review_score = min(1.0, np.log(max(1, product.review_count)) / 10)

        # Weighted combination
        business_score = (
            weights['price_weight'] * price_score +
            weights['rating_weight'] * rating_score +
            weights['review_weight'] * review_score
        )

        return business_score

    def _calculate_ml_score(self, features: List[float]) -> float:
        """Calculate score from ML feature vector."""
        if not features:
            return 0.5  # Default neutral score

        # Pad or truncate features to match model
        feature_array = np.array(features[:len(self.feature_weights)])
        if len(feature_array) < len(self.feature_weights):
            # Pad with zeros
            padding = np.zeros(len(self.feature_weights) - len(feature_array))
            feature_array = np.concatenate([feature_array, padding])

        # Linear model inference
        raw_score = np.dot(feature_array, self.feature_weights) + self.bias

        # Apply sigmoid for normalization
        return 1.0 / (1.0 + np.exp(-raw_score))

    def _calculate_confidence(self, product: ProductFeatures) -> float:
        """Calculate prediction confidence based on data quality."""
        confidence_factors = []

        # Rating confidence (more reviews = higher confidence)
        if product.review_count > 100:
            confidence_factors.append(0.9)
        elif product.review_count > 10:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Price confidence (reasonable price range)
        if 10 <= product.price <= 1000:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Feature completeness
        if len(product.features) >= 5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Rating reasonableness
        if 1 <= product.rating <= 5:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)

        return np.mean(confidence_factors)

    def predict_batch(self, products: List[ProductFeatures]) -> List[Tuple[float, float]]:
        """
        Batch inference for efficiency.

        Args:
            products: List of ProductFeatures

        Returns:
            List of (score, confidence) tuples
        """
        return [self.predict(product) for product in products]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for logging."""
        return {
            'model_version': self.model_version,
            'feature_dim': len(self.feature_weights),
            'categories_supported': list(self.category_weights.keys()),
            'inference_type': 'hybrid_business_ml'
        }


def inference_mapper(
    input_file: str,
    model_path: str = None
) -> Generator[Tuple[str, ProductScore], None, None]:
    """
    Map function: Load model and run inference on product batch.

    Key Design Decisions:
    - Load model once per mapper (not per record) for efficiency
    - Process products in batches to amortize model loading cost
    - Emit (category, ProductScore) for grouping in shuffle phase

    Args:
        input_file: Path to CSV file with product features
        model_path: Path to trained model file

    Yields:
        Tuple of (category, ProductScore) for each product
    """
    # 1. Load the ML model once per mapper (efficient for batch processing)
    model = MockMLModel(model_path)
    print(f"Mapper loaded model: {model.get_model_info()['model_version']}")

    # 2. Read product features from CSV with proper parsing
    products = []
    try:
        with open(input_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            _ = next(reader)  # Skip header row

            for row_num, row in enumerate(reader, 1):
                try:
                    product = ProductFeatures.from_csv_row(row)
                    products.append(product)
                except ValueError as e:
                    print(f"Skipping invalid row {row_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return

    print(f"Mapper processing {len(products)} products from {input_file}")

    # 3. Run batch inference for efficiency (process in chunks to manage memory)
    batch_size = 100  # Process 100 products at a time
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]

        # Use predict_batch for efficiency
        batch_results = model.predict_batch(batch)

        # 4. Yield (category, ProductScore) pairs
        for product, (score, confidence) in zip(batch, batch_results):
            product_score = ProductScore(
                product_id=product.product_id,
                category=product.category,
                score=score,
                confidence=confidence,
                model_version=model.model_version
            )
            yield (product.category, product_score)
    


def inference_shuffle(
    mapped_results: List[Generator[Tuple[str, ProductScore], None, None]]
) -> Dict[str, List[ProductScore]]:
    """
    Shuffle function: Group product scores by category.

    Groups all product scores by category for ranking in reduce phase.
    This enables category-specific recommendations and ranking.

    Args:
        mapped_results: List of generators from map phase

    Returns:
        Dictionary mapping category to list of ProductScore objects
    """
    # 1. Collect all (category, ProductScore) pairs from all mappers
    per_category_scores = defaultdict(list)

    total_products = 0
    for gen in mapped_results:
        for category, score in gen:
            per_category_scores[category].append(score)  # Group by category
            total_products += 1

    print(f"Shuffle grouped {total_products} products into {len(per_category_scores)} categories")

    # 2. Log category distribution for monitoring
    for category, scores in per_category_scores.items():
        print(f"  {category}: {len(scores)} products")

    return dict(per_category_scores)


def inference_reducer(
    category: str,
    product_scores: List[ProductScore],
    top_k: int = 100
) -> List[Tuple[str, ProductScore]]:
    """
    Reduce function: Rank products within category and select top-K.

    Sorts products by score within each category and selects the top-K
    products for the recommendation system.

    Args:
        category: Product category
        product_scores: List of ProductScore objects for this category
        top_k: Number of top products to return per category

    Returns:
        List of (category, ProductScore) tuples for top-K products
    """
    # 1. Sort products by score (descending) - highest scores first
    sorted_scores = sorted(product_scores, key=lambda x: x.score, reverse=True)

    # 2. Select top-K products for this category
    topk_scores = sorted_scores[:top_k]  # Handles K > len(products) safely

    print(f"Reducer for {category}: selected top {len(topk_scores)} from {len(product_scores)} products")

    # 3. Return formatted results as (category, ProductScore) tuples
    results = []
    for score in topk_scores:
        results.append((category, score))

    # Log best product for monitoring
    if topk_scores:
        best = topk_scores[0]
        print(f"  Best {category} product: {best.product_id} (score: {best.score:.3f})")

    return results
    


def run_distributed_inference(
    input_files: List[str],
    model_path: str,
    output_path: str,
    top_k_per_category: int = 100
) -> Dict[str, List[ProductScore]]:
    """
    Main function: Run distributed model inference using MapReduce.

    Orchestrates the complete pipeline:
    1. Map: Distribute model and run inference on product batches
    2. Shuffle: Group results by category
    3. Reduce: Rank and select top products per category

    Args:
        input_files: List of paths to product feature CSV files
        model_path: Path to trained ML model
        output_path: Path to write recommendation results
        top_k_per_category: Number of top products per category

    Returns:
        Dictionary mapping category to top ProductScore objects
    """
    print(f"Starting distributed inference on {len(input_files)} files")

    # 1. Map phase: Run inference_mapper on each input file (keep as generators)
    mapped_generators = []
    for file_path in input_files:
        print(f"Processing file: {file_path}")
        mapper_gen = inference_mapper(file_path, model_path)
        mapped_generators.append(mapper_gen)  # Store generator, don't extend

    # 2. Shuffle phase: Group results by category
    print("Starting shuffle phase...")
    shuffled_results = inference_shuffle(mapped_generators)

    # 3. Reduce phase: Rank and select top-K per category
    print("Starting reduce phase...")
    all_recommendations = []
    category_results = {}

    for category, product_scores in shuffled_results.items():
        top_products = inference_reducer(category, product_scores, top_k_per_category)
        all_recommendations.extend(top_products)

        # Store for return value (extract ProductScore objects)
        category_results[category] = [score for _, score in top_products]

    # 4. Write results to output file with proper CSV format
    print(f"Writing {len(all_recommendations)} recommendations to {output_path}")
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['category', 'product_id', 'score', 'confidence', 'model_version'])

        for category, product_score in all_recommendations:
            writer.writerow([
                category,
                product_score.product_id,
                round(product_score.score, 4),
                round(product_score.confidence, 4),
                product_score.model_version
            ])

    print(f"Distributed inference completed! Results written to {output_path}")
    return category_results


# Sample data generation for testing
def generate_sample_data(num_products: int = 1000, output_file: str = "products.csv"):
    """Generate sample product data for testing."""
    categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['product_id', 'category', 'price', 'rating', 'review_count', 'brand', 'features'])

        for i in range(num_products):
            product_id = f"PROD_{i:06d}"
            category = np.random.choice(categories)
            price = np.random.uniform(10, 1000)
            rating = np.random.uniform(1, 5)
            review_count = np.random.randint(0, 10000)
            brand = np.random.choice(brands)
            features = np.random.random(10).tolist()  # 10-dimensional feature vector

            writer.writerow([product_id, category, price, rating, review_count, brand, str(features)])


if __name__ == "__main__":
    """
    Test the distributed inference pipeline.

    1. Generate sample product data
    2. Create mock model
    3. Run inference pipeline
    4. Validate results
    """
    print("Distributed Model Inference Pipeline")
    print("=" * 50)

    # Generate test data
    print("Generating sample product data...")
    file_names = ["sample_products_1.csv"]
    for file_name in file_names:
        generate_sample_data(1000, file_name)

    # Run the inference pipeline
    print("\nRunning distributed inference pipeline...")
    topk_products = run_distributed_inference(
        input_files=file_names,
        model_path=None,
        output_path="recommendations_output.csv",
        top_k_per_category=10
    )

    # Display results summary
    print("\nPipeline Results Summary:")
    print("=" * 30)
    for category, products in topk_products.items():
        print(f"{category}: {len(products)} top products")
        if products:
            best = products[0]
            print(f"  Best: {best.product_id} (score: {best.score:.3f})")

    print(f"\nDetailed results written to: recommendations_output.csv")
  