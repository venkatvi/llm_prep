# Regression Tests

Comprehensive test suite for regression models including linear, non-linear, and transformer architectures with integration testing.

## Test Structure

### `test_configs.py`
Configuration validation tests:
- **TestRegressionModelConfig**: Linear and MLP model configuration validation
- **TestTransformerModelConfig**: Transformer-specific configuration testing
- **TestAutoregressiveDecodeConfig**: Autoregressive generation parameter validation
- **Configuration Edge Cases**: Invalid parameter combinations and validation

### `test_dataset.py`
Data handling and generation tests:
- **TestRegressionDataset**: CSV-based dataset loading and tensor conversion
- **TestDatasetFunctions**: Synthetic polynomial data generation
- **TestDataLoading**: Batch processing and DataLoader integration
- **TestDataGeneration**: Various polynomial degrees and noise levels
- **Edge Cases**: Empty datasets, malformed CSV files, invalid parameters

### `test_models.py`
Model architecture and functionality tests:
- **TestLinearRegressionModel**: Linear regression implementation validation
- **TestMLP**: Multi-layer perceptron architecture testing
- **TestTransformerModels**: Transformer variants (regression, autoregressive, encoder-decoder)
- **TestModelTraining**: Training mode setup and gradient flow
- **Architecture Validation**: Layer composition and parameter initialization

### `test_integration.py`
End-to-end workflow integration tests:
- **TestRegressionIntegration**: Complete experiment workflows
- **TestExperimentSaveLoad**: Model checkpointing and persistence
- **TestDataLoaderIntegration**: Batch training with real data loaders
- **TestOptimizerScheduler**: Various optimizer and scheduler combinations
- **TestLossFunctions**: Different loss function implementations (MSE, MAE, Huber)
- **TestReproducibility**: Deterministic results with fixed seeds
- **Edge Cases**: Small datasets, extreme parameters, error handling

## Running Tests

### All Regression Tests
```bash
cd regression/tests
python run_tests.py
```

### Individual Test Files
```bash
cd regression/tests

# Configuration tests
python -m pytest test_configs.py -v

# Dataset tests  
python -m pytest test_dataset.py -v

# Model tests
python -m pytest test_models.py -v

# Integration tests
python -m pytest test_integration.py -v
```

### Specific Test Categories
```bash
# Run specific test class
python -m pytest test_integration.py::TestRegressionIntegration -v

# Run specific test method
python -m pytest test_models.py::TestLinearRegressionModel::test_forward_pass -v

# Run with coverage
python -m pytest --cov=regression tests/
```

## Test Coverage

### Unit Tests
- **Model Architectures**: Forward/backward pass validation for all model types
- **Data Processing**: CSV loading, tensor conversion, synthetic data generation
- **Configuration Management**: Parameter validation and dataclass functionality
- **Training Components**: Optimizer setup, loss computation, metrics tracking

### Integration Tests
- **Complete Workflows**: Data generation → model training → prediction → evaluation
- **Experiment Management**: Configuration-driven experiment execution
- **Model Persistence**: Save/load functionality with state validation
- **Batch Processing**: DataLoader integration with various batch sizes
- **Hyperparameter Combinations**: Different optimizer/scheduler/loss combinations

### Regression-Specific Tests
- **Linear Regression**: Analytical solution validation where possible
- **MLP Models**: Various layer configurations and activation functions
- **Transformer Models**: All three architectures (regression, autoregressive, encoder-decoder)
- **Data Generation**: Polynomial fitting with different degrees and noise levels

## Key Test Features

### Model Architecture Validation
- **Shape Consistency**: Input/output tensor dimensions throughout pipeline
- **Parameter Counting**: Correct parameter initialization and counting
- **Gradient Flow**: Proper backpropagation through all layers
- **Activation Functions**: Various activation function integrations

### Training Pipeline Testing  
- **Optimizer Integration**: Adam, SGD, RMSprop with proper parameter updates
- **Learning Rate Scheduling**: Step, exponential, cosine annealing, ReduceLROnPlateau
- **Loss Function Validation**: MSE, MAE, Huber loss implementations
- **Validation Splits**: Proper train/validation data separation

### Experiment Framework Testing
- **Configuration Systems**: Dataclass-based configuration management
- **Logging Integration**: TensorBoard logging and metrics tracking
- **Reproducibility**: Fixed random seeds and deterministic results
- **Error Handling**: Graceful handling of invalid configurations

### Data Quality Assurance
- **Synthetic Data**: Polynomial data generation with configurable complexity
- **CSV Integration**: Real file I/O with proper error handling
- **DataLoader Compatibility**: Integration with PyTorch data loading pipeline
- **Edge Cases**: Empty datasets, single samples, large datasets

## Expected Test Results

### Linear Regression Tests
- Analytical validation against known polynomial solutions
- Gradient descent convergence for simple problems
- Proper handling of different input dimensions

### MLP Tests
- Universal approximation for polynomial functions
- Various architecture configurations (1-5 layers)
- Different activation functions (ReLU, Tanh, Sigmoid)

### Transformer Tests
- **Regression Mode**: Sequence-to-scalar prediction accuracy
- **Autoregressive Mode**: Next-token prediction and sequence generation
- **Encoder-Decoder Mode**: Sequence-to-sequence transformation

### Integration Tests
- **Training Convergence**: Loss reduction over epochs
- **Model Persistence**: Identical results after save/load cycles  
- **Batch Processing**: Consistent results across different batch sizes
- **Reproducibility**: Identical results with same random seeds

## Performance Baselines

Tests validate that models achieve reasonable performance on synthetic data:
- Linear regression: R² > 0.95 on noise-free polynomial data
- MLP models: Successful fitting of degree-3 polynomials with noise
- Transformer models: Effective sequence modeling for regression tasks

## Test Dependencies

Tests require:
- `pytest` - Test framework and fixtures
- `torch` - PyTorch for model implementation
- `pandas` - CSV data handling
- `numpy` - Numerical operations
- Standard library modules for file I/O and utilities

## Continuous Integration

These tests are part of the CI/CD pipeline ensuring:
- All model architectures remain functional
- Training pipelines work across different configurations
- Data generation and processing remain stable
- Integration between components is maintained