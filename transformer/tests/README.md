# Transformer Library Tests

Comprehensive test suite for the transformer library components.

## Test Structure

- **`test_attention.py`** - Multi-head attention mechanism tests
- **`test_ffn.py`** - Feedforward network tests  
- **`test_input_encodings.py`** - Positional encoding tests
- **`test_transformer_model.py`** - Complete transformer model tests
- **`test_integration.py`** - Integration tests with regression wrapper
- **`run_tests.py`** - Test runner with configuration

## Running Tests

### All Tests
```bash
cd transformer/tests
python run_tests.py
```

### Individual Test Files
```bash
cd transformer/tests  
python run_tests.py --individual
```

### Specific Test File
```bash
cd transformer/tests
pytest test_attention.py -v
```

### Specific Test Method
```bash
cd transformer/tests
pytest test_attention.py::TestMultiHeadAttention::test_forward_shape -v
```

## Test Coverage

### Unit Tests
- **Attention**: Shape validation, gradient flow, attention properties
- **FFN**: Layer composition, ReLU activation, dimension handling
- **Positional Encoding**: Sinusoidal patterns, sequence length handling
- **Transformer Model**: Layer stacking, global pooling, configurations

### Integration Tests
- **Data Generation**: Synthetic sequence data creation
- **Training Pipeline**: Forward/backward pass, optimizer integration
- **Model Persistence**: State dict save/load functionality
- **Configuration Validation**: Different model sizes and parameters

## Test Features

- ✅ **Deterministic**: Fixed random seeds for reproducible results
- ✅ **Comprehensive**: Unit tests for all components
- ✅ **Integration**: End-to-end regression pipeline testing
- ✅ **Edge Cases**: Zero inputs, large sequences, dimension mismatches
- ✅ **Gradient Validation**: Ensures proper backpropagation
- ✅ **Shape Verification**: Validates tensor dimensions throughout

## Dependencies

Tests require:
- `pytest` - Test framework
- `torch` - PyTorch library
- Standard library modules

## Expected Results

All tests should pass with proper transformer implementation:
- ~15 test classes
- ~50+ individual test methods
- Coverage of all transformer components
- Integration with regression framework