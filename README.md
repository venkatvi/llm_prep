# PyTorch and LLM Prep Repository

A collection of machine learning experiments and implementations with professional-grade experiment management and hyperparameter optimization.

## Structure

- `regression/` - Comprehensive regression framework with PyTorch
  - **Experiment Management**: Structured configuration system with dataclasses
  - **Hyperparameter Sweeps**: Automated grid search across parameter combinations
  - **Model Architectures**: Linear and non-linear models with flexible architectures
  - **Advanced Training**: Multiple optimizers, schedulers, loss functions, and activation functions
  - **Data Processing**: DataLoader support, batch processing, and reproducible experiments
  - **Visualization**: TensorBoard integration and comprehensive logging
  - **Professional Documentation**: Complete docstrings and usage examples

## Requirements

- Python 3.11+
- PyTorch 2.1.0
- matplotlib 3.7.2
- numpy 1.24.3
- pandas (latest)
- tensorboard 2.20.0
- Virtual environment recommended

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Individual Experiments
```bash
cd regression
python main.py --type nlinear --epochs 1000 --lr 0.001 --latent_dims "128,64,32"
```

### Experiment Framework
```bash
cd regression
python -c "
from configs import ExperimentConfig, TrainConfig, DataConfig, ModelConfig
from experiment import Experiment

config = ExperimentConfig(
    type='nlinear',
    name='demo_experiment',
    train_config=TrainConfig(epochs=500, custom_loss='mse', optimizer='adam', lr=0.001, lr_scheduler='reduceonplat'),
    data=DataConfig(use_dataloader=True, training_batch_size=32, fix_random_seed=True),
    model=ModelConfig(custom_act='relu', num_latent_layers=3, latent_dims=[128, 64, 32], allow_residual=True)
)

experiment = Experiment(config)
experiment.train()
predictions = experiment.predict()
experiment.plot_results(predictions)
"
```

### Hyperparameter Sweeps
```bash
cd regression
python experiment_sweep.py  # Runs automated grid search across parameter combinations
```

## Key Features

### 🧪 **Experiment Management**
- **Structured Configurations**: Type-safe dataclass configurations for all parameters
- **Reproducible Experiments**: Fixed random seeds and comprehensive state saving
- **Automatic Checkpointing**: Model weights, optimizer state, and loss tracking
- **TensorBoard Integration**: Real-time metrics and visualization logging

### 🔍 **Hyperparameter Optimization**
- **Grid Search**: Automated cross products of parameter arrays
- **Smart Filtering**: Validates parameter combinations automatically
- **Parallel Execution**: Run multiple experiments with different configurations
- **Comprehensive Logging**: Track all experiment variations and results

### 📊 **Professional Documentation**
- **Complete Docstrings**: Every function and class fully documented
- **Usage Examples**: Code examples for all major functionality
- **Type Annotations**: Full type hints for better development experience
- **Design Rationale**: Implementation notes and architectural decisions

## Usage

See individual module READMEs for detailed usage instructions and API documentation.

## License

MIT License - see individual files for license headers.
