# Regression Models

PyTorch regression framework with linear, non-linear, and transformer models supporting both regression and autoregressive prediction.

## Features

- **Models**: Linear regression, MLP, and Transformer (regression + autoregressive)
- **Training**: Complete pipeline with validation, optimizers, schedulers
- **Data**: Synthetic data generation with DataLoader support
- **Autoregressive**: Token-by-token generation with causal masking
- **Logging**: TensorBoard integration

## Quick Start

```bash
# Linear regression
python main.py --type linear --epochs 1000 --lr 0.01

# Non-linear MLP
python main.py --type nlinear --epochs 1000 --lr 0.001 --latent_dims "128,64,32"

# Transformer regression
python main.py --type transformer --epochs 1000 --lr 0.001

# Transformer autoregressive
python main.py --type transformer --autoregressive --epochs 1000 --lr 0.001

# Hyperparameter sweep
python experiment_sweep.py
```

## Files

- **`main.py`** - CLI entry point
- **`experiment.py`** - RegressionExperiment orchestrator
- **`e_linear_reg.py`** - Linear model
- **`e_non_linear_reg.py`** - MLP model
- **`h_transformer.py`** - Transformer regression wrapper
- **`configs.py`** - Model configurations
- **`dataset.py`** - Dataset utilities

## Usage

```python
from lib.configs import ExperimentConfig, TrainConfig
from experiment import RegressionExperiment

config = ExperimentConfig(
    type="nlinear",
    train_config=TrainConfig(epochs=1000, optimizer="adam", lr=0.001)
)

experiment = RegressionExperiment(config)
experiment.train()
predictions = experiment.predict()
```

## Key Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--type` | `linear`, `nlinear`, `transformer` | Model type |
| `--autoregressive` | flag | Enable transformer autoregressive mode |
| `--epochs` | integer | Training epochs |
| `--lr` | float | Learning rate |
| `--optimizer` | `adam`, `sgd`, `rmsprop` | Optimizer |
| `--custom_act` | `relu`, `tanh`, `gelu` | Activation |
| `--latent_dims` | comma-separated | Hidden dimensions |

## Transformer Modes

### Regression Mode (default)
- Uses global average pooling for scalar prediction
- Suitable for sequence-to-value tasks
- Example: predicting sum of sequence elements

### Autoregressive Mode (`--autoregressive`)
- Enables next-token prediction with causal masking
- Supports expanding context or sliding window generation
- Uses sequence-to-sequence architecture