# Adversarial Robustness and Feature Superposition Measurement Framework

This repository implements a comprehensive framework for studying the relationship between adversarial robustness and feature organization in neural networks, with a particular focus on superposition measurement using sparse autoencoders.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Components](#key-components)
- [Running Experiments](#running-experiments)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
# Clone the repository
git clone https://github.com/leonardbereska/superposition-measurement.git
cd superposition-measurement

# Create and activate conda environment
conda create -n superposition python=3.10 -y
conda activate superposition

# Install PyTorch (with CUDA support)
# Modify cuda version as needed (11.8, 12.1, etc.)
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y # for cuda
conda install pytorch torchvision -y  # for cpu

# Install core data science packages
conda install numpy matplotlib tqdm ipykernel -y

# Install Hugging Face datasets
pip install datasets

# Install nnsight
pip install nnsight
```

## Quick Start

The framework is designed around a three-phase workflow:

```python
# Run all phases with default parameters
from main import run_all_phases
results_dir = run_all_phases()
```

Or run each phase separately:

```python
# Run the three phases individually
from main import run_training_phase, run_evaluation_phase, run_analysis_phase

# Phase 1: Train models with different adversarial strengths
results_dir = run_training_phase()

# Phase 2: Evaluate feature organization
run_evaluation_phase(results_dir=results_dir)

# Phase 3: Analyze results and generate visualizations
run_analysis_phase(results_dir=results_dir)
```

## Key Components

### Main Functions (`main.py`)

- **`run_training_phase`**: Trains models across different adversarial strengths (ε values)
- **`run_evaluation_phase`**: Applies sparse autoencoders to measure feature organization
- **`run_analysis_phase`**: Generates visualizations and statistical summaries
- **`run_all_phases`**: Executes all three phases in sequence
- **`run_model_class_experiment`**: Runs experiments across different model types and class counts

### Configuration (`config.py`)

The `get_default_config` function returns a configuration dictionary with:

- Model parameters (`hidden_dim`, `model_type`)
- Dataset parameters (`dataset_type`, `selected_classes`)
- Training parameters (`batch_size`, `learning_rate`, `n_epochs`)
- Adversarial parameters (`train_epsilons`, `test_epsilons`)
- SAE parameters (`expansion_factor`, `l1_lambda`)

Example of custom configuration:

```python
from config import get_default_config, update_config

# Get default configuration
config = get_default_config()

# Customize configuration
custom_config = update_config(config, {
    'model': {'hidden_dim': 64, 'model_type': 'cnn'},
    'dataset': {'selected_classes': (0, 1, 2)},  # 3-class experiment
    'training': {'n_epochs': 50},
    'adversarial': {'train_epsilons': [0.0, 0.05, 0.1]}
})
```

### Dataset Utilities (`datasets.py`)

- `create_dataset`: Creates MNIST or CIFAR-10 datasets
- `create_dataloaders`: Creates DataLoader objects for training and evaluation
- `visualize_grid`: Displays a grid of dataset samples

### Model Architectures (`models.py`)

- `create_model`: Factory function for creating MLP or CNN models
- `NNsightModelWrapper`: Wrapper for extracting activations from model layers

### Training (`training.py`)

- `train_model`: Trains a model with adversarial training
- `evaluate_model_performance`: Evaluates model robustness across perturbation strengths

### Sparse Autoencoders (`sae.py`)

- `train_sae`: Trains a sparse autoencoder on model activations
- `evaluate_sae`: Evaluates feature extraction quality
- `analyze_feature_statistics`: Analyzes feature usage statistics

### Evaluation (`evaluation.py`)

- `measure_superposition`: Measures feature organization using SAEs
- `measure_superposition_on_mixed_distribution`: Analyzes features on clean vs. adversarial data

### Analysis (`analysis.py`)

- `generate_plots`: Creates standard visualizations
- `calculate_statistics`: Computes summary statistics
- `create_report`: Generates HTML report with results

## Running Experiments

### Binary vs. Multi-class Experiment

```python
from main import run_model_class_experiment

# Run experiments across model types and class counts
results = run_model_class_experiment(
    model_types=['mlp', 'cnn'],
    class_counts=[2, 3, 5, 10]
)
```

### Distribution-Aware Analysis

To properly analyze feature organization on clean vs. adversarial data:

```python
from evaluation import measure_superposition_on_mixed_distribution
from models import create_model, NNsightModelWrapper

# Load a model (example)
model = load_model(model_path)

# Analyze features on clean, adversarial, and mixed distributions
results = measure_superposition_on_mixed_distribution(
    model=model,
    clean_loader=train_loader,
    layer_name='relu',  # for MLP, or 'features.1' for CNN
    epsilon=0.1,        # perturbation strength
    expansion_factor=4, # SAE dictionary size multiplier
    save_dir=Path('results/distribution_analysis')
)
```

## Key Implementation Details

### Feature Count Calculation

The core metric for measuring superposition is based on the entropy of feature activations:

```python
def calculate_feature_metrics(p: np.ndarray) -> Dict[str, float]:
    """Calculate entropy and feature count metrics from activation distribution."""
    # Add epsilon to avoid log(0)
    p_safe = p + 1e-10
    p_safe = p_safe / p_safe.sum()  # Renormalize
    
    # Calculate entropy
    entropy = -np.sum(p_safe * np.log(p_safe))
    
    # Effective feature count (exponential of entropy)
    feature_count = np.exp(entropy)
    
    return {
        'entropy': entropy,
        'feature_count': feature_count,
    }
```

### Adversarial Training Implementation

The implementation uses FGSM (Fast Gradient Sign Method) for adversarial training:

```python
def adversarial_train_step(model, inputs, targets, optimizer, criterion, epsilon):
    """Perform a single adversarial training step using FGSM."""
    # Generate adversarial examples
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)
    loss.backward()
    
    # FGSM attack
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = torch.clamp(inputs + perturbation, 0, 1)
    
    # Train on adversarial examples
    inputs.requires_grad = False
    optimizer.zero_grad()
    adv_outputs = model(adv_inputs)
    adv_loss = criterion(adv_outputs.squeeze(), targets)
    adv_loss.backward()
    optimizer.step()
    
    return {'loss': adv_loss.item()}
```

## Troubleshooting

### Common Issues

1. **Cannot find experiment results**: Use the `search_string` parameter to specify experiment directory:

```python
run_evaluation_phase(search_string="mlp_2-class")
```

2. **Out of memory errors**: Reduce batch size or model size in configuration:

```python
config = update_config(config, {
    'training': {'batch_size': 32},
    'sae': {'batch_size': 64}
})
```

3. **SAE training instability**: Adjust L1 regularization strength:

```python
config = update_config(config, {
    'sae': {'l1_lambda': 0.01}  # Smaller value for less sparsity
})
```

4. **Slow training on CNN models**: Ensure you're using GPU acceleration:

```python
config = update_config(config, {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
})
```

## For Further Questions

If you have any issues or questions about implementing new features or running experiments, please don't hesitate to reach out.