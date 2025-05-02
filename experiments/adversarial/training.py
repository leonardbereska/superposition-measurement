# %%
"""Training utilities for adversarial robustness experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path
import json
import numpy as np

from models import create_model

# %%
def train_step(
    model: nn.Module, 
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable
) -> Dict[str, float]:
    """Perform a single training step.
    
    Args:
        model: Model to train
        inputs: Input data
        targets: Target labels
        optimizer: Optimizer
        criterion: Loss function
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)
    
    loss.backward()
    optimizer.step()
    
    return {'loss': loss.item()}

# %%
def adversarial_train_step(
    model: nn.Module, 
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable,
    epsilon: float, 
    alpha: float = 0.5
) -> Dict[str, float]:
    """Perform a single adversarial training step using FGSM.
    
    Args:
        model: Model to train
        inputs: Input data
        targets: Target labels
        optimizer: Optimizer
        criterion: Loss function
        epsilon: Perturbation size
        alpha: Weight for clean vs adversarial loss
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Calculate clean loss
    clean_outputs = model(inputs)
    clean_loss = criterion(clean_outputs.squeeze(), targets)
    
    # Generate adversarial examples
    inputs.requires_grad = True
    adv_outputs = model(inputs)
    adv_loss = criterion(adv_outputs.squeeze(), targets)
    adv_loss.backward()
    
    # FGSM
    perturbation = epsilon * inputs.grad.sign()
    adv_inputs = torch.clamp(inputs + perturbation, 0, 1)
    
    # Calculate adversarial loss
    inputs.requires_grad = False
    adv_outputs = model(adv_inputs)
    adv_loss = criterion(adv_outputs.squeeze(), targets)
    
    # Combined loss
    loss = alpha * clean_loss + (1 - alpha) * adv_loss
    loss.backward()
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'clean_loss': clean_loss.item(),
        'adv_loss': adv_loss.item()
    }

# %%
def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable,
    epsilon: float = 0.0, 
    alpha: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch with optional adversarial training.
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer
        criterion: Loss function
        epsilon: Perturbation size (0.0 for standard training)
        alpha: Weight for clean vs adversarial loss
        
    Returns:
        Dictionary with training metrics
    """
    epoch_stats = {}
    num_batches = 0
    
    for inputs, targets in dataloader:
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if epsilon > 0:
            step_stats = adversarial_train_step(
                model, inputs, targets, optimizer, criterion, epsilon, alpha
            )
        else:
            step_stats = train_step(model, inputs, targets, optimizer, criterion)
        
        # Update epoch stats
        for k, v in step_stats.items():
            epoch_stats[k] = epoch_stats.get(k, 0) + v
        
        num_batches += 1
        
    # Average stats
    for k in epoch_stats:
        epoch_stats[k] /= num_batches
        
    return epoch_stats

# %%
def evaluate_model(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: Optional[Callable] = None
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        criterion: Optional loss function
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs.squeeze(), targets)
                running_loss += loss.item()
            
            # Calculate accuracy
            if outputs.size(-1) == 1:  # Binary classification
                predictions = (torch.sigmoid(outputs) > 0.5).squeeze()
            else:  # Multi-class classification
                predictions = outputs.argmax(dim=1)
            
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            num_batches += 1
    
    metrics = {'accuracy': 100 * correct / total}
    
    if criterion is not None:
        metrics['loss'] = running_loss / num_batches
    
    return metrics

# %%
def train_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    model_type: str = 'mlp',
    hidden_dim: int = 32,
    image_size: int = 28,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    epsilon: float = 0.0,
    alpha: float = 0.5,
    verbose: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Train a model with optional adversarial training.
    
    Args:
        train_loader: DataLoader with training data
        val_loader: Optional DataLoader with validation data
        model_type: Type of model ('mlp' or 'cnn')
        hidden_dim: Hidden dimension
        image_size: Image size
        n_epochs: Number of epochs
        learning_rate: Learning rate
        epsilon: Perturbation size (0.0 for standard training)
        alpha: Weight for clean vs adversarial loss
        verbose: Whether to print progress
        device: Device to use
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine number of classes
    num_classes = len(torch.unique(train_loader.dataset.targets))
    output_dim = 1 if num_classes <= 2 else num_classes
    
    if verbose:
        print(f"Creating {model_type.upper()} model with {output_dim} output classes")
    
    # Create model
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        image_size=image_size,
        num_classes=output_dim,
        use_nnsight=False
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create criterion based on output dimension
    criterion = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(n_epochs):
        # Train for one epoch
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, epsilon, alpha
        )
        
        # Only evaluate when we're going to print progress
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            # Evaluate on training set
            train_metrics = evaluate_model(model, train_loader, criterion)
            
            # Evaluate on validation set if provided
            if val_loader is not None:
                val_metrics = evaluate_model(model, val_loader, criterion)
                
                # Print progress
                log_str = f"Epoch {epoch+1}/{n_epochs}"
                log_str += f" - Loss: {train_stats.get('loss', 0):.4f}"
                log_str += f" - Accuracy: {train_metrics['accuracy']:.2f}%"
                log_str += f" - Val Loss: {val_metrics['loss']:.4f}"
                log_str += f" - Val Accuracy: {val_metrics['accuracy']:.2f}%"
            else:
                # Print progress without validation metrics
                log_str = f"Epoch {epoch+1}/{n_epochs}"
                log_str += f" - Loss: {train_stats.get('loss', 0):.4f}"
                if 'adv_loss' in train_stats:
                    log_str += f" - Adv Loss: {train_stats.get('adv_loss', 0):.4f}"
                log_str += f" - Accuracy: {train_metrics['accuracy']:.2f}%"
                
            print(log_str)
    
    return model 

# %%
def evaluate_model_performance(
    model: nn.Module, 
    dataloader: DataLoader, 
    epsilons: List[float],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate model robustness across different perturbation sizes.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        epsilons: List of perturbation sizes to test
        device: Device to use (defaults to model's device)
        
    Returns:
        Dictionary mapping epsilon values to accuracies
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    for eps in epsilons:
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.shape[0]
            
            # For eps=0, just evaluate normally
            if eps == 0:
                with torch.no_grad():
                    outputs = model(inputs)
                    if outputs.size(-1) == 1:  # Binary
                        predictions = (torch.sigmoid(outputs) > 0.5).squeeze()
                    else:  # Multi
                        predictions = outputs.argmax(dim=1)
                        
                    correct += (predictions == targets).sum().item()
                    total += batch_size
                    continue
            
            # Generate adversarial examples with FGSM
            inputs.requires_grad = True
            outputs = model(inputs)
            
            if outputs.size(-1) == 1:  # Binary classification
               loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
            else:  # Multi-class classification
               loss = nn.CrossEntropyLoss()(outputs, targets)
           
            model.zero_grad()
            loss.backward()
           
            # FGSM attack
            perturbed_inputs = inputs + eps * inputs.grad.sign()
            perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
           
            # Evaluate on perturbed inputs
            with torch.no_grad():
                adv_outputs = model(perturbed_inputs)
                
                if adv_outputs.size(-1) == 1:  # Binary
                    predictions = (torch.sigmoid(adv_outputs) > 0.5).squeeze()
                else:  # Multi-class
                    predictions = adv_outputs.argmax(dim=1)
                    
                correct += (predictions == targets).sum().item()
                total += batch_size
        
        accuracy = 100 * correct / total
        results[str(eps)] = accuracy
        
    return results

# %%
def save_model(
    model: nn.Module,
    history: Dict[str, List[float]],
    config: Dict[str, Any],
    save_path: Path
) -> None:
    """Save model, history, and configuration.
    
    Args:
        model: Trained model
        history: Training history
        config: Model configuration
        save_path: Path to save model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'config': config
    }, save_path)

# %%
def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any]]:
    """Load model, history, and configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model onto
        
    Returns:
        Tuple of (model, history, config)
    """
    # Use default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration
    config = checkpoint['config']
    
    # Create model
    model_type = config.get('model_type', 'mlp')
    hidden_dim = config.get('hidden_dim', 32)
    image_size = config.get('image_size', 28)
    
    # Determine number of output classes from state dict
    if 'model_state' in checkpoint:
        output_dim = list(checkpoint['model_state'].items())[-1][1].size(0)
    else:
        # Default to binary classification
        output_dim = 1
    
    # Create model
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        image_size=image_size,
        num_classes=output_dim,
        use_nnsight=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state'])
    
    # Move to device
    model = model.to(device)
    
    # Get history
    history = checkpoint.get('history', {})
    
    return model, history, config