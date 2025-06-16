# %%
"""Training utilities for adversarial robustness experiments."""

import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, NamedTuple

from models import create_model, ModelConfig
from attacks import AttackConfig, generate_adversarial_examples
import logging

class TrainingConfig(NamedTuple):
    """Configuration for model training."""
    n_epochs: int
    learning_rate: float
    optimizer_type: str = 'adam'
    weight_decay: float = 0.0
    momentum: float = 0.9
    lr_schedule: Optional[str] = None
    lr_milestones: Optional[List[int]] = None
    lr_gamma: float = 0.1
    early_stopping_patience: int = 20
    min_epochs: int = 0
    use_early_stopping: bool = True  # Whether to use early stopping at all

# %%


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
    attack_config: AttackConfig,
    alpha: float = 0.5
) -> Dict[str, float]:
    """Perform a single adversarial training step.
    
    Args:
        model: Model to train
        inputs: Input data
        targets: Target labels
        optimizer: Optimizer
        criterion: Loss function
        epsilon: Perturbation size
        attack_config: Configuration for the attack
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
    adv_inputs = generate_adversarial_examples(model, inputs, targets, epsilon, attack_config)
    
    # Calculate adversarial loss
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
    attack_config: Optional[AttackConfig] = None,
    alpha: float = 0.5
) -> Dict[str, float]:
    """Train for one epoch with optional adversarial training.
    
    Args:
        model: Model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer
        criterion: Loss function
        epsilon: Perturbation size (0 for standard training)
        attack_config: Configuration for the attack (None for standard training)
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
        
        if epsilon > 0 and attack_config is not None:
            step_stats = adversarial_train_step(
                model, inputs, targets, optimizer, criterion, 
                epsilon, attack_config, alpha
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


def get_progressive_epsilon(epoch, target_epsilon, total_epochs):
    """Gradually increase epsilon during training"""
    if epoch < total_epochs * 0.2:
        return 0.33 * target_epsilon
    elif epoch < total_epochs * 0.5:
        return 0.66 * target_epsilon
    else:
        return target_epsilon
# %%
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    epsilon: float = 0.0,
    attack_config: Optional[AttackConfig] = None,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """Train a model with optional adversarial training.
    
    Args:
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        model_config: Configuration for the model
        training_config: Configuration for training
        epsilon: Perturbation size (0 for standard training)
        attack_config: Configuration for the attack (None for standard training)
        alpha: Weight for clean vs adversarial loss
        device: Device to train on
        logger: Optional logger for training progress
        
    Returns:
        Trained model
    """
    # Set up logging
    log = logger.info if logger else print
    
    # Initialize model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(
        model_type=model_config.model_type,
        input_channels=model_config.input_channels,
        hidden_dim=model_config.hidden_dim,
        image_size=model_config.image_size,
        output_dim=model_config.output_dim,
        use_nnsight=False
    ).to(device)

    # Prepare optimizer
    if training_config.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=training_config.learning_rate, 
            momentum=training_config.momentum, 
            weight_decay=training_config.weight_decay
        )
    else:  # Adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=training_config.learning_rate, 
            weight_decay=training_config.weight_decay
        )
    
    # Set up learning rate scheduler
    scheduler = None
    if training_config.lr_schedule == 'multistep' and training_config.lr_milestones is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=training_config.lr_milestones, 
            gamma=training_config.lr_gamma
        )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss() if model_config.output_dim == 1 else nn.CrossEntropyLoss()
    
    # Initialize training state
    best_val_acc = 0.0
    best_adv_acc = 0.0
    best_epoch = 0
    best_state_dict = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_adv_acc': []
    }
    
    # Training loop
    for epoch in range(training_config.n_epochs):

        # Progressively increase epsilon during training
        train_epsilon = get_progressive_epsilon(epoch, epsilon, training_config.n_epochs)

        # Train for one epoch
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, 
            train_epsilon, attack_config, alpha
        )
        
        # Step scheduler if present
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, criterion)
        
        # Track metrics
        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Get adversarial accuracy if using adversarial training
        val_adv_acc = None
        if epsilon > 0 and attack_config is not None:
            val_adv_acc = evaluate_model_performance(
                model, val_loader, [epsilon], device, attack_config
            )[str(epsilon)]
            history['val_adv_acc'].append(val_adv_acc)
            
            # Early stopping based on adversarial accuracy for adversarial training
            if val_adv_acc > best_adv_acc:
                best_adv_acc = val_adv_acc
                best_epoch = epoch
                best_state_dict = model.state_dict().copy()
            elif training_config.use_early_stopping and epoch >= training_config.min_epochs and epoch - best_epoch >= training_config.early_stopping_patience:
                log(f"\tEarly stopping at epoch {epoch+1} (adversarial accuracy hasn't improved for {training_config.early_stopping_patience} epochs)")
                model.load_state_dict(best_state_dict)
                break
        else:
            # Early stopping based on validation accuracy for clean training
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                best_state_dict = model.state_dict().copy()
            elif training_config.use_early_stopping and epoch >= training_config.min_epochs and epoch - best_epoch >= training_config.early_stopping_patience:
                log(f"\tEarly stopping at epoch {epoch+1} (validation accuracy hasn't improved for {training_config.early_stopping_patience} epochs)")
                model.load_state_dict(best_state_dict)
                break
        
        # Log progress at regular intervals
        if epoch % max(1, training_config.n_epochs // 10) == 0 or epoch == training_config.n_epochs - 1:
            log_message = f"\tEpoch {epoch+1:04d}/{training_config.n_epochs}"
            log_message += f" | Train loss: {train_stats['loss']:.4f}"
            log_message += f" | Val loss: {val_metrics['loss']:.4f}"
            log_message += f" | Val acc: {val_metrics['accuracy']:6.2f}%"
            
            if scheduler is not None:
                log_message += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
            
            if epsilon > 0 and attack_config is not None and val_adv_acc is not None:
                log_message += f" | Val adv acc: {val_adv_acc:6.2f}%"
                
            log(log_message)
    
    # Load best model if early stopping was triggered
    if training_config.use_early_stopping and best_state_dict is not None and epoch != best_epoch:
        model.load_state_dict(best_state_dict)
        if epsilon > 0 and attack_config is not None:
            log(f"\t\tBest model from epoch {best_epoch+1} with adversarial accuracy: {best_adv_acc:.2f}%")
        else:
            log(f"\t\tBest model from epoch {best_epoch+1} with validation accuracy: {best_val_acc:.2f}%")
    return model

def evaluate_model_performance(
    model: nn.Module, 
    dataloader: DataLoader, 
    epsilons: List[float],
    device: Optional[torch.device] = None,
    attack_config: Optional[AttackConfig] = None
) -> Dict[str, float]:
    """Evaluate model robustness across different perturbation sizes.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        epsilons: List of perturbation sizes to test
        device: Device to use (defaults to model's device)
        attack_config: Configuration for the attack (defaults to FGSM)
        
    Returns:
        Dictionary mapping epsilon values to accuracies
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Use default attack config if none provided
    if attack_config is None:
        attack_config = AttackConfig()
    
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
            
            # Generate adversarial examples
            perturbed_inputs = generate_adversarial_examples(
                model, inputs, targets, eps, attack_config
            )
           
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

def save_model(
    model: nn.Module,
    model_config: ModelConfig,
    save_path: Path
) -> None:
    """Save model and configuration."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config._asdict()
    }, save_path)
            
def load_model(
    model_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, ModelConfig]:
    """Load model and configuration from saved file."""
    # Use default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    model_config_dict = checkpoint['model_config']
    model_config = ModelConfig(**model_config_dict)
    
    # Create model
    model = create_model(
        model_type=model_config.model_type,
        input_channels=model_config.input_channels,
        hidden_dim=model_config.hidden_dim,
        image_size=model_config.image_size,
        output_dim=model_config.output_dim,
        use_nnsight=False
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model = model.to(device)
    
    return model, model_config