# %%
"""Training utilities for adversarial robustness experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from models import create_model

def train_step(model: nn.Module, 
               inputs: torch.Tensor, 
               targets: torch.Tensor, 
               optimizer: torch.optim.Optimizer, 
               criterion: Callable) -> Dict[str, float]:
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
    loss = criterion(outputs.squeeze(), targets)# for cifar
    #loss = criterion(outputs, targets)
    
    
    loss.backward()
    optimizer.step()
    
    return {'loss': loss.item()}

def adversarial_train_step(model: nn.Module, 
                          inputs: torch.Tensor, 
                          targets: torch.Tensor, 
                          optimizer: torch.optim.Optimizer, 
                          criterion: Callable,
                          epsilon: float, 
                          alpha: float = 0.5) -> Dict[str, float]:
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
    
    # PGD parameters
    pgd_steps = 10
    alpha_step = epsilon / pgd_steps  # step size per iteration
    adv_inputs = inputs.clone().detach().to(inputs.device)
    adv_inputs.requires_grad = True
    
    # Save the original inputs for projection
    original_inputs = inputs.detach()
    
    # PGD loop
    for _ in range(pgd_steps):
        adv_outputs = model(adv_inputs)
        adv_loss = criterion(adv_outputs.squeeze(), targets)
        adv_inputs.grad = None  # Clear any previous gradients
        adv_loss.backward()
        
        # Gradient step
        adv_inputs = adv_inputs + alpha_step * adv_inputs.grad.sign()
        
        # Project: clamp to epsilon-ball and [0,1] pixel range
        perturbation = torch.clamp(adv_inputs - original_inputs, min=-epsilon, max=epsilon)
        adv_inputs = torch.clamp(original_inputs + perturbation, min=0.0, max=1.0).detach()
        adv_inputs.requires_grad = True
    
    
    # Calculate adversarial loss
    #inputs.requires_grad = False
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

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: Callable,
                epsilon: float = 0.0, 
                alpha: float = 0.5) -> Dict[str, float]:
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

def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: Optional[Callable] = None) -> Dict[str, float]:
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

def evaluate_adversarial_robustness(model: nn.Module, 
                                   dataloader: DataLoader, 
                                   epsilons: List[float],
                                   device: Optional[torch.device] = None) -> Dict[float, float]:
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
           
            # # Generate adversarial examples with FGSM
            # inputs.requires_grad = True
            # outputs = model(inputs)
            
            # --- PGD Attack ---
            pgd_steps = 10
            alpha = eps / pgd_steps  # step size per iteration
            
            
            # Initialize adversarial examples with clean inputs
            adv_inputs = inputs.clone().detach().requires_grad_(True)
        
        
        
            for _ in range(pgd_steps):
                outputs = model(adv_inputs)
                if outputs.size(-1) == 1:
                    loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, targets)

                model.zero_grad()
                if adv_inputs.grad is not None:
                    adv_inputs.grad.zero_()
                loss.backward()

                # Gradient ascent step
                adv_inputs = adv_inputs + alpha * adv_inputs.grad.sign()

                # Project back to epsilon ball around original input
                perturbation = torch.clamp(adv_inputs - inputs, min=-eps, max=eps)
                adv_inputs = torch.clamp(inputs + perturbation, 0, 1).detach()
                adv_inputs.requires_grad = True


            # if outputs.size(-1) == 1:  # Binary classification
            #    loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
            # else:  # Multi-class classification
            #    loss = nn.CrossEntropyLoss()(outputs, targets)
           
            # model.zero_grad()
            # loss.backward()
           
            # # FGSM attack
            # perturbed_inputs = inputs + eps * inputs.grad.sign()
            # perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
           
            # Evaluate on perturbed inputs
            with torch.no_grad():
                # adv_outputs = model(perturbed_inputs)
                adv_outputs = model(adv_inputs)
                if adv_outputs.size(-1) == 1:  # Binary
                    predictions = (torch.sigmoid(adv_outputs) > 0.5).squeeze()
                else:  # Multi-class
                    predictions = adv_outputs.argmax(dim=1)
                    
                correct += (predictions == targets).sum().item()
                total += batch_size
        
        accuracy = 100 * correct / total
        results[eps] = accuracy
        
    return results

def generate_adversarial_example(
   model: nn.Module,
   image: torch.Tensor,
   true_label: torch.Tensor,
   epsilon: float = 0.1,
   targeted: bool = False,
   target_label: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   """Generate adversarial example using FGSM.
   
   Args:
       model: Trained model
       image: Input image tensor
       true_label: Original label
       epsilon: Perturbation size
       targeted: If True, optimize toward target_label
       target_label: Label to target if targeted=True
   
   Returns:
       perturbed_image, prediction, perturbation
   """
   # Prepare input
   device = next(model.parameters()).device
   image = image.clone().detach().requires_grad_(True).to(device)
   target = target_label.to(device) if targeted else true_label.to(device)
   
   # Forward pass
   output = model(image.unsqueeze(0))
   
   if output.size(-1) == 1:  # Binary classification
       loss = nn.BCEWithLogitsLoss()(output.squeeze(), target.float())
   else:  # Multi-class classification
       loss = nn.CrossEntropyLoss()(output, target)
   
   # Backward pass
   loss.backward()
   
   # Generate perturbation
   # Minus sign for targeted (move toward target)
   # Plus sign for untargeted (move away from true label)
   sign = -1 if targeted else 1
   perturbation = sign * epsilon * image.grad.sign()
   
   # Generate adversarial example
   perturbed_image = torch.clamp(image + perturbation, 0, 1)
   
   # Get prediction
   with torch.no_grad():
       adv_output = model(perturbed_image.unsqueeze(0))
       
       if adv_output.size(-1) == 1:  # Binary
           pred = torch.sigmoid(adv_output) > 0.5
       else:  # Multi-class
           pred = adv_output.argmax(dim=1)
   
   return perturbed_image, pred, perturbation

def train_model(model: nn.Module,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              lr: float = 1e-3,
              n_epochs: int = 30,
              epsilon: float = 0.0,
              alpha: float = 0.5,
              patience: int = 10,
              verbose: bool = True,
              device: Optional[torch.device] = None) -> Dict[str, List[float]]:
   """Train a model with optional adversarial training.
   
   Args:
       model: Model to train
       train_loader: DataLoader with training data
       val_loader: Optional DataLoader with validation data
       lr: Learning rate
       n_epochs: Number of training epochs
       epsilon: Perturbation size (0.0 for standard training)
       alpha: Weight for clean vs adversarial loss
       patience: Early stopping patience (0 to disable)
       verbose: Whether to print progress
       device: Device to use
       
   Returns:
       Dictionary with training history
   """
   # Set up device
   if device is None:
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   
   # Set up optimizer and criterion
   optimizer = optim.Adam(model.parameters(), lr=lr)
   
   if list(model.parameters())[-1].size(0) == 1:  # Binary classification
       criterion = nn.BCEWithLogitsLoss()
   else:  # Multi-class classification
       criterion = nn.CrossEntropyLoss()
   
   # Training logs
   history = {
       'train_loss': [],
       'train_acc': [],
       'val_loss': [],
       'val_acc': []
   }
   
   # Early stopping variables
   best_val_loss = float('inf')
   best_epoch = 0
   best_state = None
   
   # Training loop
   for epoch in range(n_epochs):
       # Train for one epoch
       train_stats = train_epoch(
           model, train_loader, optimizer, criterion, epsilon, alpha
       )
       
       # Evaluate on training set
       train_metrics = evaluate(model, train_loader, criterion)
       
       # Log training metrics
       history['train_loss'].append(train_stats.get('loss', 0))
       history['train_acc'].append(train_metrics['accuracy'])
       
       # Evaluate on validation set if provided
       if val_loader is not None:
           val_metrics = evaluate(model, val_loader, criterion)
           history['val_loss'].append(val_metrics['loss'])
           history['val_acc'].append(val_metrics['accuracy'])
           
           # Early stopping
           if patience > 0:
               if val_metrics['loss'] < best_val_loss:
                   best_val_loss = val_metrics['loss']
                   best_epoch = epoch
                   best_state = model.state_dict().copy()
               elif epoch - best_epoch >= patience:
                   if verbose:
                       print(f"Early stopping at epoch {epoch}")
                   model.load_state_dict(best_state)
                   break
       
       # Print progress
       if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
           log_str = f"Epoch {epoch+1}/{n_epochs}"
           log_str += f" - Loss: {train_stats.get('loss', 0):.4f}"
           log_str += f" - Accuracy: {train_metrics['accuracy']:.2f}%"
           
           if val_loader is not None:
               log_str += f" - Val Loss: {val_metrics['loss']:.4f}"
               log_str += f" - Val Accuracy: {val_metrics['accuracy']:.2f}%"
               
           print(log_str)
   
   return history

class SuperpositionExperiment:
   """Experiment framework for measuring superposition in adversarially trained models."""
   #Here is where we train the SAE Change the distribution here 
   def __init__(
       self,
       train_dataset,
       val_dataset,
       model_type: str = 'mlp',
       hidden_dim: int = 32,
       image_size: int = 28,
       epsilon=0,
       device: Optional[torch.device] = None
   ):
       """Initialize the experiment.
       
       Args:
           train_dataset: Training dataset
           val_dataset: Validation dataset
           model_type: Type of model to use ('mlp' or 'cnn')
           hidden_dim: Hidden dimension for MLP
           image_size: Image size
           device: Device to use
       """
       self.train_dataset = train_dataset
       self.val_dataset = val_dataset
       self.model_type = model_type
       self.hidden_dim = hidden_dim
       self.image_size = image_size
       self.device = device or torch.device(
           "cuda" if torch.cuda.is_available() else 
           "mps" if torch.backends.mps.is_available() else 
           "cpu"
       )
       
       # Create dataloaders
       self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
       self.val_loader = DataLoader(val_dataset, batch_size=64)
       #TODO: Apply L1 noise to SAE
       if(epsilon > 0): 
           pass
       # Initialize model
       # Get number of classes from dataset
       num_classes = len(set(y.item() for _, y in train_dataset))
       # Use 1 for binary classification, actual count for multi-class
       output_dim = 1 if num_classes <= 2 else num_classes
       
       print(f"Initializing {model_type.upper()} classifier over {output_dim} classes")
       self.model = create_model(
           model_type=model_type,
           hidden_dim=hidden_dim,
           image_size=image_size,
           num_classes=output_dim, 
           use_nnsight=False
       ).to(self.device)
       
       # Initialize optimizer
       self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
       
       # Initialize history
       self.history = {}
       
       # Test epsilons for evaluation
       self.test_epsilons = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]

   def save(self, path: Path) -> None:
       """Save model state, optimizer state, and experiment history."""
       path.parent.mkdir(parents=True, exist_ok=True)
       state = {
           'model_state': self.model.state_dict(),
           'optimizer_state': self.optimizer.state_dict(),
           'history': dict(self.history),
           'config': {
               'model_type': self.model_type,
               'hidden_dim': self.hidden_dim,
               'image_size': self.image_size,
               'test_epsilons': self.test_epsilons
           }
       }
       torch.save(state, path)
   
   @classmethod
   def load(cls, path: Path, train_dataset, val_dataset) -> 'SuperpositionExperiment':
       """Load experiment from saved state."""
       state = torch.load(path)
       exp = cls(
           train_dataset=train_dataset,
           val_dataset=val_dataset,
           model_type=state['config']['model_type'],
           hidden_dim=state['config']['hidden_dim'],
           image_size=state['config']['image_size']
       )
       exp.model.load_state_dict(state['model_state'])
       exp.optimizer.load_state_dict(state['optimizer_state'])
       exp.history = state['history']
       exp.test_epsilons = state['config']['test_epsilons']
       return exp
   
   def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
       """Log metrics to history."""
       for k, v in metrics.items():
           key = f"{prefix}{k}" if prefix else k
           if key not in self.history:
               self.history[key] = []
           self.history[key].append(v)
   
   def train(
       self,
       n_epochs: int,
       epsilon: float = 0.0,
       alpha: float = 0.5,
       measure_superposition: bool = False,
       measure_frequency: int = 5
   ):
       """Train model and track superposition metrics.
       
       Args:
           n_epochs: Number of training epochs
           epsilon: Perturbation size for adversarial training
           alpha: Weight for clean vs adversarial loss
           measure_superposition: Whether to measure superposition
           measure_frequency: How often to measure superposition
       """
       # Binary or multi-class criterion
       only_one_output_class = list(self.model.parameters())[-1].size(0) == 1
       if only_one_output_class:
           print("Binary classification")
           criterion = nn.BCEWithLogitsLoss()
       else:
           print("Multi-class classification")
           criterion = nn.CrossEntropyLoss()
       
       for epoch in range(n_epochs):
           # Training
           if epsilon > 0:
               stats = train_epoch(
                   self.model, self.train_loader, self.optimizer, criterion, epsilon, alpha
               )
           else:
               stats = train_epoch(
                   self.model, self.train_loader, self.optimizer, criterion
               )
           
           # Evaluation
           val_stats = evaluate(self.model, self.val_loader, criterion)
           
           # Log metrics
           self._log_metrics(stats, "train_")
           self._log_metrics(val_stats, "val_")
           
           # Evaluate adversarial robustness periodically
           if epoch % (n_epochs // measure_frequency) == 0:
               adv_accuracies = evaluate_adversarial_robustness(
                   self.model, 
                   self.val_loader,
                   self.test_epsilons
               )
               self._log_metrics(adv_accuracies, "adv_acc_")
               
               # Measure superposition
               if measure_superposition:
                   from metrics import measure_superposition
                   layer_name = 'after_relu' if self.model_type.lower() == 'mlp' else 'conv_features'
                   metrics = measure_superposition(self.model, self.train_loader, layer_name)
                   self._log_metrics(metrics, "superposition_")
               
               # Print progress
               print(f"Epoch {epoch}/{n_epochs}")
               print(f"Train loss: {stats.get('loss', 0):.4f}")
               print(f"Val accuracy: {val_stats['accuracy']:.2f}%")
               print("Adversarial accuracies:")
               for eps in self.test_epsilons:
                   print(f"ε={eps:.3f}: {adv_accuracies[eps]:.1f}%")
               if measure_superposition:
                   print(f"Superposition: {metrics['feature_count']:.2f}")
               print("-" * 40)
# %%
