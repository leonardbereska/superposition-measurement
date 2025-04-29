# %%
"""Evaluation utilities for adversarial robustness experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import json
import numpy as np
import os

from models import create_model, NNsightModelWrapper
from metrics import measure_superposition, measure_superposition_on_mixed_distribution
from utils import json_serializer

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

# %%
def evaluate_feature_organization(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    layer_name: str,
    use_mixed_distribution: bool = False,
    epsilon: float = 0.1,  # Only used for mixed distribution
    expansion_factor: int = 2,
    l1_lambda: float = 0.1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    n_epochs: int = 300,
    save_dir: Optional[Path] = None,
    max_samples: int = 10000
) -> Dict[str, Any]:
    """Measure feature organization in a trained model.
    
    Args:
        model: Trained model to evaluate
        train_loader: DataLoader with training data
        layer_name: Layer name for activation extraction
        use_mixed_distribution: Whether to use mixed distribution
        epsilon: Perturbation size for mixed distribution
        expansion_factor: SAE expansion factor
        l1_lambda: L1 regularization strength
        batch_size: Batch size for SAE training
        learning_rate: Learning rate for SAE training
        n_epochs: Number of epochs for SAE training
        save_dir: Directory to save results
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary with feature organization metrics
    """
    if use_mixed_distribution:
        return measure_superposition_on_mixed_distribution(
            model=model,
            clean_loader=train_loader,
            layer_name=layer_name,
            epsilon=epsilon,
            expansion_factor=expansion_factor,
            l1_lambda=l1_lambda,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            save_dir=save_dir,
            max_samples=max_samples
        )
    else:
        return measure_superposition(
            model=model,
            data_loader=train_loader,
            layer_name=layer_name,
            expansion_factor=expansion_factor,
            l1_lambda=l1_lambda,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            save_dir=save_dir,
            max_samples=max_samples
        )

# %%
def load_model(
    model_path: Path,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Load a model from a saved state.
    
    Args:
        model_path: Path to saved model
        device: Device to load model onto
        
    Returns:
        Loaded model
    """
    # Use default device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load state
    state = torch.load(model_path, map_location=device)
    
    # Create model
    if 'config' in state:
        config = state['config']
        
        # Get model parameters
        model_type = config.get('model_type', 'mlp')
        hidden_dim = config.get('hidden_dim', 32)
        image_size = config.get('image_size', 28)
        
        # Determine number of output classes from state dict
        if 'model_state' in state:
            output_dim = list(state['model_state'].items())[-1][1].size(0)
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
        model.load_state_dict(state['model_state'])
        
        # Move to device
        model = model.to(device)
        
        return model
    else:
        raise ValueError("Could not determine model configuration from saved state")

# %%
def evaluate_all_models_in_directory(
    results_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    layer_name: str,
    test_epsilons: List[float],
    sae_config: Dict[str, Any],
    device: torch.device,
    measure_mixed: bool = False
) -> Dict[str, Dict[float, List[float]]]:
    """Evaluate all models in a results directory.
    
    Args:
        results_dir: Path to results directory
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        layer_name: Layer name for activation extraction
        test_epsilons: List of epsilon values to test
        sae_config: SAE configuration
        device: Device to use
        measure_mixed: Whether to use mixed distribution
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize results
    results = {
        'feature_count': {},
        'robustness_score': {},
    }
    
    for test_eps in test_epsilons:
        results[f'accuracy_for_{test_eps}'] = {}
    
    # Process each epsilon directory
    for eps_dir in results_dir.glob("eps_*"):
        try:
            epsilon = float(eps_dir.name.split("_")[1])
            
            # Initialize lists for this epsilon
            for metric in results:
                results[metric][epsilon] = []
            
            # Process each run directory
            for run_dir in eps_dir.glob("run_*"):
                try:
                    # Create evaluation directory
                    eval_dir = run_dir / "evaluation"
                    eval_dir.mkdir(exist_ok=True)
                    
                    # Load model
                    model_path = run_dir / "model.pt"
                    model = load_model(model_path, device)
                    
                    # Evaluate feature organization
                    if measure_mixed:
                        # Use mixed distribution
                        feature_metrics = evaluate_feature_organization(
                            model=model,
                            train_loader=train_loader,
                            layer_name=layer_name,
                            use_mixed_distribution=True,
                            epsilon=epsilon,
                            expansion_factor=sae_config['expansion_factor'],
                            l1_lambda=sae_config['l1_lambda'],
                            batch_size=sae_config['batch_size'],
                            learning_rate=sae_config['learning_rate'],
                            n_epochs=sae_config['n_epochs'],
                            save_dir=eval_dir
                        )
                        # Use the feature count from mixed SAE on mixed distribution
                        feature_count = feature_metrics['mixed']['mixed']['feature_count']
                    else:
                        # Use standard distribution
                        feature_metrics = evaluate_feature_organization(
                            model=model,
                            train_loader=train_loader,
                            layer_name=layer_name,
                            expansion_factor=sae_config['expansion_factor'],
                            l1_lambda=sae_config['l1_lambda'],
                            batch_size=sae_config['batch_size'],
                            learning_rate=sae_config['learning_rate'],
                            n_epochs=sae_config['n_epochs'],
                            save_dir=eval_dir
                        )
                        feature_count = feature_metrics['feature_count']
                    
                    # Evaluate robustness
                    robustness = evaluate_model_performance(
                        model=model,
                        dataloader=val_loader,
                        epsilons=test_epsilons,
                        device=device
                    )
                    
                    # Calculate average robustness
                    avg_robustness = sum(float(v) for v in robustness.values()) / len(robustness)
                    
                    # Save results
                    with open(eval_dir / "results.json", 'w') as f:
                        json.dump({
                            'feature_count': feature_count,
                            'robustness_score': avg_robustness,
                            **{f'accuracy_for_{eps}': robustness[str(eps)] for eps in test_epsilons}
                        }, f, indent=4, default=json_serializer)
                    
                    # Store results
                    results['feature_count'][epsilon].append(feature_count)
                    results['robustness_score'][epsilon].append(avg_robustness)
                    for test_eps in test_epsilons:
                        results[f'accuracy_for_{test_eps}'][epsilon].append(float(robustness[str(test_eps)]))
                
                except Exception as e:
                    print(f"Error evaluating run in {run_dir}: {e}")
        
        except Exception as e:
            print(f"Error processing directory {eps_dir}: {e}")
    
    # Save combined results
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'feature_count': {str(eps): values for eps, values in results['feature_count'].items()},
            'robustness_score': {str(eps): values for eps, values in results['robustness_score'].items()},
            **{key: {str(eps): values for eps, values in values_dict.items()} 
               for key, values_dict in results.items() 
               if key.startswith('accuracy_for_')}
        }, f, indent=4, default=json_serializer)
    
    return results