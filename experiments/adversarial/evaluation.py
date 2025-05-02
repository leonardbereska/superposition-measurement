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

from sae import train_sae, evaluate_sae, analyze_feature_statistics
from models import create_model, NNsightModelWrapper
from utils import json_serializer

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

# def evaluate_feature_organization(
#     model: torch.nn.Module,
#     train_loader: torch.utils.data.DataLoader,
#     layer_name: str,
#     use_mixed_distribution: bool = False,
#     epsilon: float = 0.1,  # Only used for mixed distribution
#     expansion_factor: int = 2,
#     l1_lambda: float = 0.1,
#     batch_size: int = 128,
#     learning_rate: float = 1e-3,
#     n_epochs: int = 300,
#     save_dir: Optional[Path] = None,
#     max_samples: int = 10000
# ) -> Dict[str, Any]:
#     """Measure feature organization in a trained model.
    
#     Args:
#         model: Trained model to evaluate
#         train_loader: DataLoader with training data
#         layer_name: Layer name for activation extraction
#         use_mixed_distribution: Whether to use mixed distribution
#         epsilon: Perturbation size for mixed distribution
#         expansion_factor: SAE expansion factor
#         l1_lambda: L1 regularization strength
#         batch_size: Batch size for SAE training
#         learning_rate: Learning rate for SAE training
#         n_epochs: Number of epochs for SAE training
#         save_dir: Directory to save results
#         max_samples: Maximum number of samples to use
        
#     Returns:
#         Dictionary with feature organization metrics
#     """
#     if use_mixed_distribution:
#         return measure_superposition_on_mixed_distribution(
#             model=model,
#             clean_loader=train_loader,
#             layer_name=layer_name,
#             epsilon=epsilon,
#             expansion_factor=expansion_factor,
#             l1_lambda=l1_lambda,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             n_epochs=n_epochs,
#             save_dir=save_dir,
#             max_samples=max_samples
#         )
#     else:
#         return measure_superposition(
#             model=model,
#             data_loader=train_loader,
#             layer_name=layer_name,
#             expansion_factor=expansion_factor,
#             l1_lambda=l1_lambda,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             n_epochs=n_epochs,
#             save_dir=save_dir,
#             max_samples=max_samples
#         )


def get_feature_distribution(activations: torch.Tensor) -> np.ndarray:
    """Get distribution of activation magnitudes per feature.
    
    Args:
        activations: Tensor of activations [batch_size, feature_dim]
        
    Returns:
        Normalized distribution of feature importance
    """
    feature_norms = np.mean(np.abs(activations), axis=0)
    p = feature_norms / feature_norms.sum()
    return p

def calculate_feature_metrics(p: np.ndarray) -> Dict[str, float]:
    """Calculate entropy and feature count metrics from activation distribution.
    
    Args:
        p: Normalized distribution of feature importance
        
    Returns:
        Dictionary of metrics
    """
    # Add epsilon to avoid log(0)
    p_safe = p + 1e-10
    p_safe = p_safe / p_safe.sum()  # Renormalize
    
    # Calculate entropy
    entropy = -np.sum(p_safe * np.log(p_safe))
    
    # Effective feature count (exponential of entropy)
    feature_count = np.exp(entropy)
    
    # Additional metrics could be added here
    return {
        'entropy': entropy,
        'feature_count': feature_count,
    }

def measure_superposition(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    layer_name: str,
    sae_model=None,
    expansion_factor: int = 2,
    l1_lambda: float = 0.1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    n_epochs: int = 300,
    save_dir: Optional[Path] = None,
    max_samples: int = 10000
) -> Dict[str, float]:
    """Measure feature superposition in a model layer."""
    device = next(model.parameters()).device

    model_wrapper = NNsightModelWrapper(model)

    # limiting the number of activations to extract to avoid memory issues
    activations = model_wrapper.get_activations(data_loader, layer_name, max_activations=max_samples)

    activations = activations.detach().cpu().contiguous()

    # Check if the activations are from a convolutional layer
    is_conv = activations.dim() == 4
    
    if is_conv:
        # Reshape: [B, C, H, W] -> [B*H*W, C]
        orig_shape = activations.shape
        activations = activations.reshape(
            orig_shape[0], orig_shape[1], -1).permute(0, 2, 1).reshape(-1, orig_shape[1])
        
        # Limit number of samples for SAE training if needed
        if max_samples and len(activations) > max_samples:
            train_activations = activations[:max_samples]
        else:
            train_activations = activations
        
        # Train SAE if not provided
        if sae_model is None:
            sae_model = train_sae(
                train_activations, 
                expansion_factor=expansion_factor, 
                n_epochs=n_epochs, 
                l1_lambda=l1_lambda, 
                batch_size=batch_size, 
                learning_rate=learning_rate,
                device=device
            )
        
        # Apply SAE to get activations
        sae_acts, _ = sae_model(activations.to(device))
        sae_acts = sae_acts.detach().cpu()
        
        # Reshape back: [B*H*W, D] -> [B, H*W, D] -> [B, D, H*W]
        sae_acts = sae_acts.reshape(
            orig_shape[0], 
            -1,  # H*W
            sae_model.dictionary_dim
        ).permute(0, 2, 1)
        
        # Sum over spatial dimensions
        sae_acts = sae_acts.sum(dim=2).detach().cpu().numpy()
        
    else:
        # For fully connected layers, use the existing approach
        if sae_model is None:
            sae_model = train_sae(
                activations, 
                verbose=True,
                expansion_factor=expansion_factor, 
                n_epochs=n_epochs, 
                l1_lambda=l1_lambda, 
                batch_size=batch_size, 
                learning_rate=learning_rate,
                device=device
            )
        
        # Get SAE features
        sae_acts, _ = sae_model(activations.to(device))
        sae_acts = sae_acts.detach().cpu().numpy()
    
    # Save SAE model if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save SAE model state dict
        torch.save(sae_model.state_dict(), save_dir / f"sae_{layer_name}.pt")
        
        # Evaluate SAE and save statistics
        sae_metrics = evaluate_sae(
            sae=sae_model,
            activations=activations,
            l1_lambda=l1_lambda,
            batch_size=batch_size,
            device=device
        )

        sae_stats = analyze_feature_statistics(
            sae=sae_model,
            activations=activations,
            device=device
        )
        
        # Save statistics as JSON
        with open(save_dir / f"sae_{layer_name}_metrics.json", 'w') as f:
            json.dump({k: float(v) for k, v in sae_metrics.items()}, f, indent=4, default=json_serializer)
        with open(save_dir / f"sae_{layer_name}_stats.json", 'w') as f:
            json.dump(sae_stats, f, indent=4, default=json_serializer)
    
    # Calculate metrics
    p = get_feature_distribution(sae_acts)
    metrics = calculate_feature_metrics(p)
    
    return metrics

def measure_superposition_on_mixed_distribution(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    epsilon: float = 0.1,
    expansion_factor: int = 2,
    l1_lambda: float = 0.1,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    n_epochs: int = 300,
    save_dir: Optional[Path] = None,
    max_samples: int = 10000,
    verbose: bool = True
) -> Dict[str, float]:
    """Measure feature organization across different input distributions.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with clean data
        layer_name: Layer to extract activations from
        epsilon: Perturbation size matching training epsilon
        expansion_factor: SAE expansion factor
        l1_lambda: L1 regularization strength
        batch_size: Batch size for SAE training
        learning_rate: Learning rate for SAE training
        n_epochs: Number of epochs for SAE training
        save_dir: Directory to save results
        max_samples: Maximum number of samples to use
        verbose: Whether to print progress
        
    Returns:
        Dictionary with feature counts for clean, adversarial, and mixed distributions
    """
    device = next(model.parameters()).device
    model_wrapper = NNsightModelWrapper(model)
    
    # Extract clean activations
    clean_activations = model_wrapper.get_activations(
        dataloader, layer_name, max_activations=max_samples
    )
    clean_activations = clean_activations.detach().cpu()
    
    # Generate adversarial examples and extract their activations
    adv_activations = []
    for inputs, targets in dataloader:
        if len(adv_activations) >= max_samples:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
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
        perturbed_inputs = inputs + epsilon * inputs.grad.sign()
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # Get activations for adversarial examples
        with torch.no_grad():
            adv_acts = model_wrapper.get_layer_output(perturbed_inputs, layer_name)
            adv_activations.append(adv_acts.cpu())
    
    adv_activations = torch.cat(adv_activations[:max_samples//len(adv_activations[0])], dim=0)
    
    # Create mixed dataset (50% clean, 50% adversarial)
    min_samples = min(len(clean_activations), len(adv_activations), max_samples // 2)
    mixed_activations = torch.cat([
        clean_activations[:min_samples],
        adv_activations[:min_samples]
    ], dim=0)
    
    # Dictionary to store results
    results = {}
    
    # Define distributions to evaluate
    distributions = {
        "clean": clean_activations,
        "adversarial": adv_activations,
        "mixed": mixed_activations
    }
    
    # For each distribution, train and evaluate an SAE
    for dist_name, activations in distributions.items():
        if verbose:
            print(f"Analyzing {dist_name} distribution...")
        
        # Train SAE on this distribution
        sae = train_sae(
            activations,
            expansion_factor=expansion_factor, 
            n_epochs=n_epochs, 
            l1_lambda=l1_lambda, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            device=device,
            verbose=verbose
        )
        
        # Save SAE model if requested
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(sae.state_dict(), save_dir / f"sae_{layer_name}_{dist_name}.pt")
        
        # Evaluate SAE on the same distribution it was trained on
        sae_acts, _ = sae(activations.to(device))
        
        # Calculate feature distribution and metrics
        p = get_feature_distribution(sae_acts.detach().cpu().numpy())
        metrics = calculate_feature_metrics(p)
        
        # Store feature count
        results[dist_name+"_feature_count"] = metrics["feature_count"]
        
        if verbose:
            print(f"  Feature count: {metrics['feature_count']:.2f}")
    
    return results

# def measure_superposition_on_mixed_distribution(
#     model: nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     layer_name: str,
#     epsilon: float = 0.1,
#     expansion_factor: int = 2,
#     l1_lambda: float = 0.1,
#     batch_size: int = 128,
#     learning_rate: float = 1e-3,
#     n_epochs: int = 300,
#     save_dir: Optional[Path] = None,
#     max_samples: int = 10000,
#     verbose: bool = True
# ) -> Dict[str, Dict[str, Dict[str, float]]]:
#     """Measure superposition using SAEs trained on different distributions."""

#     device = next(model.parameters()).device
#     model_wrapper = NNsightModelWrapper(model)
    
#     # Get clean activations
#     clean_loader = dataloader
#     clean_activations = model_wrapper.get_activations(clean_loader, layer_name, max_activations=max_samples)
#     clean_activations = clean_activations.detach().cpu()
    
#     # Generate adversarial examples and get their activations
#     adv_activations = []
#     for inputs, targets in clean_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
        
#         # Generate adversarial examples with FGSM
#         inputs.requires_grad = True
#         outputs = model(inputs)
        
#         if outputs.size(-1) == 1:  # Binary classification
#             loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets)
#         else:  # Multi-class classification
#             loss = nn.CrossEntropyLoss()(outputs, targets)
        
#         model.zero_grad()
#         loss.backward()
        
#         # FGSM attack
#         perturbed_inputs = inputs + epsilon * inputs.grad.sign()
#         perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
#         # Get activations for adversarial examples
#         with torch.no_grad():
#             adv_acts = model_wrapper.get_layer_output(perturbed_inputs, layer_name)
#             adv_activations.append(adv_acts.cpu())
    
#     adv_activations = torch.cat(adv_activations, dim=0)
    
#     # Create mixed dataset (50% clean, 50% adversarial)
#     min_samples = min(len(clean_activations), len(adv_activations), max_samples // 2)
#     mixed_activations = torch.cat([
#         clean_activations[:min_samples],
#         adv_activations[:min_samples]
#     ], dim=0)
    
#     # Train SAEs and evaluate feature metrics
#     results = {}
    
#     # Define all activation distributions
#     distributions = {
#         "clean": clean_activations,
#         "adversarial": adv_activations,
#         "mixed": mixed_activations
#     }
    
#     # For each distribution, train an SAE and evaluate it on all distributions
#     for train_dist_name, train_activations in distributions.items():
#         # Train SAE on this distribution
#         sae = train_sae(
#             train_activations,
#             verbose=verbose,
#             expansion_factor=expansion_factor, 
#             n_epochs=n_epochs, 
#             l1_lambda=l1_lambda, 
#             batch_size=batch_size, 
#             learning_rate=learning_rate,
#             device=device
#         )
        
#         # Save SAE model if requested
#         if save_dir:
#             save_dir.mkdir(parents=True, exist_ok=True)
#             torch.save(sae.state_dict(), save_dir / f"sae_{layer_name}_{train_dist_name}.pt")
        
#         # Evaluate this SAE on each distribution
#         metrics = {}
#         for eval_dist_name, eval_activations in distributions.items():
#             if verbose:
#                 print(f"Evaluating SAE trained on {train_dist_name} distribution on {eval_dist_name} data...")
            
#             # Get SAE activations for this distribution
#             sae_acts, _ = sae(eval_activations.to(device))
            
#             # Calculate feature distribution and metrics
#             p = get_feature_distribution(sae_acts.detach().cpu().numpy())
#             metrics[eval_dist_name] = calculate_feature_metrics(p)
            
#             if verbose:
#                 feature_count = metrics[eval_dist_name]['feature_count']
#                 print(f"  Feature count: {feature_count:.2f}")
        
#         # Store metrics for this SAE
#         results[train_dist_name] = metrics
    
#     return results

# %%
# %%
