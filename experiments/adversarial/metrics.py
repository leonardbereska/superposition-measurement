# %%
"""Feature measurement for adversarial robustness experiments."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from sae import train_sae
from models import NNsightModelWrapper

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
    model: torch.nn.Module,
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

    model = NNsightModelWrapper(model)

    # limiting the number of activations to extract to avoid memory issues
    activations = model.get_activations(data_loader, layer_name, max_activations=max_samples)

    activations = activations.detach().cpu().contiguous()

    # Check if the activations are from a convolutional layer
    is_conv = activations.dim() == 4
    
    if is_conv:
        # Reshape: [B, C, H, W] -> [B*H*W, C]
        orig_shape = activations.shape
        activations_reshaped = activations.reshape(
            orig_shape[0], orig_shape[1], -1).permute(0, 2, 1).reshape(-1, orig_shape[1])
        
        # Limit number of samples for SAE training if needed
        if max_samples and len(activations_reshaped) > max_samples:
            train_activations = activations_reshaped[:max_samples]
        else:
            train_activations = activations_reshaped
        
        # Train SAE if not provided
        if sae_model is None:
            sae_model = train_sae(
                train_activations, 
                expansion_factor=expansion_factor, 
                n_epochs=n_epochs, 
                l1_lambda=l1_lambda, 
                batch_size=batch_size, 
                learning_rate=learning_rate
            )
        
        # Apply SAE to get activations
        sae_acts, _ = sae_model(activations_reshaped.to(device))
        sae_acts = sae_acts.detach().cpu()
        
        # Reshape back: [B*H*W, D] -> [B, H*W, D] -> [B, D, H*W]
        sae_acts = sae_acts.reshape(
            orig_shape[0], 
            -1,  # H*W
            sae_model.dictionary_dim
        ).permute(0, 2, 1)
        
        # Sum over spatial dimensions
        sae_acts_summed = sae_acts.sum(dim=2).numpy()
        
        # Calculate metrics based on feature importance across spatial dimensions
        p = get_feature_distribution(sae_acts_summed)
        
    else:
        # For fully connected layers, use the existing approach
        if sae_model is None:
            from sae import train_sae
            sae_model = train_sae(
                activations, 
                expansion_factor=expansion_factor, 
                n_epochs=n_epochs, 
                l1_lambda=l1_lambda, 
                batch_size=batch_size, 
                learning_rate=learning_rate
            )
        
        # Get SAE features
        sae_acts, _ = sae_model(activations.to(device))
        sae_acts = sae_acts.detach().cpu().numpy()
        
        # Calculate metrics with the original approach
        p = get_feature_distribution(sae_acts)
    
    # Save SAE model if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        sae_model.save(save_dir / f"sae_{layer_name}.pth")
    
    # Calculate metrics
    metrics = calculate_feature_metrics(p)
    
    return metrics

def measure_superposition_on_mixed_distribution(
    model: torch.nn.Module,
    clean_loader: torch.utils.data.DataLoader,
    epsilon: float,
    layer_name: str,
    **sae_params
) -> Dict[str, Dict[str, float]]:
    """Measure superposition using SAEs trained on different distributions."""
    device = next(model.parameters()).device
    
    # Get clean activations
    clean_activations = model.get_activations(clean_loader, layer_name)
    
    # Generate adversarial examples and get their activations
    adv_activations = []
    for inputs, targets in clean_loader:
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
            adv_acts = model.get_activations_for_input(perturbed_inputs, layer_name)
            adv_activations.append(adv_acts.cpu())
    
    adv_activations = torch.cat(adv_activations, dim=0)
    
    # Create mixed dataset (50% clean, 50% adversarial)
    mixed_activations = torch.cat([
        clean_activations[:min(len(clean_activations), len(adv_activations))],
        adv_activations[:min(len(clean_activations), len(adv_activations))]
    ], dim=0)
    
    # Train SAEs on different distributions
    results = {}
    for name, acts in [
        ("clean", clean_activations),
        ("adversarial", adv_activations),
        ("mixed", mixed_activations)
    ]:
        sae = train_sae(acts, **sae_params)
        
        # Measure feature count on each distribution using this SAE
        metrics = {}
        for eval_name, eval_acts in [
            ("clean", clean_activations),
            ("adversarial", adv_activations),
            ("mixed", mixed_activations)
        ]:
            sae_acts, _ = sae(eval_acts.to(device))
            p = get_feature_distribution(sae_acts.cpu().numpy())
            metrics[eval_name] = calculate_feature_metrics(p)
        
        results[name] = metrics
     
    return results

# %%
if __name__ == "__main__":
    # Test the measure_superposition function with a CNN model
    from models import CNN
    from datasets import MNISTDataset
    from torch.utils.data import DataLoader
    
    def test_cnn_superposition():
        # Create model and data loader
        model = CNN(28, 10)
        data_loader = DataLoader(MNISTDataset(28, 10), batch_size=128, shuffle=True)
        
        # Get convolutional feature activations (shape: [B, C, H, W])
        orig_activations = model.get_activations(data_loader, "conv_features")
        print(f"Original activations shape: {orig_activations.shape}")
        
        # Reshape activations to apply SAE on channel dimension
        # Flatten spatial dimensions: (B, C, H, W) -> (B*H*W, C)
        activations = orig_activations.view(-1, orig_activations.shape[1])
        print(f"Reshaped for SAE training: {activations.shape}")
        # only take limited number of samples for training the SAE
        activations = activations[:10000]
        print(f"Cropped for SAE training: {activations.shape}")
        # Train SAE on channel dimension (shared across spatial locations)
        from sae import train_sae
        sae_model = train_sae(
            activations, 
            expansion_factor=2, 
            n_epochs=300, 
            l1_lambda=0.1, 
            batch_size=128, 
            learning_rate=1e-3
        )
        
        # Apply SAE to get feature activations
        device = next(model.parameters()).device
        activations = activations.to(device)
        sae_model.to(device)
        sae_acts, _ = sae_model(activations)
        sae_acts = sae_acts.detach().cpu().numpy()
        
        # Reshape SAE activations back to include spatial dimensions
        # (B*H*W, D) -> (B, D, H*W)
        sae_acts = sae_acts.reshape(
            orig_activations.shape[0],  # Batch size
            -1,                         # Dictionary size (D)
            orig_activations.shape[2] * orig_activations.shape[3]  # H*W
        )
        print(f"Reshaped SAE activations: {sae_acts.shape}")
        
        # Sum over spatial dimensions to get feature importance across space
        sae_acts_summed = sae_acts.sum(axis=2)
        print(f"After summing over spatial dimension: {sae_acts_summed.shape}")
        
        # Calculate feature distribution and metrics
        p = get_feature_distribution(sae_acts_summed)
        metrics = calculate_feature_metrics(p)
        print(f"Feature metrics: {metrics}")
        
        return metrics




if __name__ == "__main__":
    test_cnn_superposition()

# %%
