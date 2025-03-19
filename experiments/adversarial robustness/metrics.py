"""Feature measurement for adversarial robustness experiments."""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

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
    save_dir: Optional[Path] = None
) -> Dict[str, float]:
    """Measure feature superposition in a model layer.
    
    Args:
        model: Model to analyze
        data_loader: DataLoader with samples for activation extraction
        layer_name: Name of layer to analyze
        sae_model: Pre-trained SAE model (if None, trains a new one)
        expansion_factor: Expansion factor for SAE dictionary size
        save_dir: Directory to save SAE model
        
    Returns:
        Dictionary of superposition metrics
    """
    device = next(model.parameters()).device
    
    # Get activations
    activations = model.get_activations(data_loader, layer_name)
    
    # Train SAE if not provided
    if sae_model is None:
        from sae import train_sae
        sae_model = train_sae(activations, expansion_factor=expansion_factor, n_epochs=n_epochs, l1_lambda=l1_lambda, batch_size=batch_size, learning_rate=learning_rate)
    
    # Get SAE features
    sae_acts, _ = sae_model(activations.to(device))
    sae_acts = sae_acts.detach().cpu().numpy()
    
    # Save SAE model if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        sae_model.save(save_dir / f"sae_{layer_name}.pth")
    
    # Calculate metrics
    p = get_feature_distribution(sae_acts)
    metrics = calculate_feature_metrics(p)
    
    return metrics