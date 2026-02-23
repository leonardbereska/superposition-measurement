"""Evaluation utilities for adversarial robustness experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import json
import numpy as np
import os
import logging

from sae import train_sae, SAEConfig
from models import NNsightModelWrapper


def measure_superposition(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    sae_config: SAEConfig,
    max_samples: int = 10000,
    save_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Measure feature organization for a given distribution.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with data (can be clean, adversarial, or mixed)
        layer_name: Layer to extract activations from
        sae_config: Configuration for the sparse autoencoder
        save_dir: Directory to save results
        max_samples: Maximum number of samples to use
        logger: Optional logger for printing information
    Returns:
        Dictionary with feature metrics
    """
    
    device = torch.device("cpu")  # because it's faster and saves memory on the GPU
    # model.to(device)
    model_wrapper = NNsightModelWrapper(model)
    
    # Extract activations
    activations = model_wrapper.get_activations(
        dataloader, layer_name, max_activations=max_samples
    )
    activations = activations.detach().cpu()
    
    # Check if the activations are from a convolutional layer
    is_conv = activations.dim() == 4
    
    if is_conv:
        # Reshape: [B, C, H, W] -> [B*H*W, C]
        orig_shape = activations.shape
        reshaped_activations = activations.reshape(
            orig_shape[0], orig_shape[1], -1).permute(0, 2, 1).reshape(-1, orig_shape[1])
        
        # Limit number of samples for SAE training if needed
        if max_samples and len(reshaped_activations) > max_samples:
            train_activations = reshaped_activations[:max_samples]
        else:
            train_activations = reshaped_activations
        
        # Train SAE on this distribution
        sae = train_sae(
            train_activations,
            sae_config=sae_config,
            device=device,
            logger=logger
        )
        
        # Apply SAE to get activations
        sae_acts, _ = sae(reshaped_activations.to(device))
        sae_acts = sae_acts.detach().cpu()
        
        # Reshape back: [B*H*W, D] -> [B, H*W, D] -> [B, D, H*W]
        sae_acts = sae_acts.reshape(
            orig_shape[0], 
            -1,  # H*W
            sae.dictionary_dim
        ).permute(0, 2, 1)
        
        # Sum over spatial dimensions
        sae_acts = sae_acts.sum(dim=2).detach().cpu().numpy()
        
    else:
        # For fully connected layers, use the existing approach
        # Train SAE on this distribution
        sae = train_sae(
            activations,
            sae_config=sae_config,
            device=device,
            logger=logger
        )
        
        # Evaluate SAE on the same distribution it was trained on
        sae_acts, _ = sae(activations.to(device))
        sae_acts = sae_acts.detach().cpu().numpy()
    
    # Save SAE model if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_dir / f"sae_{layer_name}.pt")
    
    # Calculate feature distribution and metrics
    p = get_feature_distribution(sae_acts)
    metrics = calculate_feature_metrics(p)
    
    return metrics


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
