"""Sparse autoencoder implementation for feature discovery."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any, NamedTuple
from pathlib import Path
import logging


class SAEConfig(NamedTuple):
    """Configuration for Sparse Autoencoder."""
    expansion_factor: float
    l1_lambda: float
    batch_size: int
    learning_rate: float
    n_epochs: int

class TiedSparseAutoencoder(nn.Module):
    """Sparse autoencoder with tied encoder/decoder weights."""
    
    def __init__(self, input_dim: int, dictionary_dim: int, device: Optional[torch.device] = None):
        """Initialize sparse autoencoder.
        
        Args:
            input_dim: Dimension of input activations
            dictionary_dim: Dimension of dictionary (number of features)
            device: Device to place model on
        """
        super().__init__()
        self.input_dim = input_dim
        self.dictionary_dim = dictionary_dim
        
        # Validate dimensions
        if dictionary_dim < input_dim:
            raise ValueError(f"Dictionary dimension ({dictionary_dim}) must be >= input dimension ({input_dim})")
        
        # Initialize encoder (decoder uses transposed weights)
        self.encoder = nn.Linear(input_dim, dictionary_dim)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Move to device if specified
        if device is not None:
            self.to(device)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both activations and reconstruction.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (activations, reconstruction)
        """
        # Encode input
        activations = self.encoder(x)
        
        # Apply ReLU activation
        activations_relu = F.relu(activations)
        
        # Reconstruct with tied weights
        reconstruction = F.linear(
            activations_relu, 
            self.encoder.weight.t(),
            bias=None
        )
        
        return activations_relu, reconstruction
    
    def get_dictionary(self) -> torch.Tensor:
        """Return the learned dictionary (encoder weights).
        
        Returns:
            Dictionary matrix [dictionary_dim, input_dim]
        """
        return self.encoder.weight.data
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Encoded activations [batch_size, dictionary_dim]
        """
        return F.relu(self.encoder(x))
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Reconstruction [batch_size, input_dim]
        """
        activations_relu = self.encode(x)
        return F.linear(activations_relu, self.encoder.weight.t(), bias=None)
    
    def save(self, path: Path) -> None:
        """Save model state and config.
        
        Args:
            path: Path to save model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'dictionary_dim': self.dictionary_dim
            }
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: Path, device: Optional[torch.device] = None) -> 'TiedSparseAutoencoder':
        """Load model from saved state.
        
        Args:
            path: Path to saved model
            device: Device to load model onto
            
        Returns:
            Loaded model
        """
        state = torch.load(path, map_location=device if device else torch.device('cpu'))
        model = cls(
            input_dim=state['config']['input_dim'], 
            dictionary_dim=state['config']['dictionary_dim'],
            device=device
        )
        model.load_state_dict(state['state_dict'])
        return model

def compute_loss(
    model: TiedSparseAutoencoder,
    x: torch.Tensor,
    l1_lambda: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute SAE loss with L1 regularization.
    
    Args:
        model: Sparse autoencoder
        x: Input tensor
        l1_lambda: L1 regularization strength
        
    Returns:
        Tuple of (total_loss, loss_components)
    """
    # Forward pass
    activations, reconstruction = model(x)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, x)
    
    # Sparsity loss (L1 penalty)
    sparsity_loss = l1_lambda * activations.abs().mean()
    
    # Total loss
    total_loss = recon_loss + sparsity_loss
    
    # Loss components for logging
    loss_dict = {
        'total_loss': total_loss.item(),
        'reconstruction_loss': recon_loss.item(),
        'sparsity_loss': sparsity_loss.item()
    }
    
    return total_loss, loss_dict

def train_sae_epoch(
    model: TiedSparseAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    l1_lambda: float = 0.1,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Train SAE for one epoch.
    
    Args:
        model: Sparse autoencoder
        dataloader: DataLoader with training activations
        optimizer: Optimizer
        l1_lambda: L1 regularization strength
        device: Device to use for training
        
    Returns:
        Dictionary with average losses
    """
    # Set model to training mode
    model.train()
    
    # Use model's device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Track losses
    epoch_losses = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'sparsity_loss': 0.0
    }
    
    # Process batches
    for batch in dataloader:
        # Move batch to device
        x = batch.to(device)
        
        # Compute loss
        optimizer.zero_grad()
        loss, loss_dict = compute_loss(model, x, l1_lambda)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update epoch losses
        for k, v in loss_dict.items():
            epoch_losses[k] += v
    
    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= len(dataloader)
    
    return epoch_losses

def train_sae(
    activations: torch.Tensor, 
    sae_config: SAEConfig,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> TiedSparseAutoencoder:
    """Train a Sparse Autoencoder on neural network activations.
    
    Args:
        activations: Tensor of neural network activations to train on
        sae_config: Configuration for the sparse autoencoder
        device: Device to use for training
        logger: Optional logger for training progress
        
    Returns:
        Trained sparse autoencoder model
    """
    # Set up logging
    log = logger.info if logger else print
    
    # Initialize model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    input_dim = activations.shape[1]
    dictionary_dim = int(input_dim * sae_config.expansion_factor)
    log(f"\tTraining SAE with dictionary size {dictionary_dim} on input of size {input_dim} ({activations.shape[0]} samples)")
    sae = TiedSparseAutoencoder(input_dim, dictionary_dim, device)
    
    # Prepare data and optimizer
    dataloader = torch.utils.data.DataLoader(
        activations, 
        batch_size=sae_config.batch_size,
        shuffle=True
    )
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_config.learning_rate)
    
    # Initialize training state
    best_loss = float('inf')
    best_epoch = 0
    best_state_dict = None
    patience = 10  # Default patience value
    
    # Training loop
    for epoch in range(sae_config.n_epochs):
        # Train for one epoch
        epoch_losses = train_sae_epoch(
            sae, 
            dataloader, 
            optimizer, 
            sae_config.l1_lambda, 
            device
        )
        
        # Early stopping check
        current_loss = epoch_losses['total_loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_state_dict = sae.state_dict().copy()
        elif patience > 0 and epoch - best_epoch >= patience:
            log(f"\tEarly stopping at epoch {epoch+1}")
            sae.load_state_dict(best_state_dict)
            break
        
        # Log progress at regular intervals
        if epoch % max(1, sae_config.n_epochs // 10) == 0:
            log_message = f"\t\tEpoch {epoch+1:04d}/{sae_config.n_epochs}"
            log_message += f" | Total Loss: {epoch_losses['total_loss']:.2e}"
            log_message += f" | Reconstruction: {epoch_losses['reconstruction_loss']:.2e}"
            log_message += f" | Sparsity: {epoch_losses['sparsity_loss']:.2e}"
            log(log_message)
    
    # Load best model if early stopping was triggered
    if patience > 0 and best_state_dict is not None and epoch != best_epoch:
        sae.load_state_dict(best_state_dict)
        log(f"\t\tBest model from epoch {best_epoch+1} with loss: {best_loss:.2e}")
    
    # Get feature activations for the entire dataset
    sae.eval()
    with torch.no_grad():
        all_feature_activations = []
        for batch in torch.utils.data.DataLoader(activations, batch_size=sae_config.batch_size):
            batch = batch.to(device)
            feature_activations, _ = sae(batch)
            all_feature_activations.append(feature_activations.cpu())
        
        # Concatenate all batches
        all_feature_activations = torch.cat(all_feature_activations, dim=0)
    
    # Calculate feature statistics
    feature_means = all_feature_activations.mean(dim=0)
    feature_stds = all_feature_activations.std(dim=0)
    feature_max = all_feature_activations.max(dim=0)[0]
    feature_sparsity = (all_feature_activations == 0.0).float().mean(dim=0)
    
    # Calculate feature usage (how often each feature is active)
    feature_usage = (all_feature_activations > 0.0).float().mean(dim=0)
    
    # Log summary statistics
    log(f"\t\tActivation mean: {feature_means.mean().item():.4f}")
    log(f"\t\tActivation std: {feature_stds.mean().item():.4f}")
    log(f"\t\tActivation max: {feature_max.mean().item():.4f}")
    log(f"\t\tAverage feature sparsity: {feature_sparsity.mean().item():.4f}")
    log(f"\t\tAverage feature usage: {feature_usage.mean().item():.4f}")
    
    return sae

def evaluate_sae(
    sae: TiedSparseAutoencoder,
    activations: torch.Tensor,
    l1_lambda: float = 0.1,
    batch_size: int = 128,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate SAE on activations.
    
    Args:
        model: Trained sparse autoencoder
        activations: Activations to evaluate on
        l1_lambda: L1 regularization strength
        batch_size: Batch size for evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set model to evaluation mode
    sae.eval()
    
    # Use model's device if not specified
    if device is None:
        device = next(sae.parameters()).device
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        activations, 
        batch_size=batch_size,
        shuffle=False
    )
    
    # Track metrics
    metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'sparsity_loss': 0.0,
        'activation_sparsity': 0.0,
        'feature_usage': 0.0
    }
    
    # Process batches
    with torch.no_grad():
        total_samples = 0
        
        for batch in dataloader:
            # Move batch to device
            x = batch.to(device)
            batch_size = x.size(0)
            
            # Forward pass
            activations, reconstruction = sae(x)
            
            # Compute loss components
            recon_loss = F.mse_loss(reconstruction, x)
            sparsity_loss = l1_lambda * activations.abs().mean()
            total_loss = recon_loss + sparsity_loss
            
            # Compute activation sparsity (percent of zeros)
            activation_sparsity = (activations == 0).float().mean()
            
            # Compute feature usage (percent of features used at least once)
            feature_usage = (activations.sum(dim=0) > 0).float().mean()
            
            # Update metrics (weighted by batch size)
            metrics['total_loss'] += total_loss.item() * batch_size
            metrics['reconstruction_loss'] += recon_loss.item() * batch_size
            metrics['sparsity_loss'] += sparsity_loss.item() * batch_size
            metrics['activation_sparsity'] += activation_sparsity.item() * batch_size
            metrics['feature_usage'] += feature_usage.item() * batch_size
            
            total_samples += batch_size
    
    # Average metrics
    for k in metrics:
        metrics[k] /= total_samples
    
    return metrics

def analyze_feature_statistics(
    sae: TiedSparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 128,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Analyze feature usage statistics.
    
    Args:
        model: Trained sparse autoencoder
        activations: Activations to analyze
        batch_size: Batch size for processing
        device: Device to use for processing
        
    Returns:
        Dictionary with feature statistics
    """
    # Set model to evaluation mode
    sae.eval()
    
    # Use model's device if not specified
    if device is None:
        device = next(sae.parameters()).device
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        activations, 
        batch_size=batch_size,
        shuffle=False
    )
    
    # Collect activations from all batches
    all_activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            x = batch.to(device)
            
            # Get activations
            act, _ = sae(x)
            all_activations.append(act.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)
    
    # Calculate feature activation frequencies
    feature_freq = (all_activations > 0).float().mean(dim=0)
    
    # Calculate feature magnitudes
    feature_mag = all_activations.mean(dim=0)
    
    # Calculate feature statistics
    stats = {
        'feature_freq': feature_freq.numpy(),
        'feature_mag': feature_mag.numpy(),
        'active_features': (feature_freq > 0).sum().item(),
        'mean_sparsity': (all_activations == 0).float().mean().item(),
        'dictionary_norm': sae.get_dictionary().norm(dim=1).cpu().numpy()
    }
    
    return stats