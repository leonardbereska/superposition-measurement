# %%
"""Sparse autoencoder implementation for feature discovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

# %%
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

# %%
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

# %%
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

# %%
def train_sae(
    activations: torch.Tensor, 
    expansion_factor: float = 2.0, 
    fixed_size: Optional[int] = None,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    l1_lambda: float = 0.1,
    verbose: bool = False,
    device: Optional[torch.device] = None
) -> TiedSparseAutoencoder:
    """Train a Sparse Autoencoder on neural network activations.
    
    Args:
        activations: Tensor of shape (n_samples, input_dim) containing activations to fit
        expansion_factor: Multiplier for dictionary dimension
        fixed_size: If not None, fix the dictionary dimension to this value
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        l1_lambda: L1 regularization strength
        verbose: Whether to print progress
        device: Device to use for training
        
    Returns:
        Trained TiedSparseAutoencoder
    """
    if verbose:
        print(f"Training SAE on activations with shape {activations.shape}...")
    
    # Determine dictionary size
    input_dim = activations.shape[1]
    if fixed_size:
        dictionary_dim = fixed_size
    else:
        dictionary_dim = int(input_dim * expansion_factor)
    
    # Create SAE model
    sae = TiedSparseAutoencoder(input_dim, dictionary_dim, device)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        activations, 
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    
    # Train for specified epochs
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'sparsity_loss': []
    }
    
    for epoch in range(n_epochs):
        # Train for one epoch
        epoch_losses = train_sae_epoch(sae, dataloader, optimizer, l1_lambda, device)
        
        # Update history
        for k, v in epoch_losses.items():
            history[k].append(v)
        
        # Print progress
        if verbose and epoch % max(1, n_epochs // 10) == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Total Loss: {epoch_losses['total_loss']:.2e}, "
                f"Reconstruction: {epoch_losses['reconstruction_loss']:.2e}, "
                f"Sparsity: {epoch_losses['sparsity_loss']:.2e}"
            )
    
    return sae

# %%
def evaluate_sae(
    model: TiedSparseAutoencoder,
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
    model.eval()
    
    # Use model's device if not specified
    if device is None:
        device = next(model.parameters()).device
    
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
            activations, reconstruction = model(x)
            
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

# %%
def analyze_feature_statistics(
    model: TiedSparseAutoencoder,
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
    model.eval()
    
    # Use model's device if not specified
    if device is None:
        device = next(model.parameters()).device
    
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
            act, _ = model(x)
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
        'dictionary_norm': model.get_dictionary().norm(dim=1).cpu().numpy()
    }
    
    return stats