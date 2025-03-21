"""Sparse autoencoder implementation."""

import torch
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import torch.nn.functional as F

from config import DEVICE

class BaseSparseAutoencoder(torch.nn.Module):
    """Abstract base class for sparse autoencoders."""
    
    def __init__(self, input_dim: int, dictionary_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.dictionary_dim = dictionary_dim
        assert dictionary_dim >= input_dim, "Dictionary dimension must be >= input dimension"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both activations and reconstruction."""
        raise NotImplementedError
    
    def get_dictionary(self) -> torch.Tensor:
        """Return the learned dictionary (encoder weights)."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save model state and config."""
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
    def load(cls, path: Path) -> 'BaseSparseAutoencoder':
        """Load model from saved state."""
        state = torch.load(path)
        model = cls(state['config']['input_dim'], state['config']['dictionary_dim'])
        model.load_state_dict(state['state_dict'])
        return model

class SparseAutoencoder(BaseSparseAutoencoder):
    """Standard sparse autoencoder with separate encoder/decoder weights."""
    
    def __init__(self, input_dim: int, dictionary_dim: int):
        super().__init__(input_dim, dictionary_dim)
        self.encoder = torch.nn.Linear(input_dim, dictionary_dim, bias=False)
        self.decoder = torch.nn.Linear(dictionary_dim, input_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.encoder(x)
        reconstructed = self.decoder(F.relu(activations))
        return activations, reconstructed
    
    def get_dictionary(self) -> torch.Tensor:
        return self.encoder.weight.data

class TiedSparseAutoencoder(BaseSparseAutoencoder):
    """Sparse autoencoder with tied encoder/decoder weights."""
    
    def __init__(self, input_dim: int, dictionary_dim: int):
        super().__init__(input_dim, dictionary_dim)
        self.encoder = torch.nn.Linear(input_dim, dictionary_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        activations = self.encoder(x)
        reconstructed = F.linear(
            F.relu(activations), 
            self.encoder.weight.t(),
            bias=None
        )
        return activations, reconstructed
    
    def get_dictionary(self) -> torch.Tensor:
        return self.encoder.weight.data

class SAETrainer:
    """Trainer for sparse autoencoder."""
    
    def __init__(
        self,
        model: Union[SparseAutoencoder, TiedSparseAutoencoder],
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        l1_lambda: float = 0.01,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.device = device or DEVICE
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate
        )
        
    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with detailed metrics."""
        x = x.to(self.device)
        activations, reconstructed = self.model(x)
        recon_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = self.l1_lambda * activations.abs().mean()
        total_loss = recon_loss + sparsity_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }
        return total_loss, loss_dict

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'sparsity_loss': 0.0
        }
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss, loss_dict = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= len(dataloader)
            
        return epoch_losses
    
    def train(self, dataloader: torch.utils.data.DataLoader, verbose: bool = True) -> Dict[str, list]:
        """Train the SAE.
        
        Args:
            dataloader: DataLoader with training data
            verbose: Whether to print progress
            
        Returns:
            Dictionary of loss histories
        """
        self.model.train()
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': []
        }
        
        for epoch in range(self.n_epochs):
            epoch_losses = self.train_epoch(dataloader)
            
            # Update history
            for k, v in epoch_losses.items():
                history[k].append(v)
            
            if verbose and epoch % (self.n_epochs // 5) == 0:
                print(
                    f"Epoch {epoch+1}/{self.n_epochs} - "
                    f"Total Loss: {epoch_losses['total_loss']:.2e}, "
                    f"Reconstruction: {epoch_losses['reconstruction_loss']:.2e}, "
                    f"Sparsity: {epoch_losses['sparsity_loss']:.2e}"
                )
        
        return history

def train_sae(
    activations: torch.Tensor, 
    expansion_factor: float = 2.0, 
    fixed_size: Optional[int] = None,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    l1_lambda: float = 0.1,
    verbose: bool = True
) -> TiedSparseAutoencoder:
    """Train a Sparse Autoencoder on compressed activations.
    
    Args:
        activations: Tensor of shape (n_samples, input_dim) containing activations to fit
        expansion_factor: Multiplier for dictionary dimension
        fixed_size: If not None, fix the dictionary dimension to this value
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        l1_lambda: L1 regularization strength
        verbose: Whether to print progress
        
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
    sae = TiedSparseAutoencoder(input_dim, dictionary_dim)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        activations, 
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create trainer and train
    trainer = SAETrainer(
        model=sae,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l1_lambda=l1_lambda
    )
    
    trainer.train(data_loader, verbose=verbose)
    
    return sae
