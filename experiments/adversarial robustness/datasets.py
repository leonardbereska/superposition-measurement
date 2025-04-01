# %%
"""Dataset utilities for MNIST and derivatives."""

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    """MNIST dataset with optional downsampling and class selection."""
    
    def __init__(
        self, 
        target_size=28,
        train=True,
        root='../../data',
        download=True,
        selected_classes=None  # None for all classes, list/tuple for specific classes 
    ):
        """
        Args:
            target_size: Size to downsample images to (default: original 28)
            train: If True, use training set
            root: Path to store the dataset
            download: If True, download the dataset
            selected_classes: Optional list/tuple of classes to include
        """
        # Load original MNIST
        self.original_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download
        )
        
        # Convert to torch tensor and scale to [0, 1]
        self.data = self.original_dataset.data.float() / 255.0
        
        # Add channel dimension and downsample
        self.data = self.data.unsqueeze(1)  # Add channel dimension [N, 1, H, W]
        if target_size != 28:  # Only downsample if target size is different
            self.data = F.interpolate(
                self.data,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=True
            )
        self.data = self.data.squeeze(1)  # Remove channel dimension [N, H, W]
        self.targets = self.original_dataset.targets
        
        # Filter for selected classes if specified
        if selected_classes is not None:
            mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for digit in selected_classes:
                mask = mask | (self.targets == digit)
            
            self.data = self.data[mask]
            self.targets = self.targets[mask]
            
            # Relabel sequential classes for classification (only if binary was intended)
            if len(selected_classes) == 2:
                self.targets = (self.targets == selected_classes[1]).float()
            else:
                # Create a mapping to sequential class indices
                class_map = {c: i for i, c in enumerate(sorted(selected_classes))}
                self.targets = torch.tensor([class_map[t.item()] for t in self.targets])

        # Store configuration
        self.selected_classes = selected_classes
        self.target_size = target_size
        self.train = train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def visualize_sample(self, idx):
        """Visualize a single sample using matplotlib."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(4, 4))
        plt.imshow(self.data[idx], cmap='gray')
        plt.title(f'Label: {self.targets[idx].item()}')
        plt.axis('on')
        plt.grid(True)
        plt.show()
    
    def visualize_grid(self, num_samples=25):
        """Visualize a grid of samples using matplotlib."""
        import matplotlib.pyplot as plt
        import math
        
        # Determine grid size
        grid_size = math.ceil(math.sqrt(num_samples))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        
        # Plot samples
        for i in range(min(num_samples, len(self))):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].imshow(self.data[i], cmap='gray')
            axes[row, col].axis('on')
            axes[row, col].grid(True)
            axes[row, col].set_title(f'{self.targets[i].item()}')
        
        # Remove empty subplots
        for i in range(num_samples, grid_size*grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

def get_dataloaders(batch_size=64, **dataset_kwargs):
    """Create train and test dataloaders for MNIST.
    
    Args:
        batch_size: Batch size for dataloaders
        **dataset_kwargs: Arguments passed to MNISTDataset constructor
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = MNISTDataset(train=True, **dataset_kwargs)
    test_dataset = MNISTDataset(train=False, **dataset_kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    return train_loader, test_loader

# %%
if __name__ == "__main__":
    # Test the dataset with binary digits 0 and 1
    train_loader, test_loader = get_dataloaders(batch_size=64, binary_digits=None, target_size=28)
    
    # Print dataset information
    print(f"Train dataset size: {len(train_loader.dataset)} samples")
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    
    # Get a sample batch
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_batch.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
    print(f"Sample labels: {sample_labels[:10]}")  # Show first 10 labels
    
    # Check class distribution
    unique_labels, counts = torch.unique(train_loader.dataset.targets, return_counts=True)
    print(f"\nClass distribution in training set:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label.item()}: {count.item()} samples")
    
    # Visualize a grid of samples
    print("\nVisualizing sample images from training set:")
    train_loader.dataset.visualize_grid(num_samples=16)  # Show a 4x4 grid

# %%
