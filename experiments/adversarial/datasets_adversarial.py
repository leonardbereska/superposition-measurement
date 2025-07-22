# %%
"""Dataset utilities for MNIST and CIFAR-10."""

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Union, Dict, Callable
import matplotlib.pyplot as plt
import math

# ---- Dataset Classes ----

class MNISTDataset(Dataset):
    """MNIST dataset with optional downsampling and class selection."""
    
    def __init__(
        self, 
        target_size: int = 28,
        train: bool = True,
        root: str = '../../data',
        download: bool = True,
        selected_classes = None  # None for all classes, list/tuple for specific classes 
    ) -> None:
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
        # Keep channel dimension for CNN models [N, 1, H, W]
        # self.data = self.data.squeeze(1)  # Remove channel dimension [N, H, W]
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
        self.num_channels = 1
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def get_metadata(self) -> Dict:
        """Return dataset metadata for visualization."""
        return {
            "name": "MNIST",
            "num_samples": len(self),
            "num_classes": 2 if len(self.selected_classes or []) == 2 else 
                           len(self.selected_classes) if self.selected_classes else 10,
            "image_size": self.target_size,
            "num_channels": self.num_channels,
            "cmap": "gray"  # Colormap for visualization
        }


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with optional downsampling and class selection."""
    
    def __init__(
        self, 
        target_size: int = 32,
        train: bool = True,
        root: str = 'data',
        download: bool = True,
        selected_classes = None,  # None for all classes, list/tuple for specific classes
        to_grayscale: bool = False
    ) -> None:
        """
        Args:
            target_size: Size to downsample images to (default: original 32)
            train: If True, use training set
            root: Path to store the dataset
            download: If True, download the dataset
            selected_classes: Optional list/tuple of classes to include
            to_grayscale: If True, convert images to grayscale
        """
        # Load original CIFAR-10
        self.original_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download
        )
        
        # Convert to torch tensor and scale to [0, 1]
        self.data = torch.tensor(self.original_dataset.data, dtype=torch.float32) / 255.0
        self.data = self.data.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        # Downsample
        if target_size != 32:  # Only downsample if target size is different
            self.data = F.interpolate(
                self.data,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=True
            )
        
        # Convert to grayscale if requested
        self.to_grayscale = to_grayscale
        if to_grayscale:
            # Simple average for grayscale conversion - consider using weighted conversion
            self.data = self.data.mean(dim=1, keepdim=True)  # [N, 3, H, W] -> [N, 1, H, W]
            self.num_channels = 1
        else:
            self.num_channels = 3
            
        # Convert targets to tensor
        self.targets = torch.tensor(self.original_dataset.targets)
        
        # Filter for selected classes if specified
        if selected_classes is not None:
            mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for digit in selected_classes:
                mask = mask | (self.targets == digit)
            
            self.data = self.data[mask]
            self.targets = self.targets[mask]
            
            # Relabel sequential classes for classification
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
        
    def get_metadata(self) -> Dict:
        """Return dataset metadata for visualization."""
        return {
            "name": "CIFAR-10",
            "num_samples": len(self),
            "num_classes": 2 if len(self.selected_classes or []) == 2 else 
                           len(self.selected_classes) if self.selected_classes else 10,
            "image_size": self.target_size,
            "num_channels": self.num_channels,
            "cmap": "gray" if self.to_grayscale else None  # Colormap for visualization
        }


# ---- Factory Functions ----

def create_dataset(
    dataset_type: str,
    train: bool = True,
    **kwargs
) -> Dataset:
    """Factory function to create datasets.
    
    Args:
        dataset_type: Type of dataset ('mnist', 'cifar10', etc.)
        train: Whether to create training or test dataset
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Dataset instance
    """
    if dataset_type.lower() == 'mnist':
        return MNISTDataset(
            target_size=kwargs.get('target_size', 28),
            train=train,
            selected_classes=kwargs.get('selected_classes', None),
            root=kwargs.get('data_dir', 'data')
        )
    elif dataset_type.lower() == 'cifar10':
        return CIFAR10Dataset(
            target_size=kwargs.get('target_size', 32),
            train=train,
            selected_classes=kwargs.get('selected_classes', None),
            root=kwargs.get('data_dir', 'data'),
            to_grayscale=kwargs.get('to_grayscale', False)
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def create_dataloaders(
    dataset_type: str,
    batch_size: int = 64,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Factory function to create dataloaders.
    
    Args:
        dataset_type: Type of dataset ('mnist', 'cifar10', etc.)
        batch_size: Batch size for dataloaders
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        train_loader, test_loader
    """
    train_dataset = create_dataset(dataset_type, train=True, **kwargs)
    test_dataset = create_dataset(dataset_type, train=False, **kwargs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, test_loader


# ---- Visualization Utilities ----

def visualize_sample(dataset: Dataset, idx: int, figsize: Tuple[int, int] = (4, 4)) -> None:
    """Visualize a single sample from any dataset.
    
    Args:
        dataset: Dataset containing images
        idx: Index of image to visualize
        figsize: Figure size for matplotlib
    """
    sample, label = dataset[idx]
    metadata = dataset.get_metadata() if hasattr(dataset, 'get_metadata') else {"cmap": None}
    
    plt.figure(figsize=figsize)
    
    # Handle different tensor shapes and channel configurations
    if sample.dim() == 2:  # [H, W]
        plt.imshow(sample, cmap=metadata.get("cmap", "gray"))
    elif sample.dim() == 3:  # [C, H, W] or [H, W, C]
        if sample.shape[0] in [1, 3]:  # Assume [C, H, W]
            # Convert [C, H, W] to [H, W, C] for matplotlib
            plt.imshow(sample.permute(1, 2, 0), cmap=metadata.get("cmap"))
        else:  # Assume [H, W, C]
            plt.imshow(sample, cmap=metadata.get("cmap"))
    
    plt.title(f'Label: {label.item()}')
    plt.axis('on')
    plt.grid(True)
    plt.show()

def visualize_grid(dataset: Dataset, num_samples: int = 25, figsize: Tuple[int, int] = (10, 10)) -> None:
    """Visualize a grid of samples from any dataset.
    
    Args:
        dataset: Dataset containing images
        num_samples: Number of samples to visualize
        figsize: Figure size for matplotlib
    """
    grid_size = math.ceil(math.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    metadata = dataset.get_metadata() if hasattr(dataset, 'get_metadata') else {"cmap": None}
    cmap = metadata.get("cmap")
    
    # Flatten axes for easier indexing
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i in range(min(num_samples, len(dataset))):
        sample, label = dataset[i]
        ax = axes[i]
        
        # Handle different tensor shapes and channel configurations
        if sample.dim() == 2:  # [H, W]
            ax.imshow(sample, cmap=cmap)
        elif sample.dim() == 3:  # [C, H, W] or [H, W, C]
            if sample.shape[0] in [1, 3]:  # Assume [C, H, W]
                # Convert [C, H, W] to [H, W, C] for matplotlib
                ax.imshow(sample.permute(1, 2, 0), cmap=cmap)
            else:  # Assume [H, W, C]
                ax.imshow(sample, cmap=cmap)
        
        ax.set_title(f'{label.item()}')
        ax.axis('on')
        ax.grid(True)
    
    # Remove empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset(dataset: Dataset) -> None:
    """Analyze and report statistics about a dataset.
    
    Args:
        dataset: Dataset to analyze
    """
    # Get dataset metadata
    metadata = dataset.get_metadata() if hasattr(dataset, 'get_metadata') else {}
    name = metadata.get("name", "Dataset")
    
    # Basic statistics
    print(f"\n{name} Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    # Extract labels
    all_labels = []
    for _, label in torch.utils.data.DataLoader(dataset, batch_size=128):
        all_labels.append(label)
    
    all_labels = torch.cat(all_labels)
    unique_labels, counts = torch.unique(all_labels, return_counts=True)
    
    # Class distribution
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"    Class {label.item()}: {count.item()} samples ({count.item()/len(dataset):.1%})")
    
    # Sample a few examples for shape analysis
    sample, _ = dataset[0]
    
    # Report shape information
    if sample.dim() == 2:
        shape_str = f"{sample.shape[0]}×{sample.shape[1]} (H×W)"
    elif sample.dim() == 3:
        if sample.shape[0] in [1, 3]:  # Assume [C, H, W]
            shape_str = f"{sample.shape[0]}×{sample.shape[1]}×{sample.shape[2]} (C×H×W)"
        else:  # Assume [H, W, C]
            shape_str = f"{sample.shape[0]}×{sample.shape[1]}×{sample.shape[2]} (H×W×C)"
    
    print(f"  Sample shape: {shape_str}")
    print(f"  Data type: {sample.dtype}")
    print(f"  Value range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
    
    # Visualize a sample grid
    print("\nSample grid:")
    visualize_grid(dataset, num_samples=16)  # Show a 4×4 grid


# ---- Testing Code ----

def test_dataset(dataset_type: str, **kwargs) -> None:
    """Test a dataset by creating and analyzing it.
    
    Args:
        dataset_type: Type of dataset ('mnist', 'cifar10', etc.)
        **kwargs: Additional dataset-specific parameters
    """
    print(f"Testing {dataset_type} dataset...")
    
    # Create train and test datasets
    train_dataset = create_dataset(dataset_type, train=True, **kwargs)
    test_dataset = create_dataset(dataset_type, train=False, **kwargs)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Print basic information
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    
    # Get a sample batch
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_batch.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
    print(f"Sample labels: {sample_labels[:10]}")  # Show first 10 labels
    
    # Analyze the full dataset
    analyze_dataset(train_dataset)
    
    print(f"\nCompleted testing {dataset_type} dataset.")


# %%
if __name__ == "__main__":
    # Test both MNIST and CIFAR-10 datasets
    
    # Test MNIST with binary classes
    test_dataset('mnist', selected_classes=(0, 1), target_size=28, data_dir='../../data')
    
    # Test CIFAR-10 with a few classes
    test_dataset('cifar10', selected_classes=(0, 1, 2), target_size=32, data_dir='../../data')
# %%

    # get number of classes from dataset
    train_loader, test_loader = create_dataloaders(
        dataset_type='mnist',
        batch_size=64,
        selected_classes=None,
        target_size=28,
        data_dir='../../data'
    )
    
    # for each batch in train_loader, print the number of classes

    # small function to get number of classes from dataset
    
    # print number of classes for train_loader
    print(train_loader.dataset.targets)

# %%
