# %%
"""Dataset utilities for CIFAR-10."""

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with optional downsampling and binary classification."""
    
    def __init__(
        self, 
        target_size=32,  # Is target size input size? Yes, to resize images
        train=True,
        root='../../data',
        download=True,
        binary_classes=None  # None for multi-class, tuple for binary
    ):
        """
        Args:
            target_size: Size to resize images to (default: 32)
            train: If True, use training set
            root: Path to store the dataset
            download: If True, download the dataset
            binary_classes: Optional tuple of two class indices for binary classification
        """
        # Load CIFAR-10
        self.original_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download
        )
        
        # Convert to tensor and normalize
        
        # transform = transforms.ToTensor()
        # self.data = torch.stack([transform(img).permute(2, 0, 1).float() / 255.0 for img, _ in self.original_dataset])
        transform = transforms.ToTensor()
        self.data = torch.stack([transform(img) for img, _ in self.original_dataset])
        self.targets = torch.tensor([label for _, label in self.original_dataset])
        
        # Resize images if target_size != 32
        if target_size != 32:
            self.data = F.interpolate(
                self.data, size=(target_size, target_size), mode='bilinear', align_corners=True
            )
        
        # Handle binary classification if specified
        if binary_classes is not None:
            assert len(binary_classes) == 2, "binary_classes must be a tuple of two class indices"
            mask = (self.targets == binary_classes[0]) | (self.targets == binary_classes[1])
            self.data = self.data[mask]
            self.targets = self.targets[mask]
            # Relabel targets as 0 and 1
            self.targets = (self.targets == binary_classes[1]).long()
            #self.targets = (self.targets == binary_classes[1]).float()
        else:
            self.targets = self.targets.long()
            #self.targets = self.targets.float()
        
        self.binary_classes = binary_classes
        self.target_size = target_size
        self.train = train

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def visualize_sample(self, idx):
        """Visualize a single sample using matplotlib."""
        import matplotlib.pyplot as plt
        img = self.data[idx]
        # Convert back to [H, W, C] for visualization
        img_vis = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        plt.imshow(img_vis)
        label = self.targets[idx].item()
        plt.title(f'Label: {label}')
        plt.axis('on')
        plt.grid(True)
        plt.show()
    
    def visualize_grid(self, num_samples=25):
        """Visualize a grid of samples."""
        import matplotlib.pyplot as plt
        import math
        grid_size = math.ceil(math.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        for i in range(min(num_samples, len(self))):
            row = i // grid_size
            col = i % grid_size
            img = self.data[i]
            img_vis = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            axes[row, col].imshow(img_vis)
            axes[row, col].axis('on')
            axes[row, col].set_title(f'{self.targets[i].item()}')
        for i in range(num_samples, grid_size*grid_size):
            row = i // grid_size
            col = i % grid_size
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()

def get_cifar10_dataloaders(batch_size=64, **dataset_kwargs):
    """Create train and test dataloaders for CIFAR-10."""
    train_dataset = CIFAR10Dataset(train=True, **dataset_kwargs)
    test_dataset = CIFAR10Dataset(train=False, **dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )

    return train_loader, test_loader

# %%
if __name__ == "__main__":
    # Test the dataset with binary_digits=None, resize to 28x28
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=64, binary_classes=None, target_size=32)
    
    # Print dataset information
    print(f"Train dataset size: {len(train_loader.dataset)} samples")
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    
    # # Get a sample batch
    # sample_batch, sample_labels = next(iter(train_loader))
    # print(f"\nSample batch shape: {sample_batch.shape}")
    # print(f"Sample labels shape: {sample_labels.shape}")
    # print(f"Sample labels: {sample_labels[:10]}")  # Show first 10 labels
    
    # # Check class distribution
    # targets = train_loader.dataset.targets
    # unique_labels, counts = torch.unique(targets, return_counts=True)
    # print(f"\nClass distribution in training set:")
    # for label, count in zip(unique_labels, counts):
    #     print(f"  Class {int(label.item())}: {count.item()} samples")
    
    # # Visualize a grid of samples
    # print("\nVisualizing sample images from training set:")
    # train_loader.dataset.visualize_grid(num_samples=16)  # Show a 4x4 grid

# %%
