# %%
"""Model architectures for adversarial robustness experiments."""

import torch
import torch.nn as nn
from collections import defaultdict

class BaseClassifier(nn.Module):
    """Base class for all classifiers."""
    
    def __init__(self):
        super().__init__()
        self.activation_storage = defaultdict(list)
        self.collect_activations = False
    
    def get_activations(self, dataloader: torch.utils.data.DataLoader, layer_name: str) -> torch.Tensor:
        """Collect activations for a specific layer across the dataset.
        
        Args:
            dataloader: DataLoader to collect activations from
            layer_name: Name of layer to collect activations from
        
        Returns:
            Tensor of collected activations
        """
        self.activation_storage.clear()
        self.collect_activations = True
        self.eval()
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(next(self.parameters()).device)
                _ = self(inputs)
        
        self.collect_activations = False
        
        # Concatenate all stored activations
        activations = torch.cat(self.activation_storage[layer_name], dim=0)
        self.activation_storage.clear()
        return activations

class MLP(BaseClassifier):
    """Simple MLP classifier with activation collection capabilities."""
    
    def __init__(self, hidden_dim=32, image_size=28, num_classes=1):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Define layers
        self.fc1 = nn.Linear(image_size * image_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        
        # First linear layer
        x = self.fc1(x)
        if self.collect_activations:
            self.activation_storage['before_relu'].append(x.detach().cpu())
            
        # ReLU
        x = self.relu(x)
        if self.collect_activations:
            self.activation_storage['after_relu'].append(x.detach().cpu())
            
        # Final layer
        x = self.fc2(x)
        return x

class CNN(BaseClassifier):
    """Simple CNN classifier with activation collection capabilities."""
    
    def __init__(self, image_size=28, num_classes=1):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate feature size after convolutions and pooling
        feature_size = (image_size // 4) ** 2 * 32
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        if self.collect_activations:
            self.activation_storage['conv_features'].append(x.detach().cpu())
            
        x = self.classifier(x)
        return x

def create_model(model_type='mlp', **kwargs):
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create ('mlp' or 'cnn')
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated model
    """
    if model_type.lower() == 'mlp':
        return MLP(**kwargs)
    elif model_type.lower() == 'cnn':
        return CNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
# %%

if __name__ == "__main__":
    mlp = create_model(model_type='mlp', hidden_dim=32, image_size=28, num_classes=1)
    print(mlp)

    cnn = create_model(model_type='cnn', image_size=28, num_classes=1)
    print(cnn)
# %%
