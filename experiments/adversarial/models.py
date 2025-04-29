# %%
"""Model architectures for adversarial robustness experiments."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from nnsight import NNsight

# %%
class MLP(nn.Module):
    """Simple MLP classifier."""
    
    def __init__(self, hidden_dim=32, image_size=28, num_classes=1):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Define layers
        self.fc1 = nn.Linear(image_size * image_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# %%
class CNN(nn.Module):
    """Simple CNN classifier."""
    
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
        x = self.classifier(x)
        return x

# %%
def create_model(
    model_type: str = 'mlp',
    hidden_dim: int = 32,
    image_size: int = 28,
    num_classes: int = 1,
    use_nnsight: bool = False,
    device: Optional[torch.device] = None
) -> nn.Module:
    """Create a model with specified parameters.
    
    Args:
        model_type: Type of model ('mlp' or 'cnn')
        hidden_dim: Hidden dimension for MLP
        image_size: Image size
        num_classes: Number of output classes
        use_nnsight: Whether to wrap model with NNsight
        device: Device to place model on
        
    Returns:
        Model instance
    """
    if model_type.lower() == 'mlp':
        model = MLP(hidden_dim=hidden_dim, image_size=image_size, num_classes=num_classes)
    elif model_type.lower() == 'cnn':
        model = CNN(image_size=image_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    # Wrap with NNsight if requested
    if use_nnsight:
        return NNsightModelWrapper(model)
    
    return model

# %%
def get_layer_names(model: nn.Module) -> Dict[str, str]:
    """Get dictionary of layer names and types.
    
    Args:
        model: Model to inspect
        
    Returns:
        Dictionary mapping layer names to layer types
    """
    layer_dict = {}
    
    def _explore_module(module, prefix=""):
        for name, child in module.named_children():
            # Build the full path to this layer
            full_path = f"{prefix}.{name}" if prefix else name
            
            # Store the layer type
            layer_type = child.__class__.__name__
            layer_dict[full_path] = layer_type
            
            # Recursively explore children
            _explore_module(child, full_path)
    
    # Start exploration from the root
    _explore_module(model)
    
    return layer_dict

# %%
def print_model_structure(model: nn.Module) -> None:
    """Print the model structure in a readable format.
    
    Args:
        model: Model to print
    """
    layer_dict = get_layer_names(model)
    
    print("Model Structure:")
    print("-" * 50)
    for path, layer_type in layer_dict.items():
        print(f"{path:<40} : {layer_type}")

# %%
def get_activations(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str
) -> torch.Tensor:
    """Get activations from a specific layer.
    
    Args:
        model: Model to extract activations from
        inputs: Input tensor
        layer_name: Layer name (supports dot notation)
        
    Returns:
        Activations tensor
    """
    # Wrap model with NNsight if not already wrapped
    if not isinstance(model, NNsightModelWrapper):
        model_wrapper = NNsightModelWrapper(model)
    else:
        model_wrapper = model
    
    # Get activations
    return model_wrapper.get_layer_output(inputs, layer_name)

# %%
class NNsightModelWrapper:
    """Wrapper for models to use with nnsight."""
    
    def __init__(self, model, device=None):
        self.model = NNsight(model)
        self.device = device or next(model.parameters()).device
        self._base_model = model  # Store reference to original model
    
    def get_activations(self, dataloader, layer_name, max_activations=10000):
        """Extract activations using nnsight tracing.
        
        Args:
            dataloader: DataLoader with input samples
            layer_name: Name/path to the layer
            max_activations: Maximum number of activations to extract
        Returns:
            Tensor of activations
        """
        activations = []
        
        # Process in batches
        total_samples = 0
        for i, (inputs, _) in enumerate(dataloader):
            batch_size = inputs.shape[0]
            total_samples += batch_size
            if max_activations and total_samples >= max_activations:
                break
            inputs = inputs.to(self.device)
            
            # Use nnsight trace to access intermediate activations
            with self.model.trace(inputs) as tracer:
                # Access the specified layer's output
                layer_output = self._get_layer_output(layer_name).save()
            
            activations.append(layer_output.cpu())
            
        return torch.cat(activations, dim=0)
    
    def get_layer_output(self, inputs, layer_name):
        """Get output of a specific layer for given inputs.
        
        Args:
            inputs: Input tensor
            layer_name: Layer name (supports dot notation)
            
        Returns:
            Layer output tensor
        """
        inputs = inputs.to(self.device)
        
        # Use nnsight trace to access intermediate activations
        with self.model.trace(inputs) as tracer:
            # Access the specified layer's output
            layer_output = self._get_layer_output(layer_name).save()
        
        return layer_output
    
    def _get_layer_output(self, layer_name):
        """Access a layer by name/path using dot notation."""
        # Split by dots to handle nested attributes
        parts = layer_name.split('.')
        current = self.model
        
        # Navigate through the model hierarchy
        for part in parts:
            if part.isdigit():  # Handle numerical indices for ModuleLists
                current = current[int(part)]
            else:
                current = getattr(current, part)
        
        return current.output

    def __call__(self, x):
        """Forward pass through the model."""
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Use nnsight's trace to run the model
        with self.model.trace(x) as tracer:
            output = self.model.output.save()
        
        return output