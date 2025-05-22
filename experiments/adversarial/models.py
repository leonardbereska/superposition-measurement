# %%
"""Model architectures for adversarial robustness experiments."""

import torch
import torch.nn as nn
from nnsight import NNsight
import torchvision.models as models
from typing import NamedTuple, Optional

class ModelConfig(NamedTuple):
    """Configuration for model creation."""
    model_type: str
    hidden_dim: int
    input_channels: int
    image_size: int
    output_dim: Optional[int] = None  # Will be determined from data if None

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
            layer_output = self.get_layer_output(inputs, layer_name)
            activations.append(layer_output.cpu())
            
        return torch.cat(activations, dim=0)
    
    def _access_layer_by_dot_notation(self, layer_name):
        """Access a layer by name/path using dot notation.
        
        Args:
            layer_name: Layer name with dot notation (e.g., 'features.0')
            
        Returns:
            Layer output proxy
        """
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
            layer_output = self._access_layer_by_dot_notation(layer_name).save()
        
        return layer_output

    def __call__(self, x):
        """Forward pass through the model."""
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Use nnsight's trace to run the model
        with self.model.trace(x) as tracer:
            output = self.model.output.save()
        
        return output



class MLP(nn.Module):
    """Simple MLP classifier."""
    
    def __init__(
        self,
        input_channels: int = 1,  # 1 for grayscale, 3 for RGB
        image_size: int = 28,
        hidden_dim: int = 32,
        output_dim: int = 1
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Define layers
        self.fc1 = nn.Linear(input_channels * image_size * image_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """Simple CNN model."""
    
    def __init__(
        self,
        input_channels: int = 1, # 1 for grayscale, 3 for RGB
        image_size: int = 28,
        hidden_dim: int = 32,
        output_dim: int = 1
    ):
        """Initialize CNN.
        
        Args:
            input_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),  # Use input_channels here
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate flattened size based on input dimensions
        # This assumes 28x28 input images with two 2x2 max pooling layers
        flat_size = (hidden_dim * 2) * (image_size // 4) * (image_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_model(model_type='mlp', use_nnsight=False, **kwargs):
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create ('mlp' or 'cnn')
        use_nnsight: Whether to wrap the model with NNsight
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated model, optionally wrapped with NNsight
    """
    if model_type.lower() == 'mlp':
        model = MLP(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
    elif model_type.lower() == 'cnn':
        model = CNN(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
    elif model_type.lower() == 'resnet18':
        # Simple ResNet-18 for CIFAR-10
        model = models.resnet18(pretrained=False, num_classes=kwargs['output_dim'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if use_nnsight:
        return NNsightModelWrapper(model)
    return model


def explore_model_structure(model):
    """Display the model structure with layer names accessible via nnsight.
    
    Args:
        model: Either a PyTorch model or a nnsight-wrapped model
    
    Returns:
        Dictionary mapping full layer paths to layer types
    """
    # Ensure model is wrapped with nnsight
    if not isinstance(model, NNsight) and not hasattr(model, 'model'):
        model = NNsight(model)
    elif hasattr(model, 'model') and isinstance(model.model, NNsight):
        model = model.model
    
    # Dictionary to store full paths to layers
    layer_paths = {}
    
    def _explore_module(module, prefix=""):
        # Check if we're dealing with the NNsight wrapper or a regular module
        if hasattr(module, '_model'):
            target_module = module._model
        else:
            target_module = module
            
        # Iterate through named children (immediate submodules)
        for name, child in target_module.named_children():
            # Build the full path to this layer
            full_path = f"{prefix}.{name}" if prefix else name
            
            # Store the layer type
            layer_type = child.__class__.__name__
            layer_paths[full_path] = layer_type
            
            # Recursively explore children
            _explore_module(child, full_path)
            
    # Start exploration from the root
    _explore_module(model)
    
    # Print the results in a readable format
    print("Available layer paths for activation extraction:")
    print("-" * 50)
    
    for path, layer_type in layer_paths.items():
        print(f"{path:<40} : {layer_type}")
    
    return layer_paths

# %%
# Example usage
if __name__ == "__main__":
    # Test model creation and structure exploration
    mlp = create_model('mlp', input_channels=1, hidden_dim=32, image_size=28, output_dim=1)
    explore_model_structure(mlp)

    cnn = create_model('cnn', input_channels=3, hidden_dim=64, image_size=32, output_dim=10)
    explore_model_structure(cnn)
    
    # Test with a more complex model
    resnet = models.resnet18(pretrained=False, num_classes=10)
    wrapped_resnet = NNsightModelWrapper(resnet)
    explore_model_structure(wrapped_resnet)
# %%
