# %%
"""Model architectures for adversarial robustness experiments."""

import torch
import torch.nn as nn
try:
    from nnsight import NNsight
    NNSIGHT_AVAILABLE = True
except ImportError:
    print("Warning: nnsight not available. NNsightModelWrapper will not work.")
    NNSIGHT_AVAILABLE = False
import torchvision.models as models
from typing import NamedTuple, Optional
import numpy as np

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
        if not NNSIGHT_AVAILABLE:
            raise ImportError("nnsight is not available. Please install nnsight to use NNsightModelWrapper.")
        
        self._base_model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NNsight(model)
    
    def __del__(self):
        """Clean up any weak references when the wrapper is destroyed."""
        # Remove any hooks or weak references
        if hasattr(self, '_base_model'):
            del self._base_model
    
    def get_clean_model_copy(self):
        """Create a clean copy of the base model without any hooks or weak references.
        
        This method creates a fresh model instance with the same architecture and weights,
        but without any hooks, weak references, or non-serializable attributes.
        
        Returns:
            Clean model copy suitable for deepcopy operations
        """
        from .training import create_model, ModelConfig
        
        # Get model configuration from the base model
        if hasattr(self._base_model, 'config'):
            config = self._base_model.config
        else:
            # Infer configuration from model structure
            config = ModelConfig(
                model_type=self._get_model_type(),
                input_channels=self._get_input_channels(),
                hidden_dim=self._get_hidden_dim(),
                image_size=self._get_image_size(),
                output_dim=self._get_output_dim()
            )
        
        # Create a fresh model instance
        clean_model = create_model(
            model_type=config.model_type,
            input_channels=config.input_channels,
            hidden_dim=config.hidden_dim,
            image_size=config.image_size,
            output_dim=config.output_dim,
            use_nnsight=False
        )
        
        # Copy weights from the original model
        clean_model.load_state_dict(self._base_model.state_dict())
        
        # Move to the same device
        clean_model = clean_model.to(self.device)
        
        return clean_model
    
    def _get_model_type(self):
        """Infer model type from the base model."""
        model_class = self._base_model.__class__.__name__
        if 'SimpleMLP' in model_class:
            return 'simplemlp'
        elif 'StandardCNN' in model_class:
            return 'standard_cnn'
        elif 'StandardMLP' in model_class:
            return 'mlp'
        elif 'ResNet' in model_class:
            return 'resnet18'
        else:
            return 'simplemlp'  # default
    
    def _get_input_channels(self):
        """Infer input channels from the base model."""
        # Try to get from first conv layer
        for module in self._base_model.modules():
            if isinstance(module, nn.Conv2d):
                return module.in_channels
        return 1  # default for MNIST
    
    def _get_hidden_dim(self):
        """Infer hidden dimension from the base model."""
        # Try to get from first linear layer
        for module in self._base_model.modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        return 128  # default
    
    def _get_image_size(self):
        """Infer image size from the base model."""
        # Try to infer from the first linear layer's input dimension
        for module in self._base_model.modules():
            if isinstance(module, nn.Linear):
                input_dim = module.in_features
                # Calculate image size from input dimension
                if input_dim == 784:  # MNIST: 1*28*28
                    return 28
                elif input_dim == 3072:  # CIFAR-10: 3*32*32
                    return 32
                else:
                    # Try to infer from the shape
                    input_channels = self._get_input_channels()
                    return int(np.sqrt(input_dim / input_channels))
        return 28  # default for MNIST
    
    def _get_output_dim(self):
        """Infer output dimension from the base model."""
        # Try to get from last linear layer
        for module in reversed(list(self._base_model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        return 1  # default
    
    def get_activations(self, dataloader, layer_name, max_activations=10000):
        """Extract activations with streaming and periodic memory cleanup."""
        collected_activations = []
        total_samples = 0
        consolidation_buffer = []
        consolidation_threshold = 100  # Consolidate every 100 samples
        
        for i, (inputs, _) in enumerate(dataloader):
            if max_activations and total_samples >= max_activations:
                break
                
            # Process one sample at a time for maximum memory safety
            for idx in range(inputs.shape[0]):
                if max_activations and total_samples >= max_activations:
                    break
                    
                single_input = inputs[idx:idx+1]  # Keep batch dimension
                
                # Get activation for single sample
                with torch.no_grad():  # Prevent gradient computation
                    layer_output = self.get_layer_output(single_input, layer_name)
                    consolidation_buffer.append(layer_output.cpu().clone())  # Clone to break references
                
                # Explicit memory cleanup
                del layer_output
                torch.cuda.empty_cache()
                
                total_samples += 1
                
                # Periodic consolidation to prevent list bloat
                if len(consolidation_buffer) >= consolidation_threshold:
                    # Consolidate buffer into a single tensor
                    consolidated = torch.cat(consolidation_buffer, dim=0)
                    collected_activations.append(consolidated)
                    
                    # Clear buffer and force garbage collection
                    consolidation_buffer = []
                    torch.cuda.empty_cache()
                    
                    # if total_samples % (max_activations // 5) == 0:  # Log progress
                    #     print(f"Extracted {total_samples}/{max_activations} samples")
        
        # Handle remaining samples in buffer
        if consolidation_buffer:
            consolidated = torch.cat(consolidation_buffer, dim=0)
            collected_activations.append(consolidated)
        
        # Final concatenation
        if collected_activations:
            final_result = torch.cat(collected_activations, dim=0)
        else:
            # Return empty tensor with correct shape
            sample_shape = self.get_layer_output(inputs[:1], layer_name).shape[1:]
            final_result = torch.empty(0, *sample_shape)
        
        # Final cleanup
        torch.cuda.empty_cache()
        return final_result
    
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


class StandardMLP(nn.Module):
    """Standard MLP classifier for MNIST robustness with multiple hidden layers."""
    
    def __init__(
        self,
        input_channels: int = 1,  # 1 for MNIST (grayscale)
        image_size: int = 28,  # MNIST is 28x28
        hidden_dim: int = 32,
        output_dim: int = 10  
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Define layers with 2 hidden layers for better MNIST performance
        input_dim = input_channels * image_size * image_size  # 784 for MNIST
        self.fc1 = nn.Linear(input_dim, hidden_dim * 4)  # 128 = 32 * 4
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)  # 64 = 32 * 2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)  # 32 
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)  # 10 

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x) # extract activations from here
        x = self.fc2(x)
        x = self.relu2(x) # extract activations from here
        x = self.fc3(x)
        x = self.relu3(x) # extract activations from here
        x = self.fc4(x)  
        return x


class StandardCNN(nn.Module):
    """Standard LeNet-5-style CNN model with optional batch normalization for MNIST."""
    
    def __init__(
        self,
        input_channels: int = 1,  # 1 for grayscale, 3 for RGB
        image_size: int = 28,
        hidden_dim: int = 16,
        output_dim: int = 10
    ):
        """Initialize CNN.
        
        Args:
            input_channels: Number of input image channels (1 for grayscale, 3 for RGB)
            image_size: Size of input images (assumes square images)
            hidden_dim: Number of base channels in convolutional layers
            output_dim: Number of output classes
        """
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # Output: hidden_dim × (image_size/2) × (image_size/2)
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # Output: hidden_dim*2 × (image_size/4) × (image_size/4)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Calculate flattened size based on input dimensions
        flat_size = (hidden_dim*2) * (image_size // 4) * (image_size // 4)
        
        # Classifier
        self.fc1 = nn.Sequential(
            nn.Linear(flat_size, hidden_dim*4),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, x):
        # Add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # First block
        x = self.conv1(x)  # extract activations from here
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)  # extract activations from here
        x = self.pool2(x)
        
        # Flatten and classify
        x = self.flatten(x)
        
        x = self.fc1(x)  # extract activations from here
        
        x = self.fc2(x)  
        
        return x


class SimpleMLP(nn.Module):
    """Minimal MLP classifier for MNIST."""
    
    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 128,
        output_dim: int = 10
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        input_dim = input_channels * image_size * image_size  # 784 for MNIST
        
        # Just one hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """Minimal CNN for MNIST classification."""
    
    def __init__(
        self,
        input_channels: int = 1,
        image_size: int = 28,
        hidden_dim: int = 8,
        output_dim: int = 10
    ):
        super().__init__()
        
        # Just one convolutional layer
        self.conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        # Classifier
        self.flatten = nn.Flatten()
        flat_size = hidden_dim * (image_size // 2) * (image_size // 2)  # After pooling
        self.fc = nn.Linear(flat_size, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
def get_resnet18_adapted_to_small_images(input_channels=3, output_dim=10):
    """Create a ResNet-18 model adapted for small input images (like CIFAR-10 or MNIST).
    
    This function directly modifies and returns a ResNet-18 model
    without wrapping it in another class, ensuring layer access
    works correctly for feature extraction.
    
    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        output_dim: Number of output dimensions (number of classes)
        
    Returns:
        Modified ResNet-18 model
    """
    model = models.resnet18(weights=None)  # Start fresh
    
    # Adapt first layer for small images (32x32 or 28x28)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the MaxPool layer (set to identity) for small images
    model.maxpool = nn.Identity()
    
    # Adapt final classifier
    model.fc = nn.Linear(model.fc.in_features, output_dim)
    
    return model


def get_resnet18_adapted_to_cifar10(input_channels=3, output_dim=10):
    """Create a ResNet-18 model adapted for CIFAR-10.
    
    This function directly modifies and returns a ResNet-18 model
    without wrapping it in another class, ensuring layer access
    works correctly for feature extraction.
    
    Args:
        input_channels: Number of input channels (3 for RGB images)
        output_dim: Number of output dimensions (number of classes)
        
    Returns:
        Modified ResNet-18 model
    """
    # Use the generic function for CIFAR-10
    return get_resnet18_adapted_to_small_images(input_channels, output_dim)


def create_model(model_type='mlp', use_nnsight=False, **kwargs):
    """Factory function to create models.
    
    Args:
        model_type: Type of model to create ('mlp', 'cnn', or 'resnet18')
        use_nnsight: Whether to wrap the model with NNsight
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated model, optionally wrapped with NNsight
    """
    if model_type.lower() == 'simplemlp':
        model = SimpleMLP(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
    elif model_type.lower() == 'mlp':
        model = StandardMLP(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
        print(f"MLP model created with {kwargs['hidden_dim']} hidden units")
    elif model_type.lower() == 'simplecnn':
        model = SimpleCNN(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
    elif model_type.lower() == 'cnn':  # standard CNN
        model = StandardCNN(
            input_channels=kwargs['input_channels'],
            image_size=kwargs['image_size'],
            hidden_dim=kwargs['hidden_dim'],
            output_dim=kwargs['output_dim']
        )
    elif model_type.lower() == 'resnet18':
        # Adapted ResNet-18 for small input images (CIFAR-10 or MNIST)
        model = get_resnet18_adapted_to_small_images(
            input_channels=kwargs['input_channels'],
            output_dim=kwargs['output_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if use_nnsight:
        if not NNSIGHT_AVAILABLE:
            print("Warning: nnsight not available, returning unwrapped model.")
            return model
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

    simplemlp = create_model('simplemlp', input_channels=1, hidden_dim=128, image_size=28, output_dim=10)
    explore_model_structure(simplemlp)

    simplecnn = create_model('simplecnn', input_channels=1, hidden_dim=8, image_size=28, output_dim=10)
    explore_model_structure(simplecnn)

    # mlp = create_model('mlp', input_channels=1, hidden_dim=32, image_size=28, output_dim=1)
    # cnn = create_model('cnn', input_channels=1, hidden_dim=16, image_size=28, output_dim=1)
    # mlp.eval()
    # cnn.eval()
    # mlp.to('cuda')
    # cnn.to('cuda')
    # print(f"MLP parameters: {sum(p.numel() for p in mlp.parameters())}")
    # print(f"CNN parameters: {sum(p.numel() for p in cnn.parameters())}")
    
    # Test with a more complex model
    resnet = models.resnet18(pretrained=False, num_classes=10)
    wrapped_resnet = NNsightModelWrapper(resnet)
    explore_model_structure(wrapped_resnet)
# %%
