"""Module for adversarial attack implementations."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

class AttackConfig:
    """Configuration for adversarial attacks.
    
    Attributes:
        attack_type: Type of attack ("fgsm" or "pgd")
        targeted: Whether to use targeted attacks
        pgd_steps: Number of steps for PGD attack
        pgd_alpha: Step size for PGD attack (relative to epsilon if None)
        pgd_random_start: Whether to use random initialization for PGD
    """
    def __init__(
        self,
        attack_type: str = "fgsm",
        targeted: bool = False,
        pgd_steps: int = 10,
        pgd_alpha: Optional[float] = None,
        pgd_random_start: bool = True
    ):
        self.attack_type = attack_type.lower()
        self.targeted = targeted
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.pgd_random_start = pgd_random_start
        
        # Validate attack type
        if self.attack_type not in ["fgsm", "pgd"]:
            raise ValueError(f"Unknown attack type: {attack_type}")

def attack_fgsm(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    config: AttackConfig,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate adversarial examples using FGSM attack.
    
    Args:
        model: Model to attack
        inputs: Input images [batch_size, channels, height, width]
        targets: True labels
        epsilon: Perturbation size
        config: Attack configuration
        target_labels: Labels to target if targeted=True
        
    Returns:
        Perturbed inputs
    """
    # Prepare inputs
    inputs = inputs.clone().detach().requires_grad_(True)
    target = target_labels if config.targeted else targets
    
    # Forward pass
    outputs = model(inputs)
    
    if outputs.size(-1) == 1:  # Binary classification
        loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), target.float())
    else:  # Multi-class classification
        loss = nn.CrossEntropyLoss()(outputs, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate perturbation
    sign = -1 if config.targeted else 1
    perturbed_inputs = inputs + sign * epsilon * inputs.grad.sign()
    perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    
    return perturbed_inputs

def attack_pgd(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    config: AttackConfig,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate adversarial examples using PGD attack.
    
    Args:
        model: Model to attack
        inputs: Input images [batch_size, channels, height, width]
        targets: True labels
        epsilon: Perturbation size
        config: Attack configuration
        target_labels: Labels to target if targeted=True
        
    Returns:
        Perturbed inputs
    """
    # Initialize perturbed inputs
    perturbed = inputs.clone().detach()
    
    # Determine step size (alpha)
    alpha = config.pgd_alpha if config.pgd_alpha is not None else epsilon / 4
    
    # Random start (within epsilon ball)
    if config.pgd_random_start:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0, 1)
    
    target = target_labels if config.targeted else targets
    
    for _ in range(config.pgd_steps):
        perturbed.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed)
        
        if outputs.size(-1) == 1:  # Binary classification
            loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), target.float())
        else:  # Multi-class classification
            loss = nn.CrossEntropyLoss()(outputs, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update perturbed inputs
        sign = -1 if config.targeted else 1
        with torch.no_grad():
            perturbed = perturbed.detach() + sign * alpha * perturbed.grad.sign()
            
            # Project back to epsilon ball and valid image range
            delta = torch.clamp(perturbed - inputs, -epsilon, epsilon)
            perturbed = torch.clamp(inputs + delta, 0, 1)
    
    return perturbed

def generate_adversarial_examples(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    config: AttackConfig,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate adversarial examples using the specified attack.
    
    Args:
        model: Model to attack
        inputs: Input images
        targets: True labels
        epsilon: Perturbation size
        config: Attack configuration
        target_labels: Labels to target if targeted=True
        
    Returns:
        Perturbed inputs
    """
    if config.attack_type == "fgsm":
        return attack_fgsm(model, inputs, targets, epsilon, config, target_labels)
    elif config.attack_type == "pgd":
        return attack_pgd(model, inputs, targets, epsilon, config, target_labels)
    else:
        raise ValueError(f"Unknown attack type: {config.attack_type}") 