"""Tests for adversarial attack implementations."""

import pytest
import torch
import torch.nn as nn
from attacks import (
    AttackConfig,
    generate_adversarial_examples,
    attack_fgsm,
    attack_pgd,
)
from models import create_model


@pytest.fixture
def simple_model():
    """Create a simple trained model for testing attacks."""
    torch.manual_seed(42)
    model = create_model(
        "simplemlp", input_channels=1, hidden_dim=32, image_size=28, output_dim=10
    )
    model.eval()
    return model


@pytest.fixture
def sample_batch():
    """Create a small batch of random inputs and labels."""
    torch.manual_seed(42)
    inputs = torch.rand(4, 1, 28, 28)  # Valid image range [0, 1]
    targets = torch.tensor([0, 1, 2, 3])
    return inputs, targets


class TestFGSM:
    """Tests for FGSM attack."""

    def test_perturbation_bounded(self, simple_model, sample_batch):
        """FGSM perturbations should be within epsilon bounds."""
        inputs, targets = sample_batch
        epsilon = 0.1
        config = AttackConfig(attack_type="fgsm")

        perturbed = attack_fgsm(simple_model, inputs, targets, epsilon, config)

        # Check perturbation is bounded by epsilon
        delta = (perturbed - inputs).abs()
        assert delta.max() <= epsilon + 1e-6, (
            f"Max perturbation {delta.max():.4f} exceeds epsilon {epsilon}"
        )

    def test_output_in_valid_range(self, simple_model, sample_batch):
        """Perturbed inputs should be clamped to [0, 1]."""
        inputs, targets = sample_batch
        config = AttackConfig(attack_type="fgsm")

        perturbed = attack_fgsm(simple_model, inputs, targets, 0.3, config)

        assert perturbed.min() >= 0.0, f"Min value {perturbed.min()} < 0"
        assert perturbed.max() <= 1.0, f"Max value {perturbed.max()} > 1"

    def test_zero_epsilon_unchanged(self, simple_model, sample_batch):
        """With epsilon=0, inputs should be unchanged."""
        inputs, targets = sample_batch
        config = AttackConfig(attack_type="fgsm")

        perturbed = attack_fgsm(simple_model, inputs, targets, 0.0, config)
        assert torch.allclose(perturbed, inputs, atol=1e-7)


class TestPGD:
    """Tests for PGD attack."""

    def test_perturbation_bounded(self, simple_model, sample_batch):
        """PGD perturbations should stay within the epsilon ball."""
        inputs, targets = sample_batch
        epsilon = 0.1
        config = AttackConfig(attack_type="pgd", pgd_steps=5, pgd_alpha=0.01)

        perturbed = attack_pgd(simple_model, inputs, targets, epsilon, config)

        delta = (perturbed - inputs).abs()
        assert delta.max() <= epsilon + 1e-6, (
            f"Max perturbation {delta.max():.4f} exceeds epsilon {epsilon}"
        )

    def test_output_in_valid_range(self, simple_model, sample_batch):
        """PGD outputs should be clamped to [0, 1]."""
        inputs, targets = sample_batch
        config = AttackConfig(attack_type="pgd", pgd_steps=5, pgd_alpha=0.01)

        perturbed = attack_pgd(simple_model, inputs, targets, 0.3, config)

        assert perturbed.min() >= 0.0
        assert perturbed.max() <= 1.0


class TestGenerateAdversarialExamples:
    """Tests for the dispatch function."""

    def test_fgsm_dispatch(self, simple_model, sample_batch):
        """generate_adversarial_examples should dispatch to FGSM."""
        inputs, targets = sample_batch
        config = AttackConfig(attack_type="fgsm")
        result = generate_adversarial_examples(simple_model, inputs, targets, 0.1, config)
        assert result.shape == inputs.shape

    def test_pgd_dispatch(self, simple_model, sample_batch):
        """generate_adversarial_examples should dispatch to PGD."""
        inputs, targets = sample_batch
        config = AttackConfig(attack_type="pgd", pgd_steps=3, pgd_alpha=0.01)
        result = generate_adversarial_examples(simple_model, inputs, targets, 0.1, config)
        assert result.shape == inputs.shape

    def test_unknown_attack_raises(self, simple_model, sample_batch):
        """Unknown attack type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown attack type"):
            AttackConfig(attack_type="unknown")
