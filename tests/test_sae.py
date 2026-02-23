"""Tests for sparse autoencoder implementation."""

import pytest
import torch
import numpy as np
from sae import TiedSparseAutoencoder, SAEConfig, train_sae, compute_loss


class TestTiedSparseAutoencoder:
    """Tests for the TiedSparseAutoencoder architecture."""

    def test_weight_tying(self):
        """Decoder weights should be the transpose of encoder weights."""
        sae = TiedSparseAutoencoder(input_dim=16, dictionary_dim=32)
        x = torch.randn(4, 16)
        activations, reconstruction = sae(x)

        # The reconstruction should use encoder.weight.t() (tied weights)
        expected_recon = torch.nn.functional.linear(
            activations, sae.encoder.weight.t(), bias=None
        )
        assert torch.allclose(reconstruction, expected_recon, atol=1e-6)

    def test_output_shapes(self):
        """Forward pass should return correct shapes."""
        input_dim, dict_dim, batch = 16, 32, 8
        sae = TiedSparseAutoencoder(input_dim=input_dim, dictionary_dim=dict_dim)
        x = torch.randn(batch, input_dim)

        activations, reconstruction = sae(x)

        assert activations.shape == (batch, dict_dim)
        assert reconstruction.shape == (batch, input_dim)

    def test_activations_nonnegative(self):
        """ReLU should ensure activations are non-negative."""
        sae = TiedSparseAutoencoder(input_dim=16, dictionary_dim=32)
        x = torch.randn(8, 16)
        activations, _ = sae(x)

        assert (activations >= 0).all(), "Activations should be non-negative after ReLU"

    def test_dictionary_dim_validation(self):
        """Should raise ValueError if dictionary_dim < input_dim."""
        with pytest.raises(ValueError, match="Dictionary dimension"):
            TiedSparseAutoencoder(input_dim=32, dictionary_dim=16)

    def test_reconstruction_loss_decreases(self):
        """Training should reduce reconstruction loss on random data."""
        torch.manual_seed(42)
        input_dim, dict_dim = 16, 32
        sae = TiedSparseAutoencoder(input_dim=input_dim, dictionary_dim=dict_dim)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        x = torch.randn(64, input_dim)

        # Measure initial loss
        _, initial_loss = compute_loss(sae, x, l1_lambda=0.01)
        initial = initial_loss['reconstruction_loss']

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            loss, _ = compute_loss(sae, x, l1_lambda=0.01)
            loss.backward()
            optimizer.step()

        # Measure final loss
        _, final_loss = compute_loss(sae, x, l1_lambda=0.01)
        final = final_loss['reconstruction_loss']

        assert final < initial, (
            f"Reconstruction loss should decrease: {initial:.4f} -> {final:.4f}"
        )


class TestTrainSAE:
    """Tests for the train_sae function."""

    def test_train_sae_returns_model(self):
        """train_sae should return a TiedSparseAutoencoder."""
        torch.manual_seed(42)
        activations = torch.randn(100, 16)
        config = SAEConfig(
            expansion_factor=2, l1_lambda=0.1, batch_size=32,
            learning_rate=1e-3, n_epochs=5
        )
        sae = train_sae(activations, sae_config=config, device=torch.device("cpu"))

        assert isinstance(sae, TiedSparseAutoencoder)
        assert sae.input_dim == 16
        assert sae.dictionary_dim == 32  # 16 * expansion_factor(2)
