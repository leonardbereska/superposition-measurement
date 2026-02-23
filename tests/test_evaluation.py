"""Tests for evaluation metrics (feature distribution and feature count)."""

import pytest
import numpy as np
from evaluation import calculate_feature_metrics, get_feature_distribution


class TestCalculateFeatureMetrics:
    """Tests for entropy-based feature count calculation."""

    def test_uniform_distribution_gives_max_feature_count(self):
        """Uniform distribution should give feature_count = d (number of features)."""
        d = 64
        p = np.ones(d) / d  # Uniform distribution

        metrics = calculate_feature_metrics(p)

        # exp(entropy) of uniform = d
        assert abs(metrics['feature_count'] - d) < 0.5, (
            f"Uniform distribution: expected feature_count ~ {d}, got {metrics['feature_count']:.2f}"
        )

    def test_one_hot_gives_feature_count_one(self):
        """One-hot distribution should give feature_count ~ 1."""
        d = 64
        p = np.zeros(d)
        p[0] = 1.0  # All mass on one feature

        metrics = calculate_feature_metrics(p)

        assert metrics['feature_count'] < 1.5, (
            f"One-hot distribution: expected feature_count ~ 1, got {metrics['feature_count']:.2f}"
        )

    def test_entropy_nonnegative(self):
        """Entropy should always be non-negative."""
        p = np.random.dirichlet(np.ones(32))
        metrics = calculate_feature_metrics(p)
        assert metrics['entropy'] >= 0

    def test_concentrated_less_than_uniform(self):
        """A concentrated distribution should have lower feature count than uniform."""
        d = 32
        uniform = np.ones(d) / d
        concentrated = np.zeros(d)
        concentrated[:4] = 0.25  # Mass on 4 features

        m_uniform = calculate_feature_metrics(uniform)
        m_concentrated = calculate_feature_metrics(concentrated)

        assert m_concentrated['feature_count'] < m_uniform['feature_count']


class TestGetFeatureDistribution:
    """Tests for activation-to-distribution conversion."""

    def test_output_sums_to_one(self):
        """Distribution should sum to 1."""
        activations = np.random.rand(100, 32)
        p = get_feature_distribution(activations)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_output_nonnegative(self):
        """Distribution values should be non-negative."""
        activations = np.abs(np.random.randn(100, 32))
        p = get_feature_distribution(activations)
        assert (p >= 0).all()

    def test_single_active_feature(self):
        """If only one feature is active, distribution should be concentrated."""
        activations = np.zeros((100, 16))
        activations[:, 5] = 1.0  # Only feature 5 is active
        p = get_feature_distribution(activations)

        assert p[5] == pytest.approx(1.0, abs=1e-6)
        assert all(p[i] == pytest.approx(0.0, abs=1e-6) for i in range(16) if i != 5)
