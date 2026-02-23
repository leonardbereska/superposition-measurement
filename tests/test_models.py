"""Tests for model architectures and factory function."""

import pytest
import torch
from models import create_model, ModelConfig


@pytest.mark.parametrize("model_type,kwargs,expected_out", [
    ("simplemlp", dict(input_channels=1, hidden_dim=128, image_size=28, output_dim=10), 10),
    ("simplemlp", dict(input_channels=1, hidden_dim=64, image_size=28, output_dim=1), 1),
    ("mlp", dict(input_channels=1, hidden_dim=32, image_size=28, output_dim=10), 10),
    ("cnn", dict(input_channels=1, hidden_dim=16, image_size=28, output_dim=10), 10),
    ("cnn", dict(input_channels=3, hidden_dim=16, image_size=32, output_dim=5), 5),
    ("simplecnn", dict(input_channels=1, hidden_dim=8, image_size=28, output_dim=10), 10),
    ("resnet18", dict(input_channels=3, hidden_dim=None, image_size=32, output_dim=10), 10),
])
def test_create_model_output_shape(model_type, kwargs, expected_out):
    """Verify each model type produces correct output dimensions."""
    model = create_model(model_type, **kwargs)
    model.eval()

    channels = kwargs["input_channels"]
    size = kwargs["image_size"]
    x = torch.randn(2, channels, size, size)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, expected_out), (
        f"{model_type}: expected output shape (2, {expected_out}), got {out.shape}"
    )


def test_create_model_unknown_type_raises():
    """Factory should raise ValueError for unknown model types."""
    with pytest.raises(ValueError, match="Unknown model type"):
        create_model("nonexistent", input_channels=1, hidden_dim=32, image_size=28, output_dim=10)


def test_model_config_namedtuple():
    """ModelConfig should be a valid NamedTuple with expected fields."""
    config = ModelConfig(model_type="mlp", hidden_dim=32, input_channels=1, image_size=28, output_dim=10)
    assert config.model_type == "mlp"
    assert config.hidden_dim == 32
    assert config.output_dim == 10
