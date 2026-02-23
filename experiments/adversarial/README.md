# Adversarial Robustness Experiments

See the [top-level README](../../README.md) for project overview, installation, and architecture.

## Module Reference

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point + three-phase orchestration (train/evaluate/analyze) |
| `config.py` | `get_default_config()` / `update_config()` for experiment configuration |
| `models.py` | StandardMLP, StandardCNN, SimpleMLP, SimpleCNN, ResNet-18, NNsight wrapper |
| `training.py` | Training loop with optional adversarial training (FGSM/PGD) |
| `attacks.py` | FGSM and PGD attack implementations |
| `sae.py` | Tied sparse autoencoder (`TiedSparseAutoencoder`) + training |
| `evaluation.py` | `measure_superposition()`, feature distribution, entropy-based feature count |
| `analysis.py` | Plot generation (feature counts, normalized ratios, robustness curves) |
| `datasets_adversarial.py` | MNIST and CIFAR-10 dataset loaders with class selection |
| `utils.py` | Logging, serialization, directory management |

## Troubleshooting

- **Cannot find results**: Use `--search "mlp_2-class_pgd"` or pass `--results-dir` explicitly.
- **Out of memory**: Reduce `training.batch_size` or `sae.batch_size` in config.
- **SAE instability**: Lower `sae.l1_lambda` (e.g., 0.01 instead of 0.1).
- **Slow training**: Ensure GPU is available (`torch.cuda.is_available()`).
