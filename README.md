# Superposition as Lossy Compression

Code for the paper **"Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability"** (TMLR 2025).

[[Paper]](https://arxiv.org/abs/2512.13568) [[Blog]](https://leonardbereska.github.io/blog/2025/superposition/) [[OpenReview]](https://openreview.net/forum?id=qaNP6o5qvJ)

**Authors:** Leonard Bereska, Zoe Tzifa-Kratira, Reza Samavi, Efstratios Gavves

## Overview

Neural networks encode more features than they have neurons — a phenomenon called **superposition**. We frame superposition as *lossy compression* and introduce an information-theoretic measure: apply Shannon entropy to sparse autoencoder (SAE) activations to count the **effective number of features** (virtual neurons) a layer simulates.

```
Superposition ratio:  ψ = F / N

  F = exp(H(p))    effective features ("virtual neurons")
  N                 physical neurons in the layer
  ψ > 1            → network compresses beyond lossless limit
```

We use this metric to study how adversarial training, dropout, algorithmic tasks, and grokking affect superposition.

## Key Findings

- **Adversarial training** does not uniformly reduce superposition. The effect depends on task demands vs. network capacity:
  - *Abundance regime* (simple task, ample capacity): adversarial training **increases** effective features
  - *Scarcity regime* (complex task, limited capacity): adversarial training **decreases** effective features
- **Dropout** reduces effective features by up to 50% through redundant encoding
- **Algorithmic tasks** resist superposition (ψ ≤ 1) — no input sparsity to exploit
- **Grokking** shows sharp feature consolidation at the generalization transition
- **Language models** (Pythia-70M): early MLPs show peak compression (~20x expansion), with non-monotonic layer-wise patterns

## Pipeline

```
  Phase 1: Training          Phase 2: Evaluation        Phase 3: Analysis
  ┌─────────────────┐       ┌─────────────────────┐    ┌──────────────────┐
  │ For each ε:     │       │ For each trained     │    │ Load results.json│
  │  Train model    │──────>│ model:               │───>│ Generate plots:  │
  │  with FGSM/PGD  │       │  Extract activations │    │  - feature count │
  │  adversarial    │       │  Train SAE           │    │  - normalized FC │
  │  training       │       │  Compute entropy     │    │  - robustness    │
  │  Evaluate       │       │  → feature count     │    │    curves        │
  │  robustness     │       │  (clean & adv data)  │    │                  │
  └─────────────────┘       └─────────────────────┘    └──────────────────┘
```

## Installation

```bash
git clone https://github.com/leonardbereska/superposition-measurement.git
cd superposition-measurement
pip install -e ".[dev]"
```

## Usage

```bash
# Run full pipeline (train → evaluate → analyze)
python experiments/adversarial/main.py all --dataset mnist --model cnn

# Individual phases
python experiments/adversarial/main.py train --dataset mnist --model mlp
python experiments/adversarial/main.py evaluate --results-dir results/adversarial_robustness/mnist/...
python experiments/adversarial/main.py analyze --search "mlp_2-class_pgd"

# Quick test
python experiments/adversarial/main.py quick_test --dataset mnist --model simplemlp
```

## Project Structure

```
superposition-measurement/
├── experiments/
│   ├── plotting.py              # Shared plotting utilities
│   ├── adversarial/             # Main experiment framework
│   │   ├── main.py              # CLI + orchestration (train/eval/analyze)
│   │   ├── config.py            # Experiment configuration
│   │   ├── models.py            # MLP, CNN, ResNet-18, NNsight wrapper
│   │   ├── training.py          # Training loop + adversarial training
│   │   ├── attacks.py           # FGSM and PGD implementations
│   │   ├── sae.py               # Tied sparse autoencoder
│   │   ├── evaluation.py        # Feature count measurement
│   │   ├── analysis.py          # Plotting and visualization
│   │   ├── datasets_adversarial.py  # MNIST/CIFAR-10 loaders
│   │   └── utils.py             # Logging, serialization, paths
│   ├── layerwise/               # Pythia-70m layer analysis
│   └── interventions/           # Dictionary scaling experiments
├── tests/                       # pytest test suite
├── pyproject.toml               # Package config + dependencies
├── .github/workflows/ci.yml     # CI: pytest + ruff
└── LICENSE
```

## Testing

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{bereska2025superposition,
  title={Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability},
  author={Bereska, Leonard and Tzifa-Kratira, Zoe and Samavi, Reza and Gavves, Efstratios},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=qaNP6o5qvJ}
}
```

## License

MIT
