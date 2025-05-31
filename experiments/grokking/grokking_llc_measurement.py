# -*- coding: utf-8 -*-
"""
Grokking experiment showing LLC estimation calibration in a simple modular addition example.
Adapted from Nina Panickssery and Dmitry Vaintrob's modular addition learning coefficient work.
"""

import random
from copy import deepcopy
from dataclasses import dataclass
import typing
from typing import Type, Dict, List, Optional
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce, plot_trace
from devinterp.vis_utils import EpsilonBetaAnalyzer

# SAE imports and evaluation functions
from experiments.adversarial.sae import TiedSparseAutoencoder, train_sae, SAEConfig, evaluate_sae
from experiments.adversarial.evaluation import get_feature_distribution, calculate_feature_metrics
from experiments.adversarial.analysis import ScientificPlotStyle, plot_feature_counts

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

'''
Added function: 
get_hidden_activations
extract_activations (calls sae_config = SAEConfig()
'''


@dataclass
class ExperimentParams:
    p: int = 53
    n_batches: int = 25000
    n_save_model_checkpoints: int = 100
    print_times: int = 100
    lr: float = 0.005
    batch_size: int = 128
    hidden_size: int = 48
    embed_dim: int = 12
    train_frac: float = 0.4
    # the shown grokking / llc curve behavior is robust to change of seed from my experiments, but not all seeds show grokking withying the first 100 checkpoints, NB!
    random_seed: int = 0
    device: str = DEVICE
    weight_decay: float = 0.0002

    # SAE parameters
    sae_expansion_factor: float = 4.0
    sae_l1_lambda: float = 0.1
    sae_learning_rate: float = 1e-3
    sae_n_epochs: int = 100
    sae_batch_size: int = 64


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.p, params.embed_dim)
        self.linear1r = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear2 = nn.Linear(params.hidden_size, params.p, bias=False)
        self.act = nn.GELU()
        self.vocab_size = params.p

    def forward(self, x):
        x1 = self.embedding(x[..., 0])
        x2 = self.embedding(x[..., 1])
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x)
        x = self.linear2(x)
        return x

    def get_hidden_activations(self, x):
        """Extract hidden layer activations for SAE analysis."""
        x1 = self.embedding(x[..., 0])
        x2 = self.embedding(x[..., 1])
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        hidden = x1 + x2
        hidden_activated = self.act(hidden)
        return hidden_activated


def extract_activations(model, dataloader, device):
    """Extract hidden layer activations from the model."""
    model.eval()
    activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            hidden_acts = model.get_hidden_activations(x)
            activations.append(hidden_acts.cpu())
    
    return torch.cat(activations, dim=0)

def measure_superposition_at_checkpoint(model, dataloader, params, logger=None):
    """Measure superposition using SAE at a specific model checkpoint."""
    log = logger.info if logger else print
    
    # Extract activations
    activations = extract_activations(model, dataloader, params.device)
    log(f"\tExtracted {len(activations)} activation samples")
    
    # Create SAE config
    sae_config = SAEConfig(
        expansion_factor=params.sae_expansion_factor,
        l1_lambda=params.sae_l1_lambda,
        batch_size=params.sae_batch_size,
        learning_rate=params.sae_learning_rate,
        n_epochs=params.sae_n_epochs
    )
    
    # Train SAE
    sae = train_sae(
        activations=activations,
        sae_config=sae_config,
        device=torch.device("cpu"),  # Use CPU for SAE training to save GPU memory
        logger=logger
    )
    
    # Get SAE activations
    sae.eval()
    with torch.no_grad():
        sae_activations, _ = sae(activations)
        sae_activations = sae_activations.numpy()
    
    # Calculate feature distribution and metrics
    feature_dist = get_feature_distribution(sae_activations)
    metrics = calculate_feature_metrics(feature_dist)
    
    # Additional metrics
    metrics['active_features'] = (feature_dist > 1e-6).sum()
    metrics['sparsity'] = (sae_activations == 0.0).mean()
    metrics['dictionary_size'] = sae.dictionary_dim
    
    log(f"\tFeature count: {metrics['feature_count']:.2f}")
    log(f"\tActive features: {metrics['active_features']}")
    log(f"\tSparsity: {metrics['sparsity']:.3f}")
    
    return metrics, sae

def test(model, dataset, device):
    n_correct = 0
    total_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            pred = torch.argmax(out)
            if pred == y:
                n_correct += 1
    return n_correct / len(dataset), total_loss / len(dataset)


def train(train_dataset, test_dataset, params, verbose=True, logger=None):
    log = logger.info if logger else print
    all_models = []
    sae_metrics = []  # Store SAE metrics for each checkpoint

    model = MLP(params).to(params.device)
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=params.weight_decay, lr=params.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

    print_every = params.n_batches // params.print_times
    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.n_batches // params.n_save_model_checkpoints

    loss_data = []
    if verbose:
        pbar = tqdm(total=params.n_batches, desc="Training")
    
    for i in range(params.n_batches):
        # Sample random batch of data
        batch = next(iter(train_loader))
        X, Y = batch
        X, Y = X.to(params.device), Y.to(params.device)
        # Gradient update
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()

        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            checkpoint_model = deepcopy(model)
            all_models.append(checkpoint_model)
            
            # Measure superposition at this checkpoint
            log(f"\nMeasuring superposition at checkpoint {len(all_models)}...")
            metrics, sae_model = measure_superposition_at_checkpoint(
                checkpoint_model, train_loader, params, logger
            )
            sae_metrics.append(metrics)

        if (i + 1) % print_every == 0:
            val_acc, val_loss = test(model, test_dataset, params.device)
            train_acc, train_loss = test(model, train_dataset, params.device)
            loss_data.append(
                {
                    "batch": i + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            if verbose:
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_loss:.4f}",
                        "train_acc": f"{train_acc:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                    }
                )
                pbar.update(print_every)
    if verbose:
        pbar.close()
    df = pd.DataFrame(loss_data)
    final_train_acc, final_train_loss = test(model, train_dataset, params.device)
    final_val_acc, final_val_loss = test(model, test_dataset, params.device)
    
    if verbose:
        print(f"Final Train Acc: {final_train_acc:.4f} | Final Train Loss: {final_train_loss:.4f}")
        print(f"Final Val Acc: {final_val_acc:.4f} | Final Val Loss: {final_val_loss:.4f}")
    return all_models, df, sae_metrics


def deterministic_shuffle(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst


def get_all_pairs(p):
    pairs = []
    for i in range(p):
        for j in range(p):
            pairs.append((i, j))
    return set(pairs)


def make_dataset(p):
    data = []
    pairs = get_all_pairs(p)
    for a, b in pairs:
        data.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))
    return data


def train_test_split(dataset, train_split_proportion, seed):
    l = len(dataset)
    train_len = int(train_split_proportion * l)
    idx = list(range(l))
    idx = deterministic_shuffle(idx, seed)
    train_idx = idx[:train_len]
    test_idx = idx[train_len:]
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]


def estimate_llc_given_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: typing.Callable,
    epsilon: float,
    beta: float,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    localization: float = 5.0,
    num_chains: int = 2,
    num_draws: int = 500,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    device: torch.device = DEVICE,
    online: bool = True,
    verbose: bool = False,
):

    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=dict(lr=epsilon, localization=localization, nbeta=beta),
        num_chains=num_chains,  # How many independent chains to run
        num_draws=num_draws,  # How many samples to draw per chain
        num_burnin_steps=num_burnin_steps,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=num_steps_bw_draws,  # How many steps to take between each sample
        device=device,
        online=online,
        verbose=verbose,
    )

    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats



def plot_combined_results(df, llcs, sae_metrics, params):
    """Plot combined results showing accuracy, LLC, and feature metrics."""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Accuracy curves
    ax1.plot(df["val_acc"], label="test", linewidth=ScientificPlotStyle.LINE_WIDTH, 
             color=ScientificPlotStyle.COLORS[0])
    ax1.plot(df["train_acc"], label="train", linewidth=ScientificPlotStyle.LINE_WIDTH,
             color=ScientificPlotStyle.COLORS[1])
    ScientificPlotStyle.apply_axis_style(
        ax1, "Training Accuracy", "Checkpoint", "Accuracy"
    )
    
    # Plot 2: Loss curves
    ax2.plot(df["val_loss"], label="test", linewidth=ScientificPlotStyle.LINE_WIDTH,
             color=ScientificPlotStyle.COLORS[0])
    ax2.plot(df["train_loss"], label="train", linewidth=ScientificPlotStyle.LINE_WIDTH,
             color=ScientificPlotStyle.COLORS[1])
    ScientificPlotStyle.apply_axis_style(
        ax2, "Training Loss", "Checkpoint", "Loss"
    )
    
    # Plot 3: LLC evolution
    llc_means = [llc["llc/mean"] for llc in llcs]
    ax3.plot(llc_means, linewidth=ScientificPlotStyle.LINE_WIDTH,
             color=ScientificPlotStyle.COLORS[2])
    ScientificPlotStyle.apply_axis_style(
        ax3, "LLC Evolution", "Checkpoint", "LLC (λ̂)", legend=False
    )
    
    # Plot 4: Feature metrics evolution
    feature_counts = [metrics['feature_count'] for metrics in sae_metrics]
    entropies = [metrics['entropy'] for metrics in sae_metrics]
    active_features = [metrics['active_features'] for metrics in sae_metrics]
    
    ax4.plot(feature_counts, label="Feature Count", linewidth=ScientificPlotStyle.LINE_WIDTH,
             color=ScientificPlotStyle.COLORS[0])
    ax4_twin = ax4.twinx()
    ax4_twin.plot(active_features, label="Active Features", linewidth=ScientificPlotStyle.LINE_WIDTH,
                  color=ScientificPlotStyle.COLORS[1])
    
    ScientificPlotStyle.apply_axis_style(
        ax4, "Feature Evolution", "Checkpoint", "Feature Count", legend=False
    )
    ax4_twin.set_ylabel("Active Features", fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax4_twin.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    
    # Create combined legend for plot 4
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, 
               fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best')
    
    plt.tight_layout()
    plt.savefig("experiments/grokking/combined_grokking_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a focused plot showing the relationship between accuracy and feature count
    fig, ax = plt.subplots(figsize=ScientificPlotStyle.FIGURE_SIZE)
    
    # Align data (both should have same length since we measure at same checkpoints)
    test_accs = df["val_acc"].values
    # We need to interpolate or align the indices
    checkpoint_indices = np.linspace(0, len(test_accs)-1, len(feature_counts), dtype=int)
    aligned_test_accs = test_accs[checkpoint_indices]
    
    ax.scatter(feature_counts, aligned_test_accs, 
               s=ScientificPlotStyle.MARKER_SIZE*10, alpha=0.7,
               color=ScientificPlotStyle.COLORS[0])
    
    # Add arrows to show progression
    for i in range(len(feature_counts)-1):
        ax.annotate('', xy=(feature_counts[i+1], aligned_test_accs[i+1]), 
                   xytext=(feature_counts[i], aligned_test_accs[i]),
                   arrowprops=dict(arrowstyle='->', lw=2, alpha=0.5,
                                 color=ScientificPlotStyle.COLORS[1]))
    
    ScientificPlotStyle.apply_axis_style(
        ax, "Grokking: Feature Count vs Test Accuracy", 
        "Feature Count", "Test Accuracy", legend=False
    )
    
    plt.tight_layout()
    plt.savefig("experiments/grokking/feature_count_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize parameters and seed
    params = ExperimentParams()
    torch.manual_seed(params.random_seed)

    # Create and split dataset
    dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(dataset, params.train_frac, params.random_seed)

    # Train model with SAE analysis
    print("Training model with SAE analysis...")
    all_checkpointed_models, df, sae_metrics = train(
        train_dataset=train_data, 
        test_dataset=test_data, 
        params=params,
        logger=logger
    )

    # # Plot training results
    # plt.figure(figsize=(10, 6))
    # plt.plot(df["val_acc"], label="test")
    # plt.plot(df["train_acc"], label="train")
    # plt.legend()
    # plt.ylabel("Correct answer %")
    # plt.xlabel("Checkpoint")
    # plt.title(f"Train & test correct answer % for modular addition with p={params.p}")
    # plt.savefig("experiments/grokking/accuracy_plot.png")
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.plot(df["val_loss"], label="test")
    # plt.plot(df["train_loss"], label="train")
    # plt.legend()
    # plt.ylabel("Loss")
    # plt.xlabel("Checkpoint")
    # plt.title(f"Train & test loss for modular addition with p={params.p}")
    # plt.savefig("experiments/grokking/loss_plot.png")
    # plt.close()

    # LLC estimation
    print("\nPerforming LLC estimation...")
    loader = DataLoader(train_data, shuffle=True, batch_size=params.batch_size)
    
    # Hyperparameter analysis
    analyzer = EpsilonBetaAnalyzer()
    analyzer.configure_sweep(
        llc_estimator=estimate_llc_given_model,
        llc_estimator_kwargs=dict(
            model=all_checkpointed_models[-1],
            evaluate=evaluate_ce,
            device=DEVICE,
            loader=loader,
        ),
        min_epsilon=3e-5,
        max_epsilon=3e-1,
        epsilon_samples=5,
        min_beta=None,
        max_beta=None,
        beta_samples=5,
        dataloader=loader,
    )
    analyzer.sweep()
    
    # Save analyzer plots
    plt.figure(figsize=(10, 6))
    analyzer.plot()
    plt.savefig("experiments/grokking/llc_analysis.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    analyzer.plot(div_out_beta=True)
    plt.savefig("experiments/grokking/llc_analysis_beta.png")
    plt.close()

    # Final LLC estimation with chosen parameters
    lr = 3e-3
    gamma = 5
    nbeta = 2.0
    num_draws = 500
    num_chains = 2

    print("\nEstimating final LLC values...")
    llcs = [
        estimate_learning_coeff_with_summary(
            model_checkpoint,
            loader=DataLoader(train_data, batch_size=params.batch_size, shuffle=True),
            evaluate=evaluate_ce,
            sampling_method=SGLD,
            optimizer_kwargs=dict(lr=lr, nbeta=nbeta, localization=gamma),
            num_chains=1,
            num_draws=num_draws,
            device=DEVICE,
            online=False,
        )
        for model_checkpoint in tqdm(all_checkpointed_models)
    ]

    # Generate combined plots
    print("\nGenerating analysis plots...")
    plot_combined_results(df, llcs, sae_metrics, params)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("GROKKING ANALYSIS SUMMARY")
    print("="*50)
    
    initial_feature_count = sae_metrics[0]['feature_count']
    final_feature_count = sae_metrics[-1]['feature_count']
    
    initial_test_acc = df["val_acc"].iloc[0]
    final_test_acc = df["val_acc"].iloc[-1]
    
    print(f"Initial test accuracy: {initial_test_acc:.3f}")
    print(f"Final test accuracy: {final_test_acc:.3f}")
    print(f"Accuracy improvement: {final_test_acc - initial_test_acc:.3f}")
    print()
    print(f"Initial feature count: {initial_feature_count:.2f}")
    print(f"Final feature count: {final_feature_count:.2f}")
    print(f"Feature count change: {final_feature_count - initial_feature_count:.2f}")
    print()
    
    # Find grokking point (largest increase in test accuracy)
    test_acc_diff = np.diff(df["val_acc"].values)
    grokking_point = np.argmax(test_acc_diff)
    
    print(f"Grokking appears to occur around checkpoint {grokking_point}")
    if grokking_point < len(sae_metrics):
        grok_feature_count = sae_metrics[grokking_point]['feature_count']
        print(f"Feature count at grokking: {grok_feature_count:.2f}")
    
    print("\nPlots saved to experiments/grokking/")


    # # Plot final results
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # plt.title(
    #     f"Lambdahat vs acc for modular addition p={params.p}, train_frac={params.train_frac}"
    # )
    # ax2 = ax1.twinx()
    # ax1.plot(df["val_acc"], label="test acc")
    # ax1.plot(df["train_acc"], label="train acc")
    # ax2.plot([llc["llc/mean"] for llc in llcs], color="g", label="Lambdahat")
    # ax1.set_xlabel("Checkpoint no.")
    # fig.legend(loc="center right")
    # plt.savefig("experiments/grokking/llc_vs_acc.png")
    # plt.close()

    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # plt.title(
    #     f"Lambdahat vs loss for modular addition, p={params.p}, train_frac={params.train_frac}"
    # )
    # ax2 = ax1.twinx()
    # ax1.plot(df["val_loss"], label="test loss")
    # ax1.plot(df["train_loss"], label="train loss")
    # ax2.plot([llc["llc/mean"] for llc in llcs], color="g", label="Lambdahat")
    # ax1.set_xlabel("Checkpoint no.")
    # fig.legend(loc="center right")
    # plt.savefig("experiments/grokking/llc_vs_loss.png")
    # plt.close()

if __name__ == "__main__":
    main()