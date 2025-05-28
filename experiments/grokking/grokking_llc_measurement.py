# -*- coding: utf-8 -*-
"""
Grokking experiment showing LLC estimation calibration in a simple modular addition example.
Adapted from Nina Panickssery and Dmitry Vaintrob's modular addition learning coefficient work.
"""

import random
from copy import deepcopy
from dataclasses import dataclass
import typing
from typing import Type
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce, plot_trace
from devinterp.vis_utils import EpsilonBetaAnalyzer

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def train(train_dataset, test_dataset, params, verbose=True):
    all_models = []
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
            all_models += [deepcopy(model)]

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
    train_acc, train_loss = test(model, train_dataset, params.device)
    val_acc, val_loss = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {val_acc:.4f} | Final Train Loss: {val_loss:.4f}")
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    return all_models, df


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


def main():
    # Initialize parameters and seed
    params = ExperimentParams()
    torch.manual_seed(params.random_seed)

    # Create and split dataset
    dataset = make_dataset(params.p)
    train_data, test_data = train_test_split(dataset, params.train_frac, params.random_seed)

    # Train model
    print("Training model...")
    all_checkpointed_models, df = train(
        train_dataset=train_data, test_dataset=test_data, params=params
    )

    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(df["val_acc"], label="test")
    plt.plot(df["train_acc"], label="train")
    plt.legend()
    plt.ylabel("Correct answer %")
    plt.xlabel("Checkpoint")
    plt.title(f"Train & test correct answer % for modular addition with p={params.p}")
    plt.savefig("experiments/grokking/accuracy_plot.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df["val_loss"], label="test")
    plt.plot(df["train_loss"], label="train")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Checkpoint")
    plt.title(f"Train & test loss for modular addition with p={params.p}")
    plt.savefig("experiments/grokking/loss_plot.png")
    plt.close()

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

    # Plot final results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(
        f"Lambdahat vs acc for modular addition p={params.p}, train_frac={params.train_frac}"
    )
    ax2 = ax1.twinx()
    ax1.plot(df["val_acc"], label="test acc")
    ax1.plot(df["train_acc"], label="train acc")
    ax2.plot([llc["llc/mean"] for llc in llcs], color="g", label="Lambdahat")
    ax1.set_xlabel("Checkpoint no.")
    fig.legend(loc="center right")
    plt.savefig("experiments/grokking/llc_vs_acc.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(
        f"Lambdahat vs loss for modular addition, p={params.p}, train_frac={params.train_frac}"
    )
    ax2 = ax1.twinx()
    ax1.plot(df["val_loss"], label="test loss")
    ax1.plot(df["train_loss"], label="train loss")
    ax2.plot([llc["llc/mean"] for llc in llcs], color="g", label="Lambdahat")
    ax1.set_xlabel("Checkpoint no.")
    fig.legend(loc="center right")
    plt.savefig("experiments/grokking/llc_vs_loss.png")
    plt.close()

if __name__ == "__main__":
    main()