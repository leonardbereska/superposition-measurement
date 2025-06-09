# %%
"""Evaluation utilities for adversarial robustness experiments with SAE and LLC support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any, Type, Callable
from pathlib import Path
import json
import numpy as np
import os
import logging
import socket
import random
from functools import partial

# SAE imports (existing)
from sae import train_sae, SAEConfig
from models import NNsightModelWrapper
from training import create_adversarial_dataloader, load_checkpoint

from analysis import ScientificPlotStyle
from matplotlib import pyplot as plt

# LLC imports (new)
try:
    import devinterp
    from devinterp.optim.sgld import SGLD
    from devinterp.slt.sampler import estimate_learning_coeff_with_summary
    from devinterp.utils import evaluate_ce
    from devinterp.vis_utils import EpsilonBetaAnalyzer
    LLC_AVAILABLE = True
except ImportError:
    print("Warning: devinterp not available. LLC functionality will be disabled.")
    LLC_AVAILABLE = False

# ============================================================================
# SAE FUNCTIONALITY 
# ============================================================================

def measure_superposition(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    sae_config: SAEConfig,
    max_samples: int = 10000,
    save_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Measure feature organization for a given distribution using SAE.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with data (can be clean, adversarial, or mixed)
        layer_name: Layer to extract activations from
        sae_config: Configuration for the sparse autoencoder
        save_dir: Directory to save results
        max_samples: Maximum number of samples to use
        logger: Optional logger for printing information
    Returns:
        Dictionary with feature metrics
    """
    
    device = torch.device("cpu")  # because it's faster and saves memory on the GPU
    
    model_wrapper = NNsightModelWrapper(model)
    
    # Extract activations
    activations = model_wrapper.get_activations(
        dataloader, layer_name, max_activations=max_samples
    )
    activations = activations.detach().cpu()
    
    # Check if the activations are from a convolutional layer
    is_conv = activations.dim() == 4
    
    if is_conv:
        # Reshape: [B, C, H, W] -> [B*H*W, C]
        orig_shape = activations.shape
        reshaped_activations = activations.reshape(
            orig_shape[0], orig_shape[1], -1).permute(0, 2, 1).reshape(-1, orig_shape[1])
        
        # Limit number of samples for SAE training if needed
        if max_samples and len(reshaped_activations) > max_samples:
            train_activations = reshaped_activations[:max_samples]
        else:
            train_activations = reshaped_activations
        
        # Train SAE on this distribution
        sae = train_sae(
            train_activations,
            sae_config=sae_config,
            device=device,
            logger=logger
        )
        
        # Apply SAE to get activations
        sae_acts, _ = sae(reshaped_activations.to(device))
        sae_acts = sae_acts.detach().cpu()
        
        # Reshape back: [B*H*W, D] -> [B, H*W, D] -> [B, D, H*W]
        sae_acts = sae_acts.reshape(
            orig_shape[0], 
            -1,  # H*W
            sae.dictionary_dim
        ).permute(0, 2, 1)
        
        # Sum over spatial dimensions
        sae_acts = sae_acts.sum(dim=2).detach().cpu().numpy()
        
    else:
        # For fully connected layers, use the existing approach
        # Train SAE on this distribution
        sae = train_sae(
            activations,
            sae_config=sae_config,
            device=device,
            logger=logger
        )
        
        # Evaluate SAE on the same distribution it was trained on
        sae_acts, _ = sae(activations.to(device))
        sae_acts = sae_acts.detach().cpu().numpy()
    
    # Save SAE model if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_dir / f"sae_{layer_name}.pt")
    
    # Calculate feature distribution and metrics
    p = get_feature_distribution(sae_acts)
    metrics = calculate_feature_metrics(p)
    
    return metrics


def get_feature_distribution(activations: torch.Tensor) -> np.ndarray:
    """Get distribution of activation magnitudes per feature.
    
    Args:
        activations: Tensor of activations [batch_size, feature_dim]
        
    Returns:
        Normalized distribution of feature importance
    """
    feature_norms = np.mean(np.abs(activations), axis=0)
    p = feature_norms / feature_norms.sum()
    return p

def calculate_feature_metrics(p: np.ndarray) -> Dict[str, float]:
    """Calculate entropy and feature count metrics from activation distribution.
    
    Args:
        p: Normalized distribution of feature importance
        
    Returns:
        Dictionary of metrics
    """
    # Add epsilon to avoid log(0)
    p_safe = p + 1e-10
    p_safe = p_safe / p_safe.sum()  # Renormalize
    
    # Calculate entropy
    entropy = -np.sum(p_safe * np.log(p_safe))
    
    # Effective feature count (exponential of entropy)
    feature_count = np.exp(entropy)
    
    # Additional metrics could be added here
    return {
        'entropy': entropy,
        'feature_count': feature_count,
    }

'''
def load_model is in training
def evaluate_model_performance is in training
def generate_adversarial_example is in attacks.py

def evaluate_feature_organization is nowhere
def measure_superposition is nowhere
def measure_superposition_on_mixed_distribution is nowhere
'''


# ============================================================================
# ADDITIONAL SAE FUNCTIONALITY FOR RETROACTIVE CHECKPOINT MEASUREMENTS
# ============================================================================


def analyze_checkpoints_with_sae(
    checkpoint_dir: Path,
    train_loader: DataLoader,
    sae_config: SAEConfig,
    layer_names: List[str],
    epsilons_to_test: List[float],
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Analyze saved checkpoints with SAE measurement.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_loader: Training data loader
        sae_config: SAE configuration
        layer_names: List of layers to analyze
        epsilons_to_test: List of perturbation strengths to test
        device: Device for computation
        logger: Optional logger
        
    Returns:
        Dictionary with SAE analysis results organized by epsilon, layer, and epoch
    """
    from training import load_checkpoint, create_adversarial_dataloader
    from attacks import AttackConfig
    
    log = logger.info if logger else print
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint summary
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        checkpoint_summary = json.load(f)
    
    checkpoint_paths = [Path(cp) for cp in checkpoint_summary['saved_checkpoints']]
    attack_config = AttackConfig()
    
    # Results organized by epsilon -> layer -> epoch
    results = {
        'epsilon_analysis': {},
        'layer_analysis': {},
        'evolution_analysis': {}
    }
    
    for epsilon in epsilons_to_test:
        log(f"SAE analysis for epsilon={epsilon} across {len(checkpoint_paths)} checkpoints")
        epsilon_results = {}
        
        for layer_name in layer_names:
            log(f"  Analyzing layer: {layer_name}")
            layer_results = []
            
            for checkpoint_path in checkpoint_paths:
                # Load checkpoint
                model, model_config, checkpoint_data = load_checkpoint(checkpoint_path, device)
                epoch = checkpoint_data['epoch']
                
                log(f"    Processing checkpoint from epoch {epoch}")
                
                # Create appropriate dataloader
                if epsilon > 0:
                    test_loader = create_adversarial_dataloader(
                        model, train_loader, epsilon, attack_config
                    )
                else:
                    test_loader = train_loader
                
                # Measure SAE features
                try:
                    sae_results = measure_superposition(
                        model=model,
                        dataloader=test_loader,
                        layer_name=layer_name,
                        sae_config=sae_config,
                        save_dir=checkpoint_dir / f"sae_analysis_eps_{epsilon}_layer_{layer_name}_epoch_{epoch}",
                        logger=logger
                    )
                    
                    layer_results.append({
                        'epoch': epoch,
                        'feature_count': sae_results['feature_count'],
                        'entropy': sae_results['entropy'],
                        'checkpoint_path': str(checkpoint_path)
                    })
                    
                    log(f"      Epoch {epoch}: Feature count = {sae_results['feature_count']:.2f}")
                    
                except Exception as e:
                    log(f"      Error in epoch {epoch}: {e}")
                    layer_results.append({
                        'epoch': epoch,
                        'feature_count': None,
                        'entropy': None,
                        'checkpoint_path': str(checkpoint_path),
                        'error': str(e)
                    })
            
            epsilon_results[layer_name] = layer_results
        
        results['epsilon_analysis'][str(epsilon)] = epsilon_results
    
    # Save results
    results_path = checkpoint_dir / "sae_checkpoint_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    log(f"SAE checkpoint analysis results saved to: {results_path}")
    
    return results



# ============================================================================
# LLC FUNCTIONALITIES
# ============================================================================

def evaluate_bce(model, data):
    """Evaluate function for binary classification with BCE loss."""
    x, y = data
    y_pred = model(x)
    
    # Handle shape mismatch for binary classification
    y_pred = y_pred.squeeze()  # Convert from [batch_size, 1] to [batch_size]
    y = y.float()  # Ensure targets are float for BCE
        
    return F.binary_cross_entropy_with_logits(y_pred, y), {"output": y_pred}


def initialize_distributed_training(rank=0):
    """Initialize distributed training for LLC measurement."""
    if not LLC_AVAILABLE:
        return False
        
    def find_free_port():
        # Try ports in range 29500-65535
        while True:
            port = random.randint(29500, 65535)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                sock.close()
                continue

    try:
        # Find an available port
        port = find_free_port()
        
        # Initialize process group with the free port
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{port}',
            world_size=1,
            rank=rank
        )
        return True
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        return False

def estimate_llc_given_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    evaluate: Callable,
    epsilon: float,                                     
    beta: float,                                        
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    localization: float = 5.0,
    num_chains: int = 3,
    num_draws: int = 1500,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    online: bool = True,
    verbose: bool = False,
) -> Dict:
    """Estimate LLC with given hyperparameters."""
    if not LLC_AVAILABLE:
        raise RuntimeError("LLC functionality requires devinterp library")
    
    sweep_stats = estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=sampling_method,
        optimizer_kwargs=dict(
            lr=epsilon, 
            localization=localization, 
            nbeta=beta
        ),
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        device=device,
        online=online,
        verbose=verbose,
    )

    if "llc/trace" not in sweep_stats:
        # Get the trace from wherever estimate_learning_coeff_with_summary puts it
        trace = sweep_stats.get("llc_trace", sweep_stats.get("trace", np.array([[sweep_stats.get("llc_mean", 0.0)]])))
        sweep_stats["llc/trace"] = trace
    else:
        sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    
    return sweep_stats

def find_stable_hyperparameters(sweep_df):
    """Find stable hyperparameters from LLC sweep results using stability analysis."""
    
    def analyze_trace_stability(trace, desired_stable_step=200):
        """Analyze a single trace for stability focusing on early stabilization and no drift."""
        n_steps = len(trace)
        
        # Split trace into regions
        early_region = trace[:desired_stable_step]
        stable_region = trace[desired_stable_step:]
        
        # 1. Check for zero slope in stable region (most important)
        x = np.arange(len(stable_region))
        slope, _ = np.polyfit(x, stable_region, 1)
        
        # Exponentially penalize any non-zero slope
        slope_penalty = np.exp(np.abs(slope) * 100)
        
        # 2. Check if we've stabilized by desired_stable_step
        early_mean = np.mean(early_region[-50:])  # mean of last 50 points in early region
        stable_mean = np.mean(stable_region[:50])  # mean of first 50 points in stable region
        stabilization_score = 1.0 / (np.abs(early_mean - stable_mean) + 1e-6)
        
        # 3. Check for consistent range throughout stable region
        window_size = 100
        rolling_means = [np.mean(stable_region[i:i+window_size]) 
                        for i in range(0, len(stable_region)-window_size, window_size)]
        mean_consistency = 1.0 / (np.std(rolling_means) + 1e-6)
        
        # 4. Penalize large jumps
        max_jump = np.max(np.abs(np.diff(stable_region)))
        jump_penalty = np.exp(max_jump)
        
        # Combine scores with very heavy emphasis on slope and mean consistency
        stability_score = (mean_consistency * stabilization_score) / (slope_penalty * jump_penalty)
        
        return (
            stability_score,
            slope,
            np.std(rolling_means),  # mean variation
            max_jump,
            1.0 / stabilization_score  # early stabilization penalty
        )

    # Process traces
    if isinstance(sweep_df['llc/means'].iloc[0], (list, np.ndarray)):
        stability_results = sweep_df['llc/means'].apply(analyze_trace_stability)
        sweep_df['stability_score'] = stability_results.apply(lambda x: x[0])
        sweep_df['slope'] = stability_results.apply(lambda x: x[1])
        sweep_df['mean_variation'] = stability_results.apply(lambda x: x[2])
        sweep_df['max_jump'] = stability_results.apply(lambda x: x[3])
        sweep_df['stabilization_penalty'] = stability_results.apply(lambda x: x[4])
    
    # Add epsilon preference score with stronger preference for higher values
    epsilon_values = sweep_df['epsilon'].unique()
    print("\nEpsilon values being considered:", sorted(epsilon_values))

    
    # Create a more aggressive preference for higher epsilon values
    epsilon_min, epsilon_max = sweep_df['epsilon'].min(), sweep_df['epsilon'].max()
    # Use exponential scaling to create stronger preference for higher values
    sweep_df['epsilon_preference'] = np.exp((sweep_df['epsilon'] - epsilon_min) / (epsilon_max - epsilon_min)) - 1
    
    # Print epsilon preference values for debugging
    print("\nEpsilon preference mapping:")
    for eps in sorted(epsilon_values):
        pref = np.exp((eps - epsilon_min) / (epsilon_max - epsilon_min)) - 1
        print(f"Epsilon {eps:.2e} -> Preference {pref:.2f}")
    
    # Combine stability score with epsilon preference (with stronger weight)
    alpha = 2.0  # Increase this to give more weight to epsilon preference
    sweep_df['combined_score'] = sweep_df['stability_score'] * (1 + alpha * sweep_df['epsilon_preference'])
    
    # Group by epsilon and beta
    grouped_stats = sweep_df.groupby(['epsilon', 'beta']).agg({
        'combined_score': 'mean',
        'stability_score': 'mean',
        'slope': 'mean',
        'mean_variation': 'mean',
        'max_jump': 'mean',
        'stabilization_penalty': 'mean',
        'epsilon_preference': 'first'
    }).reset_index()
    
    # Find the most stable region using combined score
    best_idx = grouped_stats['combined_score'].argmax()
    stable_epsilon = grouped_stats.iloc[best_idx]['epsilon']
    stable_beta = grouped_stats.iloc[best_idx]['beta']
    
    return stable_epsilon, stable_beta

def tune_llc_hyperparameters(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    min_epsilon: float = 3e-3,
    max_epsilon: float = 3e-1,
    epsilon_samples: int = 5,
    min_beta: Optional[float] = None,
    max_beta: Optional[float] = None,
    beta_samples: int = 5,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Find optimal hyperparameters for LLC estimation using epsilon-beta sweep."""
    if not LLC_AVAILABLE:
        raise RuntimeError("LLC functionality requires devinterp library")
    
    # Choose appropriate evaluation function
    if list(model.parameters())[-1].size(0) == 1:
        evaluate_fn = evaluate_bce
        print("Using BCE evaluation function for LLC estimation in tune_llc_hyperparameters")
    else:
        evaluate_fn = evaluate_ce
        print("Using CE evaluation function for LLC estimation in tune_llc_hyperparameters")
    
    analyzer = EpsilonBetaAnalyzer() 
    
    analyzer.configure_sweep(
        llc_estimator=estimate_llc_given_model,
        llc_estimator_kwargs=dict(
            model=model,
            evaluate=evaluate_fn,
            device=device,
            loader=loader,
            localization=9.0,
        ),
        min_epsilon=min_epsilon,
        max_epsilon=max_epsilon,
        epsilon_samples=epsilon_samples,
        min_beta=min_beta,
        max_beta=max_beta,
        beta_samples=beta_samples,
        dataloader=loader,
    )

    analyzer.sweep()

    # Generate plots for analysis
    fig1 = analyzer.plot()  # Standard plot
    fig2 = analyzer.plot(div_out_beta=True)  # Plot with beta divided out
    
    # Access sweep_df for analysis
    if hasattr(analyzer, 'sweep_df'):
        sweep_df = analyzer.sweep_df
        stable_epsilon, stable_beta = find_stable_hyperparameters(sweep_df)
    else:
        # Use default values from the grokking example
        print("No sweep_df found, USING ~DEFAULT~ VALUES")
        stable_epsilon = 0.003 # 3e-3
        stable_beta = 48.6
    
    return {
        'recommended_epsilon': stable_epsilon,
        'recommended_beta': stable_beta,
        'analyzer': analyzer,  # Return analyzer for plotting if needed
        'figures': [fig1, fig2]
    }

'''
    Online vs offline stats:
    - online stats used for a) Hyperparameter tuning with EpsilonBetaAnalyzer and b) Initial testing of sampling quality
    - offline stats used for final LLC calculation across all model checkpoints
    
    Why Both?
    Online advantages:
    - Memory efficient (running averages instead of storing all samples)
    - Can monitor convergence during sampling
    - Good for exploration/tuning where you need quick feedback

    Offline advantages:
    - More accurate statistics (can calculate on all samples at once)
    - Can perform more sophisticated statistical analysis
    - Better for final results where accuracy matters most

    In the reference notebook:
    - They use online during hyperparameter exploration to quickly test many configurations
    - They switch to offline for the final LLC calculations across all checkpoints to get the most accurate estimates for their plots
    '''

def estimate_llc(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    llc_epsilon: Optional[float] = None,
    llc_nbeta: Optional[float] = None,
    gamma: float = 5.0,
    num_chains: int = 3,
    num_draws: int = 1500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tune_hyperparams: bool = False,
    online: bool = False
) -> Dict[str, float]:
    """Estimate LLC using provided parameters."""
    if not LLC_AVAILABLE:
        raise RuntimeError("LLC functionality requires devinterp library")
    # Print debug info
    print(f"Just for debug, LLC epsilon: {llc_epsilon}, type: {type(llc_epsilon)}")
    print(f"Just for debug, LLC nbeta: {llc_nbeta}, type: {type(llc_nbeta)}")
    print(f"Just for debug, tune_hyperparams: {tune_hyperparams}")
    print(f"Using {'online' if online else 'offline'} stats")

    # Choose appropriate evaluation function
    if list(model.parameters())[-1].size(0) == 1:
        evaluate_fn = evaluate_bce
    else:
        evaluate_fn = evaluate_ce
   
    try:
        # Clean up existing process group if it exists
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Initialize new process group
        if not initialize_distributed_training():
            print("Failed to initialize distributed process group")
            return None
    
        # Estimate LLC
        llc_stats = estimate_learning_coeff_with_summary(
            model,
            loader=data_loader,
            evaluate=evaluate_fn,
            sampling_method=SGLD,
            optimizer_kwargs=dict(
                lr=llc_epsilon,
                nbeta=llc_nbeta,
                localization=gamma
            ),
            num_chains=num_chains,
            num_draws=num_draws,
            device=device,
            online=online,  
        )
        
        print("llc_stats keys:", llc_stats.keys())

        '''
        When estimate_learning_coeff_with_summary is called, the keys of the llc_stats returned are: ['init_loss', 'llc/means', 'llc/stds', 'llc/trace', 'loss/trace']
        Now, tune_llc_parameters in train.py calls estimate_llc with online=True
        When we do this: Now making a single test run with chosen hyperparameters, still on FINAL model to check if the hyperparams are good"
        The keys are : ['llc_average_mean', 'llc_average_std', 'llc_means', 'llc_stds', 'llc_trace', 'loss_trace', 'llc_epsilon', 'llc_nbeta', 'gamma']

        '''
        if "llc/trace" not in llc_stats:
            print("WHY IS IT EVEN GOING IN HERE? Because we're using offline stats and llc/trace is not in llc_stats")
            print("llc/trace not found in llc stats")
            #the keys are: dict_keys(['llc/mean', 'llc/std', 
            # 'llc-chain/0', 'llc-chain/1', 'llc-chain/2', 'loss/trace'])
            print("llc-chain/0 value:", llc_stats['llc-chain/0'])
            print("llc-chain/1 value:", llc_stats['llc-chain/1'])
            print("llc-chain/2 value:", llc_stats['llc-chain/2'])
            print("llc-chain/0 type:", type(llc_stats['llc-chain/0']))
        
        if online:
            # Online mode keys: 'init_loss', 'llc/means', 'llc/stds', 'llc/trace', 'loss/trace'
            return {
                'llc_average_mean': llc_stats['llc/means'].mean(),
                'llc_average_std': llc_stats['llc/stds'].mean(),
                'llc/means': llc_stats['llc/means'],
                'llc/stds': llc_stats['llc/stds'],
                'llc/trace': llc_stats['llc/trace'],
                'loss/trace': llc_stats['loss/trace'],
                'llc_epsilon': llc_epsilon,
                'llc_nbeta': llc_nbeta,
                'gamma': gamma,
            }
        else:
            # Offline mode keys: 'llc/mean', 'llc/std', 'llc-chain/0', 'llc-chain/1', 'llc-chain/2', 'loss/trace'
            return {
                'llc_average_mean': llc_stats['llc/mean'],
                'llc_average_std': llc_stats['llc/std'],
                'loss/trace': llc_stats['loss/trace'],
                'llc_epsilon': llc_epsilon,
                'llc_nbeta': llc_nbeta,
                'gamma': gamma,
            }
        
    except Exception as e:
        print(f"Error during LLC estimation: {e}")
        return None

    finally:
        # Clean up process group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()



# ============================================================================
#  RETROACTIVE LLC EVALUATION ON CHECKPOINTS
# ============================================================================

def analyze_checkpoints_with_llc(
    checkpoint_dir: Path,
    train_loader: DataLoader,
    llc_config: Dict[str, Any],
    epsilons_to_test: List[float],
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Analyze saved checkpoints with LLC measurement.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_loader: Training data loader
        llc_config: LLC configuration
        epsilons_to_test: List of perturbation strengths to test
        device: Device for computation
        logger: Optional logger
        
    Returns:
        Dictionary with LLC analysis results
    """
    from evaluation import estimate_llc, tune_llc_hyperparameters
    from attacks import AttackConfig
    
    log = logger.info if logger else print
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint summary
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        checkpoint_summary = json.load(f)
    
    checkpoint_paths = [Path(cp) for cp in checkpoint_summary['saved_checkpoints']]
    
    # Load the final model for hyperparameter tuning
    final_model, model_config, _ = load_checkpoint(checkpoint_paths[-1], device)
    
    # Tune LLC hyperparameters on final model
    log("Tuning LLC hyperparameters...")
    tuning_results = tune_llc_hyperparameters(
        model=final_model,
        loader=train_loader,
        min_epsilon=llc_config.get('min_epsilon', 3e-3),
        max_epsilon=llc_config.get('max_epsilon', 3e-1),
        epsilon_samples=llc_config.get('epsilon_samples', 5),
        beta_samples=llc_config.get('beta_samples', 5),
        device=device
    )
    
    llc_epsilon = tuning_results['recommended_epsilon']
    llc_nbeta = tuning_results['recommended_beta']
    
    log(f"Recommended LLC parameters: epsilon={llc_epsilon:.2e}, beta={llc_nbeta:.2f}")
    
    # Analyze each checkpoint
    results = {
        'tuning_results': tuning_results,
        'checkpoint_analysis': {},
        'epsilon_analysis': {}
    }
    
    attack_config = AttackConfig()
    
    # Analyze across different perturbation strengths
    for eps in epsilons_to_test:
        log(f"Analyzing checkpoints with epsilon={eps}")
        
        # Create adversarial dataloader if needed
        if eps > 0:
            test_loader = create_adversarial_dataloader(
                final_model, train_loader, eps, attack_config
            )
        else:
            test_loader = train_loader
        
        epsilon_results = []
        
        for checkpoint_path in checkpoint_paths:
            # Load checkpoint
            model, _, checkpoint_data = load_checkpoint(checkpoint_path, device)
            epoch = checkpoint_data['epoch']
            
            log(f"Analyzing checkpoint from epoch {epoch} with epsilon {eps}")
            
            # Estimate LLC
            llc_results = estimate_llc(
                model=model,
                data_loader=test_loader,
                llc_epsilon=llc_epsilon,
                llc_nbeta=llc_nbeta,
                gamma=llc_config.get('gamma', 5.0),
                num_chains=llc_config.get('num_chains', 3),
                num_draws=llc_config.get('num_draws', 1500),
                device=str(device),
                online=llc_config.get('online_stats', False)
            )
            
            if llc_results is not None:
                epsilon_results.append({
                    'epoch': epoch,
                    'llc_mean': llc_results['llc_average_mean'],
                    'llc_std': llc_results['llc_average_std'],
                    'checkpoint_path': str(checkpoint_path)
                })
        
        results['epsilon_analysis'][str(eps)] = epsilon_results
    
    # Save results
    results_path = checkpoint_dir / "llc_analysis_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            return obj
        
        json.dump(results, f, indent=4, default=json_serializer)
    
    log(f"LLC analysis results saved to: {results_path}")
    
    return results


# ============================================================================
# POTENTIAL FUTURE FEATURE: INFERENCE-TIME LLC MEASUREMENTS
# ============================================================================

def measure_inference_time_llc(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    llc_config: Dict[str, Any],
    save_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """Measure LLC for inference-time analysis (currently not used in main workflow).
    
    This function would be useful for generating the inference-time LLC plots:
    - plot_llc_traces, plot_llc_distribution, plot_llc_mean_std
    
    Currently our workflow only does checkpoint evolution analysis, but this could
    be added as an optional feature for comparing final trained models.
    
    Args:
        model: Final trained model
        dataloaders: Dictionary of dataloaders {'clean': clean_loader, 'eps_0.1': adv_loader, ...}
        llc_config: LLC configuration  
        save_dir: Directory to save results
        logger: Optional logger
        
    Returns:
        Dictionary with inference-time LLC results in format expected by plotting functions
    """
    if not LLC_AVAILABLE:
        if logger:
            logger.warning("LLC analysis requested but devinterp not available.")
        return None
    
    log = logger.info if logger else print
    log("Performing inference-time LLC analysis...")
    
    # Tune hyperparameters on clean data
    clean_loader = dataloaders.get('clean', list(dataloaders.values())[0])
    tuning_results = tune_llc_hyperparameters(
        model=model,
        loader=clean_loader,
        min_epsilon=llc_config.get('min_epsilon', 3e-3),
        max_epsilon=llc_config.get('max_epsilon', 3e-1),
        epsilon_samples=llc_config.get('epsilon_samples', 5),
        beta_samples=llc_config.get('beta_samples', 5),
        device=next(model.parameters()).device
    )
    
    llc_epsilon = tuning_results['recommended_epsilon']
    llc_nbeta = tuning_results['recommended_beta']
    
    # Measure LLC for each condition
    inference_results = {}
    
    for condition_name, dataloader in dataloaders.items():
        log(f"Measuring LLC for {condition_name}...")
        
        # Extract epsilon from condition name (e.g., 'eps_0.1' -> 0.1)
        if condition_name == 'clean':
            eps_value = 0.0
        else:
            try:
                eps_value = float(condition_name.split('_')[-1])
            except:
                eps_value = 0.0
        
        # Measure LLC multiple times for statistics
        llc_measurements = []
        for run in range(3):  # Multiple runs for statistical analysis
            llc_stats = estimate_llc(
                model=model,
                data_loader=dataloader,
                llc_epsilon=llc_epsilon,
                llc_nbeta=llc_nbeta,
                gamma=llc_config.get('gamma', 5.0),
                num_chains=llc_config.get('num_chains', 3),
                num_draws=llc_config.get('num_draws', 1500),
                device=str(next(model.parameters()).device),
                online=True  # Use online stats to get traces
            )
            
            if llc_stats is not None:
                llc_measurements.append(llc_stats['llc_average_mean'])
        
        if llc_measurements:
            # Store results in format expected by plotting functions
            inference_results[eps_value] = {
                'means': np.array(llc_measurements),
                'mean': np.mean(llc_measurements),
                'std': np.std(llc_measurements),
                'trace': llc_stats['loss/trace'] if llc_stats else None  # From last measurement
            }
    
    # Save results
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = save_dir / "inference_llc_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for k, v in inference_results.items():
                json_results[str(k)] = {
                    'means': v['means'].tolist() if isinstance(v['means'], np.ndarray) else v['means'],
                    'mean': float(v['mean']),
                    'std': float(v['std']),
                    'trace': v['trace'].tolist() if isinstance(v['trace'], np.ndarray) else v['trace']
                }
            json.dump(json_results, f, indent=4)
        
        log(f"Inference-time LLC results saved to: {results_path}")
    
    return inference_results

# ============================================================================
# SAE FUNCTIONALITIES FOR CHECKPOINT EVALUATION AND COMPARISON WITH LLC
# ============================================================================

def analyze_checkpoints_with_sae(
    checkpoint_dir: Path,
    train_loader: DataLoader,
    sae_config: SAEConfig,
    layer_names: List[str],
    epsilons_to_test: List[float],
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Analyze saved checkpoints with SAE measurement.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_loader: Training data loader
        sae_config: SAE configuration
        layer_names: List of layers to analyze
        epsilons_to_test: List of perturbation strengths to test
        device: Device for computation
        logger: Optional logger
        
    Returns:
        Dictionary with SAE analysis results organized by epsilon, layer, and epoch
    """
    from training import load_checkpoint, create_adversarial_dataloader
    from attacks import AttackConfig
    
    log = logger.info if logger else print
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint summary
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        checkpoint_summary = json.load(f)
    
    checkpoint_paths = [Path(cp) for cp in checkpoint_summary['saved_checkpoints']]
    attack_config = AttackConfig()
    
    # Results organized by epsilon -> layer -> epoch
    results = {
        'epsilon_analysis': {},
        'layer_analysis': {},
        'evolution_analysis': {}
    }
    
    for epsilon in epsilons_to_test:
        log(f"SAE analysis for epsilon={epsilon} across {len(checkpoint_paths)} checkpoints")
        epsilon_results = {}
        
        for layer_name in layer_names:
            log(f"  Analyzing layer: {layer_name}")
            layer_results = []
            
            for checkpoint_path in checkpoint_paths:
                # Load checkpoint
                model, model_config, checkpoint_data = load_checkpoint(checkpoint_path, device)
                epoch = checkpoint_data['epoch']
                
                log(f"    Processing checkpoint from epoch {epoch}")
                
                # Create appropriate dataloader
                if epsilon > 0:
                    test_loader = create_adversarial_dataloader(
                        model, train_loader, epsilon, attack_config
                    )
                else:
                    test_loader = train_loader
                
                # Measure SAE features
                try:
                    sae_results = measure_superposition(
                        model=model,
                        dataloader=test_loader,
                        layer_name=layer_name,
                        sae_config=sae_config,
                        save_dir=checkpoint_dir / f"sae_analysis_eps_{epsilon}_layer_{layer_name}_epoch_{epoch}",
                        logger=logger
                    )
                    
                    layer_results.append({
                        'epoch': epoch,
                        'feature_count': sae_results['feature_count'],
                        'entropy': sae_results['entropy'],
                        'checkpoint_path': str(checkpoint_path)
                    })
                    
                    log(f"      Epoch {epoch}: Feature count = {sae_results['feature_count']:.2f}")
                    
                except Exception as e:
                    log(f"      Error in epoch {epoch}: {e}")
                    layer_results.append({
                        'epoch': epoch,
                        'feature_count': None,
                        'entropy': None,
                        'checkpoint_path': str(checkpoint_path),
                        'error': str(e)
                    })
            
            epsilon_results[layer_name] = layer_results
        
        results['epsilon_analysis'][str(epsilon)] = epsilon_results
    
    # Save results
    results_path = checkpoint_dir / "sae_checkpoint_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    log(f"SAE checkpoint analysis results saved to: {results_path}")
    
    return results


def plot_combined_llc_sae_evolution(
    llc_checkpoint_results: Dict[str, Any],
    sae_checkpoint_results: Dict[str, Any],
    layer_name: str,
    epsilon: float,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Plot LLC and SAE feature count evolution side by side during training.
    
    Args:
        llc_checkpoint_results: Results from analyze_checkpoints_with_llc
        sae_checkpoint_results: Results from analyze_checkpoints_with_sae
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value to analyze
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from analysis import ScientificPlotStyle
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    if title is None:
        title = f'LLC vs SAE Evolution During Training - ε={epsilon}, Layer: {layer_name}'
    
    # Extract LLC data
    llc_data = llc_checkpoint_results['epsilon_analysis'][str(epsilon)]
    llc_epochs = [entry['epoch'] for entry in llc_data]
    llc_means = [entry['llc_mean'] for entry in llc_data]
    llc_stds = [entry['llc_std'] for entry in llc_data]
    
    # Extract SAE data
    sae_layer_data = sae_checkpoint_results['epsilon_analysis'][str(epsilon)][layer_name]
    sae_epochs = [entry['epoch'] for entry in sae_layer_data if entry['feature_count'] is not None]
    sae_features = [entry['feature_count'] for entry in sae_layer_data if entry['feature_count'] is not None]
    
    # Plot LLC evolution
    ax1.errorbar(llc_epochs, llc_means, yerr=llc_stds,
                marker='o', linewidth=ScientificPlotStyle.LINE_WIDTH,
                markersize=ScientificPlotStyle.MARKER_SIZE,
                color=ScientificPlotStyle.COLORS[0],
                capsize=ScientificPlotStyle.CAPSIZE,
                capthick=ScientificPlotStyle.CAPTHICK)
    
    ax1.set_title('LLC Evolution', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax1.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.set_ylabel('Learning Coefficient', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax1.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax1.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    # Plot SAE feature count evolution
    ax2.plot(sae_epochs, sae_features,
            marker='s', linewidth=ScientificPlotStyle.LINE_WIDTH,
            markersize=ScientificPlotStyle.MARKER_SIZE,
            color=ScientificPlotStyle.COLORS[1])
    
    ax2.set_title(f'SAE Feature Count Evolution\n({layer_name})', 
                 fontsize=ScientificPlotStyle.FONT_SIZE_LABELS, fontweight='bold')
    ax2.set_xlabel('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.set_ylabel('Feature Count', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    ax2.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    ax2.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
    
    plt.suptitle(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_llc_vs_sae_correlation_over_time(
    llc_checkpoint_results: Dict[str, Any],
    sae_checkpoint_results: Dict[str, Any], 
    layer_name: str,
    epsilon: float,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = ScientificPlotStyle.FIGURE_SIZE
) -> plt.Figure:
    """Plot correlation between LLC and SAE feature count over training time.
    
    Args:
        llc_checkpoint_results: Results from analyze_checkpoints_with_llc
        sae_checkpoint_results: Results from analyze_checkpoints_with_sae
        layer_name: Layer name for SAE analysis
        epsilon: Epsilon value to analyze
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if title is None:
        title = f'LLC vs SAE Feature Count Correlation - ε={epsilon}, Layer: {layer_name}'
    
    # Extract and align data by epoch
    llc_data = {entry['epoch']: entry['llc_mean'] 
                for entry in llc_checkpoint_results['epsilon_analysis'][str(epsilon)]}
    
    sae_data = {entry['epoch']: entry['feature_count'] 
                for entry in sae_checkpoint_results['epsilon_analysis'][str(epsilon)][layer_name]
                if entry['feature_count'] is not None}
    
    # Find common epochs
    common_epochs = sorted(set(llc_data.keys()) & set(sae_data.keys()))
    
    if len(common_epochs) < 2:
        ax.text(0.5, 0.5, 'Insufficient data for correlation plot', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    llc_values = [llc_data[epoch] for epoch in common_epochs]
    sae_values = [sae_data[epoch] for epoch in common_epochs]
    
    # Create scatter plot with epoch progression shown by color
    scatter = ax.scatter(sae_values, llc_values, 
                        c=common_epochs, cmap='viridis',
                        s=ScientificPlotStyle.MARKER_SIZE**2,
                        alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add colorbar for epochs
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Epoch', fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
    cbar.ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
    
    # Add arrows to show progression
    for i in range(len(common_epochs)-1):
        ax.annotate('', xy=(sae_values[i+1], llc_values[i+1]), 
                   xytext=(sae_values[i], llc_values[i]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.6, lw=2))
    
    # Apply styling
    ScientificPlotStyle.apply_axis_style(
        ax=ax,
        title=title,
        xlabel=f'SAE Feature Count ({layer_name})',
        ylabel='Learning Coefficient',
        legend=False
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig