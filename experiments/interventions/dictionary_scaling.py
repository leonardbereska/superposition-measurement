# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import pandas as pd

from dataset_multitasksparseparity import MultiTaskSparseParityDataset
from dropout import train_sae, calculate_feature_count, InterventionExperiment, SparseParityModel

class DictionarySizeExperiment:
    """Experiment to study how measured feature count scales with SAE dictionary size."""
    
    def __init__(
        self,
        num_tasks=3,
        bits_per_task=4,
        hidden_dims=[32, 64, 128],
        expansion_factors=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        l1_coefficients=[0.01, 0.1, 1.0],
        seeds=5,
        max_dictionary_size=1024,
        n_epochs=300,  # For model training
        sae_epochs=300,  # For SAE training
        batch_size=64,
        learning_rate=0.001,
        device=None,
        results_dir="../../results/dictionary_scaling",
        load_from_dir=None
    ):
        """Initialize the dictionary scaling experiment.
        
        Args:
            num_tasks: Number of distinct parity tasks
            bits_per_task: Number of relevant bits per task
            hidden_dims: List of hidden dimensions to test
            expansion_factors: Dictionary expansion factors relative to hidden dim
            l1_coefficients: L1 regularization strengths to test
            seeds: Number of random seeds for each configuration
            max_dictionary_size: Cap on dictionary size
            n_epochs: Training epochs for the model
            sae_epochs: Training epochs for the SAE
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            device: Device to use (defaults to cuda if available)
            results_dir: Directory to save results
            load_from_dir: Directory to load existing results and configuration from
        """
        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Check if loading from existing directory
        if load_from_dir is not None:
            self.experiment_dir = Path(load_from_dir)
            self._load_config()
            self.results = pd.read_csv(self.experiment_dir / "results.csv").to_dict('records')
        else:
            self.num_tasks = num_tasks
            self.bits_per_task = bits_per_task
            self.hidden_dims = hidden_dims
            self.expansion_factors = expansion_factors
            self.l1_coefficients = l1_coefficients
            self.seeds = seeds
            self.max_dictionary_size = max_dictionary_size
            self.n_epochs = n_epochs
            self.sae_epochs = sae_epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            
            # Set up results directory
            self.results_dir = Path(results_dir)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.experiment_dir = self.results_dir / f"dict_scaling_{timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize results storage
            self.results = []
            
            # Save configuration
            self._save_config()
    
    def _save_config(self):
        """Save experiment configuration to a JSON file."""
        import json
        
        config = {
            "num_tasks": self.num_tasks,
            "bits_per_task": self.bits_per_task,
            "hidden_dims": self.hidden_dims,
            "expansion_factors": self.expansion_factors,
            "l1_coefficients": self.l1_coefficients,
            "seeds": self.seeds,
            "max_dictionary_size": self.max_dictionary_size,
            "n_epochs": self.n_epochs,
            "sae_epochs": self.sae_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": str(self.device)
        }
        
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
    
    def _load_config(self):
        """Load experiment configuration from a JSON file."""
        import json
        
        with open(self.experiment_dir / "config.json", "r") as f:
            config = json.load(f)
        
        self.num_tasks = config["num_tasks"]
        self.bits_per_task = config["bits_per_task"]
        self.hidden_dims = config["hidden_dims"]
        self.expansion_factors = config["expansion_factors"]
        self.l1_coefficients = config["l1_coefficients"]
        self.seeds = config["seeds"]
        self.max_dictionary_size = config["max_dictionary_size"]
        self.n_epochs = config["n_epochs"]
        self.sae_epochs = config["sae_epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        # Device is already set in __init__
    
    def train_model(self, hidden_dim, seed):
        """Train a model with specified hidden dimension and seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create experiment instance
        experiment = InterventionExperiment(
            num_tasks=self.num_tasks,
            bits_per_task=self.bits_per_task,
            hidden_dim=hidden_dim,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            seed=seed
        )
        
        # Create and train model
        model = experiment.create_model()
        history, model = experiment.train_model(model)
        
        return model, experiment
    
    def measure_feature_scaling(self, model, experiment, hidden_dim, seed):
        """Measure how feature count scales with dictionary size for a trained model."""
        # Get activation samples
        train_loader = torch.utils.data.DataLoader(
            experiment.train_dataset, 
            batch_size=self.batch_size
        )
        
        activations = model.get_activation_samples(train_loader)
        
        # Compute feature count for different dictionary sizes and L1 coefficients
        for l1_coef in self.l1_coefficients:
            for expansion_factor in self.expansion_factors:
                # Calculate dictionary size
                sae_hidden_dim = int(min(self.max_dictionary_size, 
                                        hidden_dim * expansion_factor))
                
                # Train multiple SAEs with different seeds for stability measurement
                feature_counts = []
                reconstruction_errors = []
                
                for sae_seed in range(3):  # Use 3 SAE seeds for each configuration
                    # Set seed
                    np.random.seed(sae_seed)
                    torch.manual_seed(sae_seed)
                    
                    # Train SAE
                    sae = train_sae(
                        activations, 
                        sae_hidden_dim,
                        l1_coef=l1_coef, 
                        epochs=self.sae_epochs
                    )
                    
                    # Compute feature activations and reconstruction error
                    with torch.no_grad():
                        h = sae[0](activations)
                        h_relu = sae[1](h)
                        x_recon = sae[2](h_relu)
                        
                        # Compute feature norms and feature count
                        feature_norms = h_relu.sum(dim=0)
                        feature_count = calculate_feature_count(feature_norms)
                        feature_counts.append(feature_count)
                        
                        # Compute reconstruction error
                        recon_error = torch.nn.functional.mse_loss(x_recon, activations).item()
                        reconstruction_errors.append(recon_error)
                
                # Store results
                self.results.append({
                    'hidden_dim': hidden_dim,
                    'dictionary_size': sae_hidden_dim,
                    'expansion_factor': expansion_factor,
                    'l1_coefficient': l1_coef,
                    'model_seed': seed,
                    'feature_count': np.mean(feature_counts),
                    'feature_count_std': np.std(feature_counts),
                    'reconstruction_error': np.mean(reconstruction_errors),
                    'reconstruction_error_std': np.std(reconstruction_errors),
                    'normalized_feature_count': np.mean(feature_counts) / hidden_dim,
                    'normalized_dictionary_size': sae_hidden_dim / hidden_dim
                })
                
                print(f"Model: {hidden_dim}, Dict: {sae_hidden_dim}, L1: {l1_coef:.3f}, "
                      f"FC: {np.mean(feature_counts):.2f} ± {np.std(feature_counts):.2f}, "
                      f"MSE: {np.mean(reconstruction_errors):.4f}")
    
    def run_experiment(self):
        """Run the full experiment."""
        for hidden_dim in self.hidden_dims:
            for seed in range(self.seeds):
                print(f"\n=== Training model with hidden_dim={hidden_dim}, seed={seed} ===")
                
                # Train model
                model, experiment = self.train_model(hidden_dim, seed)
                
                # Measure feature scaling
                print(f"\n=== Measuring feature scaling for hidden_dim={hidden_dim}, seed={seed} ===")
                self.measure_feature_scaling(model, experiment, hidden_dim, seed)
                
                # Save intermediate results
                self.save_results()
                
        # Generate plots
        self.generate_plots()
        
        return self.results
    
    def save_results(self):
        """Save results to CSV file."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.experiment_dir / "results.csv", index=False)
    
    def generate_plots(self):
        """Generate plots from results."""
        results_df = pd.DataFrame(self.results)
        
        # Create figure directory
        fig_dir = self.experiment_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # Define scientific plot style
        class ScientificPlotStyle:
            """Standard style settings for scientific visualizations in our papers."""
            
            # Color palette - soft muted colors for data series
            COLORS = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']  # yellow, green, blue, purple
            ERROR_COLOR = '#BA898A'  # soft red for error indicators
            REFERENCE_LINE_COLOR = '#8B5C5D'  # dark red for reference lines
            
            # Typography - large sizes for readability
            FONT_SIZE_TITLE = 48     # plot titles
            FONT_SIZE_LABELS = 36    # axis labels
            FONT_SIZE_TICKS = 36     # tick labels
            FONT_SIZE_LEGEND = 28    # legend text
            
            # Plot elements
            MARKER_SIZE = 15         # data point size
            LINE_WIDTH = 5.0         # line thickness
            CAPSIZE = 12             # error bar cap size
            CAPTHICK = 5.0           # error bar cap thickness
            GRID_ALPHA = 0.3         # grid transparency
            
            # Figure dimensions
            FIGURE_SIZE = (12, 10)   # standard figure size
            COMBINED_FIG_SIZE = (20, 10)  # two-panel figure size
            
            @staticmethod
            def apply_axis_style(ax, title, xlabel, ylabel, legend=True):
                """Apply consistent styling to a matplotlib axis."""
                ax.set_title(title, fontsize=ScientificPlotStyle.FONT_SIZE_TITLE, fontweight='bold')
                ax.set_xlabel(xlabel, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
                ax.set_ylabel(ylabel, fontsize=ScientificPlotStyle.FONT_SIZE_LABELS)
                ax.tick_params(labelsize=ScientificPlotStyle.FONT_SIZE_TICKS)
                ax.grid(True, alpha=ScientificPlotStyle.GRID_ALPHA)
                if legend:
                    ax.legend(fontsize=ScientificPlotStyle.FONT_SIZE_LEGEND, loc='best')
                return ax
            
            @staticmethod
            def errorbar_kwargs(color_idx=0):
                """Return standard error bar parameters."""
                return {
                    'marker': 'o',
                    'color': ScientificPlotStyle.COLORS[color_idx % len(ScientificPlotStyle.COLORS)],
                    'markersize': ScientificPlotStyle.MARKER_SIZE,
                    'linewidth': ScientificPlotStyle.LINE_WIDTH,
                    'capsize': ScientificPlotStyle.CAPSIZE,
                    'capthick': ScientificPlotStyle.CAPTHICK,
                    'elinewidth': ScientificPlotStyle.LINE_WIDTH
                }
        
        # 1. Feature count vs dictionary size for all hidden dims and L1 coefficients
        for hidden_dim in self.hidden_dims:
            plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
            # Use log scales for better visualization of scaling relationships
            plt.yscale('log', base=2)
            plt.xscale('log', base=2)

            # Calculate limits based on the data that will be plotted
            x_min = float('inf')
            x_max = float('-inf')
            y_min = float('inf')
            y_max = float('-inf')

            for i, l1_coef in enumerate(self.l1_coefficients):
                df_subset = results_df[(results_df['hidden_dim'] == hidden_dim) & 
                                      (results_df['l1_coefficient'] == l1_coef)]
                
                if not df_subset.empty:
                    # Update limits based on this subset's data
                    x_min = min(x_min, df_subset['dictionary_size'].min())
                    x_max = max(x_max, df_subset['dictionary_size'].max())
                    y_min = min(y_min, df_subset['feature_count'].min())
                    y_max = max(y_max, df_subset['feature_count'].max())

            # Add some padding to the limits
            x_min *= 0.8
            x_max *= 1.2
            y_min *= 0.8
            y_max *= 1.2
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            for i, l1_coef in enumerate(self.l1_coefficients):
                df_subset = results_df[(results_df['hidden_dim'] == hidden_dim) & 
                                      (results_df['l1_coefficient'] == l1_coef)]
                
                if not df_subset.empty:
                    # Group by dictionary size and aggregate
                    grouped = df_subset.groupby('dictionary_size').agg({
                        'feature_count': ['mean', 'std']
                    }).reset_index()
                    
                    plt.errorbar(
                        grouped['dictionary_size'], 
                        grouped['feature_count']['mean'],
                        yerr=grouped['feature_count']['std'],
                        label=f'L1={l1_coef}',
                        **ScientificPlotStyle.errorbar_kwargs(i)
                    )
            
            ax = plt.gca()
            ScientificPlotStyle.apply_axis_style(
                ax,
                f'Feature Count vs. Dictionary Size (h={hidden_dim})',
                'Dictionary Size',
                'Feature Count'
            )
            plt.tight_layout()
            plt.savefig(fig_dir / f"feature_count_vs_dict_size_h_{hidden_dim}.pdf")
            plt.close()
        # 2. Normalized feature count vs normalized dictionary size
        # Separate plots for each hidden dimension
        for hidden_dim in self.hidden_dims:
            plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
            
            plt.yscale('log', base=2)
            plt.xscale('log', base=2)

            for i, l1_coef in enumerate(self.l1_coefficients):
                df_subset = results_df[(results_df['hidden_dim'] == hidden_dim) & 
                                       (results_df['l1_coefficient'] == l1_coef)]
                
                if not df_subset.empty:
                    # Group by normalized dictionary size and aggregate
                    grouped = df_subset.groupby('normalized_dictionary_size').agg({
                        'normalized_feature_count': ['mean', 'std']
                    }).reset_index()
                    
                    plt.errorbar(
                        grouped['normalized_dictionary_size'], 
                        grouped['normalized_feature_count']['mean'],
                        yerr=grouped['normalized_feature_count']['std'],
                        label=f'L1={l1_coef}',
                        **ScientificPlotStyle.errorbar_kwargs(i)
                    )
            
            ax = plt.gca()
            ScientificPlotStyle.apply_axis_style(
                ax,
                f'Normalized Feature Count vs. Normalized Dictionary Size (h={hidden_dim})',
                'Dictionary Size / Hidden Dimension',
                'Superposition'
            )
            plt.tight_layout()
            plt.savefig(fig_dir / f"normalized_feature_count_vs_dict_size_h_{hidden_dim}.pdf")
            plt.close()
        
        # One plot with all L1 coefficients
        # for i, l1_coef in enumerate(self.l1_coefficients):
        #     plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
        #     # plt.yscale('log', base=10)
        #     # plt.xscale('log', base=2)
            
        #     for j, hidden_dim in enumerate(self.hidden_dims):
        #         df_subset = results_df[(results_df['hidden_dim'] == hidden_dim) & 
        #                                (results_df['l1_coefficient'] == l1_coef)]
                
        #         if not df_subset.empty:
        #             # Group by normalized dictionary size and aggregate
        #             grouped = df_subset.groupby('normalized_dictionary_size').agg({
        #                 'normalized_feature_count': ['mean', 'std']
        #             }).reset_index()
                    
        #             plt.errorbar(
        #                 grouped['normalized_dictionary_size'], 
        #                 grouped['normalized_feature_count']['mean'],
        #                 yerr=grouped['normalized_feature_count']['std'],
        #                 label=f'h={hidden_dim}',
        #                 **ScientificPlotStyle.errorbar_kwargs(j)
        #             )
            
        #     ax = plt.gca()
        #     ScientificPlotStyle.apply_axis_style(
        #         ax,
        #         f'Normalized Feature Count vs. Normalized Dictionary Size (L1={l1_coef})',
        #         'Dictionary Size / Hidden Dimension',
        #         'Feature Count / Hidden Dimension'
        #     )
        #     plt.tight_layout()
        #     plt.savefig(fig_dir / f"normalized_feature_count_vs_dict_size_l1_{l1_coef}.pdf")
        #     plt.close()
        
        # 3. Feature count vs reconstruction error
        for hidden_dim in self.hidden_dims:
            plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
            plt.xscale('log')
            plt.yscale('log', base=2)
            
            for i, l1_coef in enumerate(self.l1_coefficients):
                df_subset = results_df[(results_df['hidden_dim'] == hidden_dim) & 
                                       (results_df['l1_coefficient'] == l1_coef)]
                
                if not df_subset.empty:
                    plt.scatter(
                        df_subset['reconstruction_error'],
                        df_subset['feature_count'],
                        s=ScientificPlotStyle.MARKER_SIZE * 10,
                        alpha=0.7,
                        color=ScientificPlotStyle.COLORS[i % len(ScientificPlotStyle.COLORS)],
                        label=f'L1={l1_coef}'
                    )
            
            ax = plt.gca()
            ScientificPlotStyle.apply_axis_style(
                ax,
                f'Feature Count vs. Reconstruction Error (h={hidden_dim})',
                'Reconstruction Error (MSE)',
                'Feature Count'
            )
            plt.tight_layout()
            plt.savefig(fig_dir / f"feature_count_vs_recon_error_h_{hidden_dim}.pdf")
            plt.close()

# %%
if __name__ == "__main__":
    # Run the experiment
    experiment = DictionarySizeExperiment(
        num_tasks=3,
        bits_per_task=4,
        # hidden_dims=[32, 64, 128],
        hidden_dims=[64],
        expansion_factors=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        # expansion_factors=[1.0],
        l1_coefficients=[0.01, 0.1, 1.0, 10.0],
        # l1_coefficients=[10.0],
        seeds=5,  # Use 3 seeds for each model configuration for quicker results
        max_dictionary_size=1024,  # Cap dictionary size
        n_epochs=100,  # Reduced from 300 for faster experimentation
        sae_epochs=300,  # Reduced from 300 for faster experimentation
    )
    
    results = experiment.run_experiment()
# %%
    # dict_scaling_2025-05-09_10-33-01
    # results_dir = list(Path("../../results/dictionary_scaling").glob("dict_scaling_*"))[-1]
    results_dir = Path("../../results/dictionary_scaling/dict_scaling_2025-05-09_10-33-01")
    experiment = DictionarySizeExperiment(
        load_from_dir=results_dir
    )

    # enter the directory (stored in dict_scaling_<timestamp>)
    results = pd.read_csv(results_dir / "results.csv")

    # plot the results
    experiment.generate_plots()

