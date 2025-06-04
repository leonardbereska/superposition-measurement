# %%
#!/usr/bin/env python3
"""
Robust script for analyzing layerwise feature counts in Pythia-70m using SAEs.
Fixes tokenization issues and improves error handling.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder


class LayerwiseAnalyzer:
    """Analyzer for computing feature counts across transformer layers."""
    
    def __init__(self, model_name: str = "EleutherAI/pythia-70m-deduped", 
                 dictionary_path: str = "../../data/dictionaries"):
        """Initialize the analyzer with model and SAE paths."""
        self.model_name = model_name
        self.dictionary_path = Path(dictionary_path) / model_name
        self.device = self._get_device()
        
        # Verify dictionary path exists
        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                f"Dictionary path {self.dictionary_path} does not exist. "
                f"Download dictionaries from https://github.com/saprmarks/dictionary_learning"
            )
        
        print(f"Using device: {self.device}")
        
        # Load model and SAEs
        self.model = self._load_model()
        self.saes, self.layer_names = self._load_saes()
        
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_model(self) -> LanguageModel:
        """Load the language model."""
        print(f"Loading model: {self.model_name}")
        return LanguageModel(self.model_name, device_map="auto")
    
    def _load_saes(self) -> Tuple[Dict[str, AutoEncoder], List[str]]:
        """Load all SAEs for different layers."""
        layer_specs = [
            ("embed", None),
            ("attn_out", 0), ("mlp_out", 0), ("resid_out", 0),
            ("attn_out", 1), ("mlp_out", 1), ("resid_out", 1),
            ("attn_out", 2), ("mlp_out", 2), ("resid_out", 2),
            ("attn_out", 3), ("mlp_out", 3), ("resid_out", 3),
            ("attn_out", 4), ("mlp_out", 4), ("resid_out", 4),
            ("attn_out", 5), ("mlp_out", 5), ("resid_out", 5),
        ]
        
        saes = {}
        activation_dim = 512
        dictionary_size = 64 * activation_dim
        
        for layer_type, layer_idx in layer_specs:
            layer_name = layer_type if layer_type == "embed" else f"{layer_type}_layer{layer_idx}"
            print(f"Loading SAE for {layer_name}")
            
            # Construct path to weights
            sae_subdir = layer_type if layer_type == "embed" else layer_name
            weights_path = self.dictionary_path / sae_subdir / "10_32768" / "ae.pt"
            
            if not weights_path.exists():
                print(f"Warning: SAE weights not found at {weights_path}, skipping...")
                continue
            
            # Load SAE
            sae = AutoEncoder(activation_dim, dictionary_size)
            sae.load_state_dict(torch.load(weights_path, weights_only=True, map_location=self.device))
            sae = sae.to(self.device)
            sae.eval()  # Set to evaluation mode
            saes[layer_name] = sae
        
        layer_names = list(saes.keys())
        print(f"Loaded {len(saes)} SAEs for layers: {layer_names}")
        return saes, layer_names
    
    def _get_model_output(self, layer_name: str):
        """Get the appropriate model output for a given layer."""
        if layer_name == "embed":
            return self.model.gpt_neox.embed_in.output.save()
        else:
            layer_idx = int(layer_name.split("layer")[1])
            if "mlp_out" in layer_name:
                return self.model.gpt_neox.layers[layer_idx].mlp.output.save()
            elif "attn_out" in layer_name:
                return self.model.gpt_neox.layers[layer_idx].attention.output[0].save()
            elif "resid_out" in layer_name:
                return self.model.gpt_neox.layers[layer_idx].output[0].save()
            else:
                raise ValueError(f"Unknown layer type in {layer_name}")
    
    def _process_text_safely(self, text: str, max_length: int = 512) -> Optional[torch.Tensor]:
        """Process text with robust tokenization and error handling."""
        try:
            # Truncate text to max_length characters (not tokens) to avoid extremely long sequences
            text = text[:max_length * 4]  # Conservative estimate: ~4 chars per token
            
            # Tokenize text - this returns a dict with input_ids as the key
            tokenized = self.model.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            
            # Extract input_ids and ensure they're integers
            input_ids = tokenized['input_ids']
            
            # Ensure input_ids are integers (Long tensors)
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            
            # Verify we have valid token IDs
            if input_ids.numel() == 0:
                return None
                
            return input_ids.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    
    def _calculate_feature_count(self, feature_activations: torch.Tensor) -> float:
        """Calculate feature count using entropy-based measure."""
        # Ensure we have positive activations
        if feature_activations.sum() == 0:
            return 0.0
        
        # Normalize to get probability distribution
        p = feature_activations / feature_activations.sum()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        p_safe = p + eps
        
        # Calculate entropy
        entropy = -torch.sum(p_safe * torch.log(p_safe))
        
        # Return exponential of entropy (effective feature count)
        return entropy.exp().item()
    
    def analyze_single_seed(self, seed: int, n_samples: int = 20000, 
                          dataset_name: str = "monology/pile-uncopyrighted",
                          max_length: int = 512) -> Dict[str, float]:
        """Analyze feature counts for a single random seed."""
        print(f"\nProcessing seed {seed} with {n_samples} samples")
        
        # Load dataset with specified seed
        dataset = load_dataset(dataset_name, "default", split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed).take(n_samples)
        
        # Initialize feature accumulation tensors
        feature_sums = {
            layer: torch.zeros(64 * 512, dtype=torch.float32, device=self.device)
            for layer in self.layer_names
        }
        
        # Process samples
        processed_samples = 0
        pbar = tqdm(enumerate(dataset), total=n_samples, desc=f"Seed {seed}")
        
        for i, sample in pbar:
            try:
                # Get text and process it safely
                text = sample.get('text', '')
                if not text:
                    continue
                
                # Tokenize text with error handling
                input_ids = self._process_text_safely(text, max_length)
                if input_ids is None:
                    continue
                
                # Process through model with error handling
                with torch.no_grad():
                    activations = {}
                    try:
                        with self.model.trace(input_ids) as tracer:
                            for layer_name in self.layer_names:
                                activations[layer_name] = self._get_model_output(layer_name)
                    except Exception as e:
                        print(f"Error during model trace: {e}")
                        continue
                    
                    # Process through SAEs
                    for layer_name in self.layer_names:
                        try:
                            if layer_name not in activations:
                                continue
                                
                            layer_activations = activations[layer_name].to(self.device)
                            if layer_activations.numel() == 0:
                                continue
                            
                            # Get SAE features
                            features = self.saes[layer_name].encode(layer_activations)
                            
                            # Accumulate absolute feature activations
                            feature_sums[layer_name] += features.abs().sum(dim=(0, 1)).to(self.device)
                            
                        except Exception as e:
                            print(f"Error processing layer {layer_name}: {e}")
                            continue
                    
                    # Clean up activations to prevent memory accumulation
                    del activations
                
                processed_samples += 1
                
                # Update progress bar with current processed samples
                if i % 1000 == 0:
                    pbar.set_postfix({'processed': processed_samples})
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        pbar.close()
        print(f"Successfully processed {processed_samples}/{n_samples} samples")
        
        # Calculate feature counts
        feature_counts = {}
        for layer_name in self.layer_names:
            count = self._calculate_feature_count(feature_sums[layer_name])
            feature_counts[layer_name] = count
            print(f"{layer_name} functional feature count: {count:.2f}")
        
        return feature_counts
    
    def analyze_multiple_seeds(self, seeds: List[int], n_samples: int = 20000,
                             **kwargs) -> Dict[int, Dict[str, float]]:
        """Analyze feature counts across multiple random seeds."""
        all_results = {}
        
        for i, seed in enumerate(seeds):
            print(f"\n{'='*50}")
            print(f"Processing seed {i+1}/{len(seeds)}")
            print(f"{'='*50}")
            
            try:
                results = self.analyze_single_seed(seed, n_samples, **kwargs)
                all_results[seed] = results
                
                # Force garbage collection to prevent memory issues
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error processing seed {seed}: {e}")
                print("Continuing with next seed...")
                continue
        
        return all_results
    
    def save_results(self, results: Dict[int, Dict[str, float]], 
                   output_dir: str = "../../results/layerwise",
                   n_samples: int = 20000) -> Path:
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        n_seeds = len(results)
        filename = f"feature_counts_n_samples_{n_samples}_n_seeds_{n_seeds}.json"
        results_path = output_path / filename
        
        # Convert to JSON-serializable format
        serializable_results = {
            str(seed): {layer: float(count) for layer, count in counts.items()}
            for seed, counts in results.items()
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
        return results_path


class ScientificPlotStyle:
    """Standard style settings for scientific visualizations."""
    
    COLORS = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    FONT_SIZE_TITLE = 24
    FONT_SIZE_LABELS = 18
    FONT_SIZE_TICKS = 16
    FONT_SIZE_LEGEND = 14
    MARKER_SIZE = 8
    LINE_WIDTH = 2.5
    CAPSIZE = 6
    CAPTHICK = 2.5
    GRID_ALPHA = 0.3
    FIGURE_SIZE = (12, 10)
    
    @staticmethod
    def apply_axis_style(ax, title, xlabel, ylabel, legend=True):
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
        return {
            'marker': 'o',
            'color': ScientificPlotStyle.COLORS[color_idx % len(ScientificPlotStyle.COLORS)],
            'markersize': ScientificPlotStyle.MARKER_SIZE,
            'linewidth': ScientificPlotStyle.LINE_WIDTH,
            'capsize': ScientificPlotStyle.CAPSIZE,
            'capthick': ScientificPlotStyle.CAPTHICK,
            'elinewidth': ScientificPlotStyle.LINE_WIDTH
        }


def plot_results(results_path: Path, output_dir: Optional[Path] = None, n_samples: int = 20000, n_seeds: int = 3):
    """Create publication-quality plots from results."""
    # Load results
    with open(results_path, 'r') as f:
        all_feature_counts = json.load(f)
    
    # Extract metadata from filename
    filename = results_path.stem
    # n_samples = int(filename.split('_')[3])
    # n_seeds = int(filename.split('_')[5])
    
    # Organize data by layer type
    layer_types = {"mlp_out": [], "resid_out": [], "attn_out": [], "embed": []}
    
    for seed in all_feature_counts:
        for layer_name, count in all_feature_counts[seed].items():
            for layer_type in layer_types:
                if layer_type in layer_name:
                    if layer_type == "embed":
                        layer_num = 0
                    else:
                        layer_num = int(layer_name.split("layer")[1])
                    layer_types[layer_type].append((layer_num, count))
    
    # Sort by layer number
    for layer_type in layer_types:
        layer_types[layer_type].sort(key=lambda x: x[0])
    
    # Create plot
    plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
    
    for i, layer_type in enumerate(layer_types):
        if not layer_types[layer_type]:
            continue
            
        # Group by position
        positions_dict = {}
        for pos, count in layer_types[layer_type]:
            if pos not in positions_dict:
                positions_dict[pos] = []
            positions_dict[pos].append(count)
        
        # Calculate statistics
        positions = sorted(positions_dict.keys())
        means = [np.mean(positions_dict[pos]) for pos in positions]
        stds = [np.std(positions_dict[pos]) for pos in positions]
        
        # Plot
        kwargs = ScientificPlotStyle.errorbar_kwargs(i)
        plt.errorbar(positions, means, yerr=stds, fmt='', label=layer_type, **kwargs)
    
    # Style plot
    ax = plt.gca()
    title = f'Feature Counts by Layer and Type\n(n_samples={n_samples}, n_seeds={n_seeds})'
    ScientificPlotStyle.apply_axis_style(ax, title, 'Layer Position', 'Feature Count')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = results_path.parent
    
    plot_path = output_dir / f"feature_counts_n_samples_{n_samples}_n_seeds_{n_seeds}.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    return plot_path


def main():
    """Main execution function."""
    # Configuration
    n_samples = 20000
    seeds = [42, 123, 456]  # Use fixed seeds for reproducibility
    
    try:
        # Initialize analyzer
        analyzer = LayerwiseAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_multiple_seeds(seeds, n_samples)
        
        if not results:
            print("No results obtained. Check for errors above.")
            return
        
        # Save results
        results_path = analyzer.save_results(results, n_samples=n_samples)
        
        # Create plots
        plot_results(results_path)
        
        print(f"\nAnalysis complete! Results saved and plotted.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

# %%
if __name__ == "__main__":
    main()
# %%

# Example of loading and plotting from an existing results file
if __name__ == "__main__" and "get_ipython" in globals():
    # This block only runs in interactive mode (Jupyter/IPython)
    import pathlib
    
    # Use a relative path that's more portable across different environments
    results_dir = pathlib.Path("../../results/layerwise")
    existing_results_path = results_dir / "feature_counts_n_samples_20000_n_seeds_3.json"
    
    if existing_results_path.exists():
        print(f"Loading existing results from: {existing_results_path}")
        try:
            plot_path = plot_results(existing_results_path)
            print(f"Plot generated at: {plot_path}")
        except ValueError as e:
            print(f"Error parsing filename: {e}")
            # Extract metadata directly from the filename string rather than relying on splitting
            # Pass the parameters explicitly since filename parsing is failing
            plot_path = plot_results(existing_results_path, n_samples=20000, n_seeds=3)
            print(f"Plot generated with explicit parameters at: {plot_path}")
    else:
        print(f"Results file not found: {existing_results_path}")
        print("Running main analysis instead...")
        main()

