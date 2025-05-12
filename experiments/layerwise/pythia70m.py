# %%
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Using device: {device}")
# %%
# Load model and SAE
model_name = "EleutherAI/pythia-70m-deduped"
path_to_download = f"../../data/dictionaries/{model_name}"

if not os.path.exists(path_to_download):
    raise FileNotFoundError(f"Dictionary path {path_to_download} does not exist, download dictionaries first from https://github.com/saprmarks/dictionary_learning")

model = LanguageModel(model_name, device_map="auto")

# Load SAEs for different layers
layer_specs = [
    ("embed", None),
    ("attn_out", 0),
    ("mlp_out", 0),
    ("resid_out", 0),
    ("attn_out", 1), 
    ("mlp_out", 1),
    ("resid_out", 1),
    ("attn_out", 2),
    ("mlp_out", 2), 
    ("resid_out", 2),
    ("attn_out", 3),
    ("mlp_out", 3),
    ("resid_out", 3),
    ("attn_out", 4),
    ("mlp_out", 4),
    ("resid_out", 4),
    ("attn_out", 5),
    ("mlp_out", 5),
    ("resid_out", 5),
]

saes = {}
for layer_type, layer_idx in layer_specs:
    print(f"Loading {layer_type} layer {layer_idx}")
    if layer_type == "embed":
        layer_name = layer_type
        weights_path = f"{path_to_download}/{layer_type}/10_32768/ae.pt"
    else:
        layer_name = f"{layer_type}_layer{layer_idx}"
        weights_path = f"{path_to_download}/{layer_type}_layer{layer_idx}/10_32768/ae.pt"
    
    activation_dim = 512
    dictionary_size = 64 * activation_dim
    sae = AutoEncoder(activation_dim, dictionary_size)
    sae.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    sae = sae.to(device)
    saes[layer_name] = sae
layers = list(saes.keys())
# %%
# Dataset params
dataset_name = "monology/pile-uncopyrighted"
subset_name = "default"
max_length = 512
n_samples = 2000

# Load and process dataset for multiple seeds
n_seeds = 5
seeds = list(range(n_seeds))
all_feature_counts = {}  # Store results for each seed

for i_seed in range(n_seeds):
    seed = seeds[i_seed]
    print(f"\nProcessing seed {i_seed+1}/{n_seeds}")
    print(f"n_samples: {n_samples}")
    # Load dataset with current seed
    dataset = load_dataset(dataset_name, subset_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed).take(n_samples)

    # Initialize feature sums for each layer
    feature_sums = {layer: torch.zeros(64*512, dtype=torch.float32, device=device) 
                    for layer in saes.keys()}

    def get_model_output(layer):
        if layer == "embed":
            return model.gpt_neox.embed_in.output.save()
        else:
            layer_idx = int(layer.split("layer")[1])
            if "mlp_out" in layer:
                return model.gpt_neox.layers[layer_idx].mlp.output.save()
            elif "attn_out" in layer:
                return model.gpt_neox.layers[layer_idx].attention.output[0].save()
            elif "resid_out" in layer:
                return model.gpt_neox.layers[layer_idx].output[0].save()

    # Process samples
    for i, sample in tqdm(enumerate(dataset), total=n_samples, desc=f"Seed {seed}"):
        text = sample['text'][:max_length]
        with torch.no_grad():
            activations = {}
            with model.trace(text) as tracer:
                for layer in layers:
                    activations[layer] = get_model_output(layer)
            features = {}
            for layer in layers:
                features[layer] = saes[layer].encode(activations[layer].to(device))
                feature_sums[layer] += features[layer].abs().sum(dim=(0,1)).to(device)
            del activations, features

    def calculate_functional_feature_count(feature_activations):
        """Calculate feature count using entropy"""
        p = feature_activations / feature_activations.sum()
        eps = 1e-12
        entropy = -torch.sum((p+eps) * torch.log(p+eps))
        return entropy.exp().item()

    # Calculate feature counts for current seed
    feature_counts = {}
    for layer_name in saes:
        count = calculate_functional_feature_count(feature_sums[layer_name])
        feature_counts[layer_name] = count
        print(f"{layer_name} functional feature count: {count:.2f}")

    all_feature_counts[seed] = feature_counts
# %%
# Save results
import json
import os

results_dir = "../../results/layerwise"
os.makedirs(results_dir, exist_ok=True)
# %%
results_path = os.path.join(results_dir, f"feature_counts_n_samples_{n_samples}_n_seeds_{n_seeds}.json")

# Convert all values to regular Python types for JSON serialization
serializable_results = {
    str(seed): {layer: float(count) for layer, count in counts.items()}
    for seed, counts in all_feature_counts.items()
}

with open(results_path, 'w') as f:
    json.dump(serializable_results, f, indent=2)

print(f"\nResults saved to {results_path}")

# %%
import numpy as np
import matplotlib.pyplot as plt

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


# Load results from JSON
with open(results_path, 'r') as f:
    all_feature_counts = json.load(f)

# Organize data by layer type
layer_types = {
    "mlp_out": [], 
    "resid_out": [],
    "attn_out": [],
    "embed": []
}

# Collect all values for each layer type and position
for seed in all_feature_counts:
    for layer_name, count in all_feature_counts[seed].items():
        for layer_type in layer_types:
            if layer_type in layer_name:
                if layer_type != "embed":
                    layer_num = int(layer_name.split("layer")[1])
                elif layer_type == "embed":
                    layer_num = 0
                layer_types[layer_type].append((layer_num, count))

# Sort layers by layer number
for layer_type in layer_types:
    layer_types[layer_type].sort(key=lambda x: x[0])

# Create plot with scientific style
plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)

# Prepare data for mean and std error bars
for i, layer_type in enumerate(layer_types):
    # Group points by position
    positions_dict = {}
    for pos, count in layer_types[layer_type]:
        if pos not in positions_dict:
            positions_dict[pos] = []
        positions_dict[pos].append(count)
    
    # Calculate means and std for each position
    positions = sorted(positions_dict.keys())
    means = [np.mean(positions_dict[pos]) for pos in positions]
    stds = [np.std(positions_dict[pos]) for pos in positions]
    
    # Plot mean with error bars using scientific style
    kwargs = ScientificPlotStyle.errorbar_kwargs(i)
    plt.errorbar(positions, means, yerr=stds, fmt='', label=layer_type, **kwargs)

# Apply scientific style to the plot
ax = plt.gca()
title = f'Feature Counts by Layer and Type\n(n_samples={n_samples}, n_seeds={n_seeds})'
ScientificPlotStyle.apply_axis_style(ax, title, 'Layer Position', 'Feature Count')

plt.tight_layout()

# Save figure
plt.savefig(f"{results_dir}/feature_counts_n_samples_{n_samples}_n_seeds_{n_seeds}.pdf", dpi=300)
# # %%
# %%

samples = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
feature_counts = [191, 225, 496, 831, 1421, 2413, 3253, 4193, 5024, 5598, 5914, 6149, 6268, 6320, 6328]

# plot the feature counts
# Create plot with scientific style
plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)

# Plot with scientific styling
plt.plot(samples, feature_counts, 
         color=ScientificPlotStyle.COLORS[0],
         linewidth=ScientificPlotStyle.LINE_WIDTH,
         marker='o',
         markersize=ScientificPlotStyle.MARKER_SIZE)

plt.xscale("log")

# Apply scientific style to the plot
ax = plt.gca()
title = 'Feature Count vs. Number of Samples'
ScientificPlotStyle.apply_axis_style(ax, title, 'Number of Samples', 'Feature Count', legend=False)

plt.tight_layout()
plt.savefig(f"{results_dir}/feature_count_scaling.pdf", dpi=300)
plt.show()

# %%

path = "../../data/dictionaries/pythia70m_mlp_out_layer0_features.csv"
import os
if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist")
# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns

# 1. Load the CSV file into a tensor
# csv_path = Path("all_features.csv")
csv_path = Path(path)
features = pd.read_csv(csv_path, header=None).values.flatten()
features_tensor = torch.tensor(features)

# 2. Print the percentage of nonzero entries
nonzero_count = (features_tensor > 0).sum().item()
total_count = features_tensor.numel()
nonzero_percentage = (nonzero_count / total_count) * 100

print(f"Total features: {total_count}")
print(f"Nonzero features: {nonzero_count}")
print(f"Percentage of nonzero entries: {nonzero_percentage:.2f}%")

# 3. Analyze the distribution of entries
print("\nDistribution statistics:")
print(f"Min value: {features_tensor.min().item():.4f}")
print(f"Max value: {features_tensor.max().item():.4f}")
print(f"Mean value: {features_tensor.mean().item():.4f}")
print(f"Median value: {torch.median(features_tensor).item():.4f}")
print(f"Standard deviation: {features_tensor.std().item():.4f}")

# Filter out zeros for log-scale analysis
nonzero_features = features_tensor[features_tensor > 0]
print(f"\nNonzero distribution statistics:")
print(f"Min nonzero value: {nonzero_features.min().item():.4f}")
print(f"Max nonzero value: {nonzero_features.max().item():.4f}")
print(f"Mean of nonzero values: {nonzero_features.mean().item():.4f}")
print(f"Median of nonzero values: {torch.median(nonzero_features).item():.4f}")

# 4. Visualize the distribution

# Plot 1: Histogram of all values
# plt.subplot(2, 2, 1)
# plt.hist(features, bins=50)
# plt.title('Histogram of All Feature Values')
# plt.xlabel('Feature Value')
# plt.ylabel('Count')
# plt.grid(alpha=0.3)

# # Plot 2: Histogram of nonzero values
# plt.subplot(2, 2, 2)
# plt.hist(nonzero_features, bins=50)
# plt.title('Histogram of Nonzero Feature Values')
# plt.xlabel('Feature Value')
# plt.ylabel('Count')
# plt.grid(alpha=0.3)

# Plot 1: Log-scale histogram of nonzero values
# Plot 1: Log-scale histogram of nonzero features
plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
ax1 = plt.gca()
ax1.hist(nonzero_features.numpy(), bins=50, log=True, color=ScientificPlotStyle.COLORS[0])
ScientificPlotStyle.apply_axis_style(
    ax1, 
    'Nonzero Feature Values', 
    'Feature Value',
    'Log Count',
    legend=False
)
plt.tight_layout()
plt.savefig(f"{results_dir}/nonzero_feature_values.pdf", dpi=300)
plt.show()

# Plot 2: Sorted feature values (to see distribution shape)
plt.figure(figsize=ScientificPlotStyle.FIGURE_SIZE)
ax2 = plt.gca()
sorted_features, _ = torch.sort(features_tensor, descending=True)
ax2.loglog(
    range(1, len(sorted_features) + 1), 
    sorted_features.numpy(),
    color=ScientificPlotStyle.COLORS[0],
    linewidth=ScientificPlotStyle.LINE_WIDTH
)
ScientificPlotStyle.apply_axis_style(
    ax2, 
    'Feature Values', 
    'Rank', 
    'Feature Value',
    legend=False
)
plt.tight_layout()
plt.savefig(f"{results_dir}/feature_values.pdf", dpi=300)
plt.show()

# 5. Calculate the entropy measure for the feature count
# Normalize the features to get a probability distribution
if features_tensor.sum() > 0:  # Ensure we don't divide by zero
    p = features_tensor / features_tensor.sum()
    # Calculate entropy only for nonzero probabilities
    nonzero_p = p[p > 0]
    entropy = -torch.sum(nonzero_p * torch.log(nonzero_p))
    feature_count = torch.exp(entropy).item()
    
    print(f"\nEntropy: {entropy.item():.4f}")
    print(f"Effective feature count: {feature_count:.4f}")
else:
    print("\nAll features are zero, cannot calculate entropy.")

# %%