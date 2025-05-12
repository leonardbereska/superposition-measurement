# %%
import torch
import numpy as np
import torch.utils.data
import os
import matplotlib.pyplot as plt
from dataset_multitasksparseparity import MultiTaskSparseParityDataset
# change working directory to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

"""Configuration for sparse parity intervention experiments."""
TEST_MODE = True

# Model configuration
MODEL_CONFIG = {
    "hidden_dim": 64,
    "hidden_dims": [16, 32, 64, 128, 256],  # Multiple hidden dimensions for capacity sweep
}

# Dataset configuration
DATASET_CONFIG = {
    "num_tasks": 3,
    "bits_per_task": 4,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "n_epochs": 300 if not TEST_MODE else 10,
}

# Intervention configuration
INTERVENTION_CONFIG = {
    "interventions": [
        ('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) if not TEST_MODE else ('dropout', [0.0, 0.2, 0.5])
    ],
    "seeds": 5 if not TEST_MODE else 2,
}

# SAE configuration
SAE_CONFIG = {
    "l1_coef": 0.1,
    "epochs": 300 if not TEST_MODE else 10,
    "batch_size": 128,
    "learning_rate": 0.001,
}

# Results configuration
RESULTS_DIR = "../../results/interventions"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = "intervention_results.pt"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseParityModel(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=MODEL_CONFIG["hidden_dim"], 
                 intervention='none', 
                 intervention_strength=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intervention = intervention
        self.intervention_strength = intervention_strength
        
        # Layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # First layer
        h = self.fc1(x)
        
        # Apply intervention on activations
        if self.intervention == 'dropout':
            h = torch.nn.functional.dropout(
                h, 
                p=self.intervention_strength, 
                training=self.training
            )
                
        # Apply activation function
        h = torch.nn.functional.relu(h)
            
        # Output layer
        return self.fc2(h)
    
    def get_activation_samples(self, dataloader, max_samples=1000):
        """Get activation samples for feature analysis."""
        activations = []
        count = 0
        
        self.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                # Forward pass until activation
                h = self.fc1(inputs)
                
                # Store pre-activation values
                activations.append(h.cpu())
                
                count += inputs.shape[0]
                if count >= max_samples:
                    break
        
        return torch.cat(activations, dim=0)
    

class InterventionExperiment:
    def __init__(self, 
                 num_tasks=DATASET_CONFIG["num_tasks"], 
                 bits_per_task=DATASET_CONFIG["bits_per_task"],
                 hidden_dim=MODEL_CONFIG["hidden_dim"],
                 n_epochs=TRAINING_CONFIG["n_epochs"],
                 batch_size=TRAINING_CONFIG["batch_size"],
                 learning_rate=TRAINING_CONFIG["learning_rate"],
                 device=DEVICE,
                 seed=42):
        
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Create dataset
        self.dataset = MultiTaskSparseParityDataset(
            num_tasks=num_tasks,
            bits_per_task=bits_per_task,
            seed=seed
        )
        
        self.train_dataset, self.test_dataset = self.dataset.create_train_test_split()
        
        self.input_dim = num_tasks + (num_tasks * bits_per_task)
        
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Test set size: {len(self.test_dataset)}")
            
    def create_model(self, intervention='none', strength=0.0):
        """Create model with specified intervention."""
        model = SparseParityModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            intervention=intervention,
            intervention_strength=strength
        ).to(self.device)
        
        return model
    
    def train_model(self, model):
        """Train model for a fixed number of epochs."""
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Track metrics
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        for epoch in range(self.n_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Evaluation
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item() * inputs.size(0)
                    
                    # Calculate accuracy
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    test_correct += (predicted == targets).sum().item()
                    test_total += targets.size(0)
            
            test_loss /= test_total
            test_acc = test_correct / test_total
            
            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
                
        return history, model
    
    def measure_feature_count(self, model):
        """Measure feature count using entropy-based metric."""
        # Get activation samples
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size
        )
        
        activations = model.get_activation_samples(train_loader)
        
        # Train a small sparse autoencoder
        sae_hidden_dim = min(1024, 4 * self.hidden_dim)
        sae = train_sae(activations, sae_hidden_dim)
        
        # Get feature activations
        feature_norms = compute_feature_norms(sae, activations)
        
        # Calculate entropy-based feature count
        return calculate_feature_count(feature_norms)
        
    def run_intervention_sweep(self, intervention_type, strengths, seeds=INTERVENTION_CONFIG["seeds"]):
        """Run experiment across intervention strengths."""
        results = {strength: [] for strength in strengths}
        
        for strength in strengths:
            print(f"\n=== Testing {intervention_type} with strength {strength} ===")
            
            for seed in range(seeds):
                print(f"Running seed {seed}...")
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = self.create_model(intervention_type, strength)
                history, model = self.train_model(model)
                
                # Measure feature count after training
                feature_count = self.measure_feature_count(model)
                
                # Record final metrics
                final_metrics = {
                    'feature_count': feature_count,
                    'train_acc': history['train_acc'][-1],
                    'test_acc': history['test_acc'][-1],
                    'epochs': len(history['train_acc']),
                    'history': history
                }
                
                results[strength].append(final_metrics)
                
                print(f"Seed {seed} results: Feature Count={final_metrics['feature_count']:.2f}, "
                      f"Test Acc={final_metrics['test_acc']:.4f}, "
                      f"Epochs={final_metrics['epochs']}")
        
        return results
    
    def run_capacity_intervention_sweep(self, intervention_type, strengths, hidden_dims, seeds=INTERVENTION_CONFIG["seeds"]):
        """Run experiment across both intervention strengths and model capacities."""
        results = {hidden_dim: {strength: [] for strength in strengths} for hidden_dim in hidden_dims}
        
        for hidden_dim in hidden_dims:
            print(f"\n=== Testing with hidden dimension {hidden_dim} ===")
            
            # Create a new experiment with the specified hidden dimension
            # We keep the same dataset but change the model capacity
            self.hidden_dim = hidden_dim
            
            for strength in strengths:
                print(f"\n=== Testing {intervention_type} with strength {strength} ===")
                
                for seed in range(seeds):
                    print(f"Running seed {seed}...")
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    
                    model = self.create_model(intervention_type, strength)
                    history, model = self.train_model(model)
                    
                    # Measure feature count after training
                    feature_count = self.measure_feature_count(model)
                    
                    # Record final metrics
                    final_metrics = {
                        'feature_count': feature_count,
                        'train_acc': history['train_acc'][-1],
                        'test_acc': history['test_acc'][-1],
                        'epochs': len(history['train_acc']),
                        'history': history
                    }
                    
                    results[hidden_dim][strength].append(final_metrics)
                    
                    print(f"Hidden dim {hidden_dim}, Strength {strength}, Seed {seed} results: "
                          f"Feature Count={final_metrics['feature_count']:.2f}, "
                          f"Test Acc={final_metrics['test_acc']:.4f}")
        
        return results
    

def train_sae(activations, hidden_dim, l1_coef=SAE_CONFIG["l1_coef"], epochs=SAE_CONFIG["epochs"]):
    """Train a simple SAE for feature extraction."""
    input_dim = activations.shape[1]
    
    # Create simple autoencoder
    sae = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, input_dim)
    )
    
    # Train
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_CONFIG["learning_rate"])
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(activations.shape[0])
        act_shuffled = activations[perm]
        
        # Process in batches
        batch_size = SAE_CONFIG["batch_size"]
        for i in range(0, len(act_shuffled), batch_size):
            batch = act_shuffled[i:i+batch_size]
            
            # Forward pass
            h = sae[0](batch)
            h_relu = sae[1](h)
            x_recon = sae[2](h_relu)
            
            # Loss with L1 sparsity penalty
            recon_loss = torch.nn.functional.mse_loss(x_recon, batch)
            l1_loss = l1_coef * h_relu.abs().mean()
            loss = recon_loss + l1_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return sae

def compute_feature_norms(sae, activations):
    """Compute feature norms from SAE encoding."""
    # Encode all activations
    with torch.no_grad():
        h = sae[0](activations)
        h_relu = sae[1](h)
    
    # Extract feature norms (sum across samples)
    feature_norms = h_relu.sum(dim=0)
    return feature_norms

def calculate_feature_count(feature_norms):
    """Calculate entropy-based feature count."""
    # Normalize to get probability distribution
    # Handle the case where all feature norms are zero
    if feature_norms.sum() == 0:
        # If all features are inactive, return 0 feature count
        return 0
    
    p = feature_norms / feature_norms.sum()
    # Avoid log(0) issues
    eps = 1e-10
    entropy = -torch.sum(p * torch.log(p + eps))
    
    # Feature count is exp(entropy)
    return torch.exp(entropy).item()

def capacity_sweep():
    # Create experiment
    experiment = InterventionExperiment(
        num_tasks=DATASET_CONFIG["num_tasks"],
        bits_per_task=DATASET_CONFIG["bits_per_task"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        n_epochs=TRAINING_CONFIG["n_epochs"]
    )
    
    # Run experiments
    results = {}
    from datetime import datetime
    
    # Create timestamp for the full experimental suite
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create main experiment directory
    experiment_dir = os.path.join(RESULTS_DIR, f"interventions_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Dictionary to store all intervention directories
    intervention_dirs = {}
    
    for intervention, strengths in INTERVENTION_CONFIG["interventions"]:
        print(f"\n=== Running {intervention} experiments ===")
        
        # Special case for dropout: run capacity sweep
        if intervention == 'dropout':
            print(f"\n=== Running capacity sweep for {intervention} ===")
            
            # Run the capacity intervention sweep
            capacity_results = experiment.run_capacity_intervention_sweep(
                intervention, strengths, MODEL_CONFIG["hidden_dims"], seeds=INTERVENTION_CONFIG["seeds"]
            )
            
            # Store in overall results
            results[intervention] = {"capacity_sweep": capacity_results}
            
            # Create intervention-specific directory
            intervention_dir = os.path.join(experiment_dir, f"{intervention}_capacity_sweep")
            os.makedirs(intervention_dir, exist_ok=True)
            intervention_dirs[intervention] = intervention_dir
            
            # Save intervention-specific results
            results_path = os.path.join(intervention_dir, RESULTS_FILE)
            torch.save(capacity_results, results_path)
            
            # Visualize and save plots in the intervention-specific directory
            visualize_capacity_intervention_results(intervention, capacity_results, intervention_dir)
    
    # Save the complete results in the main experiment directory
    complete_results_path = os.path.join(experiment_dir, RESULTS_FILE)
    torch.save(results, complete_results_path)

def single_capacity_test(intervention='dropout', strengths=None, hidden_dim=64, seeds=3):
    """
    Run a single capacity test with one specific hidden dimension.
    
    Args:
        intervention (str): Type of intervention to apply (e.g., 'dropout')
        strengths (list): List of intervention strengths to test
        hidden_dim (int): Single hidden dimension to use
        seeds (int): Number of random seeds to run
    """
    # Use default strengths if none provided
    if strengths is None:
        strengths = [0.0, 0.2, 0.5]
    
    # Create experiment
    experiment = InterventionExperiment(
        num_tasks=DATASET_CONFIG["num_tasks"],
        bits_per_task=DATASET_CONFIG["bits_per_task"],
        hidden_dim=hidden_dim,
        n_epochs=TRAINING_CONFIG["n_epochs"]
    )
    
    # Create timestamp for this experiment
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create experiment directory
    experiment_dir = os.path.join(RESULTS_DIR, f"single_capacity_{intervention}_{hidden_dim}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n=== Running single capacity test for {intervention} with hidden_dim={hidden_dim} ===")
    
    # Run the capacity intervention test with a single hidden dimension
    capacity_results = experiment.run_capacity_intervention_sweep(
        intervention, strengths, [hidden_dim], seeds=seeds
    )
    
    # Save results
    results_path = os.path.join(experiment_dir, RESULTS_FILE)
    torch.save(capacity_results, results_path)
    
    # Visualize and save plots
    visualize_capacity_intervention_results(intervention, capacity_results, experiment_dir)
    
    return capacity_results

def visualize_intervention_results(intervention, intervention_results, save_dir):
    """Visualize results for a specific intervention and save to the specified directory."""
    # Get all strengths and handle None values specially
    all_strengths = list(intervention_results.keys())
    
    # For plotting, we need to convert None to a string for display
    # but keep numerical values for proper ordering
    plot_strengths = []
    strength_labels = []
    
    # Calculate means and std devs
    feature_counts = []
    feature_stds = []
    test_accs = []
    test_accs_stds = []
    
    # Sort strengths, handling None specially
    if None in all_strengths:
        # Process None first
        fc_values = [r['feature_count'] for r in intervention_results[None]]
        feature_counts.append(np.mean(fc_values))
        feature_stds.append(np.std(fc_values))
        
        acc_values = [r['test_acc'] for r in intervention_results[None]]
        test_accs.append(np.mean(acc_values))
        test_accs_stds.append(np.std(acc_values))
        
        plot_strengths.append(0)  # Use 0 as placeholder for None in plot
        strength_labels.append("None")
        
        # Then process numerical values
        num_strengths = sorted([s for s in all_strengths if s is not None])
        for strength in num_strengths:
            fc_values = [r['feature_count'] for r in intervention_results[strength]]
            feature_counts.append(np.mean(fc_values))
            feature_stds.append(np.std(fc_values))
            
            acc_values = [r['test_acc'] for r in intervention_results[strength]]
            test_accs.append(np.mean(acc_values))
            test_accs_stds.append(np.std(acc_values))
            
            plot_strengths.append(strength)
            strength_labels.append(str(strength))
    else:
        # No None values, just sort normally
        sorted_strengths = sorted(all_strengths)
        for strength in sorted_strengths:
            fc_values = [r['feature_count'] for r in intervention_results[strength]]
            feature_counts.append(np.mean(fc_values))
            feature_stds.append(np.std(fc_values))
            
            acc_values = [r['test_acc'] for r in intervention_results[strength]]
            test_accs.append(np.mean(acc_values))
            test_accs_stds.append(np.std(acc_values))
            
            plot_strengths.append(strength)
            strength_labels.append(str(strength))
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Feature count
    ax1.errorbar(plot_strengths, feature_counts, yerr=feature_stds, marker='o')
    ax1.set_title(f"{intervention.capitalize()} vs Feature Count")
    ax1.set_xlabel("{} Strength".format(intervention).capitalize())
    ax1.set_ylabel("Feature Count")
    ax1.set_xticks(plot_strengths)
    ax1.set_xticklabels(strength_labels)
    ax1.grid(True)
    
    # Test accuracy
    ax2.errorbar(plot_strengths, test_accs, yerr=test_accs_stds, marker='o')
    ax2.set_title(f"{intervention.capitalize()} vs Test Accuracy")
    ax2.set_xlabel("{} Strength".format(intervention).capitalize())
    ax2.set_ylabel("Test Accuracy")
    ax2.set_xticks(plot_strengths)
    ax2.set_xticklabels(strength_labels)
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure to the intervention-specific directory
    if save_dir is not None:
        plot_path = os.path.join(save_dir, f"{intervention}_results.pdf")
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()

def visualize_capacity_intervention_results(intervention, capacity_results, save_dir):
    """Visualize results for a capacity intervention sweep and save to the specified directory.
    
    Args:
        intervention: Name of the intervention type (e.g., "Dropout")
        capacity_results: Nested dictionary of results organized by hidden_dim and strength
        save_dir: Directory path to save visualization files
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ===== CONFIGURATION =====
    # Color palette - soft muted colors
    colors = ['#F0BE5E', '#94B9A3', '#88A7B2', '#DDC6E1']
    reference_line_color = '#8B5C5D'  # Dark red for reference/threshold lines
    
    # Typography settings - much larger fonts for better readability
    FONT_SIZE_LABELS = 36
    FONT_SIZE_TITLE = 48 # Increased from 18
    FONT_SIZE_TICKS = 36   # Increased from 14
    FONT_SIZE_LEGEND = 28 # Increased from 14
    
    # Plot settings
    MARKER_SIZE = 15       # Increased from 8
    LINE_WIDTH = 5.0       # Increased from 2.0
    CAPSIZE = 12            # Increased from 4
    FIGURE_SIZE = (12, 10) # Increased from (10, 8)
    COMBINED_FIG_SIZE = (20, 10) # Increased from (16, 8)
    
    # ===== DATA PREPARATION =====
    # Extract dimensions and strengths
    hidden_dims = sorted(capacity_results.keys())
    # filter out hidden_dim = 256
    hidden_dims = [dim for dim in hidden_dims if dim != 256]
    all_strengths = list(capacity_results[hidden_dims[0]].keys())
    
    # Sort strengths, handling None specially
    if None in all_strengths:
        strengths = [None] + sorted([s for s in all_strengths if s is not None])
    else:
        strengths = sorted(all_strengths)

    
    # Process data once for all plots
    plot_data = {}
    for hidden_dim in hidden_dims:
        plot_data[hidden_dim] = {
            'plot_strengths': [],
            'feature_counts': [],
            'feature_stds': [],
            'test_accs': [],
            'test_accs_stds': []
        }
        
        for strength in strengths:
            # Handle None values for plotting
            plot_strength = 0 if strength is None else strength
            plot_data[hidden_dim]['plot_strengths'].append(plot_strength)
            
            # Calculate statistics
            fc_values = [r['feature_count'] for r in capacity_results[hidden_dim][strength]]
            plot_data[hidden_dim]['feature_counts'].append(np.mean(fc_values))
            plot_data[hidden_dim]['feature_stds'].append(np.std(fc_values))
            
            acc_values = [r['test_acc'] for r in capacity_results[hidden_dim][strength]]
            plot_data[hidden_dim]['test_accs'].append(np.mean(acc_values))
            plot_data[hidden_dim]['test_accs_stds'].append(np.std(acc_values))
    
    # Create custom x-tick labels if needed
    if None in strengths:
        strength_labels = ["None" if s is None else str(s) for s in strengths]
    
    # ===== PLOTTING FUNCTIONS =====
    def setup_axis(ax, title, xlabel, ylabel):
        """Configure a matplotlib axis with consistent styling."""
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS)
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONT_SIZE_TICKS)
        
        # Set custom x-tick labels if needed
        if None in strengths:
            first_dim = list(plot_data.keys())[0]
            ax.set_xticks(plot_data[first_dim]['plot_strengths'])
            ax.set_xticklabels(strength_labels)
    
    def plot_metric(ax, metric_name, hidden_dims):
        """Plot a specific metric (feature count or test accuracy) to the given axis."""
        for i, hidden_dim in enumerate(hidden_dims):
            data = plot_data[hidden_dim]
            
            if metric_name == 'feature':
                y_values = data['feature_counts']
                y_errors = data['feature_stds']
            else:  # accuracy
                y_values = data['test_accs']
                y_errors = data['test_accs_stds']
             
            ax.errorbar(
                data['plot_strengths'], 
                y_values, 
                yerr=y_errors, 
                marker='o', 
                color=colors[i % len(colors)], 
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                capsize=CAPSIZE,
                capthick=LINE_WIDTH,
                elinewidth=LINE_WIDTH,
                label=f"h={hidden_dim}"
            )
    
    # ===== CREATE INDIVIDUAL PLOTS =====
    # Feature count plot
    fig1, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    plot_metric(ax1, 'feature', hidden_dims)
    setup_axis(ax1, f"{intervention.capitalize()} vs Feature Count", "{} Strength".format(intervention.capitalize()), "Feature Count")
    fig1.tight_layout(pad=2.0)
    
    # Test accuracy plot
    fig2, ax2 = plt.subplots(figsize=FIGURE_SIZE)
    plot_metric(ax2, 'accuracy', hidden_dims)
    setup_axis(ax2, f"{intervention.capitalize()} vs Test Accuracy", "{} Strength".format(intervention.capitalize()), "Test Accuracy")
    fig2.tight_layout(pad=2.0)
    
    # ===== SAVE PLOTS =====
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual plots
        feature_plot_path = os.path.join(save_dir, f"{intervention}_feature_count.pdf")
        accuracy_plot_path = os.path.join(save_dir, f"{intervention}_accuracy.pdf")
        
        fig1.savefig(feature_plot_path, dpi=300, bbox_inches='tight')
        fig2.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
        
        # Create and save combined plot
        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=COMBINED_FIG_SIZE)
        
        plot_metric(ax3, 'feature', hidden_dims)
        setup_axis(ax3, "Feature Count", intervention, "Feature Count")
        
        plot_metric(ax4, 'accuracy', hidden_dims)
        setup_axis(ax4, "Test Accuracy", intervention, "Test Accuracy")
        
        fig3.tight_layout(pad=2.5)
        combined_plot_path = os.path.join(save_dir, f"{intervention}_combined.pdf")
        fig3.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig3)
    
    # Return the figure handles for further manipulation if needed
    return fig1, fig2


# %%
if __name__ == "__main__":
    single_capacity_test()
    # capacity_sweep()


# %%
    intervention = 'dropout'
    time_stamp = '2025-03-30_12-30-19'

    # Load results
    results = torch.load("../../results/interventions/interventions_{}/{}_capacity_sweep/intervention_results.pt".format(time_stamp, intervention))

    # visualize results
    visualize_capacity_intervention_results(intervention, results, "../../results/interventions/interventions_{}/{}_capacity_sweep/".format(time_stamp, intervention))
# %%
