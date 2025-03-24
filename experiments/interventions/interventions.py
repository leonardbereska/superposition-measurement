# %%
import torch
import numpy as np
import torch.utils.data
import os
import matplotlib.pyplot as plt
from datasets import MultiTaskSparseParityDataset
# change working directory to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

"""Configuration for sparse parity intervention experiments."""

# Model configuration
MODEL_CONFIG = {
    "hidden_dim": 64,
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
    "n_epochs": 300,
}

# Intervention configuration
INTERVENTION_CONFIG = {
    "interventions": [
        ('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        ('l1_activation', [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
        ('solu', [None, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]),  # Temperature values
        ('l1_weight', [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0])
    ],
    "seeds": 5,
}

# SAE configuration
SAE_CONFIG = {
    "l1_coef": 0.1,
    "epochs": 300,
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
        
        # Initialize SoLU parameter if needed
        self.softmax_temp = None
        if intervention == 'solu' and intervention_strength is not None:
            self.softmax_temp = torch.nn.Parameter(
                torch.ones(1) * intervention_strength
            )
    
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
        if self.intervention == 'solu' and self.softmax_temp is not None:
            # SoLU activation (softmax linear unit)
            h_norm = torch.nn.functional.softmax(
                h * self.softmax_temp, dim=1
            ) * h.shape[1]
            h = h * h_norm
        else:
            # Default ReLU
            h = torch.nn.functional.relu(h)
        # Store activations for L1 regularization if needed
        self.last_activations = h if self.intervention == 'l1_activation' else None
            
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
    
    def train_model(self, model, add_l1_weight_reg=0.0):
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
                
                # Add L1 weight regularization if specified
                if add_l1_weight_reg > 0:
                    l1_loss = add_l1_weight_reg * sum(
                        p.abs().sum() for p in model.parameters()
                    )
                    loss += l1_loss
                
                # Add L1 activation regularization if specified
                if model.intervention == 'l1_activation' and model.last_activations is not None:
                    l1_act_loss = model.intervention_strength * model.last_activations.abs().mean()
                    loss += l1_act_loss
                
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
                
                # For L1 weight regularization, use a different approach
                if intervention_type == 'l1_weight':
                    history, model = self.train_model(
                        self.create_model('none', 0),
                        add_l1_weight_reg=strength
                    )
                else:
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
    p = feature_norms / feature_norms.sum()
    
    # Avoid log(0) issues
    eps = 1e-10
    entropy = -torch.sum(p * torch.log(p + eps))
    
    # Feature count is exp(entropy)
    return torch.exp(entropy).item()


def main():
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
    
    for intervention, strengths in INTERVENTION_CONFIG["interventions"]:
        print(f"\n=== Running {intervention} experiments ===")
        results[intervention] = experiment.run_intervention_sweep(
            intervention, strengths, seeds=INTERVENTION_CONFIG["seeds"]
        )
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for intervention in results:
        # Create intervention-specific directory
        intervention_dir = os.path.join(RESULTS_DIR, f"{intervention}_{timestamp}")
        os.makedirs(intervention_dir, exist_ok=True)
        
        # Save intervention-specific results
        results_path = os.path.join(intervention_dir, RESULTS_FILE)
        torch.save(results[intervention], results_path)
    
    # Also save the complete results
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    torch.save(results, results_path)
    
    # Visualize
    visualize_results(results)

def visualize_results(results):
    """Visualize experiment results."""
    
    # Create figure for each intervention
    for intervention, intervention_results in results.items():
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
        ax1.set_title(f"{intervention} vs Feature Count")
        ax1.set_xlabel("Intervention Strength")
        ax1.set_ylabel("Feature Count")
        ax1.set_xticks(plot_strengths)
        ax1.set_xticklabels(strength_labels)
        ax1.grid(True)
        
        # Test accuracy
        ax2.errorbar(plot_strengths, test_accs, yerr=test_accs_stds, marker='o')
        ax2.set_title(f"{intervention} vs Test Accuracy")
        ax2.set_xlabel("Intervention Strength")
        ax2.set_ylabel("Test Accuracy")
        ax2.set_xticks(plot_strengths)
        ax2.set_xticklabels(strength_labels)
        ax2.grid(True)
        
        plt.tight_layout()
        # Save the figure to the results directory
        plt.savefig(os.path.join(RESULTS_DIR, f"{intervention}_results.pdf"))
        plt.close()  # Close the figure to free memory
        

if __name__ == "__main__":
    main()
# %%
