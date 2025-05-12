# %%
import torch
import numpy as np
from torch.utils.data import TensorDataset

class MultiTaskSparseParityDataset:
    """
    Dataset for multi-task sparse parity problems.
    
    This dataset generates binary classification problems where the label is determined
    by the parity (odd/even count of 1s) of a subset of bits specific to each task.
    Tasks are indicated by a one-hot encoding control vector.
    """
    
    def __init__(self, num_tasks=3, bits_per_task=4, seed=42):
        """
        Initialize the dataset with configurable parameters.
        
        Args:
            num_tasks (int): Number of distinct parity tasks
            bits_per_task (int): Number of relevant bits per task
            seed (int): Random seed for reproducibility
        """
        self.num_tasks = num_tasks
        self.bits_per_task = bits_per_task

        self.total_data_bits = num_tasks * bits_per_task
        self.control_size = num_tasks  # One-hot task indicator
        
        # Set seeds for reproducibility
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate all possible input patterns
        self.all_patterns = self._generate_all_data_patterns()
        self.control_vectors = torch.eye(num_tasks)
        
        print(f"Generated {len(self.all_patterns)} unique input patterns")
    
    def _generate_all_data_patterns(self):
        """
        Generate all possible binary patterns for the data bits.
        
        Returns:
            torch.Tensor: Tensor of shape [2^(bits_per_task*num_tasks), bits_per_task*num_tasks]
                containing all possible bit patterns
        """
        patterns = []
        for i in range(2**self.total_data_bits):
            # Convert integer to binary representation with padding
            bit_string = format(i, f'0{self.total_data_bits}b')
            patterns.append([int(b) for b in bit_string])
        
        return torch.tensor(patterns, dtype=torch.float)
    
    def _calculate_parity_labels(self, inputs):
        """
        Calculate parity labels for each input.
        
        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, control_size + total_data_bits]
        
        Returns:
            torch.Tensor: Binary labels with shape [batch_size, 1]
        """
        # Extract control bits and data bits
        control = inputs[:, :self.control_size]
        data = inputs[:, self.control_size:]
        
        batch_size = inputs.shape[0]
        labels = torch.zeros(batch_size, 1)
        
        # For each sample in the batch
        for i in range(batch_size):
            active_task = control[i].argmax().item()
            
            # Extract relevant bits for the active task
            task_start = active_task * self.bits_per_task
            task_end = task_start + self.bits_per_task
            task_bits = data[i, task_start:task_end]
            
            # Label is the parity (sum of bits mod 2)
            labels[i, 0] = task_bits.sum() % 2
            
        return labels
    
    def create_train_test_split(self, test_ratio=0.2):
        """
        Create stratified training and test datasets.
        
        Args:
            test_ratio (float): Proportion of data to use for testing (0.0-1.0)
        
        Returns:
            tuple: (train_dataset, test_dataset) as TensorDataset objects
        """
        all_inputs = self._create_all_task_combinations()
        all_labels = self._calculate_parity_labels(all_inputs)
        
        # Split indices, stratifying by task and label
        train_indices, test_indices = self._create_stratified_split_indices(
            all_inputs, all_labels, test_ratio
        )
        
        # Create the actual datasets
        train_dataset, test_dataset = self._create_datasets_from_indices(
            all_inputs, all_labels, train_indices, test_indices
        )

        return train_dataset, test_dataset
    
    def _create_all_task_combinations(self):
        """
        Create all possible input combinations of control vectors and data patterns.
        
        Returns:
            torch.Tensor: Tensor containing all possible inputs
        """
        all_inputs = []
        for control_idx in range(self.num_tasks):
            control = torch.zeros(self.control_size)
            control[control_idx] = 1
            
            for pattern in self.all_patterns:
                sample = torch.cat([control, pattern])
                all_inputs.append(sample)
        
        return torch.stack(all_inputs)
    
    def _create_stratified_split_indices(self, all_inputs, all_labels, test_ratio):
        """
        Create indices for a stratified train-test split.
        
        Args:
            all_inputs (torch.Tensor): All input samples
            all_labels (torch.Tensor): All labels
            test_ratio (float): Proportion for test set
        
        Returns:
            tuple: (train_indices, test_indices) as torch.Tensor objects
        """
        train_indices = []
        test_indices = []
        all_labels = all_labels.squeeze()
        generator = torch.Generator().manual_seed(self.seed)
        
        # Stratify by task and label
        for task in range(self.num_tasks):
            for label in [0, 1]:
                # Find indices matching this task and label
                mask = (all_inputs[:, task] == 1) & (all_labels == label)
                indices = torch.where(mask)[0]
                
                # Shuffle indices for randomization
                indices = indices[torch.randperm(len(indices), generator=generator)]
                
                # Split based on test_ratio
                split_idx = int(len(indices) * (1 - test_ratio))
                train_indices.append(indices[:split_idx])
                test_indices.append(indices[split_idx:])
        
        return torch.cat(train_indices), torch.cat(test_indices)
    
    def _create_datasets_from_indices(self, all_inputs, all_labels, train_indices, test_indices):
        """
        Create TensorDataset objects from the provided indices.
        
        Args:
            all_inputs (torch.Tensor): All input samples
            all_labels (torch.Tensor): All labels
            train_indices (torch.Tensor): Indices for training set
            test_indices (torch.Tensor): Indices for test set
        
        Returns:
            tuple: (train_dataset, test_dataset) as TensorDataset objects
        """
        # Create training and test sets
        train_inputs = all_inputs[train_indices]
        train_labels = all_labels[train_indices]
        test_inputs = all_inputs[test_indices]
        test_labels = all_labels[test_indices]
        
        # Convert to dataset objects
        train_dataset = TensorDataset(train_inputs, train_labels)
        test_dataset = TensorDataset(test_inputs, test_labels)
        
        return train_dataset, test_dataset
    
    def decode_sample(self, inputs, label=None):
        """
        Decode a sample into a human-readable format.
        
        Args:
            inputs (torch.Tensor): Input tensor
            label (torch.Tensor, optional): Label tensor
        
        Returns:
            dict: Information about the sample in a readable format
        """
        control = inputs[:self.control_size]
        data = inputs[self.control_size:]
        
        active_task = control.argmax().item()
        task_start = active_task * self.bits_per_task
        task_end = task_start + self.bits_per_task
        task_bits = data[task_start:task_end].tolist()
        
        result = {
            "control_bits": control.tolist(),
            "active_task": active_task,
            "task_bits": task_bits,
            "expected_parity": sum(task_bits) % 2
        }
        
        if label is not None:
            result["label"] = label.item()
            
        return result

# %%
if __name__ == "__main__":
    seed = 2

    # Test the dataset implementation
    dataset = MultiTaskSparseParityDataset(num_tasks=3, bits_per_task=4, seed=seed)
    train_dataset, test_dataset = dataset.create_train_test_split(test_ratio=0.2)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    # Display a few examples from the training set
    print("\nSample training examples:")
    n_examples = 5
    # draw random indices
    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(seed))[:n_examples]
    for i in indices:
        inputs, label = train_dataset[i]
        sample_info = dataset.decode_sample(inputs, label)
        
        print(f"Example {i+1}:")
        for key, value in sample_info.items():
            print(f"  {key}: {value}")
        print()

    # Verify label distribution
    train_labels = torch.cat([label for _, label in train_dataset])
    test_labels = torch.cat([label for _, label in test_dataset])

    print("Label distribution:")
    print(f"  Training: 0s: {(train_labels == 0).sum().item()}, 1s: {(train_labels == 1).sum().item()}")
    print(f"  Test: 0s: {(test_labels == 0).sum().item()}, 1s: {(test_labels == 1).sum().item()}")
# %%
