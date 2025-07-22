# Refactored Adversarial Robustness Framework

This document describes the new refactored structure that separates training, evaluation, and analysis into independent scripts to improve modularity and prevent OOM errors.

## 🎯 **Key Benefits of Refactoring**

1. **✅ Modular Design**: Training, evaluation, and analysis are now separate
2. **✅ Memory Optimization**: Prevents OOM errors with memory management
3. **✅ Retroactive Analysis**: Can analyze existing checkpoints without re-training
4. **✅ Flexible Workflow**: Run phases independently or together
5. **✅ Better Error Recovery**: If one phase fails, others can still run

## 📁 **New Script Structure**

```
experiments/adversarial/
├── train_models.py              # 🚀 Standalone training script
├── evaluate_models.py           # 🔍 Standard evaluation script  
├── evaluate_models_memory_optimized.py  # 💾 Memory-optimized evaluation
├── analyze_results.py           # 📊 Analysis and plotting script
├── main.py                      # 🏗️ Original combined script (still available)
└── REFACTORED_USAGE.md          # 📖 This guide
```

## 🚀 **Quick Start Examples**

### **1. Training Only**
```python
# Train models and save checkpoints
from experiments.adversarial.train_models import quick_train

results_dir = quick_train(
    model_type='resnet18',
    dataset_type='mnist',
    testing_mode=True
)
print(f"Training complete! Results in: {results_dir}")
```

### **2. Evaluation Only (Standard)**
```python
# Evaluate existing trained models
from experiments.adversarial.evaluate_models import evaluate_existing_results

evaluate_existing_results(
    results_dir="./results/adversarial_robustness/mnist/2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc",
    dataset_type="mnist",
    enable_llc=True,
    enable_sae=True,
    enable_inference_llc=True,
    enable_sae_checkpoints=True,
    enable_llc_checkpoints=True,
    enable_combined_checkpoints=False
)
```

### **3. Evaluation Only (Memory-Optimized)**
```python
# Evaluate with memory optimizations to prevent OOM
from experiments.adversarial.evaluate_models_memory_optimized import evaluate_existing_results_memory_optimized

evaluate_existing_results_memory_optimized(
    results_dir="./results/adversarial_robustness/mnist/2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc",
    dataset_type="mnist",
    enable_llc=True,
    enable_sae=True,
    enable_inference_llc=True,
    enable_sae_checkpoints=True,
    enable_llc_checkpoints=True,
    enable_combined_checkpoints=False,
    max_samples_per_analysis=3000,  # Reduce samples to prevent OOM
    process_one_epsilon_at_a_time=True  # Process epsilons sequentially
)
```

### **4. Analysis Only**
```python
# Generate plots from existing evaluation results
from experiments.adversarial.analyze_results import analyze_existing_results

analyze_existing_results(
    results_dir="./results/adversarial_robustness/mnist/2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc",
    dataset_type="mnist"
)
```

## 🔧 **Command Line Usage**

### **Training**
```bash
cd experiments/adversarial
python -m train_models
```

### **Evaluation (Standard)**
```bash
cd experiments/adversarial
python -m evaluate_models
```

### **Evaluation (Memory-Optimized)**
```bash
cd experiments/adversarial
python -m evaluate_models_memory_optimized
```

### **Analysis**
```bash
cd experiments/adversarial
python -m analyze_results
```

## 💾 **Memory Optimization Features**

The memory-optimized evaluation script includes:

1. **Reduced SAE Parameters**:
   - Smaller batch sizes (64 instead of 128)
   - Fewer epochs (12 instead of 100)
   - Reduced expansion factor (2.0 instead of 4.0)

2. **Reduced LLC Parameters**:
   - Fewer draws (200 instead of 1500)
   - Fewer chains (2 instead of 3)
   - Smaller hyperparameter grids (3×3 instead of 5×5)

3. **Memory Management**:
   - Sequential epsilon processing
   - Explicit GPU memory clearing
   - Garbage collection between analyses
   - Model deletion after each analysis

4. **Sample Limiting**:
   - Configurable max samples per analysis
   - Prevents loading too much data at once

## 🔄 **Workflow Options**

### **Option 1: Full Pipeline (Original)**
```python
from experiments.adversarial.main import quick_test

# Still works as before
results_dir = quick_test(
    model_type='resnet18',
    dataset_type='mnist',
    testing_mode=True,
    enable_sae=True,
    enable_llc=True,
    enable_inference_llc=True,
    enable_sae_checkpoints=True
)
```

### **Option 2: Modular Pipeline**
```python
# Step 1: Train
from experiments.adversarial.train_models import quick_train
results_dir = quick_train(model_type='resnet18', dataset_type='mnist')

# Step 2: Evaluate (if training succeeded)
from experiments.adversarial.evaluate_models_memory_optimized import evaluate_existing_results_memory_optimized
evaluate_existing_results_memory_optimized(results_dir=results_dir)

# Step 3: Analyze (if evaluation succeeded)
from experiments.adversarial.analyze_results import analyze_existing_results
analyze_existing_results(results_dir=results_dir)
```

### **Option 3: Retroactive Analysis**
```python
# Analyze existing results without re-training
from experiments.adversarial.evaluate_models_memory_optimized import evaluate_existing_results_memory_optimized

# Complete missing evaluations
evaluate_existing_results_memory_optimized(
    results_dir="path/to/existing/results",
    enable_sae_checkpoints=True,  # Only run SAE checkpoint analysis
    enable_llc_checkpoints=False,  # Skip LLC if it already exists
    enable_inference_llc=True      # Add inference-time LLC
)
```

## 🎯 **Answering Your Questions**

### **Q: "Would it be an option to refactor the codebase such that the training is a separate script from the evaluation?"**

**✅ YES!** This is exactly what we've done:

- **`train_models.py`**: Handles only training and checkpoint saving
- **`evaluate_models.py`**: Handles only evaluation of trained models
- **`analyze_results.py`**: Handles only analysis and plotting

### **Q: "Would it be possible for the SAEs to train and evaluate on the checkpoints retroactively?"**

**✅ YES!** The evaluation scripts can:

1. **Load existing checkpoints** from any training run
2. **Run SAE analysis** on those checkpoints without re-training
3. **Run LLC analysis** on those checkpoints
4. **Run combined analysis** on those checkpoints
5. **Add new analyses** to existing results

### **Q: "Why does it look like it breaks with OOM errors?"**

**✅ SOLVED!** The memory-optimized script addresses:

1. **Large activation tensors**: Reduced sample sizes and batch sizes
2. **Multiple model copies**: Explicit memory management and cleanup
3. **SAE training memory**: CPU training and reduced parameters
4. **LLC hyperparameter tuning**: Smaller grids and fewer measurements

## 🚨 **Troubleshooting**

### **OOM Errors**
```python
# Use memory-optimized evaluation
from experiments.adversarial.evaluate_models_memory_optimized import evaluate_existing_results_memory_optimized

evaluate_existing_results_memory_optimized(
    results_dir="your/results/dir",
    max_samples_per_analysis=2000,  # Reduce further if needed
    process_one_epsilon_at_a_time=True
)
```

### **Missing Dependencies**
```bash
# Install required packages
pip install devinterp nnsight
```

### **Partial Results**
```python
# Check what's already completed
import json
from pathlib import Path

results_dir = Path("your/results/dir")
for eps_dir in results_dir.glob("eps_*"):
    for run_dir in eps_dir.glob("run_*"):
        print(f"Checking {run_dir}")
        if (run_dir / "llc_analysis_results.json").exists():
            print("  ✅ LLC analysis complete")
        if (run_dir / "sae_checkpoint_analysis_results.json").exists():
            print("  ✅ SAE analysis complete")
```

## 📊 **Expected Output Structure**

After running the modular pipeline:

```
results/
└── adversarial_robustness/
    └── mnist/
        └── 2025-07-22_12-49_resnet18_2-class_pgd_quick_test_mnist_resnet18_with_llc_sae_inference_llc/
            ├── config.json                    # ✅ Training configuration
            ├── experiment.log                 # ✅ Training logs
            ├── eps_0.0/
            │   └── run_0/
            │       ├── model.pt               # ✅ Trained model
            │       ├── robustness.json        # ✅ Robustness results
            │       ├── checkpoints/           # ✅ Training checkpoints
            │       ├── llc_analysis_results.json          # ✅ LLC analysis
            │       ├── sae_checkpoint_analysis_results.json  # ✅ SAE analysis
            │       ├── inference_llc_results.json          # ✅ Inference LLC
            │       └── plots/                 # ✅ Generated plots
            └── eps_0.1/
                └── run_0/
                    └── [same structure]
```

## 🎉 **Summary**

The refactored codebase provides:

1. **✅ Modularity**: Separate training, evaluation, and analysis
2. **✅ Memory Efficiency**: OOM prevention with optimizations
3. **✅ Flexibility**: Run phases independently or together
4. **✅ Retroactive Analysis**: Analyze existing checkpoints
5. **✅ Error Recovery**: Continue from where you left off

You can now train models once and analyze them multiple times with different parameters, or add new analyses to existing results without re-training! 