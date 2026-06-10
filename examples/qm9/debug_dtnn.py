"""
Debug script to diagnose DTNN training issues
"""
import deepchem as dc
import numpy as np

def check_environment():
    """Print environment information"""
    print(f"DeepChem version: {dc.__version__}")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")

def check_data_statistics():
    """Analyze QM9 dataset statistics"""
    tasks, datasets, transformers = dc.molnet.load_qm9()
    train, valid, test = datasets
    
    print("\n=== Data Statistics ===")
    print(f"Train samples: {len(train)}")
    print(f"Target mean: {np.mean(train.y):.4f}")
    print(f"Target std: {np.std(train.y):.4f}")
    print(f"Target range: [{np.min(train.y):.4f}, {np.max(train.y):.4f}]")
    
    # Check for issues
    if np.isnan(train.y).any():
        print("⚠️  WARNING: NaN values in targets!")
    if np.isinf(train.y).any():
        print("⚠️  WARNING: Inf values in targets!")

if __name__ == "__main__":
    check_environment()
    check_data_statistics()
