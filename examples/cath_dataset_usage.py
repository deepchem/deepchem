"""
CATH Dataset Usage Examples
============================

This file demonstrates how to use the CATH protein structure dataset
with DeepChem for training protein backbone diffusion models.

Examples cover:
1. Basic dataset loading
2. Integration with RFDiffusion-style models
3. Custom featurization
4. Data splitting strategies
"""

import deepchem as dc
import numpy as np


def example_basic_loading():
    """Example 1: Basic CATH dataset loading."""
    print("=" * 60)
    print("Example 1: Basic CATH Dataset Loading")
    print("=" * 60)

    # Load CATH dataset with default parameters
    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone',
        splitter='random',
        reload=True)

    train, valid, test = datasets

    print(f"\nTasks: {tasks}")
    print(f"Training set size: {len(train)}")
    print(f"Validation set size: {len(valid)}")
    print(f"Test set size: {len(test)}")

    # Inspect a sample
    if len(train) > 0:
        sample = train.X[0]
        print(f"\nSample protein shape: {sample.shape}")
        print(f"  - Number of residues: {sample.shape[0]}")
        print(f"  - Atoms per residue: {sample.shape[1]} (N, CA, C)")
        print(f"  - Coordinates: {sample.shape[2]} (x, y, z)")
        print(f"\nFirst residue backbone coordinates:")
        print(f"  N atom:  {sample[0, 0]}")
        print(f"  CA atom: {sample[0, 1]}")
        print(f"  C atom:  {sample[0, 2]}")


def example_custom_max_length():
    """Example 2: Loading with custom maximum protein length."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Maximum Protein Length")
    print("=" * 60)

    # Load CATH with max_length=256 for memory-limited GPUs
    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone',
        splitter='random',
        max_length=256,
        reload=True)

    train, valid, test = datasets

    print(f"\nMax protein length: 256 residues")
    print(f"Training set size: {len(train)}")

    # Verify all proteins are within max_length
    if len(train) > 0:
        max_observed = max(sample.shape[0] for sample in train.X)
        print(f"Maximum observed length: {max_observed}")
        assert max_observed <= 256, "Truncation failed!"
        print("✓ All proteins within length limit")


def example_no_split():
    """Example 3: Loading entire dataset without splitting."""
    print("\n" + "=" * 60)
    print("Example 3: Loading Without Splitting")
    print("=" * 60)

    # Load entire dataset for unsupervised training
    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone',
        splitter=None,
        reload=True)

    dataset = datasets[0]

    print(f"\nTotal dataset size: {len(dataset)}")
    print("Use case: Unsupervised pre-training of diffusion models")


def example_integration_with_model():
    """Example 4: Integration with a simple diffusion model."""
    print("\n" + "=" * 60)
    print("Example 4: Integration with Diffusion Model")
    print("=" * 60)

    # Load dataset
    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone',
        splitter='random',
        max_length=128,
        reload=True)

    train, valid, test = datasets

    print(f"\nLoaded {len(train)} training proteins")

    # Example: Create batches for training
    # In practice, you would use a PyTorch DataLoader
    batch_size = 4
    num_batches = len(train) // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")

    # Get a batch
    if len(train) >= batch_size:
        batch_proteins = train.X[:batch_size]

        print(f"\nBatch shape info:")
        for i, protein in enumerate(batch_proteins):
            print(
                f"  Protein {i}: {protein.shape[0]} residues, shape {protein.shape}"
            )

        print("\nNote: In RFDiffusion training, you would:")
        print("1. Add noise to these coordinates at time t")
        print("2. Predict the noise or denoised coordinates")
        print("3. Compute MSE loss between prediction and target")


def example_compute_statistics():
    """Example 5: Compute dataset statistics."""
    print("\n" + "=" * 60)
    print("Example 5: Dataset Statistics")
    print("=" * 60)

    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone', splitter=None, reload=True)

    dataset = datasets[0]

    if len(dataset) > 0:
        # Compute length distribution
        lengths = [sample.shape[0] for sample in dataset.X]

        print(f"\nProtein length statistics:")
        print(f"  Mean length: {np.mean(lengths):.1f} residues")
        print(f"  Median length: {np.median(lengths):.1f} residues")
        print(f"  Min length: {np.min(lengths)} residues")
        print(f"  Max length: {np.max(lengths)} residues")
        print(f"  Std dev: {np.std(lengths):.1f} residues")

        # Compute coordinate statistics
        all_coords = np.concatenate(
            [sample.reshape(-1, 3) for sample in dataset.X], axis=0)

        print(f"\nCoordinate statistics (Angstroms):")
        print(f"  Mean: {np.mean(all_coords, axis=0)}")
        print(f"  Std:  {np.std(all_coords, axis=0)}")
        print(f"\nTotal atoms: {all_coords.shape[0]}")


if __name__ == '__main__':
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CATH Dataset Usage Examples for DeepChem")
    print("=" * 60)

    try:
        example_basic_loading()
        example_custom_max_length()
        example_no_split()
        example_integration_with_model()
        example_compute_statistics()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: {e}")
        print(
            "Make sure BioPython is installed: pip install biopython requests"
        )
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
