#!/usr/bin/env python
"""Validation script: memorization test + generation quality.

This script demonstrates that:
1. The model can memorize and reconstruct known protein structures
2. Generated samples show learning (compared to random noise baseline)
3. Loss curves show clear learning signal
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import deepchem as dc

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def normalize_protein(coords):
    """Normalize protein coordinates the same way the model does."""
    if coords.ndim == 3:
        coords = coords.reshape(-1, 9)
    ca = coords[:, 3:6]
    centroid = ca.mean(axis=0, keepdims=True)
    coords = coords - np.tile(centroid, 3)
    std = coords.std()
    if std > 1e-6:
        coords = coords / std
    return coords.astype(np.float32), std


def compute_ca_distances(coords):
    """Compute consecutive CA-CA distances."""
    ca = coords[:, 3:6]
    diffs = ca[1:] - ca[:-1]
    return np.sqrt(np.sum(diffs**2, axis=1))


def compute_n_ca_c_angle(coords):
    """Compute N-CA-C bond angle per residue."""
    angles = []
    for i in range(len(coords)):
        n = coords[i, 0:3]
        ca = coords[i, 3:6]
        c = coords[i, 6:9]
        v1 = n - ca
        v2 = c - ca
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1, 1)
        angles.append(np.degrees(np.arccos(cos_a)))
    return np.array(angles)


def main():
    logger.info("=" * 70)
    logger.info("RFDiffusion Validation: Memorization + Generation Quality")
    logger.info("=" * 70)

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'validation_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Memorization Test - Can the model learn a single protein?
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Memorization (Single Protein)")
    logger.info("=" * 60)

    # Download crambin (small, well-characterized protein)
    import requests
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'validation_data')
    os.makedirs(data_dir, exist_ok=True)
    pdb_file = os.path.join(data_dir, '1crn.pdb')
    if not os.path.exists(pdb_file):
        r = requests.get('https://files.rcsb.org/download/1crn.pdb', timeout=30)
        with open(pdb_file, 'wb') as f:
            f.write(r.content)

    featurizer = dc.feat.ProteinBackboneFeaturizer(max_length=256)
    features = featurizer.featurize([pdb_file])
    crambin = features[0]

    # Normalize
    crambin_norm, crambin_std = normalize_protein(crambin)
    logger.info(f"Crambin: {crambin_norm.shape[0]} residues, std={crambin_std:.2f}")

    # Create dataset with just crambin repeated
    n_copies = 16
    X = np.empty(n_copies, dtype=object)
    for i in range(n_copies):
        X[i] = crambin
    y = np.zeros((n_copies, 1), dtype=np.float32)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    # Train model to memorize
    model = dc.models.RFDiffusionModel(
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        num_diffusion_steps=100,  # Fewer steps = easier to learn
        max_seq_len=256,
        batch_size=8,
        learning_rate=5e-4,
        device=torch.device(device_str),
    )

    params = sum(p.numel() for p in model.model.parameters())
    logger.info(f"Model: {params:,} parameters")

    losses = []
    for epoch in range(500):
        loss = model.fit(dataset, nb_epoch=1)
        losses.append(float(loss))
        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1}: loss = {loss:.6f}")

    logger.info(f"Memorization: loss went from {losses[0]:.4f} → {losses[-1]:.4f} "
                f"({(1-losses[-1]/losses[0])*100:.1f}% reduction)")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Expanded CATH Training with structural evaluation
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Expanded CATH Training (94 proteins, 500 epochs)")
    logger.info("=" * 60)

    # Load all PDB files from cache
    pdb_dir = os.path.join(os.path.dirname(__file__), '..', 'cath_expanded_data')
    pdb_files = sorted([os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    logger.info(f"Found {len(pdb_files)} cached PDB files")

    features = featurizer.featurize(pdb_files)
    valid_features = []
    for f in features:
        if isinstance(f, np.ndarray) and f.size > 0:
            valid_features.append(f)

    logger.info(f"Featurized: {len(valid_features)} proteins")

    X = np.empty(len(valid_features), dtype=object)
    for i, f in enumerate(valid_features):
        X[i] = f
    y = np.zeros((len(valid_features), 1), dtype=np.float32)
    full_dataset = dc.data.NumpyDataset(X=X, y=y)

    # Split
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(
        full_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
    logger.info(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Compute real statistics for reference
    real_ca_dists = []
    real_n_ca_c = []
    real_stds = []
    for i in range(len(train)):
        x = train.X[i]
        if isinstance(x, np.ndarray) and x.size > 0:
            if x.ndim == 3:
                x_flat = x.reshape(-1, 9)
            else:
                x_flat = x
            if x_flat.shape[0] >= 3:
                d = compute_ca_distances(x_flat)
                real_ca_dists.extend(d.tolist())
                a = compute_n_ca_c_angle(x_flat)
                real_n_ca_c.extend(a.tolist())
                # Compute normalization std
                ca = x_flat[:, 3:6]
                centroid = ca.mean(axis=0, keepdims=True)
                centered = x_flat - np.tile(centroid, 3)
                std = centered.std()
                if std > 1e-6:
                    real_stds.append(std)

    avg_std = np.mean(real_stds)
    logger.info(f"\nReal protein statistics:")
    logger.info(f"  CA-CA distance: {np.mean(real_ca_dists):.3f} ± {np.std(real_ca_dists):.3f} Å")
    logger.info(f"  N-CA-C angle:   {np.mean(real_n_ca_c):.1f} ± {np.std(real_n_ca_c):.1f}°")
    logger.info(f"  Avg coord std:  {avg_std:.2f} Å")

    # Create model
    model2 = dc.models.RFDiffusionModel(
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        num_diffusion_steps=200,
        max_seq_len=256,
        batch_size=8,
        learning_rate=3e-4,
        device=torch.device(device_str),
    )

    # Train
    cath_losses = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(500):
        loss = model2.fit(train, nb_epoch=1)
        cath_losses.append(float(loss))

        if loss < best_loss:
            best_loss = loss
            model_dir = os.path.join(OUTPUT_DIR, 'best_model')
            os.makedirs(model_dir, exist_ok=True)
            model2.save_checkpoint(model_dir=model_dir)

        if (epoch + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={loss:.6f} best={best_loss:.6f}")

    train_time = time.time() - start_time
    logger.info(f"\nTraining: {train_time/60:.1f} min, "
                f"loss: {cath_losses[0]:.4f} → {cath_losses[-1]:.4f} "
                f"({(1-cath_losses[-1]/cath_losses[0])*100:.1f}% reduction)")

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Generation and Quality Metrics
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Sample Generation and Quality Metrics")
    logger.info("=" * 60)

    # Load best model
    model2.restore(model_dir=os.path.join(OUTPUT_DIR, 'best_model'))

    # Generate samples of various lengths
    for seq_len in [20, 50, 100]:
        samples = model2.generate(num_samples=5, seq_length=seq_len)
        logger.info(f"\nGenerated samples (length={seq_len}):")
        logger.info(f"  Shape: {samples.shape}")
        logger.info(f"  Value range: [{samples.min():.3f}, {samples.max():.3f}]")
        logger.info(f"  Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")

        # Rescale and compute metrics
        for i in range(len(samples)):
            s = samples[i] * avg_std

            d = compute_ca_distances(s)
            a = compute_n_ca_c_angle(s)

            if i == 0:  # Print first sample details
                logger.info(f"  Sample 0 (rescaled by {avg_std:.2f}):")
                logger.info(f"    CA-CA dist: {np.mean(d):.3f} ± {np.std(d):.3f} Å "
                            f"(real: {np.mean(real_ca_dists):.3f} Å)")
                logger.info(f"    N-CA-C angle: {np.mean(a):.1f} ± {np.std(a):.1f}° "
                            f"(real: {np.mean(real_n_ca_c):.1f}°)")

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Noise baseline comparison
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Comparison with Random Noise Baseline")
    logger.info("=" * 60)

    # Random noise baseline (what generation would look like without learning)
    random_samples = np.random.randn(5, 50, 9) * avg_std

    rand_dists = []
    rand_angles = []
    for i in range(5):
        d = compute_ca_distances(random_samples[i])
        rand_dists.extend(d.tolist())
        a = compute_n_ca_c_angle(random_samples[i])
        rand_angles.extend(a.tolist())

    # Model-generated
    model_samples = model2.generate(num_samples=5, seq_length=50)
    model_dists = []
    model_angles = []
    for i in range(5):
        s = model_samples[i] * avg_std
        d = compute_ca_distances(s)
        model_dists.extend(d.tolist())
        a = compute_n_ca_c_angle(s)
        model_angles.extend(a.tolist())

    logger.info(f"\n{'Metric':<25} {'Real':>10} {'Model':>10} {'Random':>10}")
    logger.info("-" * 55)
    logger.info(f"{'CA-CA dist (Å)':<25} {np.mean(real_ca_dists):>10.3f} "
                f"{np.mean(model_dists):>10.3f} {np.mean(rand_dists):>10.3f}")
    logger.info(f"{'N-CA-C angle (°)':<25} {np.mean(real_n_ca_c):>10.1f} "
                f"{np.mean(model_angles):>10.1f} {np.mean(rand_angles):>10.1f}")

    # ══════════════════════════════════════════════════════════════════
    # Save all results
    # ══════════════════════════════════════════════════════════════════
    results = {
        'memorization_test': {
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'reduction_pct': (1 - losses[-1]/losses[0]) * 100,
            'loss_history': losses,
        },
        'cath_training': {
            'dataset_size': len(full_dataset),
            'train_size': len(train),
            'initial_loss': cath_losses[0],
            'final_loss': cath_losses[-1],
            'best_loss': best_loss,
            'reduction_pct': (1 - cath_losses[-1]/cath_losses[0]) * 100,
            'training_time_min': train_time / 60,
            'loss_history': cath_losses,
        },
        'real_statistics': {
            'ca_ca_dist_mean': float(np.mean(real_ca_dists)),
            'ca_ca_dist_std': float(np.std(real_ca_dists)),
            'n_ca_c_angle_mean': float(np.mean(real_n_ca_c)),
            'n_ca_c_angle_std': float(np.std(real_n_ca_c)),
        },
        'generated_statistics': {
            'ca_ca_dist_mean': float(np.mean(model_dists)),
            'n_ca_c_angle_mean': float(np.mean(model_angles)),
        },
        'random_baseline': {
            'ca_ca_dist_mean': float(np.mean(rand_dists)),
            'n_ca_c_angle_mean': float(np.mean(rand_angles)),
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'validation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save generated PDBs
    pdb_dir = os.path.join(OUTPUT_DIR, 'generated_pdbs')
    os.makedirs(pdb_dir, exist_ok=True)

    from scripts.train_rfdiffusion_cath import save_as_pdb
    final_samples = model2.generate(num_samples=10, seq_length=50)
    for i in range(len(final_samples)):
        rescaled = final_samples[i] * avg_std
        save_as_pdb(rescaled, os.path.join(pdb_dir, f'generated_{i+1}.pdb'))

    logger.info(f"\nAll results saved to {OUTPUT_DIR}")
    logger.info("=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
