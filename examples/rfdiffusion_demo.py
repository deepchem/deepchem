"""RFDiffusion Model Demo: Training and Evaluation on CATH Dataset.

This script demonstrates the full DeepChem-native workflow for training
an RFDiffusion model on protein backbone structures:

1. Load CATH dataset using dc.molnet.load_cath()
2. Train RFDiffusionModel(TorchModel) using model.fit(dataset)
3. Run a memorization test on a small subset
4. Generate new protein backbone structures
5. Evaluate generated samples

Usage:
    python examples/rfdiffusion_demo.py
"""

import numpy as np
import time
import deepchem as dc

print("=" * 70)
print("RFDiffusion Model - DeepChem Integration Demo")
print("=" * 70)

# ============================================================
# 1. Load CATH Dataset
# ============================================================
print("\n[1/5] Loading CATH protein backbone dataset...")
try:
    tasks, datasets, transformers = dc.molnet.load_cath(
        featurizer='ProteinBackbone', splitter='random')
    train_dataset, valid_dataset, test_dataset = datasets
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")

    # Show some statistics
    train_lengths = [train_dataset.X[i].shape[0]
                     for i in range(len(train_dataset))]
    print(f"  Protein lengths: min={min(train_lengths)}, "
          f"max={max(train_lengths)}, "
          f"mean={np.mean(train_lengths):.0f}")
    use_cath = True
except Exception as e:
    print(f"  Could not load CATH dataset: {e}")
    print("  Falling back to synthetic data...")
    use_cath = False

# ============================================================
# 2. Create and Train Model
# ============================================================
print("\n[2/5] Creating RFDiffusionModel...")

model = dc.models.RFDiffusionModel(
    embed_dim=128,
    time_dim=64,
    num_layers=4,
    num_heads=4,
    num_diffusion_steps=200,
    max_seq_len=256,
    dropout=0.1,
    batch_size=4,
    learning_rate=1e-4,
)

n_params = sum(p.numel() for p in model.model.parameters())
print(f"  Model parameters: {n_params:,}")
print(f"  Device: {model.device}")

if use_cath:
    dataset = train_dataset
else:
    # Create synthetic dataset
    n_samples = 20
    proteins = [np.random.randn(np.random.randint(20, 80), 3, 3).astype(np.float32)
                for _ in range(n_samples)]
    X = np.empty(n_samples, dtype=object)
    for i, p in enumerate(proteins):
        X[i] = p
    y = np.zeros((n_samples, 1), dtype=np.float32)
    dataset = dc.data.NumpyDataset(X=X, y=y)
    print(f"  Using synthetic dataset with {n_samples} samples")

print("\n[3/5] Training model...")
all_losses = []
num_epochs = 50
epochs_per_checkpoint = 10

for epoch_block in range(0, num_epochs, epochs_per_checkpoint):
    start_time = time.time()
    loss = model.fit(dataset, nb_epoch=epochs_per_checkpoint)
    elapsed = time.time() - start_time
    all_losses.append(loss)
    total_epochs = epoch_block + epochs_per_checkpoint
    print(f"  Epochs {total_epochs:3d}/{num_epochs} | "
          f"Loss: {loss:.6f} | "
          f"Time: {elapsed:.1f}s")

print(f"\n  Final loss: {all_losses[-1]:.6f}")
if len(all_losses) > 1:
    improvement = (all_losses[0] - all_losses[-1]) / all_losses[0] * 100
    print(f"  Loss reduction: {improvement:.1f}%")

# ============================================================
# 4. Memorization Test
# ============================================================
print("\n[4/5] Memorization test (overfit on 4 samples)...")

# Create tiny dataset for memorization
if use_cath:
    # Use first 4 training samples
    tiny_X = train_dataset.X[:4]
    tiny_y = np.zeros((4, 1), dtype=np.float32)
else:
    tiny_proteins = [np.random.randn(20, 3, 3).astype(np.float32)
                     for _ in range(4)]
    tiny_X = np.empty(4, dtype=object)
    for i, p in enumerate(tiny_proteins):
        tiny_X[i] = p
    tiny_y = np.zeros((4, 1), dtype=np.float32)

tiny_dataset = dc.data.NumpyDataset(X=tiny_X, y=tiny_y)

memo_model = dc.models.RFDiffusionModel(
    embed_dim=64,
    num_layers=2,
    num_heads=4,
    num_diffusion_steps=100,
    batch_size=4,
    learning_rate=1e-3,
)

print("  Training on 4 samples for 200 epochs...")
memo_losses = []
for step in range(0, 200, 50):
    loss = memo_model.fit(tiny_dataset, nb_epoch=50)
    memo_losses.append(loss)
    print(f"    Epoch {step+50:3d}/200 | Loss: {loss:.6f}")

if len(memo_losses) > 1:
    memo_reduction = (memo_losses[0] - memo_losses[-1]) / memo_losses[0] * 100
    print(f"  Memorization loss reduction: {memo_reduction:.1f}%")
    if memo_reduction > 50:
        print("  ✓ Model successfully memorizes small dataset")
    else:
        print("  ⚠ Model may need more training for full memorization")

# ============================================================
# 5. Generate Samples
# ============================================================
print("\n[5/5] Generating protein backbone structures...")
start_time = time.time()
num_gen_samples = 3
seq_length = 30

samples = model.generate(num_samples=num_gen_samples, seq_length=seq_length)
gen_time = time.time() - start_time

print(f"  Generated {num_gen_samples} proteins of length {seq_length}")
print(f"  Output shape: {samples.shape}")
print(f"  Generation time: {gen_time:.1f}s")
print(f"  Values - min: {samples.min():.3f}, max: {samples.max():.3f}, "
      f"mean: {samples.mean():.3f}, std: {samples.std():.3f}")
print(f"  All finite: {np.isfinite(samples).all()}")

# Check basic structural validity
for i in range(num_gen_samples):
    coords = samples[i].reshape(-1, 3, 3)  # (L, 3_atoms, 3_xyz)
    n_coords = coords[:, 0]   # N atoms
    ca_coords = coords[:, 1]  # CA atoms
    c_coords = coords[:, 2]   # C atoms

    # CA-CA distances
    ca_dists = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
    print(f"  Sample {i+1}: CA-CA dist mean={ca_dists.mean():.2f}, "
          f"std={ca_dists.std():.2f}")

print("\n" + "=" * 70)
print("Demo complete.")
print("=" * 70)
print("\nUsage in your own code:")
print("  import deepchem as dc")
print("  tasks, datasets, _ = dc.molnet.load_cath(featurizer='ProteinBackbone')")
print("  train, valid, test = datasets")
print("  model = dc.models.RFDiffusionModel(embed_dim=256, num_layers=8)")
print("  model.fit(train, nb_epoch=100)")
print("  samples = model.generate(num_samples=5, seq_length=50)")
