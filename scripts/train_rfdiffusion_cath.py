#!/usr/bin/env python
"""Train RFDiffusion on an expanded CATH dataset and evaluate generated samples.

This script:
1. Downloads a larger set of CATH-representative PDB structures (~100 proteins)
2. Featurizes them using ProteinBackboneFeaturizer
3. Trains RFDiffusionModel on GPU
4. Generates protein backbone samples
5. Evaluates structural quality (bond distances, bond angles, Ramachandran-like metrics)
6. Saves results and generated structures
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path

# Add deepchem to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import deepchem as dc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_log.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# ── Expanded CATH-representative PDB set ────────────────────────────────
# ~100 diverse proteins spanning all 4 CATH classes:
#   Class 1: Mainly Alpha
#   Class 2: Mainly Beta
#   Class 3: Alpha Beta (mixed and alternating)
#   Class 4: Few Secondary Structures (small or irregular)
EXPANDED_PDB_IDS = [
    # === Class 1: Mainly Alpha ===
    "1MBN",  # Myoglobin (globin fold)
    "256B",  # Cytochrome b562
    "1HRC",  # Cytochrome c
    "1A6M",  # Calmodulin
    "1LMB",  # Lambda repressor
    "2HHB",  # Hemoglobin
    "1ECA",  # Erythrocruorin
    "1UTG",  # Uteroglobin
    "1RGE",  # Ribonuclease
    "3ICB",  # Calbindin
    "1MBC",  # Myoglobin (CO bound)
    "1CYO",  # Cytochrome b5
    "1HLE",  # Lactalbumin
    "1GAB",  # GA module
    "1VII",  # Villin headpiece
    "1R69",  # 434 repressor
    "4HHB",  # Hemoglobin deoxy
    "1BAB",  # Hemoglobin (bovine)
    "1DLW",  # Cytochrome c2
    "1OPC",  # Apolipoprotein
    "1M6T",  # Ferritin
    "1BZR",  # Apolipoprotein E
    "1ALV",  # Albumin fragment
    "1BFD",  # Lactoferrin fragment
    "1AIG",  # Immunoglobulin-like

    # === Class 2: Mainly Beta ===
    "1UBQ",  # Ubiquitin (beta-grasp)
    "2IG2",  # Immunoglobulin
    "1TEN",  # Tenascin (FN3)
    "1FNF",  # Fibronectin
    "1TTF",  # Tumor necrosis factor
    "1CD8",  # CD8 alpha
    "1RIE",  # Ribonuclease inhibitor
    "1PKN",  # Protein kinase
    "1CPN",  # Cupredoxin
    "1SRL",  # SRC SH3
    "1FKB",  # FKBP12
    "1WIT",  # WW domain
    "1MJC",  # Major cold shock protein
    "1PLT",  # Plastocyanin
    "1TIT",  # Titin I27
    "2AIT",  # Concanavalin A
    "1GEN",  # Gene V protein
    "1PGA",  # Protein G (B1)
    "1IGD",  # Protein G (D1)
    "1CSE",  # Subtilisin (inhibitor)
    "1PGX",  # Protein G-X
    "1EMR",  # Endothelin-1 receptor
    "1PAZ",  # Pseudoazurin
    "1NXB",  # Neurotoxin
    "2PTL",  # PTB domain

    # === Class 3: Alpha-Beta ===
    "1CRN",  # Crambin
    "1YCR",  # MDM2 (p53 complex)
    "1A3N",  # HLA-A2
    "1BKR",  # Beta-lactamase
    "1PHT",  # Phosphotransferase
    "1E0L",  # Enolase
    "1LYZ",  # Lysozyme
    "2RN2",  # Ribonuclease H
    "1RNB",  # Barnase
    "4GCR",  # Gamma-crystallin
    "1AKE",  # Adenylate kinase
    "1CSP",  # Cold shock protein
    "1CHD",  # Cytochrome cd1
    "1SN3",  # Scorpion neurotoxin
    "1L2Y",  # Trp-cage miniprotein
    "3LZM",  # T4 lysozyme
    "1HEL",  # Hen lysozyme
    "1IOB",  # Iron-binding protein
    "2LZM",  # Lysozyme variant
    "1HEW",  # Hen egg-white lysozyme
    "1RN1",  # Ribonuclease A
    "1OVA",  # Ovalbumin
    "1AXN",  # Annexin V
    "1PLC",  # Phospholipase C
    "1LAP",  # Leucine aminopeptidase

    # === Class 4: Few Secondary Structures / Small ===
    "1LE0",  # Metallothionein
    "1ZAA",  # Zinc finger
    "1EDN",  # Endothelin
    "1BPI",  # BPTI
    "1ROO",  # Rubredoxin
    "1HIP",  # Histidine-containing phosphocarrier
    "2ERO",  # Erabutoxin
    "1EJG",  # Carbonic anhydrase
    "1CBN",  # Crambin variant
    "1RDG",  # Rubredoxin-like
    "1PTX",  # Pertussis toxin
    "5PTI",  # Pancreatic trypsin inhibitor
    "1CLB",  # Calbindin D9K
    "6PTI",  # BPTI variant
    "1FME",  # Met-J
    "1WHI",  # S. aureus protein
    "1COA",  # Coat protein
    "4RXN",  # Rubredoxin
    "1I6C",  # Insulin-like
    "1CTF",  # Chymotrypsin fragment
]


def download_pdbs(pdb_ids, data_dir):
    """Download PDB files and return successful paths."""
    import requests
    os.makedirs(data_dir, exist_ok=True)
    pdb_files = []
    ids = []

    for pdb_id in pdb_ids:
        pdb_code = pdb_id.lower()
        pdb_file = os.path.join(data_dir, f"{pdb_code}.pdb")

        if os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 100:
            pdb_files.append(pdb_file)
            ids.append(pdb_id)
            continue

        try:
            url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(pdb_file, 'wb') as f:
                    f.write(response.content)
                pdb_files.append(pdb_file)
                ids.append(pdb_id)
                logger.info(f"  Downloaded {pdb_id}")
            else:
                logger.warning(f"  Failed {pdb_id}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"  Error {pdb_id}: {e}")

    return pdb_files, ids


def create_dataset(pdb_files, ids, max_length=256):
    """Featurize PDB files and create DeepChem dataset."""
    featurizer = dc.feat.ProteinBackboneFeaturizer(max_length=max_length)
    features = featurizer.featurize(pdb_files)

    valid_data = []
    for i, f in enumerate(features):
        if isinstance(f, np.ndarray) and f.size > 0:
            valid_data.append((f, ids[i]))

    logger.info(f"Successfully featurized {len(valid_data)}/{len(pdb_files)} proteins")

    features_list = [d[0] for d in valid_data]
    valid_ids = [d[1] for d in valid_data]

    # Compute statistics
    lengths = [f.shape[0] for f in features_list]
    logger.info(f"Protein lengths: min={min(lengths)}, max={max(lengths)}, "
                f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")

    # Convert to object array for variable-length support
    X = np.empty(len(features_list), dtype=object)
    for i, f in enumerate(features_list):
        X[i] = f

    y = np.zeros((len(features_list), 1), dtype=np.float32)
    dataset = dc.data.NumpyDataset(X=X, y=y, ids=np.array(valid_ids))
    return dataset, lengths


def compute_ca_distances(coords):
    """Compute consecutive CA-CA distances from backbone coords.

    Parameters
    ----------
    coords : np.ndarray
        Shape (num_residues, 9) — N, CA, C for each residue.

    Returns
    -------
    np.ndarray
        CA-CA distances for consecutive residues.
    """
    # CA is atoms index 1 (columns 3:6)
    ca_coords = coords[:, 3:6]
    diffs = ca_coords[1:] - ca_coords[:-1]
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return distances


def compute_bond_angles(coords):
    """Compute CA-CA-CA pseudo bond angles.

    Parameters
    ----------
    coords : np.ndarray
        Shape (num_residues, 9).

    Returns
    -------
    np.ndarray
        Angles in degrees for triplets of consecutive CA atoms.
    """
    ca_coords = coords[:, 3:6]
    if len(ca_coords) < 3:
        return np.array([])

    angles = []
    for i in range(len(ca_coords) - 2):
        v1 = ca_coords[i] - ca_coords[i + 1]
        v2 = ca_coords[i + 2] - ca_coords[i + 1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    return np.array(angles) if angles else np.array([])


def compute_n_ca_c_angle(coords):
    """Compute N-CA-C bond angle for each residue.

    The ideal N-CA-C angle is ~111 degrees.
    """
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
    return np.array(angles) if angles else np.array([])


def compute_ca_c_n_distance(coords):
    """Compute CA-C and C-N(next) peptide bond distances."""
    ca_c_dists = []
    c_n_dists = []
    for i in range(len(coords)):
        ca = coords[i, 3:6]
        c = coords[i, 6:9]
        ca_c_dists.append(np.linalg.norm(c - ca))
        if i < len(coords) - 1:
            n_next = coords[i + 1, 0:3]
            c_n_dists.append(np.linalg.norm(n_next - c))
    return np.array(ca_c_dists), np.array(c_n_dists)


def evaluate_samples(samples, real_dataset, output_dir):
    """Evaluate generated samples against real protein statistics.

    Parameters
    ----------
    samples : np.ndarray
        Generated backbone coords, shape (num_samples, seq_length, 9).
    real_dataset : Dataset
        Real protein dataset for comparison.
    output_dir : str
        Directory to save evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # ── Real protein statistics ──
    real_ca_dists = []
    real_angles = []
    real_n_ca_c = []
    real_ca_c = []
    real_c_n = []

    for i in range(len(real_dataset)):
        x = real_dataset.X[i]
        if not isinstance(x, np.ndarray) or x.size == 0:
            continue
        # Handle both (L, 3, 3) and (L, 9) shapes
        if x.ndim == 3 and x.shape[1] == 3 and x.shape[2] == 3:
            x = x.reshape(-1, 9)
        if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 9:
            continue
        d = compute_ca_distances(x)
        real_ca_dists.extend(d.tolist())
        a = compute_bond_angles(x)
        real_angles.extend(a.tolist())
        nca = compute_n_ca_c_angle(x)
        real_n_ca_c.extend(nca.tolist())
        cac, cn = compute_ca_c_n_distance(x)
        real_ca_c.extend(cac.tolist())
        real_c_n.extend(cn.tolist())

    real_ca_dists = np.array(real_ca_dists) if real_ca_dists else np.array([0.0])
    real_angles = np.array(real_angles) if real_angles else np.array([0.0])
    real_n_ca_c = np.array(real_n_ca_c) if real_n_ca_c else np.array([0.0])
    real_ca_c = np.array(real_ca_c) if real_ca_c else np.array([0.0])
    real_c_n = np.array(real_c_n) if real_c_n else np.array([0.0])

    results['real'] = {
        'ca_ca_dist': {'mean': float(np.mean(real_ca_dists)), 'std': float(np.std(real_ca_dists))},
        'ca_ca_ca_angle': {'mean': float(np.mean(real_angles)), 'std': float(np.std(real_angles))},
        'n_ca_c_angle': {'mean': float(np.mean(real_n_ca_c)), 'std': float(np.std(real_n_ca_c))},
        'ca_c_dist': {'mean': float(np.mean(real_ca_c)), 'std': float(np.std(real_ca_c))},
        'c_n_dist': {'mean': float(np.mean(real_c_n)), 'std': float(np.std(real_c_n))},
    }

    # Expected ideal values from protein chemistry
    # CA-CA: ~3.8 Å, N-CA-C: ~111°, CA-C: ~1.52 Å, C-N: ~1.33 Å
    logger.info("=== Real Protein Statistics ===")
    logger.info(f"  CA-CA distance: {np.mean(real_ca_dists):.3f} ± {np.std(real_ca_dists):.3f} Å (ideal: ~3.8 Å)")
    logger.info(f"  CA-CA-CA angle: {np.mean(real_angles):.1f} ± {np.std(real_angles):.1f}°")
    logger.info(f"  N-CA-C angle:   {np.mean(real_n_ca_c):.1f} ± {np.std(real_n_ca_c):.1f}° (ideal: ~111°)")
    logger.info(f"  CA-C distance:  {np.mean(real_ca_c):.3f} ± {np.std(real_ca_c):.3f} Å (ideal: ~1.52 Å)")
    logger.info(f"  C-N distance:   {np.mean(real_c_n):.3f} ± {np.std(real_c_n):.3f} Å (ideal: ~1.33 Å)")

    # ── Generated sample statistics ──
    # Generated samples are in normalized space (centered, unit variance).
    # For distance comparison, we scale back using the average std from real data.
    # For angular comparison, no scaling needed (angles are scale-invariant).
    
    # Compute average std from real proteins for rescaling
    real_stds = []
    for i in range(len(real_dataset)):
        x = real_dataset.X[i]
        if isinstance(x, np.ndarray) and x.size > 0:
            if x.ndim == 3 and x.shape[1] == 3 and x.shape[2] == 3:
                x = x.reshape(-1, 9)
            if x.ndim == 2 and x.shape[0] >= 3 and x.shape[1] >= 9:
                ca = x[:, 3:6]
                centroid = ca.mean(axis=0, keepdims=True)
                centered = x - np.tile(centroid, 3)
                std = centered.std()
                if std > 1e-6:
                    real_stds.append(std)
    avg_std = np.mean(real_stds) if real_stds else 10.0  # fallback
    logger.info(f"  Average coordinate std for rescaling: {avg_std:.2f} Å")

    gen_ca_dists = []
    gen_angles = []
    gen_n_ca_c = []
    gen_ca_c = []
    gen_c_n = []

    for i in range(len(samples)):
        # Rescale generated samples back to Angstrom space
        s = samples[i] * avg_std
        d = compute_ca_distances(s)
        gen_ca_dists.extend(d.tolist())
        a = compute_bond_angles(s)
        gen_angles.extend(a.tolist())
        nca = compute_n_ca_c_angle(s)
        gen_n_ca_c.extend(nca.tolist())
        cac, cn = compute_ca_c_n_distance(s)
        gen_ca_c.extend(cac.tolist())
        gen_c_n.extend(cn.tolist())

    gen_ca_dists = np.array(gen_ca_dists) if gen_ca_dists else np.array([0.0])
    gen_angles = np.array(gen_angles) if gen_angles else np.array([0.0])
    gen_n_ca_c = np.array(gen_n_ca_c) if gen_n_ca_c else np.array([0.0])
    gen_ca_c = np.array(gen_ca_c) if gen_ca_c else np.array([0.0])
    gen_c_n = np.array(gen_c_n) if gen_c_n else np.array([0.0])

    results['generated'] = {
        'ca_ca_dist': {'mean': float(np.mean(gen_ca_dists)), 'std': float(np.std(gen_ca_dists))},
        'ca_ca_ca_angle': {'mean': float(np.mean(gen_angles)), 'std': float(np.std(gen_angles))},
        'n_ca_c_angle': {'mean': float(np.mean(gen_n_ca_c)), 'std': float(np.std(gen_n_ca_c))},
        'ca_c_dist': {'mean': float(np.mean(gen_ca_c)), 'std': float(np.std(gen_ca_c))},
        'c_n_dist': {'mean': float(np.mean(gen_c_n)), 'std': float(np.std(gen_c_n))},
    }

    logger.info("\n=== Generated Sample Statistics ===")
    logger.info(f"  CA-CA distance: {np.mean(gen_ca_dists):.3f} ± {np.std(gen_ca_dists):.3f} Å")
    logger.info(f"  CA-CA-CA angle: {np.mean(gen_angles):.1f} ± {np.std(gen_angles):.1f}°")
    logger.info(f"  N-CA-C angle:   {np.mean(gen_n_ca_c):.1f} ± {np.std(gen_n_ca_c):.1f}°")
    logger.info(f"  CA-C distance:  {np.mean(gen_ca_c):.3f} ± {np.std(gen_ca_c):.3f} Å")
    logger.info(f"  C-N distance:   {np.mean(gen_c_n):.3f} ± {np.std(gen_c_n):.3f} Å")

    # ── Quality scores ──
    # How close are generated distributions to real?
    def safe_rel_error(gen_val, real_val):
        if abs(real_val) < 1e-6:
            return 1.0
        return abs(gen_val - real_val) / abs(real_val)

    ca_dist_error = safe_rel_error(np.mean(gen_ca_dists), np.mean(real_ca_dists))
    angle_error = safe_rel_error(np.mean(gen_angles), np.mean(real_angles))
    n_ca_c_error = safe_rel_error(np.mean(gen_n_ca_c), np.mean(real_n_ca_c))
    ca_c_error = safe_rel_error(np.mean(gen_ca_c), np.mean(real_ca_c))
    c_n_error = safe_rel_error(np.mean(gen_c_n), np.mean(real_c_n))

    overall_quality = 1.0 - np.mean([ca_dist_error, angle_error, n_ca_c_error, ca_c_error, c_n_error])
    overall_quality = max(0, overall_quality)

    results['quality'] = {
        'ca_dist_relative_error': float(ca_dist_error),
        'angle_relative_error': float(angle_error),
        'n_ca_c_relative_error': float(n_ca_c_error),
        'ca_c_relative_error': float(ca_c_error),
        'c_n_relative_error': float(c_n_error),
        'overall_quality_score': float(overall_quality),
    }

    logger.info("\n=== Quality Assessment ===")
    logger.info(f"  CA-CA distance error:  {ca_dist_error*100:.1f}%")
    logger.info(f"  CA-CA-CA angle error:  {angle_error*100:.1f}%")
    logger.info(f"  N-CA-C angle error:    {n_ca_c_error*100:.1f}%")
    logger.info(f"  CA-C distance error:   {ca_c_error*100:.1f}%")
    logger.info(f"  C-N distance error:    {c_n_error*100:.1f}%")
    logger.info(f"  Overall quality score: {overall_quality*100:.1f}%")

    # Save results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save generated coordinates
    np.save(os.path.join(output_dir, 'generated_samples.npy'), samples)

    return results


def save_as_pdb(coords, filename, chain_id='A'):
    """Save backbone coordinates as a PDB file for visualization.

    Parameters
    ----------
    coords : np.ndarray
        Shape (num_residues, 9).
    filename : str
        Output PDB file path.
    chain_id : str
        Chain identifier.
    """
    atom_names = ['N', 'CA', 'C']
    with open(filename, 'w') as f:
        f.write(f"REMARK   Generated by RFDiffusion (DeepChem)\n")
        f.write(f"REMARK   {len(coords)} residues\n")
        atom_num = 1
        for res_idx in range(len(coords)):
            for atom_idx, atom_name in enumerate(atom_names):
                x = coords[res_idx, atom_idx * 3]
                y = coords[res_idx, atom_idx * 3 + 1]
                z = coords[res_idx, atom_idx * 3 + 2]
                # Standard PDB ATOM format
                f.write(f"ATOM  {atom_num:5d}  {atom_name:<3s} ALA {chain_id}{res_idx+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n")
                atom_num += 1
        f.write("TER\nEND\n")


def main():
    import torch

    logger.info("=" * 70)
    logger.info("RFDiffusion Training on Expanded CATH Dataset")
    logger.info("=" * 70)

    # ── Configuration ──
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'cath_expanded_data')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'rfdiffusion_results')
    MAX_LENGTH = 256  # Max protein length (K80s have limited memory)

    # Model hyperparameters
    EMBED_DIM = 128
    NUM_LAYERS = 6
    NUM_HEADS = 8
    NUM_DIFFUSION_STEPS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 200
    GENERATE_SEQ_LEN = 50
    NUM_GENERATE = 10

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_str}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Download and featurize ──
    logger.info("\n--- Step 1: Downloading PDB structures ---")
    pdb_files, ids = download_pdbs(EXPANDED_PDB_IDS, DATA_DIR)
    logger.info(f"Downloaded {len(pdb_files)} / {len(EXPANDED_PDB_IDS)} PDB files")

    logger.info("\n--- Step 2: Featurizing proteins ---")
    dataset, lengths = create_dataset(pdb_files, ids, max_length=MAX_LENGTH)
    logger.info(f"Dataset size: {len(dataset)} proteins")

    # Split into train/valid/test
    splitter = dc.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)

    logger.info(f"Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # ── Step 3: Create model ──
    logger.info("\n--- Step 3: Creating RFDiffusion model ---")
    model = dc.models.RFDiffusionModel(
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_seq_len=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=torch.device(device_str),
    )

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── Step 4: Train ──
    logger.info(f"\n--- Step 4: Training for {NUM_EPOCHS} epochs ---")
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        loss = model.fit(train, nb_epoch=1)
        epoch_time = time.time() - epoch_start
        loss_history.append(float(loss))

        if loss < best_loss:
            best_loss = loss
            # Save best model
            model_dir = os.path.join(OUTPUT_DIR, 'best_model')
            os.makedirs(model_dir, exist_ok=True)
            model.save_checkpoint(model_dir=model_dir)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS}: loss={loss:.6f} "
                        f"best={best_loss:.6f} ({epoch_time:.1f}s)")

    total_time = time.time() - start_time
    logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
    logger.info(f"Final loss: {loss_history[-1]:.6f}, Best loss: {best_loss:.6f}")
    logger.info(f"Loss reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")

    # ── Step 5: Generate samples ──
    logger.info(f"\n--- Step 5: Generating {NUM_GENERATE} samples (length={GENERATE_SEQ_LEN}) ---")

    # Load best model
    model.restore(model_dir=os.path.join(OUTPUT_DIR, 'best_model'))

    gen_start = time.time()
    samples = model.generate(num_samples=NUM_GENERATE, seq_length=GENERATE_SEQ_LEN)
    gen_time = time.time() - gen_start
    logger.info(f"Generated {NUM_GENERATE} samples in {gen_time:.1f}s")
    logger.info(f"Sample shape: {samples.shape}")

    # Save generated structures as PDB files
    # First compute rescaling factor from training data
    real_stds = []
    for i in range(len(train)):
        x = train.X[i]
        if isinstance(x, np.ndarray) and x.size > 0:
            if x.ndim == 3 and x.shape[1] == 3 and x.shape[2] == 3:
                x = x.reshape(-1, 9)
            if x.ndim == 2 and x.shape[0] >= 3 and x.shape[1] >= 9:
                ca = x[:, 3:6]
                centroid = ca.mean(axis=0, keepdims=True)
                centered = x - np.tile(centroid, 3)
                std = centered.std()
                if std > 1e-6:
                    real_stds.append(std)
    avg_std = np.mean(real_stds) if real_stds else 10.0

    pdb_dir = os.path.join(OUTPUT_DIR, 'generated_pdbs')
    os.makedirs(pdb_dir, exist_ok=True)
    for i in range(len(samples)):
        # Save both raw (normalized) and rescaled versions
        rescaled = samples[i] * avg_std
        save_as_pdb(rescaled, os.path.join(pdb_dir, f'generated_{i+1}.pdb'))
    logger.info(f"Saved {len(samples)} PDB files to {pdb_dir} (rescaled by {avg_std:.2f})")

    # ── Step 6: Evaluate ──
    logger.info(f"\n--- Step 6: Evaluating generated samples ---")
    results = evaluate_samples(samples, train, OUTPUT_DIR)

    # Save training history
    history = {
        'loss_history': loss_history,
        'total_training_time_sec': total_time,
        'config': {
            'embed_dim': EMBED_DIM,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'num_diffusion_steps': NUM_DIFFUSION_STEPS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'train_size': len(train),
            'valid_size': len(valid),
            'test_size': len(test),
            'total_params': total_params,
            'device': device_str,
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Dataset: {len(dataset)} proteins from CATH")
    logger.info(f"Training: {NUM_EPOCHS} epochs, final loss = {loss_history[-1]:.6f}")
    logger.info(f"Loss reduction: {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
    logger.info(f"Generated: {NUM_GENERATE} protein backbones of length {GENERATE_SEQ_LEN}")
    logger.info(f"Quality score: {results['quality']['overall_quality_score']*100:.1f}%")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
