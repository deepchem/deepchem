import pytest
import numpy as np
from flaky import flaky

try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


def rotation_matrix(axis, angle):
    """Generate a 3D rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[
        a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)
    ], [
        2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)
    ], [
        2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c
    ]])


def apply_rotation(x, axis, angle):
    """Apply a 3D rotation to the positions."""
    R = rotation_matrix(axis, angle)
    return torch.tensor(np.dot(x.numpy(), R), dtype=torch.float32)


@flaky
@pytest.mark.torch
def test_tfn_overfitting():
    """
    Test overfitting for the TFN model
    """
    from rdkit import Chem
    from deepchem.models.torch_models import TFNModel
    import numpy as np
    import deepchem as dc
    import shutil
    import os

    smiles = ['C#C', 'C#N']
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=False,
                                                    embeded=True)

    features = featurizer.featurize(mols)
    assert len(features) == 2

    labels = np.array([[
        -0.2845,
    ], [
        -0.3604,
    ]], dtype=np.float32)

    weights = np.ones_like(labels, dtype=np.float32)

    dataset = dc.data.NumpyDataset(X=features, y=labels, w=weights)

    model = TFNModel(
        num_layers=7,
        atom_feature_size=6,
        num_workers=4,
        num_channels=32,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        batch_size=12,
    )

    loss = model.fit(dataset, nb_epoch=200)
    preds = model.predict(dataset).reshape(-1)

    if os.path.exists("cache"):
        shutil.rmtree("cache")

    assert loss < 1e-02
    assert np.allclose(labels, preds, atol=0.15)


@pytest.mark.torch
def test_tfn_equivariance():
    """
    Test rotation equivariance for the TFN model
    """
    from rdkit import Chem
    from deepchem.models.torch_models import TFNModel
    import numpy as np
    import deepchem as dc
    import shutil
    import os

    smiles = ['C#C', 'C#N']
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=False,
                                                    embeded=True)

    features = featurizer.featurize(mols)
    assert len(features) == 2

    labels = np.array([[
        -0.2845,
    ], [
        -0.3604,
    ]], dtype=np.float32)

    weights = np.ones_like(labels, dtype=np.float32)

    dataset = dc.data.NumpyDataset(X=features, y=labels, w=weights)

    model = TFNModel(
        num_layers=7,
        atom_feature_size=6,
        num_workers=4,
        num_channels=32,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        batch_size=12,
    )

    _ = model.fit(dataset, nb_epoch=10)
    preds = model.predict(dataset) 

    axis = np.array([1.0, 1.0, 1.0])  # Rotate around (1,1,1) axis
    angle = np.pi / 4  # 45-degree rotation
    new_coords = apply_rotation(
        torch.tensor([i.edge_features for i in features]), axis, angle).numpy()
    for i, feats in enumerate(features):
        feats.edge_features = new_coords[i]

    # Since in this test predictions are scalar values, checking for equivariance
    # is the same as checking for invariance.
    preds_rot = model.predict(dataset)

    if os.path.exists("cache"):
        shutil.rmtree("cache")

    assert np.allclose(preds_rot, preds, atol=1e-05)


@pytest.mark.torch
def test_tfn_save_restore():
    """
    Test saving and restoring the TFN model
    """
    from deepchem.models.torch_models import TFNModel
    import tempfile
    from rdkit import Chem
    import deepchem as dc
    import os
    import shutil

    # Generate random data for testing model saving and loading
    smiles = ['C#C', 'C#N']
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    featurizer = dc.feat.EquivariantGraphFeaturizer(fully_connected=False,
                                                    embeded=True)

    features = featurizer.featurize(mols)
    assert len(features) == 2

    labels = np.array([[
        -0.2845,
    ], [
        -0.3604,
    ]], dtype=np.float32)

    weights = np.ones_like(labels, dtype=np.float32)

    dataset = dc.data.NumpyDataset(X=features, y=labels, w=weights)
    model_dir = tempfile.mkdtemp()

    model = TFNModel(
        num_layers=7,
        atom_feature_size=6,
        num_workers=4,
        num_channels=32,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        pooling='max',
        model_dir=model_dir,
    )
    # Initialize model and set a temporary directory for saving

    # Train and get predictions from the model
    model.fit(dataset, nb_epoch=1)
    pred_before_restore = model.predict(dataset)

    # Save and restore model, then compare predictions
    model.save()
    reloaded_model = TFNModel(
        num_layers=7,
        atom_feature_size=6,
        num_workers=4,
        num_channels=32,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        batch_size=12,
        model_dir=model_dir,
    )
    reloaded_model.restore()
    pred_after_restore = reloaded_model.predict(dataset)

    if os.path.exists("cache"):
        shutil.rmtree("cache")

    # Ensure predictions before and after restoring are close
    assert np.allclose(pred_before_restore, pred_after_restore, atol=1e-05)
