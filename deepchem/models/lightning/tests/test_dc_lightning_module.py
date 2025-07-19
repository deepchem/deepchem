import pytest
import numpy as np
import deepchem as dc
from copy import deepcopy
from pathlib import Path
try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False
pytestmark = pytest.mark.skipif(not gpu_available,
                                reason="No GPU available for testing")
try:
    import lightning as L
    from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule
    from deepchem.models.lightning.dc_lightning_module import DCLightningModule
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

pytestmark = pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                                reason="PyTorch Lightning is not installed")

np.random.seed(42)
torch.manual_seed(42)
L.seed_everything(42)


@pytest.fixture(scope="function")
def gcn_data():
    """
    Fixture to load the BACE dataset for a GCNModel.
    This runs only once per test module, saving time.
    """
    from deepchem.models.tests.test_graph_models import get_dataset
    from deepchem.feat import MolGraphConvFeaturizer
    tasks, dataset, transformers, metric = get_dataset(
        'classification', featurizer=MolGraphConvFeaturizer())
    dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                             dataset.ids)

    # Using the validation set for faster testing, as in the reference file
    return {
        "dataset": dataset,
        "n_tasks": tasks,
        "transformers": transformers,
        "metric": metric
    }


@pytest.fixture(scope="function")
def gcn_model(gcn_data):
    """
    Fixture to create a fresh GCNModel for each test function.
    This ensures tests are independent and don't share a trained state.
    """
    tasks = gcn_data["n_tasks"]
    return dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )


@pytest.mark.torch
def test_gcn_fit_predict_workflow(gcn_model, gcn_data):
    """
    Tests if the fit and predict workflow works for a GCNModel.
    This validates the custom DataModule's collate function and the LightningModule's
    training and prediction steps with complex graph data.
    """
    dataset = gcn_data["dataset"]

    # Setup DataModule and LightningModule
    data_module = DCLightningDatasetModule(dataset=dataset,
                                           batch_size=10,
                                           model=gcn_model)
    lightning_model = DCLightningModule(dc_model=gcn_model)

    # Setup Trainer
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=-1,  # use all available GPUs
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
    )

    # Test fit
    trainer.fit(model=lightning_model, datamodule=data_module)

    # Test predict
    prediction_batches = trainer.predict(model=lightning_model,
                                         datamodule=data_module)

    # The trainer returns a list of outputs from each batch, so we concatenate them
    predictions = np.concatenate([p for p in prediction_batches])

    # Verify prediction output
    assert isinstance(prediction_batches, list)
    assert len(prediction_batches) > 0

    assert isinstance(predictions, np.ndarray)
    # The final prediction shape should be (n_samples, n_tasks, n_classes=2)
    assert predictions.shape == (10, 2, 2)


@pytest.mark.torch
def test_gcn_checkpointing_and_loading(gcn_model, gcn_data, tmp_path="temp"):
    """
    Tests that a GCNModel can be saved via a checkpoint and reloaded correctly.
    It verifies that the model state is identical before saving and after loading.
    """
    dataset = gcn_data["dataset"]

    # Setup modules and trainer with a temporary directory for checkpoints
    data_module = DCLightningDatasetModule(dataset=dataset,
                                           batch_size=10,
                                           model=gcn_model)
    lightning_model = DCLightningModule(dc_model=gcn_model)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=-1,  # use all available GPUs
        default_root_dir=str(tmp_path),  # Save checkpoints to a temp dir
        enable_progress_bar=False,
    )

    # Store model state *before* training for comparison
    state_before_training = deepcopy(lightning_model.pt_model.state_dict())

    # Train the model for one epoch, which will create a checkpoint
    trainer.fit(model=lightning_model, datamodule=data_module)

    # Get model state *after* training
    state_after_training = lightning_model.pt_model.state_dict()

    # --- Correctness Check 1: Before Saving ---
    # Verify that training actually changed the model's weights
    weight_changed = False
    for key in state_before_training:
        if not torch.allclose(state_before_training[key].detach().cpu(),
                              state_after_training[key].detach().cpu()):
            weight_changed = True
            break
    assert weight_changed, "Model weights did not change after one epoch of training."

    # Find the saved checkpoint file
    checkpoint_dir = Path(
        tmp_path) / "lightning_logs" / "version_0" / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    assert len(checkpoint_files) > 0, "Checkpoint file was not created."
    ckpt_path = checkpoint_files[0]

    # Load the model from the checkpoint
    # This is the standard Lightning way to reload a model
    reloaded_model = DCLightningModule.load_from_checkpoint(ckpt_path,
                                                            dc_model=gcn_model)
    state_reloaded = reloaded_model.pt_model.state_dict()

    # --- Correctness Check 2: After Loading ---
    # Verify that the reloaded state dict has the same keys
    assert state_after_training.keys() == state_reloaded.keys()

    # Verify that the reloaded weights are identical to the saved weights
    for key in state_after_training:
        torch.testing.assert_close(
            state_after_training[key].detach().cpu(),
            state_reloaded[key].detach().cpu(),
            msg=f"Weight mismatch for key {key} after reloading.",
        )

    # --- Correctness Check 3: Functional Equivalence ---
    # Predict with both models and compare results to ensure they are identical
    original_preds_batches = trainer.predict(lightning_model,
                                             datamodule=data_module)
    reloaded_preds_batches = trainer.predict(reloaded_model,
                                             datamodule=data_module)

    original_preds = np.concatenate([p[0] for p in original_preds_batches])
    reloaded_preds = np.concatenate([p[0] for p in reloaded_preds_batches])

    np.testing.assert_allclose(
        original_preds,
        reloaded_preds,
        err_msg="Predictions from original and reloaded models do not match.",
    )
