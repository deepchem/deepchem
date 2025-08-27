import pytest
import deepchem as dc
import numpy as np
import os
import shutil
try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

pytestmark = [
    pytest.mark.skipif(not gpu_available,
                       reason="No GPU available for testing"),
    pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                       reason="PyTorch Lightning is not installed")
]


@pytest.mark.torch
def test_manual_save_restore():
    L.seed_everything(42)
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    # Create first model and trainer
    model1 = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                           n_features=1024,
                                           layer_sizes=[1000],
                                           dropouts=0.2,
                                           learning_rate=0.0001,
                                           device="cpu",
                                           batch_size=16)

    # Use a specific model_dir for this test to avoid conflicts
    trainer1 = LightningTorchModel(
        model=model1,
        batch_size=16,
        accelerator="cuda",
        devices=-1,
        log_every_n_steps=1,
    )

    # Train first model
    trainer1.fit(valid_dataset, nb_epoch=3,
                 checkpoint_interval=0)  # disable auto checkpointing

    trainer1.save_checkpoint(model_dir="test_dir")

    y1 = trainer1.predict(valid_dataset)

    # Create second model and trainer with same model_dir
    model2 = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                           n_features=1024,
                                           layer_sizes=[1000],
                                           dropouts=0.2,
                                           learning_rate=0.0001,
                                           device="cpu",
                                           batch_size=16)

    trainer2 = LightningTorchModel(model=model2,
                                   batch_size=16,
                                   model_dir="test_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1)

    # Restore from specific checkpoint name
    trainer2.restore()

    # Now they should produce similar results
    y2 = trainer2.predict(valid_dataset)
    assert np.allclose(y1, y2, atol=1e-3)

    # Clean up
    try:
        if os.path.exists("test_dir"):
            shutil.rmtree("test_dir")
    except:
        pass  # Ignore cleanup errors


@pytest.mark.torch
def test_multitask_classifier_restore_correctness():
    L.seed_everything(42)
    tasks, datasets, _ = dc.molnet.load_clintox()
    _, valid_dataset, _ = datasets

    model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                          n_features=1024,
                                          layer_sizes=[1000],
                                          dropouts=0.2,
                                          learning_rate=0.0001,
                                          device="cpu",
                                          batch_size=16)

    trainer = LightningTorchModel(model=model,
                                  batch_size=16,
                                  model_dir="test_multitask_restore_dir",
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp")

    trainer.fit(valid_dataset, nb_epoch=3)

    # Store original model state dictionary for comparison
    original_state_dict = trainer.model.model.state_dict()

    # Create new model and trainer instance
    restore_model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                                  n_features=1024,
                                                  layer_sizes=[1000],
                                                  dropouts=0.2,
                                                  learning_rate=0.0001,
                                                  device="cpu",
                                                  batch_size=16)

    trainer2 = LightningTorchModel(model=restore_model,
                                   batch_size=16,
                                   model_dir="test_multitask_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1,
                                   strategy="fsdp")

    # Restore from checkpoint
    trainer2.restore()

    # Run prediction to ensure model is properly loaded
    _ = trainer2.predict(valid_dataset)

    # Get restored model state dictionary for comparison
    restored_state_dict = trainer2.model.model.state_dict()

    # Verify that the restored state dict has the same keys
    assert original_state_dict.keys() == restored_state_dict.keys()

    # Verify that the restored weights are identical to the original weights
    for key in original_state_dict:
        torch.testing.assert_close(
            original_state_dict[key].detach().cpu(),
            restored_state_dict[key].detach().cpu(),
            msg=f"Weight mismatch for key {key} after restore operation.")

    # Clean up
    try:
        if os.path.exists("test_multitask_restore_dir"):
            shutil.rmtree("test_multitask_restore_dir")
    except:
        pass  # Ignore cleanup errors caused by file locks


@pytest.mark.torch
def test_gcn_model_restore_correctness():
    L.seed_everything(42)
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks, all_dataset, _ = dc.molnet.load_bace_classification(featurizer)
    _, valid_dataset, _ = all_dataset

    model = dc.models.GCNModel(mode='classification',
                               n_tasks=len(tasks),
                               batch_size=16,
                               learning_rate=0.001,
                               device="cpu")

    trainer = LightningTorchModel(model=model,
                                  batch_size=16,
                                  model_dir="test_gcn_restore_dir",
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp")

    trainer.fit(valid_dataset, nb_epoch=3)

    # Store original model state dictionary for comparison
    original_state_dict = trainer.model.model.state_dict()

    # Create new model and trainer instance
    restore_model = dc.models.GCNModel(mode='classification',
                                       n_tasks=len(tasks),
                                       batch_size=16,
                                       learning_rate=0.001,
                                       device="cpu")

    trainer2 = LightningTorchModel(model=restore_model,
                                   batch_size=16,
                                   model_dir="test_gcn_restore_dir",
                                   accelerator="cuda",
                                   devices=-1,
                                   log_every_n_steps=1,
                                   strategy="fsdp")

    # Restore from checkpoint - look for checkpoint1.ckpt in model_dir
    trainer2.restore()

    # Run prediction to ensure model is properly loaded
    _ = trainer2.predict(valid_dataset)

    # Get restored model state dictionary for comparison
    restored_state_dict = trainer2.model.model.state_dict()

    # Verify that the restored state dict has the same keys
    assert original_state_dict.keys() == restored_state_dict.keys()

    # Verify that the restored weights are identical to the original weights
    for key in original_state_dict:
        torch.testing.assert_close(
            original_state_dict[key].detach().cpu(),
            restored_state_dict[key].detach().cpu(),
            msg=f"Weight mismatch for key {key} after restore operation.")

    # Clean up
    try:
        if os.path.exists("test_gcn_restore_dir"):
            shutil.rmtree("test_gcn_restore_dir")
    except:
        pass  # Ignore cleanup errors caused by file locks


@pytest.mark.torch
def test_gcn_model_overfit_and_checkpointing():
    np.random.seed(42)
    torch.manual_seed(42)
    L.seed_everything(42)

    # Load the BACE dataset for GCNModel
    from deepchem.models.tests.test_graph_models import get_dataset
    from deepchem.feat import MolGraphConvFeaturizer
    tasks, dataset, transformers, metric = get_dataset(
        'classification', featurizer=MolGraphConvFeaturizer())
    dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w,
                                             dataset.ids)
    gcn_model = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=5,
        learning_rate=0.0003,
        device='cpu',
    )

    lightning_trainer = LightningTorchModel(model=gcn_model,
                                            batch_size=5,
                                            accelerator="cuda",
                                            strategy="fsdp",
                                            devices=-1,
                                            enable_checkpointing=True)

    # Train the model
    lightning_trainer.fit(dataset,
                          max_checkpoints_to_keep=3,
                          checkpoint_interval=20,
                          nb_epoch=70)

    # evaluate the checkpoints availablity 3 + 1 represents the 3 checkpoints plus the final model
    checkpoints = os.listdir(
        os.path.join(lightning_trainer.model_dir, "checkpoints"))
    assert len(
        checkpoints) == 3 + 1, "No checkpoints were created during training."

    # After training, create a new LightningTorchModel instance for prediction and load the best checkpoint

    # Create a new model instance and load weights
    gcn_model_pred = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )

    # Load weights from checkpoint using the same model_dir
    lightning_trainer_pred = LightningTorchModel(
        model=gcn_model_pred,
        batch_size=10,
        model_dir=lightning_trainer.model_dir,
        accelerator="cuda",
        logger=False,
        devices=1,
        enable_progress_bar=False)

    lightning_trainer_pred.restore()

    scores_multi = lightning_trainer_pred.evaluate(dataset, [metric],
                                                   transformers)

    assert scores_multi[
        "mean-roc_auc_score"] > 0.85, "Model did not learn anything during training."

    try:
        if os.path.exists(lightning_trainer.model_dir):
            shutil.rmtree(lightning_trainer.model_dir)
    except:
        pass  # Ignore cleanup errors caused by file locks
