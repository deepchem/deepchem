import pytest
import deepchem as dc
import numpy as np
import os
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
def test_multitask_classifier_reload_correctness():
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
                                  max_epochs=30,
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp",
                                  fast_dev_run=True)

    trainer.fit(valid_dataset)
    # get a some 10 weights for assertion
    weights = trainer.model.model.layers[0].weight[:10].detach().cpu().numpy()

    trainer.save_checkpoint("multitask_classifier.ckpt")

    # Reload model and checkpoint
    reload_model = dc.models.MultitaskClassifier(n_tasks=len(tasks),
                                                 n_features=1024,
                                                 layer_sizes=[1000],
                                                 dropouts=0.2,
                                                 learning_rate=0.0001,
                                                 device="cpu",
                                                 batch_size=16)

    trainer = LightningTorchModel.load_checkpoint("multitask_classifier.ckpt",
                                                  model=reload_model,
                                                  batch_size=16,
                                                  max_epochs=10,
                                                  accelerator="cuda",
                                                  devices=-1,
                                                  log_every_n_steps=1,
                                                  strategy="fsdp",
                                                  fast_dev_run=True)

    # get a some 10 weights for assertion
    reloaded_weights = trainer.model.model.layers[0].weight[0][:10].detach(
    ).cpu().numpy()

    _ = trainer.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)


@pytest.mark.torch
def test_gcn_model_reload_correctness():
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
                                  max_epochs=10,
                                  accelerator="cuda",
                                  devices=-1,
                                  log_every_n_steps=1,
                                  strategy="fsdp",
                                  fast_dev_run=True)

    trainer.fit(valid_dataset)

    # get a some 10 weights for assertion
    weights = trainer.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[:10].detach().cpu().numpy()

    trainer.save_checkpoint("gcn_model.ckpt")

    # Reload model and checkpoint
    reload_model = dc.models.GCNModel(mode='classification',
                                      n_tasks=len(tasks),
                                      batch_size=16,
                                      learning_rate=0.001,
                                      device="cpu")

    trainer = LightningTorchModel.load_checkpoint("gcn_model.ckpt",
                                                  model=reload_model,
                                                  batch_size=16,
                                                  max_epochs=10,
                                                  accelerator="cuda",
                                                  devices=-1,
                                                  log_every_n_steps=1,
                                                  fast_dev_run=True)

    # get a some 10 weights for assertion
    reloaded_weights = trainer.model.model.model.gnn.gnn_layers[
        0].res_connection.weight[0][:10].detach().cpu().numpy()

    _ = trainer.predict(valid_dataset)

    # make it equal with a tolerance of 1e-5
    assert torch.allclose(torch.tensor(weights),
                          torch.tensor(reloaded_weights),
                          atol=1e-5)


@pytest.mark.torch
def test_gcn_overfit_with_lightning_trainer():
    """
    Tests if the GCN model can overfit to a small dataset using Lightning trainer.
    This validates that the Lightning training loop works correctly and the model
    can learn from the data.
    """

    np.random.seed(42)  # Ensure reproducibility for numpy operations
    torch.manual_seed(42)  # Ensure reproducibility for PyTorch operations
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
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )

    lightning_trainer = LightningTorchModel(
        model=gcn_model,
        batch_size=10,
        max_epochs=70,
        accelerator="cuda",
        strategy="fsdp",
        devices=-1,
        logger=False,
        enable_progress_bar=False,
    )

    # Train the model
    lightning_trainer.fit(dataset)

    # After training, create a new LightningTorchModel instance for prediction and load the best checkpoint
    # Find the latest checkpoint
    lightning_trainer.save_checkpoint("best_model.ckpt")

    # Create a new model instance and load weights
    gcn_model_pred = dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),
        number_atom_features=30,
        batch_size=10,
        learning_rate=0.0003,
        device='cpu',
    )

    # Load weights from checkpoint
    lightning_trainer_pred = LightningTorchModel.load_checkpoint(
        "best_model.ckpt",
        devices=1,
        model=gcn_model_pred,
        batch_size=10,
        accelerator="cuda",
        logger=False,
        enable_progress_bar=False)

    scores_multi = lightning_trainer_pred.evaluate(dataset, [metric],
                                                   transformers)
    os.remove("best_model.ckpt")
    assert scores_multi[
        "mean-roc_auc_score"] > 0.85, "Model did not learn anything during training."
