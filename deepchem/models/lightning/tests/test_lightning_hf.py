import pytest
import numpy as np
import deepchem as dc
from deepchem.data import DiskDataset
from copy import deepcopy
import os
try:
    import torch
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
except ImportError:
    gpu_available = False

try:
    import lightning as L
    from deepchem.models.lightning.trainer import LightningTorchModel
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True

try:
    from deepchem.models.torch_models.chemberta import Chemberta
    os.environ[
        "TOKENIZERS_PARALLELISM"] = "false"  # to avoid deadlocks in tokenization due to parallel processing already done by dataloader.
    CHEMBERTA_IMPORT_FAILED = False
except ImportError:
    CHEMBERTA_IMPORT_FAILED = True

pytestmark = [
    pytest.mark.skipif(not gpu_available,
                       reason="No GPU available for testing"),
    pytest.mark.skipif(PYTORCH_LIGHTNING_IMPORT_FAILED,
                       reason="PyTorch Lightning is not installed")
]


@pytest.fixture(scope="function")
def smiles_data(tmp_path_factory):
    """
    Fixture to create a small SMILES dataset for ChemBERTa testing.
    """
    # Small set of SMILES strings for testing
    smiles = [
        'CCO', 'CCC', 'CC(C)O', 'C1=CC=CC=C1', 'CCN(CC)CC', 'CC(=O)O',
        'C1=CC=C(C=C1)O', 'CCO[Si](OCC)(OCC)OCC', 'CC(C)(C)O', 'CC(C)C',
        'CCC(C)C', 'CCCC', 'CCCCC', 'CCCCCC', 'CC(C)CC', 'CCC(C)(C)C'
    ]

    # molecular weight predictions
    labels = [
        46.07, 44.10, 60.10, 78.11, 101.19, 60.05, 94.11, 208.33, 74.12, 58.12,
        72.15, 58.12, 72.15, 86.18, 72.15, 86.18
    ]

    data_dir = tmp_path_factory.mktemp("dataset")

    dataset = DiskDataset.from_numpy(X=np.array(smiles),
                                     y=np.array(labels).reshape(-1, 1),
                                     w=np.ones(len(smiles)),
                                     ids=np.arange(len(smiles)),
                                     data_dir=str(data_dir))

    return {"dataset": dataset}


@pytest.mark.torch
def test_chemberta_masked_lm_workflow(smiles_data):
    dataset = smiles_data["dataset"]
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    dc_hf_model = Chemberta(task='mlm',
                            tokenizer_path=tokenizer_path,
                            device='cpu',
                            batch_size=2,
                            learning_rate=0.0001)

    # Setup LightningTorchModel for MLM pretraining with FSDP
    trainer = LightningTorchModel(
        model=dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="fsdp",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
    )

    # Test MLM training
    trainer.fit(train_dataset=dataset, checkpoint_interval=0, nb_epoch=1)

    trainer.save_checkpoint(model_dir="chemberta_mlm_checkpoint")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='mlm',
                                tokenizer_path=tokenizer_path,
                                device='cpu',
                                batch_size=2,
                                learning_rate=0.0001)

    # Load the checkpoint into the new model instance
    reloaded_trainer = LightningTorchModel(
        model=new_dc_hf_model,
        batch_size=2,
        model_dir="chemberta_mlm_checkpoint",
        accelerator="gpu",
        devices=1,
    )
    reloaded_trainer.restore()

    # Test MLM prediction using the reloaded trainer
    prediction_batches = reloaded_trainer.predict(dataset=dataset,
                                                  num_workers=0)

    # Verify prediction output
    assert len(prediction_batches) > 0

    # For MLM, predictions should be token logits
    if prediction_batches and prediction_batches[0] is not None:
        predictions = prediction_batches[0]
        assert isinstance(predictions, np.ndarray)


@pytest.mark.torch
def test_chemberta_regression_workflow(smiles_data):
    dataset = smiles_data["dataset"]

    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    dc_hf_model = Chemberta(task='regression',
                            tokenizer_path=tokenizer_path,
                            device='cpu',
                            batch_size=2,
                            learning_rate=0.0001)
    # Setup LightningTorchModel for regression training with FSDP
    trainer = LightningTorchModel(
        model=dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=-1,
        strategy="fsdp",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
    )

    # Test regression training
    trainer.fit(train_dataset=dataset,
                num_workers=4,
                checkpoint_interval=0,
                nb_epoch=1)
    trainer.save_checkpoint(model_dir="chemberta_reg_checkpoint")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='regression',
                                tokenizer_path=tokenizer_path,
                                device='cpu',
                                batch_size=2,
                                learning_rate=0.0001)

    # Load the checkpoint into the new model instance
    reloaded_trainer = LightningTorchModel(
        model=new_dc_hf_model,
        batch_size=2,
        model_dir="chemberta_reg_checkpoint",
        accelerator="gpu",
        devices=1,
    )
    reloaded_trainer.restore()

    # Test regression prediction using the reloaded trainer
    prediction = reloaded_trainer.predict(dataset=dataset)

    assert len(prediction) > 0

    predictions = np.concatenate(prediction)
    assert isinstance(predictions, np.ndarray)

    # For regression, predictions should match the number of samples and tasks
    assert predictions.shape == (16,)  # single regression task


@pytest.mark.torch
def test_chemberta_classification_workflow(smiles_data, tmp_path):
    dataset = smiles_data["dataset"]

    # Convert regression labels to binary classification
    y_binary = (dataset.y > np.median(dataset.y)).astype(int)

    # Create DiskDataset for classification instead of NumpyDataset
    classification_dataset = DiskDataset.from_numpy(
        X=dataset.X,
        y=y_binary,
        w=dataset.w,
        ids=dataset.ids,
        data_dir=str(tmp_path / "classification_data"))

    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    # Load ChemBERTa model for binary classification using Chemberta
    dc_hf_model = Chemberta(task='classification',
                            tokenizer_path=tokenizer_path,
                            device='cpu',
                            batch_size=2,
                            learning_rate=0.0001)

    # Setup LightningTorchModel for classification training with FSDP
    trainer = LightningTorchModel(
        model=dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=-1,
        strategy="fsdp",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        fast_dev_run=True,
        precision="16-mixed",
    )

    # Test classification training
    trainer.fit(train_dataset=classification_dataset,
                num_workers=4,
                checkpoint_interval=0,
                nb_epoch=1)

    trainer.save_checkpoint(model_dir="chemberta_classification_checkpoint")

    # Create a new model instance for loading
    new_dc_hf_model = Chemberta(task='classification',
                                tokenizer_path=tokenizer_path,
                                device='cpu',
                                batch_size=2,
                                learning_rate=0.0001)

    # Load the checkpoint into the new model instance
    reloaded_trainer = LightningTorchModel(
        model=new_dc_hf_model,
        batch_size=2,
        model_dir="chemberta_classification_checkpoint",
        accelerator="gpu",
        devices=1,
    )
    reloaded_trainer.restore()

    # Test classification prediction using the reloaded trainer
    prediction = reloaded_trainer.predict(dataset=classification_dataset)

    assert len(prediction) > 0

    assert isinstance(prediction, np.ndarray)

    # For binary classification, predictions should be probabilities for 2 classes
    assert prediction.shape[1] == 2  # binary classification probabilities


@pytest.mark.torch
def test_chemberta_checkpointing_and_loading(smiles_data):
    dataset = smiles_data["dataset"]
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"

    # Load ChemBERTa model for regression using Chemberta
    dc_hf_model = Chemberta(task='regression',
                            tokenizer_path=tokenizer_path,
                            device='cpu',
                            batch_size=2,
                            learning_rate=0.0001)

    # Setup LightningTorchModel
    trainer = LightningTorchModel(
        model=dc_hf_model,
        batch_size=2,
        accelerator="gpu",
        devices=-1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
    )

    # Store model state before training for comparison
    state_before_training = deepcopy(
        trainer.lightning_model.pt_model.state_dict())

    # Train the model for one epoch, which will create a checkpoint
    trainer.fit(train_dataset=dataset, checkpoint_interval=0, nb_epoch=3)

    # Get model state after training
    state_after_training = trainer.lightning_model.pt_model.state_dict()

    # --- Correctness Check 1: Before Saving ---
    # Verify that training actually changed the model's weights
    weight_changed = False
    for key in state_before_training:
        if not torch.allclose(state_before_training[key].detach().cpu(),
                              state_after_training[key].detach().cpu(),
                              rtol=1e-4,
                              atol=1e-6):
            weight_changed = True
            break
    assert weight_changed, "Model weights did not change (no training occurred)."

    trainer.save_checkpoint(model_dir="model_checkpoint")

    # Create a new model instance for loading
    dc_hf_model_new = Chemberta(task='regression',
                                tokenizer_path=tokenizer_path,
                                device='cpu',
                                batch_size=2,
                                learning_rate=0.0001)

    # Load the model from the checkpoint using LightningTorchModel
    reloaded_trainer = LightningTorchModel(model=dc_hf_model_new,
                                           batch_size=2,
                                           model_dir="model_checkpoint",
                                           accelerator="gpu",
                                           devices=-1,
                                           logger=False,
                                           enable_progress_bar=False)
    reloaded_trainer.restore()
    state_reloaded = reloaded_trainer.lightning_model.pt_model.state_dict()

    # --- Correctness Check 2: After Loading ---
    # Verify that the reloaded state dict has the same keys
    assert state_after_training.keys() == state_reloaded.keys()

    # Verify that the reloaded weights are identical to the saved weights
    for key in state_after_training:
        torch.testing.assert_close(
            state_after_training[key].detach().cpu(),
            state_reloaded[key].detach().cpu(),
            msg=f"Weight mismatch for key {key} after reloading.",
            rtol=1e-4,
            atol=1e-6)

    # --- Correctness Check 3: Functional Equivalence ---
    # Predict with both models and compare results to ensure they are identical
    original_preds = trainer.predict(dataset=dataset, num_workers=0)
    reloaded_preds = reloaded_trainer.predict(dataset=dataset, num_workers=0)

    np.testing.assert_allclose(
        original_preds,
        reloaded_preds,
        err_msg="Predictions from original and reloaded models do not match.",
        rtol=1e-4,
        atol=1e-6)


@pytest.mark.torch
def test_chemberta_overfit_with_lightning_trainer(smiles_data):
    L.seed_everything(42)

    dataset = smiles_data["dataset"]
    tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_60k"
    mae_metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

    # Create ChemBERTa model for classification
    dc_hf_model = Chemberta(
        task='regression',
        tokenizer_path=tokenizer_path,
        device='cpu',
        batch_size=1,  # Smaller batch size to ensure all samples are processed
        learning_rate=0.0005)

    # Create Lightning trainer
    lightning_trainer = LightningTorchModel(
        model=dc_hf_model,
        batch_size=1,
        accelerator="gpu",
        strategy="fsdp",
        devices=-1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        precision="16-mixed",
    )

    lightning_trainer.fit(train_dataset=dataset,
                          checkpoint_interval=0,
                          nb_epoch=70)

    # Save checkpoint after training
    lightning_trainer.save_checkpoint(model_dir="chemberta_overfit_best")

    new_dc_hf_model = Chemberta(
        task='regression',
        tokenizer_path=tokenizer_path,
        device='cpu',
        batch_size=1,  # Match model batch size
        learning_rate=0.0005)

    eval_before = new_dc_hf_model.evaluate(dataset=dataset,
                                           metrics=[mae_metric])

    # Load the checkpoint into the new model instance
    reloaded_trainer = LightningTorchModel(
        model=new_dc_hf_model,
        batch_size=1,
        model_dir="chemberta_overfit_best",
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
    )
    reloaded_trainer.restore()

    # Evaluate the model on the training set
    eval_score = reloaded_trainer.evaluate(dataset=dataset,
                                           metrics=[mae_metric])
    # If the model overfits the mae score should be significantly lower than before training
    assert eval_before[mae_metric.name] > eval_score[
        mae_metric.name] * 2, "Model did not overfit as expected"
