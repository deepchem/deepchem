import pytest
import torch
import numpy as np
import lightning as L
import deepchem as dc

# Import the custom modules you are testing
from deepchem.models.lightning.new_dc_lightning_dataset_module import DeepChemLightningDataModule
from deepchem.models.lightning.new_dc_lightning_module import DeepChemLightningModule

# Check if a GPU is available for testing, otherwise skip GPU tests
try:
    import torch.cuda
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
except ImportError:
    gpu_available = False

pytestmark = pytest.mark.skipif(not gpu_available, reason="Tests require a GPU.")

np.random.seed(42)  # Ensure reproducibility for numpy operations
torch.manual_seed(42)  # Ensure reproducibility for PyTorch operations

L.seed_everything(42)

@pytest.fixture(scope="module")
def gcn_data():
    """
    Fixture to load the BACE dataset for a GCNModel.
    This runs only once per test module, saving time.
    """
    from deepchem.models.tests.test_graph_models import get_dataset
    from deepchem.feat import MolGraphConvFeaturizer
    tasks, dataset, transformers, metric = get_dataset('classification', featurizer=MolGraphConvFeaturizer())
    dataset = dc.data.DiskDataset.from_numpy(dataset.X, dataset.y, dataset.w, dataset.ids)
    
    # Using the validation set for faster testing, as in the reference file
    return {"dataset": dataset, "n_tasks": tasks, "transformers": transformers,"metric": metric}

@pytest.fixture(scope="function")
def gcn_model(gcn_data):
    """
    Fixture to create a fresh GCNModel for each test function.
    This ensures tests are independent and don't share a trained state.
    """
    tasks = gcn_data["n_tasks"]
    return dc.models.GCNModel(
        mode='classification',
        n_tasks=len(tasks),  # This will be 1 now
        number_atom_features=30,  # Same as reference
        batch_size=10,  # Same as reference
        learning_rate=0.0003,  # Same as reference
        device='cpu',  # Use GPU for training
    )

def test_gcn_overfit_with_lightning_trainer(gcn_model, gcn_data):
    """
    Tests if the GCN model can overfit to a small dataset using Lightning trainer.
    This validates that the Lightning training loop works correctly and the model
    can learn from the data by achieving good performance on the training set.
    Also tests the new multi-GPU prediction capability.
    """
    import lightning as L
    from deepchem.models.lightning.trainer2 import DeepChemLightningTrainer
    import shutil
    import os
    np.random.seed(42)  # Ensure reproducibility for numpy operations
    torch.manual_seed(42)  # Ensure reproducibility for PyTorch operations

    L.seed_everything(42)


    dataset = gcn_data["dataset"]
    tasks = gcn_data["n_tasks"]
    transformers = gcn_data["transformers"]
    metric = gcn_data["metric"]
    

    # Create Lightning trainer with parameters similar to reference test
    # Define a custom checkpoint directory
    checkpoint_dir = "my_custom_checkpoints"
    lightning_trainer = DeepChemLightningTrainer(
        model=gcn_model,
        batch_size=10,  # Same as reference
        max_epochs=100,  # Reduce for debugging
        accelerator="cuda",
        strategy="fsdp",
        devices=-1,  
        logger=False,
        enable_progress_bar=False,
        default_root_dir=checkpoint_dir,  # Save checkpoints to this directory
    )

    
    # Debug: Print dataset info
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Dataset y shape: {dataset.y.shape}")
    # print(f"Dataset w shape: {dataset.w.shape}")

    # Train the model
    lightning_trainer.fit(dataset)

    # After training, create a new DeepChemLightningTrainer instance for prediction and load the best checkpoint
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
    lightning_trainer_pred = DeepChemLightningTrainer.load_checkpoint("best_model.ckpt", gcn_model_pred, 10,accelerator="cuda",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=checkpoint_dir,)


    # Now test evaluation (which uses prediction internally)
    # try:
    scores_multi = lightning_trainer_pred.evaluate(dataset, [metric], transformers)
    print(f"Multi-GPU evaluation successful!")
    print(f"Multi-GPU ROC score: {scores_multi.get('mean-roc_auc_score', 'N/A')}")
    # except Exception as e:
    #     print(f"Multi-GPU evaluation failed: {e}")
    #     scores_multi = None
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.remove("best_model.ckpt")
    assert scores_multi["mean-roc_auc_score"] > 0.85, "Model did not learn anything during training."
