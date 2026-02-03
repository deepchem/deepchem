import pytest
import numpy as np
from deepchem.data import NumpyDataset

try:
    from transformers import BertConfig
    from deepchem.models.torch_models.geneformer import DeepChemGeneformer

    has_pytorch_and_transformers = True
except ImportError:
    has_pytorch_and_transformers = False


@pytest.mark.torch
def test_geneformer_overfit():
    """Test that DeepChemGeneformer can overfit a small dataset."""
    if not has_pytorch_and_transformers:
        pytest.skip("PyTorch and Transformers are required for this test.")

    n_samples = 10
    n_genes = 200

    X = np.random.rand(n_samples, n_genes).astype(np.float32)

    # Edge Case: Add a sample with all zero gene expression values
    X[-1, :] = 0.0

    # Rank-Value Verification-Identify index of highest value gene for the first sample
    highest_gene_idx = np.argmax(X[0])

    y = np.random.randint(0, 2, size=(n_samples, 1)).astype(np.float32)

    dataset = NumpyDataset(X, y)

    # Create a dummy config
    config = BertConfig(
        vocab_size=n_genes,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=512,
    )

    model = DeepChemGeneformer(
        hf_model_name="dummy",
        model_config=config,
        n_tasks=1,
        mode="classification",
        learning_rate=0.01,
        batch_size=2,
    )

    loss = model.fit(dataset, nb_epoch=20)

    assert isinstance(loss, float), "Loss should be a float value"

    preds = model.predict(dataset)

    assert preds.shape == (
        n_samples,
        2,
    ), f"Expected prediction shape {(n_samples, 2)}, but got {preds.shape}"

    # Check RankV alue encoding logic via default_generator
    gen = model.default_generator(dataset)
    inputs, labels, weights = next(gen)
    # inputs[0] is input_ids, inputs[1] is attention_mask
    assert (inputs[0].shape[0] == 2
           ), f"Expected batch size 2, but got {inputs[0].shape[0]}"
    assert (inputs[0].dtype == np.int64
           ), f"Expected input_ids dtype int64, but got {inputs[0].dtype}"

    # Verify that the first token ID of the first sample matches the highest-value gene index
    assert (
        inputs[0][0, 0] == highest_gene_idx
    ), f"Expected first token ID {highest_gene_idx}, but got {inputs[0][0, 0]}"


@pytest.mark.torch
def test_geneformer_reload(tmp_path):
    """Test that DeepChemGeneformer can be saved and reloaded."""
    if not has_pytorch_and_transformers:
        pytest.skip("PyTorch and Transformers are required for this test.")

    n_samples = 5
    n_genes = 50
    n_tasks = 1
    X = np.random.rand(n_samples, n_genes).astype(np.float32)
    y = np.random.rand(n_samples, n_tasks).astype(np.float32)
    dataset = NumpyDataset(X, y)

    config = BertConfig(
        vocab_size=100,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )

    model = DeepChemGeneformer(
        hf_model_name="dummy",
        model_config=config,
        n_tasks=n_tasks,
        mode="regression",
        model_dir=str(tmp_path),
    )

    model.fit(dataset, nb_epoch=1)
    pred_original = model.predict(dataset)

    # Save and reload
    model.save_checkpoint()

    model2 = DeepChemGeneformer(
        hf_model_name="dummy",
        model_config=config,
        n_tasks=n_tasks,
        mode="regression",
        model_dir=str(tmp_path),
    )
    model2.restore()

    pred_reloaded = model2.predict(dataset)

    # Shape validation
    assert pred_reloaded.shape == (
        n_samples,
        n_tasks,
    ), f"Expected prediction shape {(n_samples, n_tasks)}, but got {pred_reloaded.shape}"

    # Numeric stability check
    assert np.allclose(
        pred_original, pred_reloaded, atol=1e-5
    ), "Predictions after reloading do not match original predictions"
