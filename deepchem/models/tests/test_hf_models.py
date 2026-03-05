"""Unit tests for HuggingFaceModel (deepchem/models/torch_models/hf_models.py).

Tests cover:
  - Basic model initialisation for every supported task
  - Gradient-checkpointing flag (enabled / disabled)
  - Graceful no-op when the underlying model does not support gradient
    checkpointing
  - One training step (fit) on a minimal in-memory dataset
  - Prediction on a minimal in-memory dataset

All tests use the smallest possible transformer configuration so that the
suite runs in a few seconds on CPU without network access.

Running
-------
    pytest deepchem/models/tests/test_hf_models.py -v
"""

import logging

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Optional-dependency guards – same pattern as the rest of DeepChem  #
# ------------------------------------------------------------------ #
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

try:
    import transformers  # noqa: F401
    has_transformers = True
except ImportError:
    has_transformers = False

# Both torch and transformers are required for every test in this file.
pytestmark = [
    pytest.mark.torch,
    pytest.mark.skipif(not has_torch, reason="PyTorch is not installed"),
    pytest.mark.skipif(not has_transformers,
                       reason="transformers is not installed"),
]

# ------------------------------------------------------------------ #
# Shared fixtures                                                      #
# ------------------------------------------------------------------ #


def _make_tiny_roberta_config():
    """Return a RobertaConfig with the smallest sensible architecture."""
    from transformers.models.roberta import RobertaConfig
    return RobertaConfig(
        vocab_size=128,  # tiny vocabulary – no real tokeniser needed
        hidden_size=32,  # 2 attention heads x 16
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )


def _make_smiles_dataset(n_samples: int = 4,
                         task: str = "regression",
                         n_tasks: int = 1):
    """Create a minimal NumpyDataset of dummy SMILES strings.

    Parameters
    ----------
    n_samples:
        Number of rows.
    task:
        'regression', 'classification', or 'mlm'.
    n_tasks:
        Number of label columns.
    """
    import deepchem as dc

    # Raw SMILES stored in X; DummyFeaturizer passes them through unchanged.
    smiles = np.array([
        "CCO",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CN",
        "CCN",
        "CCCO",
        "c1ccncc1",
    ][:n_samples])

    if task == "classification":
        y = np.random.randint(0, 2, size=(n_samples, n_tasks)).astype(np.int64)
    else:
        y = np.random.rand(n_samples, n_tasks).astype(np.float32)

    w = np.ones((n_samples, n_tasks), dtype=np.float32)
    return dc.data.NumpyDataset(X=smiles, y=y, w=w, ids=smiles)


def _make_tokenizer():
    """Build a minimal ByteLevelBPETokenizer and wrap it as RobertaTokenizerFast.

    Uses only in-memory SMILES strings so there is no file I/O dependency
    beyond a temporary directory.
    """
    import os
    import tempfile
    from tokenizers import ByteLevelBPETokenizer
    from transformers.models.roberta import RobertaTokenizerFast

    smiles_corpus = [
        "CCO",
        "CCC",
        "c1ccccc1",
        "CC(=O)O",
        "CN",
        "CCN",
        "CCCO",
        "c1ccncc1",
    ]

    tmpdir = tempfile.mkdtemp()
    corpus_path = os.path.join(tmpdir, "corpus.txt")
    with open(corpus_path, "w") as fh:
        fh.write("\n".join(smiles_corpus))

    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=corpus_path,
        vocab_size=128,
        min_frequency=1,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    tok_dir = os.path.join(tmpdir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    bpe.save_model(tok_dir)

    return RobertaTokenizerFast.from_pretrained(tok_dir)


# ------------------------------------------------------------------ #
# Tests                                                                #
# ------------------------------------------------------------------ #


class TestHuggingFaceModelInit:
    """Tests that cover model initialisation for every supported task."""

    def test_init_mlm_task(self, tmp_path):
        """HuggingFaceModel initialises without error for task='mlm'."""
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
        )

        assert hf_model.task == "mlm"
        assert hf_model.data_collator is not None

    def test_init_regression_task(self, tmp_path):
        """HuggingFaceModel initialises without error for task='regression'."""
        from transformers.models.roberta import (
            RobertaConfig, RobertaForSequenceClassification)
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        # problem_type + num_labels required for regression head
        config.problem_type = "regression"
        config.num_labels = 1
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="regression",
            model_dir=str(tmp_path),
        )

        assert hf_model.task == "regression"
        assert hf_model.data_collator is None

    def test_init_classification_task(self, tmp_path):
        """HuggingFaceModel initialises without error for task='classification'."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.num_labels = 2
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="classification",
            model_dir=str(tmp_path),
        )

        assert hf_model.task == "classification"

    def test_init_no_task(self, tmp_path):
        """HuggingFaceModel initialises when task is None (raw output mode)."""
        from transformers.models.roberta import RobertaModel
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaModel(config)
        tokenizer = _make_tokenizer()

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task=None,
            model_dir=str(tmp_path),
        )

        assert hf_model.task is None


class TestGradientCheckpointing:
    """Tests for the enable_gradient_checkpointing parameter."""

    def test_gradient_checkpointing_disabled_by_default(self, tmp_path):
        """enable_gradient_checkpointing defaults to False; model trains normally."""
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()

        # Should not raise regardless of default value.
        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
        )

        # Gradient checkpointing should not be active by default.
        assert not model.is_gradient_checkpointing

    def test_gradient_checkpointing_enabled_does_not_crash(self, tmp_path):
        """Passing enable_gradient_checkpointing=True must not raise an error.

        This is the primary regression test: any model that supports the
        ``gradient_checkpointing_enable`` method must initialise cleanly when
        the flag is set.
        """
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()

        # Must not raise.
        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
            enable_gradient_checkpointing=True,
        )

        assert hf_model is not None

    def test_gradient_checkpointing_actually_enabled(self, tmp_path):
        """When the flag is True the underlying model reports checkpointing active."""
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()

        HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
            enable_gradient_checkpointing=True,
        )

        # After wrapping, the underlying PreTrainedModel should report that
        # gradient checkpointing is active.
        assert model.is_gradient_checkpointing

    def test_gradient_checkpointing_unsupported_model_no_crash(
            self, tmp_path, caplog):
        """Models without gradient_checkpointing_enable must not crash; a warning is logged.

        This test creates a plain ``torch.nn.Module`` (which does not inherit
        from ``PreTrainedModel``) to simulate a model that lacks the method.
        We then wrap it with a mock tokeniser to verify the guard works.
        """
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        # Minimal nn.Module without gradient_checkpointing_enable.
        class TinyModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, **kwargs):
                return type('Output', (), {'loss': torch.tensor(0.0)})()

        model = TinyModel()
        tokenizer = _make_tokenizer()

        with caplog.at_level(logging.WARNING,
                             logger="deepchem.models.torch_models.hf_models"):
            # Must not raise.
            hf_model = HuggingFaceModel(
                model=model,
                tokenizer=tokenizer,
                task=None,
                model_dir=str(tmp_path),
                enable_gradient_checkpointing=True,
            )

        assert hf_model is not None
        # A warning should have been emitted.
        assert any(
            "gradient_checkpointing_enable" in record.message
            for record in caplog.records
        ), "Expected a warning about missing gradient_checkpointing_enable"

    def test_gradient_checkpointing_with_regression(self, tmp_path):
        """Gradient checkpointing works for regression task (not just mlm)."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.problem_type = "regression"
        config.num_labels = 1
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()

        # Must not raise.
        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="regression",
            model_dir=str(tmp_path),
            enable_gradient_checkpointing=True,
        )

        assert model.is_gradient_checkpointing

    def test_gradient_checkpointing_with_classification(self, tmp_path):
        """Gradient checkpointing works for classification task."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.num_labels = 2
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="classification",
            model_dir=str(tmp_path),
            enable_gradient_checkpointing=True,
        )

        assert model.is_gradient_checkpointing


class TestHuggingFaceModelFit:
    """Smoke tests: verify that fit() runs without error on tiny datasets."""

    def test_fit_mlm(self, tmp_path):
        """fit() completes one epoch for masked-language-model training."""
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="mlm")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
            batch_size=2,
        )

        loss = hf_model.fit(dataset, nb_epoch=1)
        assert isinstance(loss, float)

    def test_fit_regression(self, tmp_path):
        """fit() completes one epoch for a regression task."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.problem_type = "regression"
        config.num_labels = 1
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="regression")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="regression",
            model_dir=str(tmp_path),
            batch_size=2,
        )

        loss = hf_model.fit(dataset, nb_epoch=1)
        assert isinstance(loss, float)

    def test_fit_classification(self, tmp_path):
        """fit() completes one epoch for a classification task."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.num_labels = 2
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="classification")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="classification",
            model_dir=str(tmp_path),
            batch_size=2,
        )

        loss = hf_model.fit(dataset, nb_epoch=1)
        assert isinstance(loss, float)

    def test_fit_with_gradient_checkpointing_enabled(self, tmp_path):
        """fit() works end-to-end when gradient checkpointing is active."""
        from transformers.models.roberta import RobertaForMaskedLM
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        model = RobertaForMaskedLM(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="mlm")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="mlm",
            model_dir=str(tmp_path),
            batch_size=2,
            enable_gradient_checkpointing=True,
        )

        loss = hf_model.fit(dataset, nb_epoch=1)
        assert isinstance(loss, float)


class TestHuggingFaceModelPredict:
    """Smoke tests for predict()."""

    def test_predict_regression(self, tmp_path):
        """predict() returns an array of the correct shape for regression."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.problem_type = "regression"
        config.num_labels = 1
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="regression")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="regression",
            model_dir=str(tmp_path),
            batch_size=2,
        )

        predictions = hf_model.predict(dataset)
        assert predictions.shape[0] == 4

    def test_predict_classification(self, tmp_path):
        """predict() returns an array of the correct shape for classification."""
        from transformers.models.roberta import RobertaForSequenceClassification
        from deepchem.models.torch_models.hf_models import HuggingFaceModel

        config = _make_tiny_roberta_config()
        config.num_labels = 2
        model = RobertaForSequenceClassification(config)
        tokenizer = _make_tokenizer()
        dataset = _make_smiles_dataset(n_samples=4, task="classification")

        hf_model = HuggingFaceModel(
            model=model,
            tokenizer=tokenizer,
            task="classification",
            model_dir=str(tmp_path),
            batch_size=2,
        )

        predictions = hf_model.predict(dataset)
        assert predictions.shape[0] == 4
