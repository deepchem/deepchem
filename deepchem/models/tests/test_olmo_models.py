"""
Unit and integration tests for OLMo DeepChem integration.

Run with:
    pytest deepchem/models/tests/test_olmo_models.py -v

All tests that require GPU resources or a network download are marked with
``@pytest.mark.slow`` and can be skipped in CI via:
    pytest -m "not slow"

Tests that require the actual ``allenai/OLMo-7B`` weights are marked
``@pytest.mark.huggingface`` and skip unless the environment variable
``DEEPCHEM_TEST_HF_MODELS=1`` is set.
"""

from __future__ import annotations

import os
import unittest
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional imports – tests are skipped gracefully when deps are missing
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from deepchem.models.torch_models.olmo import (
        OLMO_SUPPORTED_TASKS,
        OLMoModel,
        RegressionHead,
    )
    from deepchem.models.torch_models.olmo_generation import (
        MolecularTextDataset,
        OLMoGenerationFeaturizer,
        OLMoPretrainer,
        evaluate_generation,
    )
    from deepchem.models.torch_models.olmo_finetune import (
        OLMoFineTuner,
        OLMoSupervisedDataset,
        evaluate_classification,
        evaluate_regression,
    )

    HAS_OLMO = True
except ImportError:
    HAS_OLMO = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_NO_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
_SKIP_NO_TF = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)
_SKIP_NO_OLMO = pytest.mark.skipif(not HAS_OLMO, reason="OLMo module not importable")

_TEST_SMILES: List[str] = [
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
    "c1ccccc1",                  # Benzene
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "OC(=O)c1ccccc1O",           # Salicylic acid
]

_TEST_LABELS_CLS = np.array([1, 0, 1, 0, 1], dtype=np.float32)
_TEST_LABELS_REG = np.array([2.1, -0.5, 1.3, 0.8, 1.9], dtype=np.float32)

# ---------------------------------------------------------------------------
# Lightweight mock tokenizer (no HF download needed)
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(vocab_size: int = 128) -> MagicMock:
    """Return a MagicMock that behaves like a minimal HuggingFace tokenizer."""
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.eos_token = "[EOS]"
    tok.eos_token_id = 1
    tok.vocab_size = vocab_size

    def _call(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        batch = len(texts)
        length = min(max_length, 32)
        result = MagicMock()
        result.__getitem__ = lambda self, k: {
            "input_ids": torch.zeros(batch, length, dtype=torch.long),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
        }[k]
        result.to = lambda device: result
        return result

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


# ---------------------------------------------------------------------------
# Module-level tests (no HF weights required)
# ---------------------------------------------------------------------------

@_SKIP_NO_OLMO
class TestRegressionHead(unittest.TestCase):
    """Tests for the standalone RegressionHead module."""

    def test_forward_single_task(self) -> None:
        head = RegressionHead(hidden_size=64, n_tasks=1)
        x = torch.randn(4, 64)
        out = head(x)
        self.assertEqual(out.shape, (4, 1))

    def test_forward_multi_task(self) -> None:
        head = RegressionHead(hidden_size=64, n_tasks=3)
        x = torch.randn(8, 64)
        out = head(x)
        self.assertEqual(out.shape, (8, 3))

    def test_dropout_training(self) -> None:
        head = RegressionHead(hidden_size=32, n_tasks=1, dropout=0.5)
        head.train()
        x = torch.randn(2, 32)
        # Two forward passes with dropout should differ
        o1 = head(x)
        o2 = head(x)
        self.assertFalse(torch.allclose(o1, o2))

    def test_no_nan_in_output(self) -> None:
        head = RegressionHead(hidden_size=16, n_tasks=2)
        x = torch.randn(6, 16)
        out = head(x)
        self.assertFalse(torch.any(torch.isnan(out)))


@_SKIP_NO_OLMO
class TestOLMoSupportedTasks(unittest.TestCase):
    """Smoke tests for task-validation logic (no weight download)."""

    def test_supported_tasks_constant(self) -> None:
        self.assertIn("generation", OLMO_SUPPORTED_TASKS)
        self.assertIn("classification", OLMO_SUPPORTED_TASKS)
        self.assertIn("regression", OLMO_SUPPORTED_TASKS)

    @patch("deepchem.models.torch_models.olmo.AutoTokenizer")
    @patch("deepchem.models.torch_models.olmo.AutoModelForCausalLM")
    def test_invalid_task_raises(self, mock_model, mock_tok) -> None:
        with self.assertRaises(ValueError):
            OLMoModel(task="unsupported_task", model_name="allenai/OLMo-1B")


# ---------------------------------------------------------------------------
# MolecularTextDataset tests
# ---------------------------------------------------------------------------

@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
class TestMolecularTextDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.tok = _make_mock_tokenizer()

    def test_len(self) -> None:
        ds = MolecularTextDataset(_TEST_SMILES, self.tok, max_length=64)
        self.assertEqual(len(ds), len(_TEST_SMILES))

    def test_item_keys(self) -> None:
        ds = MolecularTextDataset(_TEST_SMILES, self.tok, max_length=64)
        item = ds[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)

    def test_labels_mask_padding(self) -> None:
        ds = MolecularTextDataset(_TEST_SMILES, self.tok, max_length=64)
        item = ds[0]
        # All positions are non-padding in our mock → labels should equal input_ids
        attn = item["attention_mask"]
        labels = item["labels"]
        pad_positions = (attn == 0)
        if pad_positions.any():
            self.assertTrue((labels[pad_positions] == -100).all())


# ---------------------------------------------------------------------------
# OLMoGenerationFeaturizer tests
# ---------------------------------------------------------------------------

@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
class TestOLMoGenerationFeaturizer(unittest.TestCase):

    def setUp(self) -> None:
        self.tok = _make_mock_tokenizer()
        # Make the tokenizer callable
        self.tok.side_effect = None

        def _tok_call(text, **kwargs):
            result = MagicMock()
            result.__getitem__ = lambda s, k: {
                "input_ids": np.zeros((1, 32), dtype=np.int64),
            }[k]
            return result

        self.tok.side_effect = _tok_call

    def test_featurize_returns_array(self) -> None:
        feat = OLMoGenerationFeaturizer(self.tok, max_length=32)
        # _featurize is called internally by featurize()
        # We test the output shape directly
        arr = feat._featurize(_TEST_SMILES[0])
        self.assertEqual(arr.shape, (32,))
        self.assertEqual(arr.dtype, np.int64)


# ---------------------------------------------------------------------------
# OLMoSupervisedDataset tests
# ---------------------------------------------------------------------------

@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
class TestOLMoSupervisedDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.tok = _make_mock_tokenizer()

        def _tok_call(text, **kwargs):
            r = MagicMock()
            r.__getitem__ = lambda s, k: {
                "input_ids": torch.zeros(1, 32, dtype=torch.long),
                "attention_mask": torch.ones(1, 32, dtype=torch.long),
            }[k]
            return r

        self.tok.side_effect = _tok_call

    def test_classification_dataset(self) -> None:
        ds = OLMoSupervisedDataset(
            _TEST_SMILES, _TEST_LABELS_CLS, self.tok,
            max_length=32, task="classification",
        )
        self.assertEqual(len(ds), 5)
        item = ds[0]
        self.assertIn("labels", item)
        self.assertEqual(item["labels"].shape, (1,))

    def test_regression_dataset(self) -> None:
        ds = OLMoSupervisedDataset(
            _TEST_SMILES, _TEST_LABELS_REG, self.tok,
            max_length=32, task="regression",
        )
        item = ds[2]
        self.assertAlmostEqual(item["labels"][0].item(), _TEST_LABELS_REG[2], places=4)

    def test_multitask_labels(self) -> None:
        labels_2d = np.stack([_TEST_LABELS_CLS, _TEST_LABELS_REG], axis=1)
        ds = OLMoSupervisedDataset(
            _TEST_SMILES, labels_2d, self.tok,
            max_length=32, task="regression",
        )
        item = ds[0]
        self.assertEqual(item["labels"].shape, (2,))


# ---------------------------------------------------------------------------
# Integration-level tests (marked slow / HuggingFace)
# ---------------------------------------------------------------------------

_RUN_HF = os.environ.get("DEEPCHEM_TEST_HF_MODELS", "0") == "1"
_SKIP_NO_HF_FLAG = pytest.mark.skipif(
    not _RUN_HF,
    reason="Set DEEPCHEM_TEST_HF_MODELS=1 to run HuggingFace weight tests.",
)

_MINI_MODEL = "allenai/OLMo-1B"  # smallest available OLMo checkpoint


@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
@_SKIP_NO_TF
@pytest.mark.slow
@pytest.mark.huggingface
class TestOLMoModelIntegration(unittest.TestCase):
    """Integration tests downloading actual OLMo weights (slow, requires HF)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _RUN_HF:
            pytest.skip("HuggingFace model tests disabled.")
        cls.gen_model = OLMoModel(
            task="generation",
            model_name=_MINI_MODEL,
            torch_dtype="float32",
        )

    def test_generation_model_loads(self) -> None:
        self.assertIsNotNone(self.gen_model.model)

    def test_generate_molecules(self) -> None:
        mols = self.gen_model.generate_molecules(
            prompt="CC(=O)",
            n_molecules=3,
            max_new_tokens=32,
        )
        self.assertEqual(len(mols), 3)
        for m in mols:
            self.assertIsInstance(m, str)

    def test_classification_model_loads(self) -> None:
        cls_model = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1,
            n_classes=2,
            torch_dtype="float32",
        )
        self.assertIsNotNone(cls_model.model)

    def test_regression_model_loads(self) -> None:
        reg_model = OLMoModel(
            task="regression",
            model_name=_MINI_MODEL,
            n_tasks=1,
            torch_dtype="float32",
        )
        self.assertIsNotNone(reg_model.model)

    def test_tokenizer_pad_token(self) -> None:
        self.assertIsNotNone(self.gen_model.tokenizer.pad_token)

    def test_repr(self) -> None:
        r = repr(self.gen_model)
        self.assertIn("OLMoModel", r)
        self.assertIn("generation", r)

    def test_trainable_param_count(self) -> None:
        n = self.gen_model.get_num_trainable_params()
        self.assertGreater(n, 0)

    def test_lora_classification(self) -> None:
        pytest.importorskip("peft")
        lora_model = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1,
            n_classes=2,
            use_lora=True,
            lora_r=4,
            torch_dtype="float32",
        )
        # LoRA should reduce trainable params relative to full fine-tuning
        n_lora = lora_model.get_num_trainable_params()
        full = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1,
            n_classes=2,
            use_lora=False,
            torch_dtype="float32",
        )
        n_full = full.get_num_trainable_params()
        self.assertLess(n_lora, n_full)


@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
@_SKIP_NO_TF
@pytest.mark.slow
@pytest.mark.huggingface
class TestOLMoPretrainerIntegration(unittest.TestCase):
    """Integration tests for continued pre-training."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _RUN_HF:
            pytest.skip("HuggingFace model tests disabled.")
        import tempfile
        cls.tmp_dir = tempfile.mkdtemp()
        cls.model = OLMoModel(
            task="generation",
            model_name=_MINI_MODEL,
            torch_dtype="float32",
        )

    def test_pretrainer_one_step(self) -> None:
        trainer = OLMoPretrainer(
            self.model,
            output_dir=self.tmp_dir,
            batch_size=2,
            max_epochs=1,
            save_every_n_steps=100,
        )
        losses = trainer.train_on_smiles(
            _TEST_SMILES,
            max_length=64,
        )
        self.assertIsInstance(losses, list)
        for l in losses:
            self.assertFalse(np.isnan(l))

    def test_pretrainer_wrong_task_raises(self) -> None:
        cls_model = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1, n_classes=2,
        )
        with self.assertRaises(ValueError):
            OLMoPretrainer(cls_model, output_dir=self.tmp_dir)


@_SKIP_NO_OLMO
@_SKIP_NO_TORCH
@_SKIP_NO_TF
@pytest.mark.slow
@pytest.mark.huggingface
class TestOLMoFineTunerIntegration(unittest.TestCase):
    """Integration tests for classification / regression fine-tuning."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _RUN_HF:
            pytest.skip("HuggingFace model tests disabled.")
        import tempfile
        cls.tmp_dir = tempfile.mkdtemp()

    def test_classification_one_epoch(self) -> None:
        model = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1, n_classes=2,
            torch_dtype="float32",
        )
        tuner = OLMoFineTuner(
            model, output_dir=self.tmp_dir,
            max_epochs=1, batch_size=2,
        )
        history = tuner.fit(
            _TEST_SMILES, _TEST_LABELS_CLS,
            max_length=64,
        )
        self.assertIn("train_loss", history)
        self.assertGreater(len(history["train_loss"]), 0)

    def test_regression_one_epoch(self) -> None:
        model = OLMoModel(
            task="regression",
            model_name=_MINI_MODEL,
            n_tasks=1,
            torch_dtype="float32",
        )
        tuner = OLMoFineTuner(
            model, output_dir=self.tmp_dir,
            loss_fn="huber", max_epochs=1, batch_size=2,
        )
        history = tuner.fit(
            _TEST_SMILES, _TEST_LABELS_REG,
            max_length=64,
        )
        self.assertIn("train_loss", history)

    def test_val_metric_computed(self) -> None:
        model = OLMoModel(
            task="classification",
            model_name=_MINI_MODEL,
            n_tasks=1, n_classes=2,
            torch_dtype="float32",
        )
        tuner = OLMoFineTuner(
            model, output_dir=self.tmp_dir,
            max_epochs=1, batch_size=2,
        )
        history = tuner.fit(
            _TEST_SMILES, _TEST_LABELS_CLS,
            val_smiles=_TEST_SMILES,
            val_labels=_TEST_LABELS_CLS,
            max_length=64,
        )
        self.assertIn("val_metric", history)
        self.assertGreater(len(history["val_metric"]), 0)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
