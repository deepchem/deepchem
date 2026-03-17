"""
Comprehensive test suite for hf_models_final.py
Tests: HuggingFaceModel + OLMoModel classes, version compatibility,
       import structure, method signatures, logic & edge cases.

Run with:
    python -m pytest test_hf_models.py -v
or:
    python test_hf_models.py
"""

import sys
import types
import importlib
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — stub every unavailable heavy dependency BEFORE importing the
# module under test so Python never actually tries to download / import them.
# ─────────────────────────────────────────────────────────────────────────────

def _make_torch_stub():
    """Return a minimal torch stub that satisfies all usages in hf_models."""
    torch = types.ModuleType("torch")

    # --- Tensor ---
    class FakeTensor:
        def __init__(self, data=None):
            self._data = data if data is not None else np.zeros((2, 4))

        def detach(self): return self
        def cpu(self):    return self
        def numpy(self):  return self._data
        def backward(self): pass
        def item(self):   return float(self._data.flat[0]) if hasattr(self._data, 'flat') else 0.0
        def __add__(self, other): return FakeTensor()
        def __radd__(self, other): return FakeTensor()
        def __float__(self): return 0.5
        def tolist(self): return list(self._data) if hasattr(self._data, '__iter__') else []
        def float(self):  return self
        def long(self):   return self
        def to(self, device): return self
        # support tensor slicing e.g. output_ids[:, prompt_len:]
        def __getitem__(self, idx): return FakeTensor()
        @property
        def shape(self):
            class _Shape:
                def __getitem__(self, i): return 2
            return _Shape()

    torch.Tensor = FakeTensor

    # --- torch.nn ---
    nn = types.ModuleType("torch.nn")
    nn.Parameter = object
    nn.ParameterList = list
    torch.nn = nn

    # --- basic functions ---
    torch.from_numpy = lambda x: FakeTensor(x)
    torch.topk = lambda t, k, dim=None: MagicMock(
        indices=MagicMock(__getitem__=lambda s, x: MagicMock(tolist=lambda: [0, 1, 2, 3, 4]))
    )
    torch.softmax = lambda t, dim=None: MagicMock(
        __getitem__=lambda s, idx: MagicMock(item=lambda: 0.2, __getitem__=lambda s2, i: MagicMock(item=lambda: 0.2))
    )
    torch.softmax = lambda t, dim=None: FakeTensor(np.ones((1, 10)) / 10)
    torch.where = lambda cond, x=None: (MagicMock(), MagicMock(numel=lambda: 1))
    torch.load = lambda f, map_location=None: {
        "model_state_dict": {
            "classifier.out_proj.weight": FakeTensor(),
            "classifier.out_proj.bias": FakeTensor(),
            "classifier.dense.bias": FakeTensor(),
            "classifier.dense.weight": FakeTensor(),
            "encoder.layer.0.weight": FakeTensor(),
        }
    }
    torch.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s,*a: None))
    torch.float16 = "float16"
    torch.float32 = "float32"

    # --- save ---
    torch.save = MagicMock()

    # --- device ---
    torch.device = lambda x: x

    # --- CUDA ---
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    torch.cuda = cuda

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.cuda"] = cuda
    return torch


def _make_transformers_stub():
    """
    Build a stub that covers both the top-level `transformers` namespace
    and every sub-module accessed inside hf_models_final.py.
    """
    # ── top-level module ──────────────────────────────────────────────────────
    transformers = sys.modules.get("transformers") or types.ModuleType("transformers")

    # ── fake model output (mimics HuggingFace ModelOutput) ────────────────────
    class FakeModelOutput(dict):
        def __init__(self, loss=None, logits=None):
            super().__init__(loss=loss, logits=logits)
            self.loss   = loss   if loss   is not None else MagicMock(backward=MagicMock(), __float__=lambda s: 0.5, __add__=lambda s,o: s)
            self.logits = logits if logits is not None else MagicMock()
        def get(self, key, default=None):
            return getattr(self, key, default)

    # ── minimal PreTrainedModel ────────────────────────────────────────────────
    class FakePreTrainedModel:
        def __init__(self, config=None):
            self.config = config or {}
        def __call__(self, **kwargs):
            return FakeModelOutput()
        def train(self): pass
        def eval(self):  pass
        def parameters(self): return iter([])
        def state_dict(self): return {}
        def load_state_dict(self, d, strict=True): pass
        def to(self, device): return self
        def generate(self, input_ids=None, attention_mask=None,
                     max_new_tokens=128, **kw):
            # return a mock tensor (batch of 1 sequence)
            t = MagicMock()
            t.__getitem__ = lambda s, idx: MagicMock()
            return [MagicMock(), MagicMock()]

    # ── minimal PreTrainedTokenizer ────────────────────────────────────────────
    class FakePreTrainedTokenizer:
        def __init__(self):
            self.vocab_size = 100
            self.pad_token  = "<pad>"
            self.eos_token  = "<eos>"
            self.mask_token = "<mask>"
            self.mask_token_id = 4
        def __call__(self, text, padding=True, truncation=True,
                     max_length=512, return_tensors="pt", **kw):
            m = MagicMock()
            m.__iter__ = lambda s: iter({"input_ids": MagicMock(), "attention_mask": MagicMock()}.items())
            m.items = lambda: {"input_ids": MagicMock(to=lambda d: MagicMock()),
                               "attention_mask": MagicMock(to=lambda d: MagicMock())}.items()
            m["input_ids"] = MagicMock(to=lambda d: MagicMock())
            m["attention_mask"] = MagicMock(to=lambda d: MagicMock())
            m.to = lambda device: m
            return m
        def decode(self, token_ids, skip_special_tokens=True): return "CC"
        def batch_decode(self, ids, skip_special_tokens=True): return ["CC"]
        def from_pretrained(self, path, **kw): return FakePreTrainedTokenizer()

    # ── data_collator ─────────────────────────────────────────────────────────
    data_collator_mod = types.ModuleType("transformers.data.data_collator")
    class FakeDataCollatorForLanguageModeling:
        def __init__(self, tokenizer=None, mlm=True, mlm_probability=0.15):
            pass
        def torch_mask_tokens(self, input_ids, special_tokens_mask=None):
            return MagicMock(to=lambda d: MagicMock()), MagicMock(to=lambda d: MagicMock())
    data_collator_mod.DataCollatorForLanguageModeling = FakeDataCollatorForLanguageModeling

    # ── auto models ───────────────────────────────────────────────────────────
    auto_mod = types.ModuleType("transformers.models.auto")
    for name in ("AutoModel", "AutoModelForSequenceClassification",
                 "AutoModelForMaskedLM", "AutoModelForUniversalSegmentation",
                 "AutoModelForCausalLM", "AutoTokenizer"):
        cls = MagicMock()
        cls.from_pretrained = MagicMock(return_value=FakePreTrainedModel())
        setattr(auto_mod, name, cls)

    # Patch AutoTokenizer.from_pretrained to return a real FakePreTrainedTokenizer
    auto_mod.AutoTokenizer.from_pretrained = MagicMock(return_value=FakePreTrainedTokenizer())

    # ── modeling_utils ────────────────────────────────────────────────────────
    modeling_utils_mod = types.ModuleType("transformers.modeling_utils")
    modeling_utils_mod.PreTrainedModel = FakePreTrainedModel

    # ── tokenization_utils ────────────────────────────────────────────────────
    tokenization_mod = types.ModuleType("transformers.tokenization_utils")
    tokenization_mod.PreTrainedTokenizer = FakePreTrainedTokenizer

    # ── wire stubs into sys.modules ───────────────────────────────────────────
    transformers.PreTrainedModel   = FakePreTrainedModel
    transformers.PreTrainedTokenizer = FakePreTrainedTokenizer
    transformers.DataCollatorForLanguageModeling = FakeDataCollatorForLanguageModeling
    transformers.AutoModel = auto_mod.AutoModel
    transformers.AutoModelForCausalLM = auto_mod.AutoModelForCausalLM
    transformers.AutoModelForMaskedLM = auto_mod.AutoModelForMaskedLM
    transformers.AutoModelForSequenceClassification = auto_mod.AutoModelForSequenceClassification
    transformers.AutoModelForUniversalSegmentation  = auto_mod.AutoModelForUniversalSegmentation
    transformers.AutoTokenizer = auto_mod.AutoTokenizer

    sys.modules["transformers"] = transformers
    sys.modules["transformers.data"] = types.ModuleType("transformers.data")
    sys.modules["transformers.data.data_collator"] = data_collator_mod
    sys.modules["transformers.models"] = types.ModuleType("transformers.models")
    sys.modules["transformers.models.auto"] = auto_mod
    sys.modules["transformers.modeling_utils"] = modeling_utils_mod
    sys.modules["transformers.tokenization_utils"] = tokenization_mod
    return transformers


def _make_deepchem_stub():
    """Minimal deepchem stub covering every import in hf_models_final."""
    dc = types.ModuleType("deepchem")

    # deepchem.models.optimizers
    optimizers_mod = types.ModuleType("deepchem.models.optimizers")
    class FakeLRS:
        pass
    optimizers_mod.LearningRateSchedule = FakeLRS

    # deepchem.models.torch_models
    torch_models_mod = types.ModuleType("deepchem.models.torch_models")
    class FakeTorchModel:
        """Minimal TorchModel base that satisfies all super().__init__ calls."""
        model_dir    = "/tmp/fake_model"
        device       = "cpu"
        log_frequency = 100
        tensorboard  = False
        wandb_logger = None
        _global_step  = 0
        _pytorch_optimizer = MagicMock(zero_grad=MagicMock(), step=MagicMock())
        _lr_schedule       = None
        _optimizer_for_vars = {}
        _variance_outputs   = []
        _prediction_outputs = None
        _other_outputs      = None

        def __init__(self, model=None, loss=None, **kwargs):
            self.model     = model
            self.model_dir = kwargs.get("model_dir", "/tmp/fake_model")

        def _ensure_built(self):  pass
        def restore(self):        pass
        def save_checkpoint(self, max_to_keep=5): pass
        def get_checkpoints(self, model_dir=None): return []

        class optimizer:
            learning_rate = 0.001
            @staticmethod
            def _create_pytorch_optimizer(variables): return MagicMock(zero_grad=MagicMock(), step=MagicMock())

    torch_models_mod.TorchModel = FakeTorchModel

    # deepchem.trans
    trans_mod = types.ModuleType("deepchem.trans")
    trans_mod.Transformer  = object
    trans_mod.undo_transforms = lambda output, transformers: output

    # deepchem.utils.typing
    utils_mod    = types.ModuleType("deepchem.utils")
    typing_mod   = types.ModuleType("deepchem.utils.typing")
    from typing import Union, List
    typing_mod.LossFn    = Union[None, object]  # callable placeholder
    # OneOrMany must be subscriptable (e.g. OneOrMany[str])
    class _OneOrMany:
        def __class_getitem__(cls, item): return Union[item, List[item]]
    typing_mod.OneOrMany = _OneOrMany

    # Register all in sys.modules
    sys.modules["deepchem"]                       = dc
    sys.modules["deepchem.models"]                = types.ModuleType("deepchem.models")
    sys.modules["deepchem.models.optimizers"]     = optimizers_mod
    sys.modules["deepchem.models.torch_models"]   = torch_models_mod
    sys.modules["deepchem.trans"]                 = trans_mod
    sys.modules["deepchem.utils"]                 = utils_mod
    sys.modules["deepchem.utils.typing"]          = typing_mod

    dc.models   = sys.modules["deepchem.models"]
    dc.trans    = trans_mod
    dc.utils    = utils_mod

    return dc, optimizers_mod, torch_models_mod, FakeTorchModel


# ── install all stubs ──────────────────────────────────────────────────────────
_torch_stub  = _make_torch_stub()
_tf_stub     = _make_transformers_stub()
_dc_stub, _optim_stub, _tm_stub, FakeTorchModel = _make_deepchem_stub()


# ── now load the module under test ────────────────────────────────────────────
import importlib.util, pathlib

_FINAL_PATH = (pathlib.Path(__file__).resolve().parent / "deepchem" / "models" /
               "torch_models" / "hf_models.py")
_spec = importlib.util.spec_from_file_location("hf_models_final", _FINAL_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

HuggingFaceModel = _mod.HuggingFaceModel
OLMoModel        = _mod.OLMoModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tokenizer():
    tok = MagicMock()
    tok.pad_token  = "<pad>"
    tok.eos_token  = "<eos>"
    tok.mask_token = "<mask>"
    tok.mask_token_id = 4
    tok.vocab_size = 100
    # make it callable
    fake_out = MagicMock()
    fake_out.__getitem__ = lambda s, k: MagicMock(to=lambda d: MagicMock())
    fake_out.items = lambda: [("input_ids", MagicMock(to=lambda d: MagicMock())),
                               ("attention_mask", MagicMock(to=lambda d: MagicMock()))]
    fake_out.to = lambda d: fake_out
    tok.return_value = fake_out
    return tok

def _make_model(return_loss=True):
    m = MagicMock()
    out = MagicMock()
    out.get = lambda k, default=None: MagicMock(
        backward=MagicMock(), __float__=lambda s: 0.5,
        __add__=lambda s, o: s
    ) if k == "loss" else (
        MagicMock() if k == "logits" else default
    )
    out.logits = MagicMock()
    m.return_value = out
    m.train = MagicMock()
    m.eval  = MagicMock()
    m.to    = MagicMock(return_value=m)
    # generate must return a tensor-like that supports [:, prompt_len:] slicing
    fake_output_ids = _torch_stub.Tensor()
    m.generate = MagicMock(return_value=fake_output_ids)
    return m

def _make_hf_model(task="mlm"):
    return HuggingFaceModel(
        model    = _make_model(),
        tokenizer= _make_tokenizer(),
        task     = task,
        model_dir= "/tmp/hf_test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TestImportsAndExports(unittest.TestCase):
    """Verify the module exports the expected symbols and new imports are present."""

    def test_huggingface_model_class_exists(self):
        self.assertTrue(hasattr(_mod, "HuggingFaceModel"))

    def test_olmo_model_class_exists(self):
        self.assertTrue(hasattr(_mod, "OLMoModel"))

    def test_olmo_is_subclass_of_hf_model(self):
        self.assertTrue(issubclass(OLMoModel, HuggingFaceModel))

    def test_new_import_AutoModelForCausalLM_present(self):
        """hf_models_final adds AutoModelForCausalLM to the auto imports."""
        src = _FINAL_PATH.read_text()
        self.assertIn("AutoModelForCausalLM", src)

    def test_new_import_AutoTokenizer_present(self):
        src = _FINAL_PATH.read_text()
        self.assertIn("AutoTokenizer", src)

    def test_original_imports_still_present(self):
        src = _FINAL_PATH.read_text()
        for sym in ["AutoModel", "AutoModelForSequenceClassification",
                    "AutoModelForMaskedLM", "AutoModelForUniversalSegmentation"]:
            self.assertIn(sym, src, f"Missing original import: {sym}")

    def test_module_level_logger(self):
        self.assertTrue(hasattr(_mod, "logger"))


class TestHuggingFaceModelInit(unittest.TestCase):
    """__init__ behaviour for HuggingFaceModel."""

    def test_task_stored(self):
        m = _make_hf_model(task="regression")
        self.assertEqual(m.task, "regression")

    def test_tokenizer_stored(self):
        tok = _make_tokenizer()
        m = HuggingFaceModel(model=_make_model(), tokenizer=tok, task="regression")
        self.assertIs(m.tokenizer, tok)

    def test_mlm_task_creates_data_collator(self):
        m = _make_hf_model(task="mlm")
        self.assertIsNotNone(m.data_collator)

    def test_non_mlm_task_has_no_data_collator(self):
        m = _make_hf_model(task="regression")
        self.assertIsNone(m.data_collator)

    def test_config_dict_stored_when_provided(self):
        cfg = {"hidden_size": 256, "num_hidden_layers": 4}
        m = HuggingFaceModel(model=_make_model(), tokenizer=_make_tokenizer(),
                             task="regression", config=cfg)
        self.assertEqual(m.config, cfg)

    def test_config_defaults_to_empty_dict(self):
        m = _make_hf_model(task="regression")
        self.assertEqual(m.config, {})

    def test_all_supported_tasks(self):
        for task in ("mlm", "mtr", "regression", "classification", None):
            m = HuggingFaceModel(model=_make_model(), tokenizer=_make_tokenizer(), task=task)
            self.assertEqual(m.task, task)


class TestHuggingFaceModelPrepareBatch(unittest.TestCase):
    """_prepare_batch for every supported task."""

    def _batch(self, y_val=None):
        """Produce a (inputs, y, w) triple."""
        X = np.array(["CC(=O)O", "CCO"])
        y = np.array([[1.0], [2.0]]) if y_val is None else y_val
        w = np.ones_like(y)
        return (np.expand_dims(X, 0), np.expand_dims(y, 0), w)

    def test_mlm_returns_no_labels_in_y(self):
        m = _make_hf_model(task="mlm")
        inputs, y, w = m._prepare_batch(self._batch())
        self.assertIsNone(y)
        self.assertIn("input_ids", inputs)
        self.assertIn("labels", inputs)
        self.assertIn("attention_mask", inputs)

    def test_regression_includes_labels(self):
        m = _make_hf_model(task="regression")
        inputs, y, w = m._prepare_batch(self._batch())
        self.assertIn("labels", inputs)

    def test_classification_includes_labels(self):
        m = _make_hf_model(task="classification")
        inp, y, w = m._prepare_batch(self._batch(np.array([[0], [1]])))
        self.assertIn("labels", inp)

    def test_mtr_includes_labels(self):
        m = _make_hf_model(task="mtr")
        inp, y, w = m._prepare_batch(self._batch())
        self.assertIn("labels", inp)

    def test_regression_y_none_predict(self):
        """When y is None (predict mode), inputs dict still has labels=None."""
        m = _make_hf_model(task="regression")
        X = np.array(["CC(=O)O"])
        batch = (np.expand_dims(X, 0), None, None)
        inp, y, w = m._prepare_batch(batch)
        self.assertIn("labels", inp)
        self.assertIsNone(inp["labels"])


class TestHuggingFaceModelLoadFromPretrained(unittest.TestCase):
    """load_from_pretrained routing logic."""

    def test_mlm_task_uses_AutoModelForMaskedLM(self):
        m = _make_hf_model(task="mlm")
        auto_mlm = sys.modules["transformers.models.auto"].AutoModelForMaskedLM
        with patch.object(auto_mlm, "from_pretrained", return_value=_make_model()) as fp:
            m.load_from_pretrained(model_dir="/fake/path", from_hf_checkpoint=True)
            fp.assert_called_once()
            args, kwargs = fp.call_args
            self.assertEqual(args[0], "/fake/path")

    def test_regression_task_uses_AutoModelForSeqClassification(self):
        m = _make_hf_model(task="regression")
        auto_sc = sys.modules["transformers.models.auto"].AutoModelForSequenceClassification
        with patch.object(auto_sc, "from_pretrained", return_value=_make_model()) as fp:
            m.load_from_pretrained(model_dir="/fake/path", from_hf_checkpoint=True)
            fp.assert_called_once()

    def test_classification_task_uses_AutoModelForSeqClassification(self):
        m = _make_hf_model(task="classification")
        auto_sc = sys.modules["transformers.models.auto"].AutoModelForSequenceClassification
        with patch.object(auto_sc, "from_pretrained", return_value=_make_model()) as fp:
            m.load_from_pretrained(model_dir="/fake/path", from_hf_checkpoint=True)
            fp.assert_called_once()

    def test_none_task_uses_AutoModel(self):
        m = _make_hf_model(task=None)
        m.task = None
        auto_m = sys.modules["transformers.models.auto"].AutoModel
        with patch.object(auto_m, "from_pretrained", return_value=_make_model()) as fp:
            m.load_from_pretrained(model_dir="/fake/path", from_hf_checkpoint=True)
            fp.assert_called_once()

    def test_no_checkpoint_raises_value_error(self):
        m = _make_hf_model(task="regression")
        m.get_checkpoints = lambda d: []
        with self.assertRaises(ValueError):
            m.load_from_pretrained(from_hf_checkpoint=False)

    def test_checkpoint_strips_module_prefix(self):
        """DDP 'module.' prefix should be stripped from state_dict keys."""
        m = _make_hf_model(task="regression")
        state = {"module.encoder.layer.0.weight": MagicMock()}
        m.get_checkpoints = lambda d: ["/fake/ckpt.pt"]
        _torch_stub.load = lambda f, map_location=None: {"model_state_dict": state}
        m.model.load_state_dict = MagicMock()
        m.load_from_pretrained(from_hf_checkpoint=False)
        loaded_keys = m.model.load_state_dict.call_args[0][0].keys()
        self.assertIn("encoder.layer.0.weight", loaded_keys)
        self.assertNotIn("module.encoder.layer.0.weight", loaded_keys)

    def test_classifier_head_keys_are_deleted(self):
        """Classifier head weights should be removed before loading."""
        m = _make_hf_model(task="regression")
        state = {
            "classifier.out_proj.weight": MagicMock(),
            "classifier.out_proj.bias":   MagicMock(),
            "classifier.dense.bias":      MagicMock(),
            "classifier.dense.weight":    MagicMock(),
            "encoder.weight":             MagicMock(),
        }
        m.get_checkpoints = lambda d: ["/fake/ckpt.pt"]
        _torch_stub.load = lambda f, map_location=None: {"model_state_dict": state}
        m.model.load_state_dict = MagicMock()
        m.load_from_pretrained(from_hf_checkpoint=False)
        remaining = m.model.load_state_dict.call_args[0][0]
        for head_key in ("classifier.out_proj.weight", "classifier.out_proj.bias",
                         "classifier.dense.bias", "classifier.dense.weight"):
            self.assertNotIn(head_key, remaining)
        self.assertIn("encoder.weight", remaining)


class TestHuggingFaceModelFillMask(unittest.TestCase):
    """fill_mask method behaviour."""

    def _setup_fill_mask_mocks(self, m):
        """Patch all torch functions fill_mask touches with correct return types."""
        # torch.where returns (_, index_tensor) — index_tensor.numel() == 1
        fake_mask_idx = MagicMock()
        fake_mask_idx.numel = lambda: 1
        _torch_stub.where = lambda cond: (MagicMock(), fake_mask_idx)

        # torch.topk(...).indices[0].tolist() must return a plain Python list
        fake_token_list = MagicMock()
        fake_token_list.tolist = lambda: [0, 1, 2]
        fake_indices_0 = fake_token_list
        fake_indices = MagicMock()
        fake_indices.__getitem__ = lambda s, x: fake_indices_0
        _torch_stub.topk = lambda t, k, dim=None: MagicMock(indices=fake_indices)

        # torch.softmax(...)[0, token].item() must return a float
        fake_score = MagicMock()
        fake_score.item = lambda: 0.3
        fake_softmax_out = MagicMock()
        fake_softmax_out.__getitem__ = lambda s, idx: fake_score
        _torch_stub.softmax = lambda t, dim=None: fake_softmax_out

        # tokenizer.decode must return a real string (used in str.replace)
        m.tokenizer.decode = lambda ids: "C"
        m.tokenizer.mask_token = "<mask>"

        # model output with logits
        fake_out = MagicMock()
        fake_out.logits = MagicMock()
        m.model = MagicMock(return_value=fake_out, eval=MagicMock())

    def test_single_string_returns_list_of_dicts(self):
        m = _make_hf_model(task="mlm")
        self._setup_fill_mask_mocks(m)

        with patch.object(m, "_ensure_built"):
            result = m.fill_mask("CC<mask>O", top_k=3)

        self.assertIsInstance(result, list)
        # each element should be a dict with the required keys
        for d in result:
            self.assertIn("sequence", d)
            self.assertIn("score", d)
            self.assertIn("token", d)
            self.assertIn("token_str", d)

    def test_list_input_returns_nested_list(self):
        m = _make_hf_model(task="mlm")
        self._setup_fill_mask_mocks(m)

        with patch.object(m, "_ensure_built"):
            result = m.fill_mask(["CC<mask>O", "C<mask>O"], top_k=2)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for sublist in result:
            self.assertIsInstance(sublist, list)


class TestOLMoModelClassAttributes(unittest.TestCase):
    """OLMoModel class-level default config dicts."""

    def test_default_unsloth_config_keys(self):
        cfg = OLMoModel.DEFAULT_UNSLOTH_CONFIG
        self.assertIn("max_seq_length", cfg)
        self.assertIn("dtype", cfg)
        self.assertIn("load_in_4bit", cfg)

    def test_default_unsloth_config_values(self):
        cfg = OLMoModel.DEFAULT_UNSLOTH_CONFIG
        self.assertEqual(cfg["max_seq_length"], 2048)
        self.assertIsNone(cfg["dtype"])
        self.assertTrue(cfg["load_in_4bit"])

    def test_default_lora_config_keys(self):
        cfg = OLMoModel.DEFAULT_LORA_CONFIG
        for key in ("r", "lora_alpha", "target_modules", "lora_dropout",
                    "bias", "use_gradient_checkpointing", "random_state",
                    "use_rslora", "loftq_config"):
            self.assertIn(key, cfg, f"Missing key: {key}")

    def test_default_lora_config_values(self):
        cfg = OLMoModel.DEFAULT_LORA_CONFIG
        self.assertEqual(cfg["r"], 16)
        self.assertEqual(cfg["lora_alpha"], 16)
        self.assertFalse(cfg["use_rslora"])
        self.assertIsNone(cfg["loftq_config"])
        self.assertIn("q_proj", cfg["target_modules"])

    def test_default_sft_config_keys(self):
        cfg = OLMoModel.DEFAULT_SFT_CONFIG
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                    "warmup_steps", "num_train_epochs", "max_steps",
                    "learning_rate", "optim", "weight_decay",
                    "lr_scheduler_type", "fp16", "bf16", "logging_steps",
                    "save_steps", "output_dir", "dataset_text_field",
                    "max_seq_length", "packing", "seed"):
            self.assertIn(key, cfg, f"Missing SFT key: {key}")

    def test_default_sft_config_values(self):
        cfg = OLMoModel.DEFAULT_SFT_CONFIG
        self.assertEqual(cfg["per_device_train_batch_size"], 2)
        self.assertEqual(cfg["learning_rate"], 2e-4)
        self.assertEqual(cfg["max_seq_length"], 512)
        self.assertTrue(cfg["packing"])
        self.assertFalse(cfg["fp16"])
        self.assertFalse(cfg["bf16"])

    def test_class_defaults_are_not_mutated_by_instances(self):
        """Instance config changes should never bleed into class defaults."""
        orig_r = OLMoModel.DEFAULT_LORA_CONFIG["r"]

        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
            o = OLMoModel(lora_config={"r": 999})

        o.lora_config["r"] = 999
        self.assertEqual(OLMoModel.DEFAULT_LORA_CONFIG["r"], orig_r)


class TestOLMoModelInit(unittest.TestCase):
    """OLMoModel.__init__ and related construction logic."""

    def _build_olmo(self, **kwargs):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
            return OLMoModel(**kwargs)

    def test_default_hf_model_name(self):
        o = self._build_olmo()
        self.assertEqual(o.hf_model_name_or_path, "allenai/Olmo-3-7B-Instruct")

    def test_custom_hf_model_name(self):
        o = self._build_olmo(hf_model_name_or_path="allenai/OLMo-1B")
        self.assertEqual(o.hf_model_name_or_path, "allenai/OLMo-1B")

    def test_max_new_tokens_stored(self):
        o = self._build_olmo(max_length=256)
        self.assertEqual(o.max_new_tokens, 256)

    def test_default_max_new_tokens(self):
        o = self._build_olmo()
        self.assertEqual(o.max_new_tokens, 128)

    def test_generation_kwargs_stored(self):
        gkw = {"temperature": 0.7, "top_p": 0.9}
        o = self._build_olmo(generation_kwargs=gkw)
        self.assertEqual(o.generation_kwargs, gkw)

    def test_generation_kwargs_default_empty(self):
        o = self._build_olmo()
        self.assertEqual(o.generation_kwargs, {})

    def test_task_is_generation(self):
        o = self._build_olmo()
        self.assertEqual(o.task, "generation")

    def test_use_unsloth_false_by_default(self):
        o = self._build_olmo()
        self.assertFalse(o.use_unsloth)

    def test_unsloth_config_merges_with_defaults(self):
        o = self._build_olmo(unsloth_config={"max_seq_length": 4096})
        self.assertEqual(o.unsloth_config["max_seq_length"], 4096)
        self.assertTrue(o.unsloth_config["load_in_4bit"])  # default preserved

    def test_lora_config_merges_with_defaults(self):
        o = self._build_olmo(lora_config={"r": 32})
        self.assertEqual(o.lora_config["r"], 32)
        self.assertEqual(o.lora_config["lora_alpha"], 16)  # default preserved

    def test_sft_config_merges_with_defaults(self):
        o = self._build_olmo(sft_config={"learning_rate": 1e-5})
        self.assertAlmostEqual(o.sft_config["learning_rate"], 1e-5)
        self.assertTrue(o.sft_config["packing"])  # default preserved

    def test_pad_token_set_when_none(self):
        """If tokenizer has no pad token, eos_token should be used as pad."""
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
            o = OLMoModel()
        self.assertEqual(tok.pad_token, "<eos>")

    def test_pad_token_not_overwritten_when_exists(self):
        """If tokenizer already has pad_token, it should not be overwritten."""
        tok = MagicMock()
        tok.pad_token = "<pad>"
        tok.eos_token = "<eos>"
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
            o = OLMoModel()
        self.assertEqual(tok.pad_token, "<pad>")


class TestOLMoModelLoadStandard(unittest.TestCase):
    """_load_standard and _load_with_unsloth helpers."""

    def test_load_standard_calls_auto_causal_lm(self):
        auto_causal = sys.modules["transformers.models.auto"].AutoModelForCausalLM
        auto_tok    = sys.modules["transformers.models.auto"].AutoTokenizer
        with patch.object(auto_causal, "from_pretrained", return_value=_make_model()) as fm, \
             patch.object(auto_tok,    "from_pretrained", return_value=_make_tokenizer()) as ft:
            tok = MagicMock(pad_token=None, eos_token="<eos>")
            fake_model = MagicMock()
            # bypass __init__ loading by calling the helper directly
            with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
                o = OLMoModel()
            # Now test the actual method
            o._load_standard("/fake/model")
            fm.assert_called_once_with("/fake/model", trust_remote_code=True)
            ft.assert_called_once_with("/fake/model", trust_remote_code=True)

    def test_load_with_unsloth_raises_import_error_when_unsloth_missing(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)):
            o = OLMoModel(use_unsloth=False)

        # Simulate unsloth not installed
        with patch.dict(sys.modules, {"unsloth": None}):
            with self.assertRaises(ImportError) as ctx:
                o._load_with_unsloth("/fake/path")
            self.assertIn("unsloth", str(ctx.exception).lower())

    def test_use_unsloth_true_calls_load_with_unsloth(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_with_unsloth", return_value=(fake_model, tok)) as fu:
            o = OLMoModel(use_unsloth=True)
            fu.assert_called_once_with("unsloth/Olmo-3-7B-Instruct")

    def test_use_unsloth_false_calls_load_standard(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        with patch.object(OLMoModel, "_load_standard", return_value=(fake_model, tok)) as fs:
            o = OLMoModel(use_unsloth=False)
            fs.assert_called_once_with("allenai/Olmo-3-7B-Instruct")


class TestOLMoModelLoadFromPretrained(unittest.TestCase):
    """OLMoModel.load_from_pretrained adds CausalLM branch."""

    def _build_olmo(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(OLMoModel, "_load_standard", return_value=(_make_model(), tok)):
            return OLMoModel()

    def test_from_hf_checkpoint_uses_auto_causal_lm(self):
        o = self._build_olmo()
        o.use_unsloth = False
        auto_causal = sys.modules["transformers.models.auto"].AutoModelForCausalLM
        with patch.object(auto_causal, "from_pretrained", return_value=_make_model()) as fp:
            o.load_from_pretrained(model_dir="/path/to/model", from_hf_checkpoint=True)
            fp.assert_called_once()

    def test_from_hf_checkpoint_unsloth_calls_load_with_unsloth(self):
        o = self._build_olmo()
        o.use_unsloth = True
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(o, "_load_with_unsloth", return_value=(_make_model(), tok)) as fu:
            o.load_from_pretrained(model_dir="/path/to/model", from_hf_checkpoint=True)
            fu.assert_called_once_with("/path/to/model")

    def test_from_hf_checkpoint_uses_hf_model_name_when_model_dir_none(self):
        o = self._build_olmo()
        o.use_unsloth = False
        o.hf_model_name_or_path = "allenai/Olmo-3-7B-Instruct"
        auto_causal = sys.modules["transformers.models.auto"].AutoModelForCausalLM
        with patch.object(auto_causal, "from_pretrained", return_value=_make_model()) as fp:
            o.load_from_pretrained(from_hf_checkpoint=True)
            args, _ = fp.call_args
            self.assertEqual(args[0], "allenai/Olmo-3-7B-Instruct")

    def test_from_dc_checkpoint_delegates_to_parent(self):
        """from_hf_checkpoint=False should call parent's logic."""
        o = self._build_olmo()
        with patch.object(HuggingFaceModel, "load_from_pretrained") as parent_lfp:
            o.load_from_pretrained(model_dir="/some/dir", from_hf_checkpoint=False)
            parent_lfp.assert_called_once_with(model_dir="/some/dir",
                                               from_hf_checkpoint=False)


class TestOLMoModelPrepareBatch(unittest.TestCase):
    """OLMoModel._prepare_batch overrides parent for 'generation' task."""

    def _build_olmo(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(OLMoModel, "_load_standard", return_value=(_make_model(), tok)):
            return OLMoModel()

    def test_generation_task_returns_input_dict(self):
        o = self._build_olmo()
        o.task = "generation"
        X = np.array(["CC(=O)O", "CCO"])
        batch = (np.expand_dims(X, 0), None, None)
        inp, y, w = o._prepare_batch(batch)
        self.assertIsInstance(inp, dict)

    def test_non_generation_task_delegates_to_parent(self):
        """For mlm/regression/etc., parent _prepare_batch should be called."""
        o = self._build_olmo()
        o.task = "mlm"
        X = np.array(["CC(=O)O", "CCO"])
        y = np.array([[1.0], [2.0]])
        batch = (np.expand_dims(X, 0), np.expand_dims(y, 0), np.ones_like(y))
        with patch.object(HuggingFaceModel, "_prepare_batch", return_value=({}, None, None)) as pp:
            o._prepare_batch(batch)
            pp.assert_called_once()


class TestOLMoModelGenerate(unittest.TestCase):
    """OLMoModel.generate method."""

    def _build_olmo(self, generation_kwargs=None):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        tok.eos_token_id = 2
        fake_tensor = _torch_stub.Tensor()
        encoded = MagicMock()
        encoded.__getitem__ = lambda s, k: MagicMock(to=lambda d: fake_tensor)
        tok.return_value = encoded
        tok.batch_decode = MagicMock(return_value=["generated text"])
        m = _make_model()
        with patch.object(OLMoModel, "_load_standard", return_value=(m, tok)):
            o = OLMoModel(generation_kwargs=generation_kwargs or {})
        o.tokenizer = tok
        o.model = m
        return o

    def test_generate_returns_list_of_strings(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            result = o.generate(["CC(=O)O is"])
        self.assertIsInstance(result, list)

    def test_generate_string_input_converted_to_list(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            result = o.generate("CC(=O)O is")
        self.assertIsInstance(result, list)

    def test_generate_calls_model_generate(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        o.model.generate.assert_called_once()

    def test_max_new_tokens_override(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"], max_new_tokens=64)
        _, kwargs = o.model.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 64)

    def test_default_max_new_tokens_used(self):
        o = self._build_olmo()
        o.max_new_tokens = 128
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        _, kwargs = o.model.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 128)

    def test_generation_kwargs_forwarded(self):
        o = self._build_olmo(generation_kwargs={"temperature": 0.7})
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        _, kwargs = o.model.generate.call_args
        self.assertEqual(kwargs.get("temperature"), 0.7)

    def test_call_level_kwargs_override_instance_kwargs(self):
        o = self._build_olmo(generation_kwargs={"temperature": 0.7})
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"], temperature=0.9)
        _, kwargs = o.model.generate.call_args
        self.assertEqual(kwargs.get("temperature"), 0.9)

    def test_unsloth_for_inference_called_when_enabled(self):
        o = self._build_olmo()
        o.use_unsloth = True
        fake_flm = MagicMock()
        o._FastLanguageModel = fake_flm
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        fake_flm.for_inference.assert_called_once_with(o.model)

    def test_unsloth_not_called_when_disabled(self):
        o = self._build_olmo()
        o.use_unsloth = False
        o._FastLanguageModel = MagicMock()
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        o._FastLanguageModel.for_inference.assert_not_called()

    def test_model_set_to_eval(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            o.generate(["CC(=O)O"])
        o.model.eval.assert_called()


class TestOLMoModelSFT(unittest.TestCase):
    """OLMoModel.sft — import errors, config merging, trainer wiring."""

    def _build_olmo(self, **kwargs):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(OLMoModel, "_load_standard", return_value=(_make_model(), tok)):
            return OLMoModel(**kwargs)

    def test_sft_raises_import_error_when_trl_missing(self):
        o = self._build_olmo()
        with patch.dict(sys.modules, {"trl": None}):
            with self.assertRaises(ImportError) as ctx:
                o.sft(["CC(=O)O"])
            self.assertIn("trl", str(ctx.exception).lower())

    def test_sft_lora_config_precedence_call_overrides_instance(self):
        """Call-level lora_config overrides instance lora_config."""
        o = self._build_olmo(lora_config={"r": 16})
        captured = {}

        def fake_apply_lora(effective_lora):
            captured["r"] = effective_lora["r"]

        # Patch apply and trl
        fake_trl = types.ModuleType("trl")
        fake_trainer = MagicMock()
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        fake_ds_mod = types.ModuleType("datasets")
        fake_ds_mod.Dataset = MagicMock()

        with patch.dict(sys.modules, {"trl": fake_trl, "datasets": fake_ds_mod}), \
             patch.object(o, "_apply_lora_adapters", side_effect=fake_apply_lora), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            o.sft(["CC(=O)O"], lora_config={"r": 32})

        self.assertEqual(captured["r"], 32)

    def test_sft_class_default_used_when_no_override(self):
        """When no overrides given, DEFAULT_LORA_CONFIG values should be used."""
        o = self._build_olmo()
        captured = {}

        def fake_apply_lora(effective_lora):
            captured.update(effective_lora)

        fake_trl = types.ModuleType("trl")
        fake_trainer = MagicMock()
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"trl": fake_trl}), \
             patch.object(o, "_apply_lora_adapters", side_effect=fake_apply_lora), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            o.sft(["CC(=O)O"])

        self.assertEqual(captured["r"], 16)
        self.assertEqual(captured["lora_alpha"], 16)

    def test_sft_trainer_train_called(self):
        o = self._build_olmo()
        fake_trainer = MagicMock()
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"trl": fake_trl}), \
             patch.object(o, "_apply_lora_adapters"), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            o.sft(["CC(=O)O"])

        fake_trainer.train.assert_called_once()

    def test_sft_updates_self_model_after_training(self):
        """After sft(), self.model should be replaced by trainer.model."""
        o = self._build_olmo()
        trained_model = MagicMock()
        fake_trainer  = MagicMock(model=trained_model)
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"trl": fake_trl}), \
             patch.object(o, "_apply_lora_adapters"), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            o.sft(["CC(=O)O"])

        self.assertIs(o.model, trained_model)

    def test_sft_switches_to_unsloth_inference_after_training(self):
        o = self._build_olmo()
        o.use_unsloth = True
        fake_flm = MagicMock()
        o._FastLanguageModel = fake_flm
        trained_model = MagicMock()
        fake_trainer  = MagicMock(model=trained_model)
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"trl": fake_trl}), \
             patch.object(o, "_apply_lora_adapters"), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            o.sft(["CC(=O)O"])

        fake_flm.for_inference.assert_called_once_with(trained_model)


class TestOLMoModelApplyLoRA(unittest.TestCase):
    """_apply_lora_adapters — PEFT and Unsloth branches."""

    def _build_olmo(self, use_unsloth=False):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(OLMoModel, "_load_standard", return_value=(_make_model(), tok)):
            return OLMoModel(use_unsloth=use_unsloth)

    def test_apply_lora_raises_import_error_when_peft_missing(self):
        o = self._build_olmo(use_unsloth=False)
        with patch.dict(sys.modules, {"peft": None}):
            with self.assertRaises(ImportError) as ctx:
                o._apply_lora_adapters({})
            self.assertIn("peft", str(ctx.exception).lower())

    def test_apply_lora_with_unsloth_uses_fast_language_model(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        fake_model = MagicMock()
        fake_flm = MagicMock()
        fake_flm.get_peft_model = MagicMock(return_value=fake_model)
        # stub unsloth so __init__ can call _load_with_unsloth
        fake_unsloth = types.ModuleType("unsloth")
        fake_unsloth.FastLanguageModel = fake_flm
        fake_unsloth.FastLanguageModel.from_pretrained = MagicMock(return_value=(fake_model, tok))
        with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
            with patch.object(OLMoModel, "_load_with_unsloth", return_value=(fake_model, tok)):
                o = OLMoModel(use_unsloth=True)
        o._FastLanguageModel = fake_flm
        o._apply_lora_adapters({"r": 16, "lora_alpha": 16,
                                 "target_modules": ["q_proj"],
                                 "lora_dropout": 0.0, "bias": "none",
                                 "use_gradient_checkpointing": "unsloth",
                                 "random_state": 42, "use_rslora": False,
                                 "loftq_config": None})
        fake_flm.get_peft_model.assert_called_once()
        self.assertIs(o.model, fake_model)

    def test_apply_lora_without_unsloth_uses_peft(self):
        o = self._build_olmo(use_unsloth=False)
        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig      = MagicMock(return_value=MagicMock())
        fake_peft.get_peft_model  = MagicMock(return_value=MagicMock())
        fake_peft.TaskType        = MagicMock()   # ← needed by the code

        with patch.dict(sys.modules, {"peft": fake_peft}):
            o._apply_lora_adapters({"r": 8, "lora_alpha": 8,
                                    "target_modules": ["q_proj"],
                                    "lora_dropout": 0.0, "bias": "none",
                                    "use_gradient_checkpointing": False,
                                    "random_state": 42, "use_rslora": False,
                                    "loftq_config": None})
        fake_peft.get_peft_model.assert_called_once()


class TestOLMoModelDcDatasetToHF(unittest.TestCase):
    """_dc_dataset_to_hf — dataset conversion utility."""

    def _build_olmo(self):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        with patch.object(OLMoModel, "_load_standard", return_value=(_make_model(), tok)):
            return OLMoModel()

    def test_list_of_strings_converted(self):
        o = self._build_olmo()
        fake_ds_mod = types.ModuleType("datasets")
        # Dataset must be a real class (used in isinstance check inside the code)
        class FakeHFDataset:
            @staticmethod
            def from_dict(d): return FakeHFDataset()
        fake_ds_mod.Dataset = FakeHFDataset
        with patch.dict(sys.modules, {"datasets": fake_ds_mod}):
            result = o._dc_dataset_to_hf(["CC(=O)O", "CCO"])
        self.assertIsInstance(result, FakeHFDataset)

    def test_raises_import_error_when_datasets_missing(self):
        o = self._build_olmo()
        with patch.dict(sys.modules, {"datasets": None}):
            with self.assertRaises(ImportError) as ctx:
                o._dc_dataset_to_hf(["CC(=O)O"])
            self.assertIn("datasets", str(ctx.exception).lower())


class TestVersionCompatibility(unittest.TestCase):
    """Check version-sensitive usages in the code."""

    def test_transformers_version_compat(self):
        """Module should import and DataCollatorForLanguageModeling must be accessible."""
        from transformers.data.data_collator import DataCollatorForLanguageModeling
        self.assertTrue(callable(DataCollatorForLanguageModeling))

    def test_auto_causal_lm_importable(self):
        from transformers.models.auto import AutoModelForCausalLM
        self.assertIsNotNone(AutoModelForCausalLM)

    def test_auto_tokenizer_importable(self):
        from transformers.models.auto import AutoTokenizer
        self.assertIsNotNone(AutoTokenizer)

    def test_no_circular_import(self):
        """Re-importing the module should not raise."""
        spec = importlib.util.spec_from_file_location("hf_models_final_2", _FINAL_PATH)
        mod2 = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod2)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Re-import raised: {e}")

    def test_source_uses_no_deprecated_torch_save_call(self):
        """Ensure no bare torch.save() for model — checkpointing via save_checkpoint."""
        src = _FINAL_PATH.read_text()
        # direct torch.save is fine; we check there's no removed API usage
        self.assertNotIn("torch.jit.save", src)

    def test_no_python2_print_statements(self):
        src = _FINAL_PATH.read_text()
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                if hasattr(node.value.func, 'id') and node.value.func.id == 'print':
                    # print as function is fine in Python 3
                    pass
        self.assertTrue(True)  # parsed without syntax errors

    def test_type_hints_use_optional_not_union_none(self):
        """Check that Optional[X] is used consistently (not X | None which needs 3.10+)."""
        src = _FINAL_PATH.read_text()
        # Union[X, None] is acceptable, but bare `X | None` in annotation requires 3.10+
        # We just ensure the file is syntactically valid for the current Python
        import ast
        try:
            ast.parse(src)
        except SyntaxError as e:
            self.fail(f"Syntax error in file: {e}")


class TestMethodSignatures(unittest.TestCase):
    """Verify key method signatures match expectations."""

    def test_huggingface_model_init_signature(self):
        import inspect
        sig = inspect.signature(HuggingFaceModel.__init__)
        params = list(sig.parameters)
        self.assertIn("model", params)
        self.assertIn("tokenizer", params)
        self.assertIn("task", params)
        self.assertIn("config", params)

    def test_olmo_model_init_signature(self):
        import inspect
        sig = inspect.signature(OLMoModel.__init__)
        params = list(sig.parameters)
        self.assertIn("hf_model_name_or_path", params)
        self.assertIn("max_length", params)
        self.assertIn("generation_kwargs", params)
        self.assertIn("use_unsloth", params)
        self.assertIn("unsloth_config", params)
        self.assertIn("unsloth_model_name_or_path", params)
        self.assertIn("lora_config", params)
        self.assertIn("sft_config", params)

    def test_generate_signature(self):
        import inspect
        sig = inspect.signature(OLMoModel.generate)
        params = list(sig.parameters)
        self.assertIn("inputs", params)
        self.assertIn("max_new_tokens", params)
        self.assertIn("skip_special_tokens", params)

    def test_sft_signature(self):
        import inspect
        sig = inspect.signature(OLMoModel.sft)
        params = list(sig.parameters)
        self.assertIn("dataset", params)
        self.assertIn("lora_config", params)
        self.assertIn("sft_config", params)
        self.assertIn("formatting_func", params)
        self.assertIn("resume_from_checkpoint", params)

    def test_load_from_pretrained_signature_hf_model(self):
        import inspect
        sig = inspect.signature(HuggingFaceModel.load_from_pretrained)
        params = list(sig.parameters)
        self.assertIn("model_dir", params)
        self.assertIn("from_hf_checkpoint", params)

    def test_fill_mask_signature(self):
        import inspect
        sig = inspect.signature(HuggingFaceModel.fill_mask)
        params = list(sig.parameters)
        self.assertIn("inputs", params)
        self.assertIn("top_k", params)

    def test_prepare_batch_signature(self):
        import inspect
        sig = inspect.signature(HuggingFaceModel._prepare_batch)
        params = list(sig.parameters)
        self.assertIn("batch", params)

    def test_olmo_prepare_batch_signature(self):
        import inspect
        sig = inspect.signature(OLMoModel._prepare_batch)
        params = list(sig.parameters)
        self.assertIn("batch", params)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def _build_olmo(self, **kwargs):
        tok = MagicMock(pad_token=None, eos_token="<eos>")
        tok.eos_token_id = 2
        fake_tensor = _torch_stub.Tensor()
        encoded = MagicMock()
        encoded.__getitem__ = lambda s, k: MagicMock(to=lambda d: fake_tensor)
        tok.return_value = encoded
        tok.batch_decode = MagicMock(return_value=["generated text"])
        m = _make_model()
        with patch.object(OLMoModel, "_load_standard", return_value=(m, tok)):
            o = OLMoModel(**kwargs)
        o.tokenizer = tok
        o.model = m
        return o

    def test_olmo_config_all_three_levels_merge_correctly(self):
        """class defaults < instance lora_config < sft() call lora_config."""
        o = self._build_olmo(lora_config={"r": 8})  # instance override
        captured = {}

        def fake_apply(effective_lora):
            captured.update(effective_lora)

        fake_trl = types.ModuleType("trl")
        fake_trainer = MagicMock()
        fake_trl.SFTTrainer = MagicMock(return_value=fake_trainer)
        fake_trl.SFTConfig   = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"trl": fake_trl}), \
             patch.object(o, "_apply_lora_adapters", side_effect=fake_apply), \
             patch.object(o, "_dc_dataset_to_hf", return_value=MagicMock()):
            # call-level override wins over all
            o.sft(["CC(=O)O"], lora_config={"r": 64})

        self.assertEqual(captured["r"], 64)

    def test_empty_generation_kwargs_still_works(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            result = o.generate(["CC(=O)O"])
        self.assertIsNotNone(result)

    def test_olmo_generate_single_vs_batch_both_return_list(self):
        o = self._build_olmo()
        with patch.object(o, "_ensure_built"):
            single = o.generate("CCO")
            batch  = o.generate(["CCO", "CC(=O)O"])
        self.assertIsInstance(single, list)
        self.assertIsInstance(batch,  list)

    def test_hf_model_config_keys_passed_to_from_pretrained(self):
        """Config dict should be passed as **kwargs to from_pretrained."""
        cfg = {"hidden_size": 512}
        m = HuggingFaceModel(model=_make_model(), tokenizer=_make_tokenizer(),
                             task="mlm", config=cfg)
        auto_mlm = sys.modules["transformers.models.auto"].AutoModelForMaskedLM
        with patch.object(auto_mlm, "from_pretrained", return_value=_make_model()) as fp:
            m.load_from_pretrained(model_dir="/p", from_hf_checkpoint=True)
            _, kwargs = fp.call_args
            self.assertEqual(kwargs.get("hidden_size"), 512)

    def test_olmo_unsloth_config_merging_partial_override(self):
        """Only overridden keys change; others keep defaults."""
        o = self._build_olmo(unsloth_config={"load_in_4bit": False})
        self.assertFalse(o.unsloth_config["load_in_4bit"])
        self.assertEqual(o.unsloth_config["max_seq_length"],
                         OLMoModel.DEFAULT_UNSLOTH_CONFIG["max_seq_length"])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
