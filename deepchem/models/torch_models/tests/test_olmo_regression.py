import gc
import os

import pytest
import deepchem as dc
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    pass

SMILES = [
    "CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1",
]


def quantization_config():
    if not torch.cuda.is_available():
        # bitsandbytes 4-bit quantization requires a CUDA GPU.
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


@pytest.mark.hf
def test_olmo_single_label_regression():
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=1,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 quantization_config=quantization_config(),
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0], [0.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 1)
    del model
    gc.collect()


@pytest.mark.hf
def test_olmo_single_label_regression_overfit():
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=1,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 quantization_config=quantization_config(),
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0], [0.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 1)
    del model
    gc.collect()


@pytest.mark.hf
def test_olmo_single_label_regression_load_from_pretrained(tmpdir):
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')

    pretrain_model = Olmo(task_type="regression",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=1,
                          torch_dtype=torch.float16
                          if torch.cuda.is_available() else torch.float32,
                          skip_weight_init=True,
                          gradient_checkpointing=True)
    pretrain_model.load_from_pretrained("allenai/OLMo-1B-hf",
                                        from_hf_checkpoint=True)

    pretrain_model.save_checkpoint(model_dir=pretrain_model_dir)
    finetune_model = Olmo(task_type="regression",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=1,
                          torch_dtype=torch.float16
                          if torch.cuda.is_available() else torch.float32,
                          model_dir=finetune_model_dir,
                          skip_weight_init=True,
                          gradient_checkpointing=True)
    finetune_model.load_from_pretrained(pretrain_model_dir)
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys()
        if key.startswith('model.')
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)
    del pretrain_model, finetune_model
    gc.collect()


@pytest.mark.hf
def test_olmo_single_label_regression_save_reload(tmpdir):
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo
    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=1,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 model_dir=tmpdir,
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)
    model._ensure_built()
    model.save_checkpoint()

    model_new = Olmo(task_type="regression",
                     tokenizer_path="allenai/OLMo-1B-hf",
                     n_tasks=1,
                     torch_dtype=torch.float16
                     if torch.cuda.is_available() else torch.float32,
                     model_dir=tmpdir,
                     skip_weight_init=True,
                     gradient_checkpointing=True)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)
    del model, model_new
    gc.collect()


@pytest.mark.hf
def test_olmo_multi_label_regression():
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="regression",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=2,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 quantization_config=quantization_config(),
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0, 0.0], [0.0, 1.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 2)
    del model
    gc.collect()


@pytest.mark.hf
def test_olmo_multi_label_regression_overfit():
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    model = Olmo(task_type="mtr",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=2,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 quantization_config=quantization_config(),
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)

    dataset = dc.data.NumpyDataset(SMILES, np.array([[1.0, 0.0], [0.0, 1.0]]))

    loss = model.fit(dataset, nb_epoch=1)
    assert loss is not None

    predictions = model.predict(dataset)
    assert predictions.shape == (len(SMILES), 2)
    del model
    gc.collect()


@pytest.mark.hf
def test_olmo_multi_label_regression_load_from_pretrained(tmpdir):
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo

    pretrain_model_dir = os.path.join(tmpdir, 'pretrain')
    finetune_model_dir = os.path.join(tmpdir, 'finetune')

    pretrain_model = Olmo(task_type="mtr",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=2,
                          torch_dtype=torch.float16
                          if torch.cuda.is_available() else torch.float32,
                          skip_weight_init=True,
                          gradient_checkpointing=True)
    pretrain_model.load_from_pretrained("allenai/OLMo-1B-hf",
                                        from_hf_checkpoint=True)

    pretrain_model.save_checkpoint(model_dir=pretrain_model_dir)
    finetune_model = Olmo(task_type="mtr",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=2,
                          torch_dtype=torch.float16
                          if torch.cuda.is_available() else torch.float32,
                          model_dir=finetune_model_dir,
                          skip_weight_init=True,
                          gradient_checkpointing=True)
    finetune_model.load_from_pretrained(pretrain_model_dir)
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys()
        if key.startswith('model.')
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)
    del pretrain_model, finetune_model
    gc.collect()


@pytest.mark.hf
def test_olmo_multi_label_regression_save_reload(tmpdir):
    pytest.skip(
        "Skipping slow OLMo test that downloads and runs the full model.")
    from deepchem.models.torch_models.olmo import Olmo
    model = Olmo(task_type="mtr",
                 tokenizer_path="allenai/OLMo-1B-hf",
                 n_tasks=2,
                 torch_dtype=torch.float16
                 if torch.cuda.is_available() else torch.float32,
                 model_dir=tmpdir,
                 skip_weight_init=True,
                 gradient_checkpointing=True)
    model.load_from_pretrained("allenai/OLMo-1B-hf", from_hf_checkpoint=True)
    model._ensure_built()
    model.save_checkpoint()

    model_new = Olmo(task_type="mtr",
                     tokenizer_path="allenai/OLMo-1B-hf",
                     n_tasks=2,
                     torch_dtype=torch.float16
                     if torch.cuda.is_available() else torch.float32,
                     model_dir=tmpdir,
                     skip_weight_init=True,
                     gradient_checkpointing=True)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)
    del model, model_new
    gc.collect()
