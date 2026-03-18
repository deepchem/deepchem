import pytest
import numpy as np
from deepchem.data import ImageDataset
try:
    import torch
    from deepchem.models.torch_models.oneformer import OneFormer
except ModuleNotFoundError:
    pass


@pytest.mark.hf
def test_oneformer_train():
    from transformers import OneFormerConfig

    # micro config for testing
    config = OneFormerConfig().from_pretrained(
        'shi-labs/oneformer_ade20k_swin_tiny', is_training=True)
    config.encoder_layers = 2
    config.text_encoder_num_layers = 2
    config.decoder_layers = 2
    config.num_attention_heads = 2
    config.encoder_feedforward_dim = 16
    config.dim_feedforward = 16

    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      model_config=config,
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1)
    X = np.random.randint(0, 255, (3, 224, 224, 3))
    y = np.random.randint(0, 1, (3, 224, 224))

    dataset = ImageDataset(X, y)
    avg_loss = model.fit(dataset, nb_epoch=2)

    assert isinstance(avg_loss, float)


@pytest.mark.hf
def test_oneformer_predict():
    from transformers import OneFormerConfig

    # micro config for testing
    config = OneFormerConfig().from_pretrained(
        'shi-labs/oneformer_ade20k_swin_tiny', is_training=True)
    config.encoder_layers = 2
    config.text_encoder_num_layers = 2
    config.decoder_layers = 2
    config.num_attention_heads = 2
    config.encoder_feedforward_dim = 16
    config.dim_feedforward = 16

    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      model_config=config,
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1)
    X = np.random.randint(0, 255, (3, 224, 224, 3))
    y = np.random.randint(0, 1, (3, 224, 224))

    dataset = ImageDataset(X, y)
    preds = model.predict(dataset)
    preds = np.array(preds)

    assert np.array(preds).shape == y.shape


@pytest.mark.hf
def test_oneformer_save_reload(tmpdir):
    from transformers import OneFormerConfig
    # micro config for testing
    config = OneFormerConfig().from_pretrained(
        'shi-labs/oneformer_ade20k_swin_tiny', is_training=True)
    config.encoder_layers = 2
    config.text_encoder_num_layers = 2
    config.decoder_layers = 2
    config.num_attention_heads = 2
    config.encoder_feedforward_dim = 16
    config.dim_feedforward = 16

    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      model_config=config,
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1,
                      model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                          model_config=config,
                          segmentation_task="semantic",
                          torch_dtype=torch.float16,
                          batch_size=1,
                          model_dir=tmpdir)
    model_new.restore()

    old_state = model.model.state_dict()
    new_state = model_new.model.state_dict()
    matches = [
        torch.allclose(old_state[key], new_state[key])
        for key in old_state.keys()
    ]

    # all keys values should match
    assert all(matches)


@pytest.mark.hf
def test_oneformer_overfit():
    from transformers import OneFormerConfig
    import deepchem as dc

    # micro config for testing
    config = OneFormerConfig().from_pretrained(
        'shi-labs/oneformer_ade20k_swin_tiny', is_training=True)
    config.encoder_layers = 2
    config.text_encoder_num_layers = 2
    config.decoder_layers = 2
    config.num_attention_heads = 2
    config.encoder_feedforward_dim = 16
    config.dim_feedforward = 16

    model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                      model_config=config,
                      segmentation_task="semantic",
                      torch_dtype=torch.float16,
                      batch_size=1)

    X = np.random.randint(0, 255, (3, 224, 224, 3))
    y = np.random.randint(0, 1, (3, 224, 224))
    dataset = ImageDataset(X, y)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    model.fit(dataset, nb_epoch=3)
    eval_score = model.evaluate(dataset, [classification_metric])

    assert eval_score[classification_metric.name] > 0.8, "Failed to overfit"
