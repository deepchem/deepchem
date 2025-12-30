import pytest
import numpy as np
import tempfile
import os
import deepchem as dc
try:
    import torch
except ModuleNotFoundError:
    pass


def get_dataset(mode="classification",
                featurizer="smiles2img",
                data_points=10,
                img_spec='std',
                img_size=80,
                n_tasks=5):
    from deepchem.feat import SmilesToImage
    from deepchem.molnet.load_function.chembl25_datasets import CHEMBL25_TASKS
    np.random.seed(123)

    dataset_file = os.path.join(os.path.dirname(__file__), "assets",
                                "chembl_25_small.csv")

    if featurizer == "smiles2img":
        img_size = img_size
        img_spec = img_spec
        res = 0.5
        feat = SmilesToImage(img_size=img_size, img_spec=img_spec, res=res)

    loader = dc.data.CSVLoader(tasks=CHEMBL25_TASKS,
                               feature_field='smiles',
                               featurizer=feat)
    dataset = loader.create_dataset(inputs=[dataset_file],
                                    shard_size=10000,
                                    data_dir=tempfile.mkdtemp())
    w = np.ones(shape=(data_points, n_tasks))
    if mode == 'classification':
        y = np.random.randint(0, 2, size=(data_points, n_tasks))

        metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
                                   np.mean,
                                   mode="classification")
    else:
        y = np.random.normal(size=(data_points, n_tasks))
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error,
                                   mode="regression")
    dataset = dc.data.NumpyDataset(dataset.X[:data_points], y, w,
                                   dataset.ids[:data_points])
    return dataset, metric


@pytest.mark.torch
def test_chemception_forward():
    """To check that the model class beneath ChemCeptionModel is working properly by ensuring its output has the expected shape"""
    import torch.nn as nn
    from deepchem.models.torch_models.chemnet_layers import Stem, InceptionResnetA, InceptionResnetB, InceptionResnetC, ReductionA, ReductionB
    from deepchem.models.torch_models import ChemCeption

    DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}
    base_filters = 16
    img_spec = 'std'
    img_size = 80
    n_tasks = 10
    n_classes = 2
    in_channels = 1 if img_spec == "std" else 4
    mode = 'classification'

    components = {}
    components['stem'] = Stem(in_channels=in_channels,
                              out_channels=base_filters)
    components['inception_resnet_A'] = nn.Sequential(*[
        InceptionResnetA(base_filters, base_filters)
        for _ in range(DEFAULT_INCEPTION_BLOCKS['A'])
    ])
    components['reduction_A'] = ReductionA(base_filters, base_filters)

    components['inception_resnet_B'] = nn.Sequential(*[
        InceptionResnetB(4 * base_filters, base_filters)
        for _ in range(DEFAULT_INCEPTION_BLOCKS['B'])
    ])
    components['reduction_B'] = ReductionB(4 * base_filters, base_filters)

    current_channels = int(
        torch.floor(torch.tensor(7.875 * base_filters)).item())

    components['inception_resnet_C'] = nn.Sequential(*[
        InceptionResnetC(current_channels, base_filters)
        for _ in range(DEFAULT_INCEPTION_BLOCKS['C'])
    ])
    components['global_avg_pool'] = nn.AdaptiveAvgPool2d(1)
    if mode == "classification":
        components['fc_classification'] = nn.Linear(current_channels,
                                                    n_tasks * n_classes)
    else:
        components['fc_regression'] = nn.Linear(current_channels, n_tasks)

    smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    featurizer = dc.feat.SmilesToImage(img_size=img_size, img_spec='std')
    images = featurizer.featurize(smiles)
    image = torch.tensor(images, dtype=torch.float32)
    if mode == 'classification':
        output_layer = components['fc_classification']
    else:
        output_layer = components['fc_regression']
    input = image.permute(
        0, 3, 1, 2
    )  # to convert from channel last  (N,H,W,C) to pytorch default channel first (N,C,H,W) representation
    model = ChemCeption(stem=components['stem'],
                        inception_resnet_A=components['inception_resnet_A'],
                        reduction_A=components['reduction_A'],
                        inception_resnet_B=components['inception_resnet_B'],
                        reduction_B=components['reduction_B'],
                        inception_resnet_C=components['inception_resnet_C'],
                        global_avg_pool=components['global_avg_pool'],
                        output_layer=output_layer,
                        mode=mode,
                        n_tasks=n_tasks,
                        n_classes=n_classes)
    output = model(input)

    # preditions
    assert output[0].shape == (1, n_tasks, n_classes)
    # logits
    assert output[1].shape == (1, n_tasks, n_classes)


@pytest.mark.torch
def test_chemception_regression_overfit():
    """Overfit test the model to ensure it can learn a simple task."""
    from deepchem.models.torch_models import ChemCeptionModel
    torch.manual_seed(123)

    n_tasks = 1
    img_size = 80
    mode = 'regression'
    img_spec = "std"
    dataset, metric = get_dataset(mode=mode,
                                  featurizer="smiles2img",
                                  img_spec=img_spec,
                                  img_size=img_size,
                                  n_tasks=n_tasks)

    model = ChemCeptionModel(n_tasks=n_tasks,
                             img_spec=img_spec,
                             img_size=img_size,
                             augment=False,
                             mode=mode)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean_absolute_error'] < 0.1


@pytest.mark.torch
def test_chemception_classification_overfit():
    """Overfit test the model to ensure it can learn a simple task."""
    from deepchem.models.torch_models import ChemCeptionModel
    torch.manual_seed(123)

    n_tasks = 5
    img_size = 80
    mode = 'classification'
    img_spec = "std"
    dataset, metric = get_dataset(mode=mode,
                                  featurizer="smiles2img",
                                  img_spec=img_spec,
                                  img_size=img_size,
                                  n_tasks=n_tasks)

    model = ChemCeptionModel(n_tasks=n_tasks,
                             img_spec=img_spec,
                             img_size=img_size,
                             augment=False,
                             mode=mode)

    model.fit(dataset, nb_epoch=100)
    scores = model.evaluate(dataset, [metric])
    assert scores['mean-roc_auc_score'] >= 0.9


@pytest.mark.torch
def test_chemception_compare_with_tf_impl():
    """Compare the ouputs of tensorflow and torch 1implementations when model parameters are equal"""
    from deepchem.models.torch_models import ChemCeptionModel
    tf_weights_dir = os.path.join(os.path.dirname(__file__),
                                  "assets/chemception/")
    tf_regression_output = os.path.join(
        os.path.dirname(__file__),
        "assets/chemception/tf_regression_output.npy")

    n_tasks = 5
    mode = 'regression'
    img_spec = "std"
    dataset, metric = get_dataset(mode=mode,
                                  featurizer="smiles2img",
                                  img_spec=img_spec,
                                  n_tasks=n_tasks)
    torch_model = ChemCeptionModel(n_tasks=n_tasks,
                                   img_spec="std",
                                   model_dir=None,
                                   augment=False,
                                   mode=mode)

    layers = [
        'stem', 'inception_resnet_A', 'reduction_A', 'inception_resnet_B',
        'reduction_B', 'inception_resnet_C', 'output_layer'
    ]

    for layer_name in layers:
        layer_tf_weights = np.load(f'{tf_weights_dir}{layer_name}.npz')
        weigths_list = [layer_tf_weights[key] for key in layer_tf_weights.files]

        for i in range(len(weigths_list)):
            if len(weigths_list[i].shape) == 4:
                weigths_list[i] = np.transpose(weigths_list[i], (3, 2, 0, 1))
            elif len(weigths_list[i].shape) == 2:
                weigths_list[i] = np.transpose(weigths_list[i], (1, 0))

        count = 0
        with torch.no_grad():
            for name, param in torch_model.model.named_parameters():
                if layer_name in name:
                    if len(param.shape) == 1:
                        if weigths_list[count - 1].shape == param.shape:
                            param.copy_(
                                torch.from_numpy(weigths_list[count - 1]))
                    else:
                        if weigths_list[count + 1].shape == param.shape:
                            param.copy_(
                                torch.from_numpy(weigths_list[count + 1]))
                    count += 1
        count

    tf_model_predictions = np.load(tf_regression_output)
    torch_model_predictions = torch_model.predict(dataset)

    diff = tf_model_predictions - torch_model_predictions
    print(f"Max absolute difference: {np.max(np.abs(diff))}")
    print(f"Mean difference: {np.mean(diff)}")

    assert np.allclose(tf_model_predictions,
                       torch_model_predictions,
                       atol=1e-6,
                       rtol=1e-5)


@pytest.mark.torch
def test_chemception_modular_fit_restore():
    """Tests that the pretrainer can be restored from a checkpoint and resume training."""
    from deepchem.models.torch_models import ChemCeptionModel
    torch.manual_seed(123)

    n_tasks = 1
    mode = 'regression'
    img_spec = "std"
    img_size = 80
    dataset, metric = get_dataset(mode=mode,
                                  featurizer="smiles2img",
                                  img_spec=img_spec,
                                  img_size=img_size,
                                  n_tasks=n_tasks)

    chemception1 = ChemCeptionModel(n_tasks=n_tasks,
                                    img_spec=img_spec,
                                    img_size=img_size,
                                    augment=False,
                                    mode=mode)

    chemception1.fit(dataset, nb_epoch=300)

    # Create an identical model, do a single step of fitting with restore=True and make sure it got restored correctly.
    chemception2 = ChemCeptionModel(n_tasks=n_tasks,
                                    img_spec=img_spec,
                                    img_size=img_size,
                                    model_dir=chemception1.model_dir,
                                    augment=False,
                                    mode=mode)

    chemception2.fit(dataset, nb_epoch=1, restore=True)

    prediction = np.squeeze(chemception2.predict_on_batch(dataset.X), axis=-1)
    assert np.allclose(dataset.y, prediction, atol=1e-2)


@pytest.mark.torch
def test_chemception_load_from_pretrained():
    """Test to ensure the model can be pretrained in classification mode,
    reloaded in regression mode and weigts of all layers except the prediction head are copied"""
    from deepchem.models.torch_models import ChemCeptionModel

    n_tasks = 5
    img_size = 80
    img_spec = "std"

    dataset_pt, _ = get_dataset(mode='classification',
                                featurizer="smiles2img",
                                img_spec=img_spec,
                                img_size=img_size,
                                n_tasks=n_tasks)
    dataset_ft, _ = get_dataset(mode='regression',
                                featurizer="smiles2img",
                                img_spec=img_spec,
                                img_size=img_size,
                                n_tasks=n_tasks)

    model_pt = ChemCeptionModel(n_tasks=n_tasks,
                                img_spec=img_spec,
                                img_size=img_size,
                                augment=False,
                                mode='classification')
    model_pt.fit(dataset_pt, nb_epoch=1)

    model_ft = ChemCeptionModel(n_tasks=n_tasks,
                                img_spec=img_spec,
                                img_size=img_size,
                                augment=False,
                                mode='regression')

    # asserting that weights are not same before reloading
    for param_name in model_pt.model.state_dict().keys():
        if 'output_layer' not in param_name:
            assert not np.allclose(
                model_pt.model.get_parameter(param_name).detach().cpu(),
                model_ft.model.get_parameter(param_name).detach().cpu())

    # loading pretrained weights
    model_ft.load_from_pretrained(source_model=model_pt,
                                  components=[
                                      'stem', 'inception_resnet_A',
                                      'inception_resnet_B',
                                      'inception_resnet_C', 'reduction_A',
                                      'reduction_B'
                                  ])

    # asserting that weight matches after loading
    for param_name in model_pt.model.state_dict().keys():
        if 'output_layer' not in param_name:
            assert np.allclose(
                model_pt.model.get_parameter(param_name).detach().cpu(),
                model_ft.model.get_parameter(param_name).detach().cpu())

    model_ft.fit(dataset_ft, nb_epoch=1)
