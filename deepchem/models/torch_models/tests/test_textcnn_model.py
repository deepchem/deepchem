import pytest
import numpy as np
import deepchem as dc
import os
try:
    import torch
    from deepchem.models.torch_models import TextCNNModel
    from deepchem.models.text_cnn import default_dict
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_textcnn_module():
    model = TextCNNModel(1, default_dict, 1)
    assert model.seq_length == max(model.kernel_sizes)
    large_length = 500
    model = TextCNNModel(1, default_dict, large_length)
    assert model.seq_length == large_length


@pytest.mark.torch
def test_overfit_classification():
    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_classification.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)

    char_dict, length = TextCNNModel.build_char_dict(dataset)
    batch_size = 10

    model = TextCNNModel(n_tasks,
                         char_dict=char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="classification",
                         log_frequency=10)

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [classification_metric])

    assert scores[classification_metric.name] > .8


@pytest.mark.torch
def test_overfit_regression():
    """Test textCNN model overfits tiny data."""
    np.random.seed(123)
    n_tasks = 1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Load mini log-solubility dataset.
    featurizer = dc.feat.RawFeaturizer()
    tasks = ["outcome"]
    input_file = os.path.join(current_dir,
                              "../../tests/assets/example_regression.csv")
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="smiles",
                               featurizer=featurizer)
    dataset = loader.create_dataset(input_file)

    regression_metric = dc.metrics.Metric(dc.metrics.pearson_r2_score,
                                          task_averager=np.mean)

    char_dict, length = TextCNNModel.build_char_dict(dataset)
    batch_size = 10

    model = TextCNNModel(n_tasks,
                         char_dict,
                         seq_length=length,
                         batch_size=batch_size,
                         learning_rate=0.001,
                         use_queue=False,
                         mode="regression")

    # Fit trained model
    model.fit(dataset, nb_epoch=200)

    # Eval model on train
    scores = model.evaluate(dataset, [regression_metric])

    assert scores[regression_metric.name] > .9


# @pytest.mark.torch
# def test_overfit():
#     delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
#         featurizer='Raw', split='index')
#     train_dataset, valid_dataset, test_dataset = delaney_datasets
#     # print("TEST DATASET")
#     # print(test_dataset)
#     # print(test_dataset.X)
#     # smiles = np.array(['CC', 'CCC'])
#     # dataset = dc.data.NumpyDataset(X=smiles,y=np.array([1.0,0.0]),ids=smiles)
#     char_dict, length = TextCNNModel.build_char_dict(test_dataset)

#     model = TextCNNModel(n_tasks=len(delaney_tasks),
#                          char_dict=char_dict,
#                          seq_length=length,
#                          mode='regression',
#                          learning_rate=1e-3,
#                          batch_size=10,
#                          use_queue=False)
#     # print(model.summary())
#     model.fit(test_dataset, nb_epoch=100)

# embedding shape  (None, 73, 75)
# conv output shape:  (None, 73, 100)
# conv output shape:  (None, 72, 200)
# conv output shape:  (None, 71, 200)
# conv output shape:  (None, 70, 200)
# conv output shape:  (None, 69, 200)
# conv output shape:  (None, 68, 100)
# conv output shape:  (None, 67, 100)
# conv output shape:  (None, 66, 100)
# conv output shape:  (None, 65, 100)
# conv output shape:  (None, 64, 100)
# conv output shape:  (None, 59, 160)
# conv output shape:  (None, 54, 160)
