import os

import deepchem as dc
import pytest
import numpy as np
try:
    import torch
    from deepchem.models.torch_models.prot_bert import ProtBERT
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    pass
np.random.seed(32)
torch.manual_seed(32)

class SimpleCNN(nn.Module):

    def __init__(self, input_dim=1024, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_dim // 2),
                             num_classes)  # Adjusting for the pooling layer

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding channel dimension
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


@pytest.mark.torch
def test_prot_bert_load():
    # Test to ensure all variants of the model are supported
    model = ProtBERT(task='mlm', model_pretrain_dataset="uniref100", n_tasks=1)
    model = ProtBERT(task='mlm', model_pretrain_dataset="bfd", n_tasks=1)

    custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512),
                                                 nn.ReLU(), nn.Linear(512, 256),
                                                 nn.ReLU(), nn.Linear(256, 15))
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="uniref100",
                     cls_task="custom",
                     cls_head=custom_torch_cls_seq_network,
                     n_tasks=1,
                     n_classes=15)

    custom_torch_CNN_network = SimpleCNN()
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="custom",
                     cls_head=custom_torch_CNN_network,
                     n_tasks=1,
                     n_classes=2)

    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="membrane",
                     n_tasks=1)
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="subcellular location",
                     n_tasks=1)

    assert model


@pytest.mark.torch
def test_prot_bert_pretraining_mlm(protein_classification_dataset):
    model = ProtBERT(task='mlm', model_pretrain_dataset="uniref100", n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    assert loss

    model = ProtBERT(task='mlm', model_pretrain_dataset="bfd", n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    assert loss


@pytest.mark.torch
def test_prot_bert_finetuning(protein_classification_dataset):

    custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512),
                                                 nn.ReLU(), nn.Linear(512, 2))
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="custom",
                     cls_head=custom_torch_cls_seq_network,
                     n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    eval_score = model.evaluate(protein_classification_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.accuracy_score))
    assert eval_score, loss
    prediction = model.predict(protein_classification_dataset)
    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)

    custom_torch_CNN_network = SimpleCNN()
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="custom",
                     cls_head=custom_torch_CNN_network,
                     n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    eval_score = model.evaluate(protein_classification_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.accuracy_score))
    assert eval_score, loss
    prediction = model.predict(protein_classification_dataset)
    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)

    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="membrane",
                     n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    eval_score = model.evaluate(protein_classification_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.accuracy_score),
                                n_classes=2)
    assert eval_score, loss
    prediction = model.predict(protein_classification_dataset)
    assert prediction.shape == (protein_classification_dataset.y.shape[0], 2)

    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="subcellular location",
                     n_tasks=1)
    loss = model.fit(protein_classification_dataset, nb_epoch=1)
    eval_score = model.evaluate(protein_classification_dataset,
                                metrics=dc.metrics.Metric(
                                    dc.metrics.accuracy_score),
                                n_classes=10)
    assert eval_score, loss
    prediction = model.predict(protein_classification_dataset)
    assert prediction.shape == (protein_classification_dataset.y.shape[0], 10)


@pytest.mark.torch
def test_protbert_load_from_pretrained(tmpdir):

    pretrain_data_type = "uniref100"
    pretrain_model_dir = os.path.join(tmpdir,
                                      'pretrain_{}'.format(pretrain_data_type))
    finetune_model_dir = os.path.join(tmpdir,
                                      'finetune_{}'.format(pretrain_data_type))
    pretrain_model = ProtBERT(task='mlm',
                              model_pretrain_dataset=pretrain_data_type,
                              n_tasks=1,
                              model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()
    custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512),
                                                 nn.ReLU(), nn.Linear(512, 2))
    finetune_model = ProtBERT(task='classification',
                              model_pretrain_dataset=pretrain_data_type,
                              cls_task="custom",
                              cls_head=custom_torch_cls_seq_network,
                              n_tasks=1,
                              model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)

    pretrain_data_type = "bfd"
    pretrain_model_dir = os.path.join(tmpdir,
                                      'pretrain_{}'.format(pretrain_data_type))
    finetune_model_dir = os.path.join(tmpdir,
                                      'finetune_{}'.format(pretrain_data_type))
    pretrain_model = ProtBERT(task='mlm',
                              model_pretrain_dataset=pretrain_data_type,
                              n_tasks=1,
                              model_dir=pretrain_model_dir)
    pretrain_model.save_checkpoint()

    finetune_model = ProtBERT(task='classification',
                              model_pretrain_dataset="bfd",
                              cls_task="membrane",
                              n_tasks=1,
                              model_dir=finetune_model_dir)
    finetune_model.load_from_pretrained(pretrain_model_dir)

    # check weights match
    pretrain_model_state_dict = pretrain_model.model.state_dict()
    finetune_model_state_dict = finetune_model.model.state_dict()

    pretrain_base_model_keys = [
        key for key in pretrain_model_state_dict.keys() if 'bert' in key
    ]
    matches = [
        torch.allclose(pretrain_model_state_dict[key],
                       finetune_model_state_dict[key])
        for key in pretrain_base_model_keys
    ]

    assert all(matches)


@pytest.mark.torch
def test_protbert_save_reload(tmpdir):
    custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512),
                                                 nn.ReLU(), nn.Linear(512, 2))
    model = ProtBERT(task='classification',
                     model_pretrain_dataset="bfd",
                     cls_task="custom",
                     cls_head=custom_torch_cls_seq_network,
                     n_tasks=1,
                     model_dir=tmpdir)
    model._ensure_built()
    model.save_checkpoint()

    model_new = ProtBERT(task='classification',
                         model_pretrain_dataset="bfd",
                         cls_task="custom",
                         cls_head=custom_torch_cls_seq_network,
                         n_tasks=1,
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


@pytest.mark.torch
def test_protbert_overfit():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    featurizer = dc.feat.DummyFeaturizer()
    tasks = ["outcome"]
    loader = dc.data.CSVLoader(tasks=tasks,
                               feature_field="protein",
                               featurizer=featurizer)
    dataset = loader.create_dataset(
        os.path.join(current_dir,
                     "../../tests/assets/example_protein_classification.csv"))

    pretrain_data_type = "bfd"
    custom_torch_cls_seq_network = nn.Sequential(nn.Linear(1024, 512),
                                                 nn.ReLU(), nn.Linear(512, 2))
    finetune_model_bfd = ProtBERT(task='classification',
                                  model_pretrain_dataset=pretrain_data_type,
                                  cls_task="custom",
                                  cls_head=custom_torch_cls_seq_network,
                                  n_tasks=1,
                                  batch_size=5,
                                  learning_rate=1e-5)
    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    finetune_model_bfd.fit(dataset, nb_epoch=100)
    eval_score_bfd = finetune_model_bfd.evaluate(dataset,
                                                 [classification_metric])
    assert eval_score_bfd[classification_metric.name] > 0.9

    pretrain_data_type = "uniref100"

    finetune_model_uniref = ProtBERT(task='classification',
                                     model_pretrain_dataset=pretrain_data_type,
                                     cls_task="custom",
                                     cls_head=custom_torch_cls_seq_network,
                                     n_tasks=1,
                                     batch_size=1,
                                     learning_rate=1e-5)
    classification_metric = dc.metrics.Metric(dc.metrics.accuracy_score)
    finetune_model_uniref.fit(dataset, nb_epoch=10)
    eval_score_uniref = finetune_model_uniref.evaluate(dataset,
                                                       [classification_metric])
    assert eval_score_uniref[classification_metric.name] > 0.9
