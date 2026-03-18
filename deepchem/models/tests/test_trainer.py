import pytest
import deepchem as dc


@pytest.mark.torch
def testDistributedTrainer():
    import torch
    from deepchem.models.torch_models import GCNModel
    from deepchem.models.trainer import DistributedTrainer

    train_smiles = [
        "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC",
        "C1CCC1", "CCC"
    ]
    train_labels = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

    model = GCNModel(mode='regression',
                     n_tasks=1,
                     batch_size=2,
                     learning_rate=0.001,
                     device=torch.device('cpu'))
    featurizer = dc.feat.MolGraphConvFeaturizer()
    X = featurizer.featurize(train_smiles)
    dataset = dc.data.NumpyDataset(X=X, y=train_labels)

    trainer = DistributedTrainer(max_epochs=1, batch_size=64, accelerator='cpu')
    trainer.fit(model, dataset)
