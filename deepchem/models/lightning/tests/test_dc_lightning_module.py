import unittest
import deepchem as dc
import numpy as np

try:
    from deepchem.models import GCNModel, MultitaskClassifier
    from deepchem.models.lightning.dc_lightning_module import DCLightningModule
    from deepchem.models.lightning.dc_lightning_dataset_module import DCLightningDatasetModule, collate_dataset_wrapper
    from deepchem.metrics import to_one_hot
    import lightning as L  # noqa
    PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
    PYTORCH_LIGHTNING_IMPORT_FAILED = True


class TestDCLightningModule(unittest.TestCase):

    @unittest.skipIf(PYTORCH_LIGHTNING_IMPORT_FAILED,
                     'PyTorch Lightning is not installed')
    def test_multitask_classifier(self):

        class TestMultitaskDatasetBatch:

            def __init__(self, batch):
                X = [batch[0]]
                y = [
                    np.array([
                        to_one_hot(b.flatten(), 2).reshape(2, 2)
                        for b in batch[1]
                    ])
                ]
                w = [batch[2]]
                self.batch_list = [X, y, w]

        def collate_dataset_wrapper(batch):
            return TestMultitaskDatasetBatch(batch)

        tasks, datasets, _ = dc.molnet.load_clintox()
        _, valid_dataset, _ = datasets

        model = MultitaskClassifier(n_tasks=len(tasks),
                                    n_features=1024,
                                    layer_sizes=[1000],
                                    dropouts=0.2,
                                    learning_rate=0.0001)

        molnet_dataloader = DCLightningDatasetModule(valid_dataset, 6,
                                                     collate_dataset_wrapper)
        lightning_module = DCLightningModule(model)
        trainer = L.Trainer(max_epochs=1)
        trainer.fit(lightning_module, molnet_dataloader)

    @unittest.skipIf(PYTORCH_LIGHTNING_IMPORT_FAILED,
                     'PyTorch Lightning is not installed')
    def test_gcn_model(self):

        train_smiles = [
            "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC", "C1CCC1", "CCC",
            "C1CCC1", "CCC"
        ]
        train_labels = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]

        model = GCNModel(mode='classification',
                         n_tasks=1,
                         batch_size=2,
                         learning_rate=0.001)

        featurizer = dc.feat.MolGraphConvFeaturizer()
        X = featurizer.featurize(train_smiles)
        sample = dc.data.NumpyDataset(X=X, y=train_labels)

        smiles_datasetmodule = DCLightningDatasetModule(
            sample, 2, collate_dataset_wrapper)

        lightning_module = DCLightningModule(model)
        trainer = L.Trainer(max_epochs=1)
        trainer.fit(lightning_module, smiles_datasetmodule)
