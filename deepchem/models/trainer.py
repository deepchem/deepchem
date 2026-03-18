from typing import Union, List

import deepchem as dc


class DistributedTrainer():
    r"""Perform distributed training of DeepChem models

    DistributedTrainer provides an interface for scaling the training of DeepChem
    model to multiple GPUs and nodes. To achieve this, it uses
    `Lightning <https://lightning.ai/>`_ under the hood. A DeepChem model
    is converted to a `LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_
    using :class:`~deepchem.models.lightning.DCLightningModule` and a DeepChem dataset
    is converted to a PyTorch iterable dataset using :class:`~deepchem.models.lightning.DCLightningDatasetModule`.
    The model and dataset are then trained using Lightning `Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Example
    -------
    .. code-block:: python

        import deepchem as dc
        from deepchem.models.trainer import DistributedTrainer

        dataset = dc.data.DiskDataset('zinc100k')

        atom_vocab = GroverAtomVocabularyBuilder.load('zinc100k_atom_vocab.json')
        bond_vocab = GroverBondVocabularyBuilder.load('zinc100k_bond_vocab.json')

        model = GroverModel(task='pretraining',
                            mode='pretraining',
                            node_fdim=151,
                            edge_fdim=165,
                            features_dim=2048,
                            functional_group_size=85,
                            hidden_size=128,
                            learning_rate=0.0001,
                            batch_size=1,
                            dropout=0.1,
                            ffn_num_layers=2,
                            ffn_hidden_size=5,
                            num_attn_heads=8,
                            attn_hidden_size=128,
                            atom_vocab=atom_vocab,
                            bond_vocab=bond_vocab)

        trainer = DistributedTrainer(max_epochs=1,
                                     batch_size=64,
                                     num_workers=0,
                                     accelerator='gpu',
                                     distributed_strategy='ddp')
        trainer.fit(model, dataset)

    """

    def __init__(self,
                 max_epochs: int,
                 batch_size: int,
                 devices: Union[str, int, List[int]] = 'auto',
                 accelerator: str = 'auto',
                 distributed_strategy: str = 'auto'):
        """Initialise DistributedTrainer

        Parameters
        ----------
        max_epochs: int
            Maximum number of training epochs
        batch_size: int
            Batch size
        devices: str, optional (default: 'auto')
            Number of devices to train on (int) or which devices to train on (list or str).
        accelerator: str, optional (default: 'auto')
            Specify the accelerator to train the models (cpu, gpu, mps)
        distributed_strategy: str, (default: 'auto')
            Specify training strategy (ddp - distributed data parallel, fsdp - fully sharded data parallel, etc).
        When strategy is auto, the Trainer chooses the default strategy for the hardware (ddp for torch). Also, refer to Lightning's docs on strategy `here <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>`_ .
        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.devices = devices
        self.accelerator = accelerator
        self.distributed_strategy = distributed_strategy

    def fit(self, model: 'dc.models.Model', dataset: 'dc.data.Dataset') -> None:
        """Train the model using DistributedTrainer

        Parameters
        ----------
        model: dc.models.Model
            A deepchem model to train
        dataset: dc.data.DiskDataset
            A deepchem Dataset for training
        """
        import lightning as L
        from deepchem.models.lightning import DCLightningModule, DCLightningDatasetModule
        lit_model = DCLightningModule(model)
        lit_dataset = DCLightningDatasetModule(dataset,
                                               batch_size=self.batch_size)
        trainer = L.Trainer(max_epochs=self.max_epochs,
                            devices=self.devices,
                            accelerator=self.accelerator)
        trainer.fit(lit_model, lit_dataset)
        return
