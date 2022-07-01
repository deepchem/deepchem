try:
  import torch
  import pytorch_lightning as pl  # noqa
  PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
  PYTORCH_LIGHTNING_IMPORT_FAILED = True


class DCLightningModule(pl.LightningModule):
  """DeepChem Lightning Module to be used with Lightning trainer.

  Example code
  >>> import deepchem as dc
  >>> from deepchem.models import MultitaskClassifier
  >>> import numpy as np
  >>> import torch
  >>> from torch.utils.data import DataLoader
  >>> from deepchem.models.lightning.dc_lightning_module
  ...   import DCLightningModule
  >>> model = MultitaskClassifier(params)
  >>> valid_dataloader = DataLoader(test_dataset)
  >>> lightning_module = DCLightningModule(model)
  >>> trainer = pl.Trainer(max_epochs=1)
  >>> trainer.fit(lightning_module, valid_dataloader)

  The lightning module is a wrapper over deepchem's torch model.
  This module directly works with pytorch lightning trainer
  which runs training for multiple epochs and also is responsible
  for setting up and training models on multiple GPUs.
  """

  def __init__(self, dc_model):
    """Create a new DCLightningModule

    Parameters
    ----------
    dc_model: deepchem.models.torch_models.torch_model.TorchModel
      TorchModel to be wrapped inside the lightning module.
    """
    super().__init__()
    self.dc_model = dc_model

    self.pt_model = self.dc_model.model
    self.loss = self.dc_model._loss_fn

  def configure_optimizers(self):
    """Configure optimizers, for details refer to:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
    """
    return self.dc_model.optimizer._create_pytorch_optimizer(
        self.pt_model.parameters(),)

  def training_step(self, batch, batch_idx):
    """For details refer to:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule  # noqa

    Args:
        batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
            The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
        batch_idx (``int``): Integer displaying index of this batch
        optimizer_idx (``int``): When using multiple optimizers, this argument will also be present.
        hiddens (``Any``): Passed in if
            :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` > 0.

    Return:
        Any of.

        - :class:`~torch.Tensor` - The loss tensor
        - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
        - ``None`` - Training will skip to the next batch. This is only for automatic optimization.
            This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.
    """
    batch = batch.batch_list
    inputs, labels, weights = self.dc_model._prepare_batch(batch)

    outputs = self.pt_model(inputs[0])

    if isinstance(outputs, torch.Tensor):
      outputs = [outputs]

    if self.dc_model._loss_outputs is not None:
      outputs = [outputs[i] for i in self.dc_model._loss_outputs]

    loss_outputs = self.loss(outputs, labels, weights)

    self.log(
        "train_loss",
        loss_outputs,
        on_epoch=True,
        sync_dist=True,
        reduce_fx="mean",
        prog_bar=True,
    )

    return loss_outputs
