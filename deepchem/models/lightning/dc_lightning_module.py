try:
  import torch
  import pytorch_lightning as pl  # noqa
  PYTORCH_LIGHTNING_IMPORT_FAILED = False
except ImportError:
  PYTORCH_LIGHTNING_IMPORT_FAILED = True


class DCLightningModule(pl.LightningModule):
  """DeepChem Lightning Module to be used with Lightning trainer.

  TODO: Add dataloader, example code and fit, once datasetmodule
  is ready
  The lightning module is a wrapper over deepchem's torch model.
  This module directly works with pytorch lightning trainer
  which runs training for multiple epochs and also is responsible
  for setting up and training models on multiple GPUs.
  https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule

  Notes
  -----
  This class requires PyTorch to be installed.
  """

  def __init__(self, dc_model):
    """Create a new DCLightningModule.

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
    return self.dc_model.optimizer._create_pytorch_optimizer(
        self.pt_model.parameters(),)

  def training_step(self, batch, batch_idx):
    """Perform a training step.

    Parameters
    ----------
    batch: A tensor, tuple or list.
    batch_idx: Integer displaying index of this batch
    optimizer_idx: When using multiple optimizers, this argument will also be present.

    Returns
    -------
    loss_outputs: outputs of losses.
    """
    batch = batch.batch_list
    inputs, labels, weights = self.dc_model._prepare_batch(batch)
    if isinstance(inputs, list):
      assert len(inputs) == 1
      inputs = inputs[0]

    outputs = self.pt_model(inputs)

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
