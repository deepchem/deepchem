import pytorch_lightning as pl
import torch


class DCLightningModule(pl.LightningModule):

  def __init__(self, dc_model):
    super().__init__()
    self.dc_model = dc_model

    self.pt_model = self.dc_model.model
    self.loss = self.dc_model._loss_fn

  def configure_optimizers(self):
    return self.dc_model.optimizer._create_pytorch_optimizer(
        self.pt_model.parameters(),)

  def training_step(self, batch, batch_idx):
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
