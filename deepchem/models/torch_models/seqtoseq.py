"""Sequence to sequence translation models."""

from deepchem.models.torch_models import layers
import torch
import torch.nn as nn


class VariationalRandomizer(nn.Module):
    """Add random noise to the embedding and include a corresponding loss."""

    def __init__(self, embedding_dimension, annealing_start_step,
                 annealing_final_step, **kwargs):
        super(VariationalRandomizer, self).__init__(**kwargs)
        self._embedding_dimension = embedding_dimension
        self._annealing_final_step = annealing_final_step
        self._annealing_start_step = annealing_start_step
        self.dense_mean = nn.LazyLinear(embedding_dimension)
        self.dense_stddev = nn.LazyLinear(embedding_dimension)
        self.combine = layers.CombineMeanStd(training_only=True)

    def forward(self, inputs, training=True):
        input, global_step = inputs
        embedding_mean = self.dense_mean(input)
        embedding_stddev = self.dense_stddev(input)
        embedding = self.combine([embedding_mean, embedding_stddev],
                                 training=training)
        mean_sq = embedding_mean * embedding_mean
        stddev_sq = embedding_stddev * embedding_stddev
        kl = mean_sq + stddev_sq - torch.log(stddev_sq + 1e-20) - 1
        anneal_steps = self._annealing_final_step - self._annealing_start_step
        if anneal_steps > 0:
            current_step = global_step.to(
                torch.float64) - self._annealing_start_step
            anneal_frac = torch.maximum(0.0, current_step) / anneal_steps
            kl_scale = torch.minimum(1.0, anneal_frac * anneal_frac)
        else:
            kl_scale = 1.0
        loss = 0.5 * kl_scale * torch.mean(kl)
        self.kl_loss = loss.item()
        return embedding

    def get_loss(self):
        return self.kl_loss
