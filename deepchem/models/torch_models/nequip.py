"""DeepChem batching scaffold for future NequIP models."""

from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch

from deepchem.data import Dataset
from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.utils.typing import LossFn


class NequIPModel(TorchModel):
    """Prepare variable-size atomistic batches for a future NequIP backbone.

    This class is the DeepChem integration scaffold for NequIP-style models. It
    batches :class:`GraphData` samples, preserves the requested task ordering,
    and prepares graph-level energies and atom-level forces. It does not
    implement the NequIP architecture or a physical loss.

    Parameters
    ----------
    tasks : sequence of str
        Any non-duplicated ordering of ``"energy"`` and ``"forces"``. An
        empty sequence is supported for unlabeled prediction datasets.
    model : torch.nn.Module
        The atomistic model to receive a torch-converted
        :class:`BatchGraphData`. Until the native NequIP backbone is added,
        callers must supply this module.
    loss : callable
        A custom TorchModel loss callable accepting ``(outputs, labels,
        weights)``. Labels and weights are lists in ``tasks`` order.
    batch_size : int, default 100
        Number of structures per batch. Final batches are never padded.
    learning_rate : float or LearningRateSchedule, default 0.001
        Learning rate passed to :class:`TorchModel`.
    **kwargs
        Additional arguments for :class:`TorchModel`.

    Notes
    -----
    Energy tensors have shape ``(B, 1)``. Force tensors are concatenated in
    graph order and have shape ``(N_total, 3)``; force weights have shape
    ``(N_total, 1)``. Unlabeled batches return empty label and weight lists.
    Backend conversion and all NequIP mathematics belong to later work.
    """

    _SUPPORTED_TASKS = {"energy", "forces"}

    def __init__(self,
                 tasks: Sequence[str],
                 model: torch.nn.Module,
                 loss: LossFn,
                 batch_size: int = 100,
                 learning_rate: Union[float, LearningRateSchedule] = 0.001,
                 **kwargs: Any) -> None:
        if isinstance(tasks, str):
            raise ValueError("tasks must be a sequence of task names")
        self.tasks: List[str] = list(tasks)
        unsupported = [
            task for task in self.tasks if task not in self._SUPPORTED_TASKS
        ]
        if unsupported:
            raise ValueError(
                "Unsupported NequIPModel tasks: %s. Supported tasks are %s." %
                (unsupported, sorted(self._SUPPORTED_TASKS)))
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("NequIPModel tasks must not contain duplicates.")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not callable(loss):
            raise TypeError("loss must be a custom callable")

        self.task_to_index = {
            task: index for index, task in enumerate(self.tasks)
        }
        output_types = kwargs.pop("output_types", ["prediction"])
        super().__init__(model,
                         loss=loss,
                         output_types=output_types,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         **kwargs)

    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = "fit",
        deterministic: bool = True,
        pad_batches: bool = True
    ) -> Iterable[Tuple[List[Any], List[Any], List[Any]]]:
        """Iterate over structures without padding the final graph batch."""
        del mode, pad_batches
        for _ in range(epochs):
            for X_b, y_b, w_b, _ in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=False):
                yield [X_b], [y_b], [w_b]

    def _prepare_batch(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[BatchGraphData, List[torch.Tensor], List[torch.Tensor]]:
        """Batch graphs and prepare targets and weights in ``tasks`` order."""
        inputs, labels, weights = batch
        if inputs is None or len(inputs) != 1:
            raise ValueError("inputs must contain one GraphData batch")
        graphs = list(inputs[0])
        if not graphs or not all(
                isinstance(graph, GraphData) for graph in graphs):
            raise ValueError(
                "inputs must contain at least one GraphData sample")
        graph_batch = BatchGraphData(graphs).numpy_to_torch(self.device)

        if not self.tasks or self._is_missing(labels):
            return graph_batch, [], []

        label_array = np.asarray(labels[0], dtype=object)
        expected_layout = (len(graphs), len(self.tasks))
        if label_array.shape != expected_layout:
            raise ValueError(
                "Label array has shape %s. Expected %s for tasks %s." %
                (label_array.shape, expected_layout, self.tasks))

        if self._is_missing(weights):
            weight_array = np.ones(expected_layout, dtype=np.float32)
        else:
            try:
                weight_array = np.asarray(weights[0], dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Weights must be numeric structure/task values") from error
            if weight_array.shape != expected_layout:
                raise ValueError("Weight array has shape %s. Expected %s." %
                                 (weight_array.shape, expected_layout))
            if not np.isfinite(weight_array).all():
                raise ValueError("Weights must be finite")

        prepared_labels: List[torch.Tensor] = []
        prepared_weights: List[torch.Tensor] = []
        for task in self.tasks:
            task_index = self.task_to_index[task]
            if task == "energy":
                energies = []
                for sample_index in range(len(graphs)):
                    try:
                        energy = np.asarray(label_array[sample_index,
                                                        task_index],
                                            dtype=np.float32)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Energy label for sample %d must be a numeric scalar"
                            % sample_index) from error
                    if energy.shape != () or not np.isfinite(energy).item():
                        raise ValueError(
                            "Energy label for sample %d must be a finite scalar"
                            % sample_index)
                    energies.append(energy.item())
                task_labels = torch.tensor(energies,
                                           dtype=torch.float32,
                                           device=self.device).view(-1, 1)
                task_weights = torch.as_tensor(weight_array[:, task_index,
                                                            None],
                                               device=self.device)
            else:
                force_labels = []
                force_weights = []
                for sample_index, graph in enumerate(graphs):
                    try:
                        forces = np.asarray(label_array[sample_index,
                                                        task_index],
                                            dtype=np.float32)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Force labels for sample %d must be numeric" %
                            sample_index) from error
                    expected_shape = (graph.num_nodes, 3)
                    if forces.shape != expected_shape:
                        raise ValueError(
                            "Force labels for sample %d have shape %s. Expected %s."
                            % (sample_index, forces.shape, expected_shape))
                    if not np.isfinite(forces).all():
                        raise ValueError(
                            "Force labels for sample %d must be finite" %
                            sample_index)
                    force_labels.append(forces)
                    force_weights.append(
                        np.full((graph.num_nodes, 1),
                                weight_array[sample_index, task_index],
                                dtype=np.float32))
                task_labels = torch.as_tensor(np.concatenate(force_labels),
                                              device=self.device)
                task_weights = torch.as_tensor(np.concatenate(force_weights),
                                               device=self.device)
            prepared_labels.append(task_labels)
            prepared_weights.append(task_weights)

        return graph_batch, prepared_labels, prepared_weights

    @staticmethod
    def _is_missing(values: Any) -> bool:
        return values is None or len(values) == 0 or values[0] is None
