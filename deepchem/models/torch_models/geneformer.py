import logging
from typing import Optional, Union, List, Tuple, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from deepchem.models.torch_models import TorchModel
from deepchem.data import Dataset

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel
except ImportError:
    pass


class GeneformerModule(nn.Module):
    """
     The PyTorch Module representing the Geneformer wrapper.

    I'm using the Geneformer backbone because it’s already pre-trained on a massive amount
    of single-cell data.It’s based on BERT, so it uses self-attention to rank gene expression patterns
    rather than just looking at raw counts.I've wrapped the main model with a classification head so we can use
    those gene patterns to actually predict (cell types/disease states/etc.)

     Parameters
     ----------
     hf_model_name: str
         Name of the Hugging Face model to load.
     config: Any, optional
         Configuration object for the model (e.g. BertConfig). If provided,
         model is initialized from config instead of pre-trained weights.
     n_tasks: int
         Number of tasks/classes.
     mode: str
         'classification' or 'regression'.
    """

    def __init__(
        self,
        hf_model_name: str,
        config: Optional[Any] = None,
        n_tasks: int = 1,
        mode: str = "classification",
    ):
        super(GeneformerModule, self).__init__()
        self.n_tasks = n_tasks
        self.mode = mode

        if config is not None:
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_pretrained(hf_model_name)

        hidden_size = self.model.config.hidden_size

        # For classification with 1 task we use CrossEntropyLoss which expects (batch, 2) output for binary classification.
        # basically even if it is a single binary classification we are putting as 2 classes.
        if mode == "classification" and n_tasks == 1:
            self.classifier = nn.Linear(hidden_size, 2)
        else:
            self.classifier = nn.Linear(hidden_size, n_tasks)

    def forward(
            self, inputs: Union[List[torch.Tensor],
                                torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs: Union[List[torch.Tensor], torch.Tensor]
            A list containing [input_ids, attention_mask] or just input_ids.

        Returns
        -------
        torch.Tensor
            Logits or predictions.
        """
        if isinstance(inputs, (list, tuple)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
        else:
            input_ids = inputs
            attention_mask = None

        # Forward pass through HF model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Geneformer is BERT based so without using CLS I'm using mean pooling

        # outputs.last_hidden_state: (Batch, Seq_Len, Hidden)
        if attention_mask is not None:
            # Mask out padding tokens from the average
            # Expanding mask from (Batch, Seq_Len) to (Batch, Seq_Len, 1) so that element wise multiplation
            input_mask_expanded = (attention_mask.unsqueeze(-1).expand(
                outputs.last_hidden_state.size()).float())
            sum_embeddings = torch.sum(
                outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
        else:
            # Simple mean if no mask provided
            embedding = torch.mean(outputs.last_hidden_state, dim=1)

        logits = self.classifier(embedding)
        return logits


class DeepChemGeneformer(TorchModel):
    """
    DeepChem wrapper for Geneformer foundation model.

    I have added a DeepChem compatible wrapper for Geneformer to make it easier to run on the custom datasets.
    The goal here is to let users plug Geneformer's pre-trained biology weights directly into the existing pipelines,
    specifically for things like classifying cell types or predicting expression levels without having to rewrite
    the training loop from scratch

    Parameters
    ----------
    hf_model_name: str, default 'ctheodoris/Geneformer'
        The HuggingFace model identifier.
    model_config: Any, optional
        A configuration object (e.g. transformers.BertConfig) to initialize the
        model randomly without downloading weights. Useful for testing.
    mode: str, default 'classification'
        The task mode: 'classification' or 'regression'.
    n_tasks: int, default 1
        Number of tasks.
    max_length: int, default 2048
        Maximum sequence length for the Geneformer model.
    **kwargs:
        Additional arguments passed to the TorchModel constructor.

    Example
    -------
    >>> from deepchem.models.torch_models.geneformer import DeepChemGeneformer
    >>> from transformers import BertConfig
    >>> config = BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    >>> model = DeepChemGeneformer(hf_model_name='ctheodoris/Geneformer', model_config=config, n_tasks=1, mode='classification')

    """

    def __init__(self,
                 hf_model_name: str = "ctheodoris/Geneformer",
                 model_config: Optional[Any] = None,
                 mode: str = "classification",
                 n_tasks: int = 1,
                 max_length: int = 2048,
                 **kwargs):
        self.mode = mode
        self.n_tasks = n_tasks
        self.hf_model_name = hf_model_name
        self.model_config = model_config
        self.max_length = max_length

        module = GeneformerModule(hf_model_name=hf_model_name,
                                  config=model_config,
                                  n_tasks=n_tasks,
                                  mode=mode)

        if mode == "classification":
            if n_tasks == 1:
                criterion = nn.CrossEntropyLoss(reduction="none")
                self.loss_fn = lambda outputs, labels, weights: (criterion(
                    outputs[0], labels) * weights.squeeze()).mean()
            else:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
                self.loss_fn = lambda outputs, labels, weights: (criterion(
                    outputs[0], labels) * weights).mean()
        elif mode == "regression":
            criterion = nn.MSELoss(reduction="none")
            self.loss_fn = lambda outputs, labels, weights: (criterion(
                outputs[0], labels) * weights).mean()
        else:
            raise ValueError("mode must be 'classification' or 'regression'")

        super(DeepChemGeneformer, self).__init__(model=module,
                                                 loss=self.loss_fn,
                                                 **kwargs)

    def default_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = "fit",
        deterministic: bool = True,
        pad_batches: bool = True,
    ) -> Iterable[Tuple[Any, Any, Any]]:
        """
        Create a generator that yields batches of data from the dataset.

        Performs Rank-Value encoding for gene expression counts. Genes are
        sorted by their expression values in descending order.

        Parameters
        ----------
        dataset: Dataset
            The DeepChem dataset to iterate over.
        epochs: int
            Number of epochs to iterate.
        mode: str
            Iteration mode.
        deterministic: bool
            Whether to iterate deterministically.
        pad_batches: bool
            Whether to pad batches to the batch size.

        Returns
        -------
        Iterable[Tuple[Any, Any, Any]]
            A generator yielding (inputs, labels, weights).
        """
        for batch in dataset.iterbatches(
                batch_size=self.batch_size,
                epochs=epochs,
                deterministic=deterministic,
                pad_batches=pad_batches,
        ):
            X, y, w, ids = batch

            is_counts = False
            if isinstance(X, np.ndarray) and np.issubdtype(
                    X.dtype, np.floating):
                is_counts = True

            processed_seqs = []
            for i in range(len(X)):
                sample = X[i]
                if is_counts:
                    # Rank-Value Encoding: Filter zero-expression genes and sort
                    # indices by expression value descending.
                    # We assume indices correspond to Geneformer token IDs for MVP.
                    nz_indices = np.flatnonzero(sample)
                    nz_values = sample[nz_indices]
                    sorted_nz_indices = nz_indices[np.argsort(-nz_values)]
                    seq = sorted_nz_indices[:self.max_length]
                else:
                    seq = sample[:self.max_length]
                processed_seqs.append(seq)

            # Determine max length in this batch for padding
            batch_max_len = max(
                len(s) for s in processed_seqs) if processed_seqs else 0
            # Ensure at least 1 for empty samples
            batch_max_len = max(1, batch_max_len)

            input_ids = np.zeros((len(X), batch_max_len), dtype=np.int64)
            attention_mask = np.zeros((len(X), batch_max_len), dtype=np.int64)

            for i, seq in enumerate(processed_seqs):
                length = len(seq)
                if length > 0:
                    input_ids[i, :length] = seq
                    attention_mask[i, :length] = 1

            inputs = [input_ids, attention_mask]
            yield (inputs, y, w)

    def _prepare_batch(self, batch: Tuple[Any, Any,
                                          Any]) -> Tuple[Any, Any, Any]:
        """
        Prepare batch for the model by casting to appropriate types and moving to device.

        Parameters
        ----------
        batch: Tuple[Any, Any, Any]
            The batch of data (inputs, labels, weights).

        Returns
        -------
        Tuple[Any, Any, Any]
            The processed batch.
        """
        inputs, y, w = batch
        input_ids, attention_mask = inputs

        # Cast to torch.long for HF compatibility
        input_ids_t = torch.as_tensor(input_ids,
                                      dtype=torch.long,
                                      device=self.device)
        attention_mask_t = torch.as_tensor(attention_mask,
                                           dtype=torch.long,
                                           device=self.device)

        processed_inputs = [input_ids_t, attention_mask_t]

        y_t = None
        if y is not None:
            y_t = torch.as_tensor(y, device=self.device)
            if self.mode == "classification":
                if self.n_tasks == 1:
                    y_t = y_t.long()
                    # Squeeze (N, 1) -> (N) for CrossEntropyLoss
                    if y_t.dim() == 2 and y_t.shape[1] == 1:
                        y_t = y_t.squeeze(1)
                else:
                    y_t = y_t.float()
            else:
                y_t = y_t.float()

        if w is not None:
            w_t = torch.as_tensor(w, device=self.device, dtype=torch.float)
        else:
            w_t = None

        return processed_inputs, y_t, w_t
