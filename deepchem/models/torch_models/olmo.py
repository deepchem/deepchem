import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, OlmoPreTrainedModel, OlmoForCausalLM
from transformers.modeling_layers import GenericForSequenceClassification
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import OneOrMany
from typing import Any, Tuple, Optional, Iterable, List


class OlmoForSequenceClassification(GenericForSequenceClassification,
                                    OlmoPreTrainedModel):
    """OLMo model with a linear scoring head for classification or regression.

    Adds a linear layer on top of the last token's hidden state.

    Parameters
    ----------
    config : OlmoConfig
        Must have num_labels and problem_type set before constructing.

    Examples
    --------
    >>> model = OlmoForSequenceClassification.from_pretrained(
    ...     "allenai/OLMo-1B-hf", num_labels=1,
    ...     problem_type="multi_label_classification",
    ...     torch_dtype=torch.bfloat16)
    """

    base_model_prefix = "model"

    def __init__(self, config):
        super(GenericForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        # Linear layer initialises in full precision even when the backbone is
        # half precision, causing dtype mismatches. Cast to match the backbone.
        self.score = self.score.to(next(self.parameters()).dtype)
        self.post_init()


class Olmo(HuggingFaceModel):
    """OLMo wrapper for classification, regression, and causal language modelling.

    Loads a pretrained OLMo checkpoint with the appropriate head for the task:
    classification/mtc uses BCEWithLogitsLoss with sigmoid outputs,
    regression/mtr uses MSELoss, and causal_lm uses OlmoForCausalLM.
    The model is loaded in bfloat16 with gradient checkpointing enabled for memory efficiency.

    Parameters
    ----------
    model : str
        Loads a HuggingFace model ID, local path, or an instantiated model.
    tokenizer : AutoTokenizer or None
        Tokenizer for SMILES strings. Loaded from model path if None.
    task_type : str, default 'classification'
        One of 'classification', 'regression', 'causal_lm', 'mtc', or 'mtr'.
    n_tasks : int, default 1
        Number of output labels. Ignored for causal_lm.
    **kwargs
        Forwarded to HuggingFaceModel.

    Example
    --------
    >>> import deepchem as dc
    >>> import numpy as np
    >>> from deepchem.models.torch_models.olmo import Olmo
    >>> model = Olmo(model="allenai/OLMo-1B-hf", tokenizer=None,
    ...              task_type="classification", n_tasks=1)
    >>> dataset = dc.data.NumpyDataset(
    ...     ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F",
    ...      "CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"],
    ...     np.array([[1.0], [0.0]]))
    >>> loss = model.fit(dataset, nb_epoch=1)
    >>> predictions = model.predict(dataset)
    """

    def __init__(self,
                 model,
                 tokenizer,
                 task_type: str = "classification",
                 n_tasks: int = 1,
                 **kwargs):
        self.n_tasks = n_tasks
        if task_type not in ("classification", "regression", "causal_lm", "mtc",
                             "mtr"):
            raise ValueError(
                f"task_type must be 'classification', 'regression', 'causal_lm', 'mtc', or 'mtr', "
                f"got '{task_type}'")

        if isinstance(model, str):
            model_path = model
            if task_type == "causal_lm":
                model = OlmoForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.bfloat16)
            else:
                if task_type in ("classification", "mtc"):
                    problem_type = "multi_label_classification"
                else:
                    problem_type = "regression"
                model = OlmoForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=n_tasks,
                    problem_type=problem_type,
                    torch_dtype=torch.bfloat16)
            model.gradient_checkpointing_enable()
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)

        super().__init__(model=model,
                         tokenizer=tokenizer,
                         task=task_type,
                         **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Tokenize inputs and cast labels to the model dtype for each task.

        Labels are cast to bfloat16 to match the head output, as required by
        MSELoss and BCEWithLogitsLoss. causal_lm is used from the parent.

        Parameters
        ----------
        batch : tuple of (inputs, labels, weights)
            Raw batch where inputs[0] is a numpy array of SMILES strings,
            labels has shape (1, batch_size, n_tasks) or None during predict.

        Returns
        -------
        inputs : dict
            Tokenized inputs with a labels key, ready for model forward().
        y : torch.Tensor or None
            Label tensor on device, or None during prediction.
        w : torch.Tensor or None
            Sample weight tensor on device, or None.
        """

        smiles_batch, y, w = batch

        if w is not None:
            w = torch.tensor(w, dtype=torch.float).to(self.device)

        tokens = self.tokenizer(smiles_batch[0].tolist(),
                                padding=True,
                                return_tensors="pt")

        if self.task in ['regression', 'classification', 'mtr', 'mtc']:
            if y is not None:
                # y is None during predict
                model_dtype = next(self.model.parameters()).dtype
                y = torch.from_numpy(y[0]).to(dtype=model_dtype,
                                              device=self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, 'labels': y}
            return inputs, y, w
        else:
            return super()._prepare_batch(batch)

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer], uncertainty: bool,
                 other_output_types: Optional[OneOrMany[str]]):
        """Run inference and return numpy predictions.

        Overrides the parent to cast bfloat16 outputs to float32 before
        calling .numpy(), since NumPy does not support bfloat16.

        Parameters
        ----------
        generator : iterable
            Batch generator from the dataset.
        transformers : list of Transformer
            Inverse transformers for label unscaling. Only supported for
            single-output models.
        uncertainty : bool
            Not supported; raises ValueError if True.
        other_output_types : str or list of str or None
            Additional output names to return instead of predictions.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, n_tasks).
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
            )
        if uncertainty:
            if self._variance_outputs is None or len(
                    self._variance_outputs) == 0:
                raise ValueError('This model cannot compute uncertainties')
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    'The number of variances must exactly match the number of outputs'
                )
        if other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    'This model cannot compute other outputs since no other output_types were specified.'
                )
        self._ensure_built()
        self.model.eval()
        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            output = self.model(**inputs)
            logits = output.logits
            if self.task in ('classification', 'mtc'):
                output_values = torch.sigmoid(logits)
            else:
                output_values = logits

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [
                t.float().detach().cpu().numpy() for t in output_values
            ]
            # Apply tranformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
            access_values = []
            if other_output_types:
                access_values += self._other_outputs
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError(
                        "predict() does not support Transformers for models with multiple outputs."
                    )
                elif len(output_values) == 1:
                    output_values = [
                        undo_transforms(output_values[0], transformers)
                    ]
            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))

        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)
