from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
try:
    import torch
    has_torch = True
except:
    has_torch = False


class Geneformer(HuggingFaceModel):
    """DeepChem wrapper for the Geneformer single-cell foundation model.

    This class provides a DeepChem-compatible interface for Geneformer,
    a transformer-based foundation model pre-trained on large-scale
    single-cell transcriptomic data. The model uses Rank-Value Encoding
    to represent gene expression patterns as token sequences.

    Unlike text-based models (e.g., ChemBERTa), Geneformer receives
    pre-featurized integer token sequences from the dataset rather than
    raw strings. The GeneformerFeaturizer should be used during dataset
    creation to convert gene expression counts to token IDs.

    Supported Tasks
    ---------------
    - 'classification': Single or multi-label classification (e.g., cell type)
    - 'regression': Continuous value prediction (e.g., gene expression levels)
    - 'mtr': Multi-task regression

    Parameters
    ----------
    task : str
        The learning task type. Must be one of 'classification', 'regression',
        or 'mtr' (multi-task regression).
    hf_model_name : str, default 'ctheodoris/Geneformer'
        The HuggingFace model identifier for loading pre-trained weights.
    n_tasks : int, default 1
        Number of prediction targets. For classification with n_tasks=1,
        binary classification is assumed.
    config : Dict[str, Any], optional
        Additional configuration parameters passed to the HuggingFace model.
        Useful for customizing model architecture or initializing from scratch.

    Examples
    --------
    >>> from deepchem.models.torch_models.geneformer import Geneformer
    >>> from transformers import BertConfig
    >>> # Initialize with custom config for testing (avoids downloading weights)
    >>> config = {'vocab_size': 100, 'hidden_size': 16, 'num_hidden_layers': 1,
    ...           'num_attention_heads': 2, 'intermediate_size': 32}
    >>> model = Geneformer(task='classification', n_tasks=1, config=config)

    Notes
    -----
    - Input data should be pre-featurized using GeneformerFeaturizer
    - The model expects input_ids as integer arrays (not raw text)
    - Attention masks are computed automatically based on padding tokens (0)

    References
    ----------
    .. [1] Theodoris, C.V., et al. "Transfer learning enables predictions
        in network biology." Nature (2023).

    See Also
    --------
    deepchem.feat.molecule_featurizers.GeneformerFeaturizer : Featurizer for
        converting gene expression counts to token sequences.
    """

    def __init__(
        self,
        task: str,
        hf_model_name: str = 'ctheodoris/Geneformer',
        n_tasks: int = 1,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Initialize the Geneformer model.

        Parameters
        ----------
        task : str
            The learning task type ('classification', 'regression', or 'mtr').
        hf_model_name : str, default 'ctheodoris/Geneformer'
            HuggingFace model identifier.
        n_tasks : int, default 1
            Number of prediction targets.
        config : Dict[str, Any], optional
            Model configuration parameters.
        **kwargs
            Additional arguments passed to HuggingFaceModel.
        """
        self.n_tasks = n_tasks
        self.hf_model_name = hf_model_name
        self._config_dict = config if config else {}

        # Validate task
        valid_tasks = ['classification', 'regression', 'mtr']
        if task not in valid_tasks:
            raise ValueError(
                f"Invalid task '{task}'. Must be one of {valid_tasks}"
            )

        # Build the HuggingFace model
        model = self._build_model(task, n_tasks)

        # Initialize parent class
        # Note: HuggingFaceModel expects a tokenizer, but Geneformer uses
        # pre-featurized inputs. We pass None and override _prepare_batch.
        super(Geneformer, self).__init__(
            model=model,
            tokenizer=None,  # type: ignore
            task=task,
            config=self._config_dict,
            **kwargs
        )

    def _build_model(self, task: str, n_tasks: int) -> 'PreTrainedModel':
        """Build the HuggingFace model based on task type.

        Parameters
        ----------
        task : str
            The learning task type.
        n_tasks : int
            Number of prediction targets.

        Returns
        -------
        PreTrainedModel
            The configured HuggingFace model.
        """
        if self._config_dict:
            # Initialize from config (random weights)
            hf_config = AutoConfig.for_model(
                model_type='bert',
                **self._config_dict
            )

            if task == 'classification':
                if n_tasks == 1:
                    hf_config.problem_type = 'single_label_classification'
                    hf_config.num_labels = 2  # Binary classification
                else:
                    hf_config.problem_type = 'multi_label_classification'
                    hf_config.num_labels = n_tasks
            elif task in ['regression', 'mtr']:
                hf_config.problem_type = 'regression'
                hf_config.num_labels = n_tasks

            model = AutoModelForSequenceClassification.from_config(hf_config)
        else:
            # Load pre-trained model
            if task == 'classification':
                if n_tasks == 1:
                    problem_type = 'single_label_classification'
                    num_labels = 2
                else:
                    problem_type = 'multi_label_classification'
                    num_labels = n_tasks
            else:
                problem_type = 'regression'
                num_labels = n_tasks

            model = AutoModelForSequenceClassification.from_pretrained(
                self.hf_model_name,
                problem_type=problem_type,
                num_labels=num_labels,
                trust_remote_code=True
            )

        return model

    def _prepare_batch(
        self,
        batch: Tuple[Any, Any, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare a batch of pre-featurized data for model input.

        Unlike text-based models that tokenize strings in this method,
        Geneformer receives pre-featurized token IDs from the dataset.
        This method converts the numpy arrays to PyTorch tensors,
        generates attention masks, and moves data to the appropriate device.

        Parameters
        ----------
        batch : Tuple[Any, Any, Any]
            A tuple of (inputs, labels, weights) where:
            - inputs: numpy array of shape (batch_size, seq_length) containing
              integer token IDs
            - labels: numpy array of target values (or None for prediction)
            - weights: numpy array of sample weights

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            A tuple containing:
            - inputs_dict: Dictionary with 'input_ids', 'attention_mask', and
              optionally 'labels' for the HuggingFace model
            - labels: PyTorch tensor of labels (or None)
            - weights: PyTorch tensor of weights (or None)
        """
        X, y, w = batch

        # Handle case where X might be wrapped in a list/tuple
        if isinstance(X, (list, tuple)):
            input_ids = X[0]
        else:
            input_ids = X

        # Convert to numpy if needed
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        # Create attention mask: 1 for non-padding tokens, 0 for padding
        # Assumes padding token is 0
        attention_mask = (input_ids != 0).astype(np.int64)

        # Convert to PyTorch tensors and move to device
        input_ids_t = torch.as_tensor(
            input_ids,
            dtype=torch.long,
            device=self.device
        )
        attention_mask_t = torch.as_tensor(
            attention_mask,
            dtype=torch.long,
            device=self.device
        )

        # Build inputs dictionary for HuggingFace model
        inputs_dict: Dict[str, torch.Tensor] = {
            'input_ids': input_ids_t,
            'attention_mask': attention_mask_t,
        }

        # Process labels
        y_t: Optional[torch.Tensor] = None
        if y is not None:
            # Handle wrapped labels
            if isinstance(y, (list, tuple)):
                y = y[0]
            y_t = torch.as_tensor(y, device=self.device)

            if self.task == 'classification':
                if self.n_tasks == 1:
                    # Binary classification with CrossEntropyLoss
                    y_t = y_t.long()
                    # Squeeze (N, 1) -> (N) for CrossEntropyLoss
                    if y_t.dim() == 2 and y_t.shape[1] == 1:
                        y_t = y_t.squeeze(1)
                else:
                    # Multi-label classification with BCEWithLogitsLoss
                    y_t = y_t.float()
            else:
                # Regression
                y_t = y_t.float()

            inputs_dict['labels'] = y_t

        # Process weights
        w_t: Optional[torch.Tensor] = None
        if w is not None:
            w_t = torch.as_tensor(w, dtype=torch.float, device=self.device)

        return inputs_dict, y_t, w_t

    def load_from_pretrained(
        self,
        model_dir: Optional[str] = None,
        from_hf_checkpoint: bool = False
    ) -> None:
        """Load model weights from a pretrained checkpoint.

        Parameters
        ----------
        model_dir : str, optional
            Directory containing the model checkpoint. If None, uses self.model_dir.
        from_hf_checkpoint : bool, default False
            If True, loads directly from a HuggingFace checkpoint using
            from_pretrained(). If False, loads from a DeepChem checkpoint.

        Notes
        -----
        When loading from a HuggingFace checkpoint, the model architecture
        should match the checkpoint being loaded.
        """
        if model_dir is None:
            model_dir = self.model_dir

        if from_hf_checkpoint:
            # Determine configuration based on current task
            if self.task == 'classification':
                if self.n_tasks == 1:
                    problem_type = 'single_label_classification'
                    num_labels = 2
                else:
                    problem_type = 'multi_label_classification'
                    num_labels = self.n_tasks
            else:
                problem_type = 'regression'
                num_labels = self.n_tasks

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                problem_type=problem_type,
                num_labels=num_labels,
                trust_remote_code=True,
                **self._config_dict
            )
        else:
            # Use parent class method for DeepChem checkpoints
            super().load_from_pretrained(model_dir, from_hf_checkpoint=False)

