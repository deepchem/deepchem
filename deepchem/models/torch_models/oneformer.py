from PIL import Image
import numpy as np

import time
import logging
from collections.abc import Sequence as SequenceCollection
from typing import Union, Dict, Tuple, Iterable, List, Optional, Callable, Any

from deepchem.models.optimizers import LearningRateSchedule
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import LossFn, OneOrMany

import torch
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoProcessor, AutoModelForUniversalSegmentation, OneFormerConfig

logger = logging.getLogger(__name__)


class OneFormer(HuggingFaceModel):
    """
    Wrapper class that wraps the OneFormer model as a DeepChem model.

    The OneFormer model was proposed in OneFormer: One Transformer to Rule Universal Image Segmentation [1]
    by Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi. OneFormer is a universal
    image segmentation framework that can be trained on a single panoptic dataset to perform semantic, instance,
    and panoptic segmentation tasks.

    Whilst all official HuggingFace model weights by shi-labs [2] are supported, the current implementation only
    supports the OneFormer model for semantic segmentation.

    Instance and panoptic segmentation tasks are not supported yet.

    Usage Example:
    --------------
    >> from PIL import Image
    >> import numpy as np
    >> import torch
    >> from deepchem.data import NumpyDataset, ImageDataset
    >> from deepchem.models.torch_models import OneFormer
    >> from deepchem.metrics import Metric, mean_absolute_error, jaccaard_index

    >> # Prepare the dataset
    >> X = np.random.randint(0, 255, (3, 512, 512, 3))
    >> y = np.random.randint(0, 1, (3, 512, 512))
    >> dataset = ImageDataset(X, y)
    >> id2label = {0: "label-A", 1: "label-B"}

    >> # Create the model
    >> model = OneFormer(segmentation_task="semantic", model_path='shi-labs/oneformer_ade20k_swin_tiny',
    ...                  id2label=id2label, torch_dtype=torch.float16, batch_size=2)

    >> # Train the model
    >> avg_loss = model.fit(dataset, nb_epoch=3)

    >> # Predict the model
    >> preds = model.predict(dataset)

    >> # Evaluate the model
    >> mae_metric = Metric(mean_absolute_error)
    >> iou_metric = Metric(jaccaard_index)
    >> iou = iou_metric.compute_metric(dataset.y, preds)
    >> mae = mae_metric.compute_metric(np.array(dataset.y).flatten(), np.array(preds).flatten())

    References
    ----------
    .. [1] Jain, J., Li, J., Chiu, M., Hassani, A., Orlov, N., & Shi, H. (2022, November 10). OneFormer: One Transformer to rule universal image segmentation. arXiv.org. https://arxiv.org/abs/2211.06220
    .. [2] https://huggingface.co/shi-labs
    """

    def __init__(self,
                 segmentation_task: str = "semantic",
                 model_path: str = 'shi-labs/oneformer_ade20k_swin_tiny',
                 model_config: Optional[OneFormerConfig] = None,
                 model_processor: Optional[AutoProcessor] = None,
                 id2label: Dict = {
                     0: "unlabelled",
                     1: "labelled"
                 },
                 torch_dtype: torch.dtype = torch.float32,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        segmentation_task: str
            The segmentation task to perform. The available tasks are
            - `semantic` - semantic segmentation (default)
            - `instance` - instance segmentation (not supported yet)
            - `panoptic` - panoptic segmentation (not supported yet)
        model_path: str
            Path to the OneFormer HuggingFace model; HF Model Hub or local
            - ex: 'shi-labs/oneformer_ade20k_swin_tiny'
        model_config: OneFormerConfig
            Optional configuration for the OneFormer model. If not provided, the configuration
            will be loaded from the model_path. If provided, the configuration will be used
            to initialize the model instead of the configuration from the model_path.
        model_processor: AutoProcessor
            Optional processor for the OneFormer model. If not provided, the processor
            will be loaded from the model_path. If provided, the processor will be used
            to initialize the model instead of the processor from the model_path.
        id2label: dict
            A dictionary mapping class indices to class labels.
        torch_dtype: torch.dtype
            The torch data type to use for the model. The supported data types are
            - `torch.float32` (default)
            - `torch.float16`

        Note
        ----
        If a custom model configuration and processor are provided, ensure that
        `model_processor.image_processor.num_text == model_config.num_queries - model_config.text_encoder_n_ctx` for
        contrastive learning to work correctly during training.
        """

        self.model_path = model_path
        self.segmentation_task = segmentation_task
        self.task = "universal_segmentation"
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.torch_dtype = torch_dtype

        if model_config is None and model_path is not None:
            self.model_config = OneFormerConfig().from_pretrained(
                model_path, is_training=True, torch_dtype=torch_dtype)
        else:
            self.model_config = model_config

        if model_processor is None and model_path is not None:
            self.model_processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.model_processor = model_processor

        if (self.model_path is None) and (self.model_config is None or
                                          self.model_processor is None):
            raise ValueError(
                "Please provide either a model path or a model configuration and processor."
            )

        self.model_config.id2label = self.id2label
        self.model_config.label2id = self.label2id

        self.model = AutoModelForUniversalSegmentation.from_config(
            self.model_config)

        self.model_processor.image_processor.num_text = self.model.config.num_queries - self.model.config.text_encoder_n_ctx
        assert self.model_processor.image_processor.num_text == self.model.config.num_queries - self.model.config.text_encoder_n_ctx

        super().__init__(model=self.model,
                         task=self.task,
                         tokenizer=None,
                         **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """
        Preprocess and prepare the batch for the model.

        Parameters
        ----------
        batch: Tuple
            A tuple of the form (X, y, w) where X is the input data, y is the label and w is the weight.

        Returns
        -------
        processed_inputs: Dict
            A dictionary containing the processed inputs.
        y: torch.Tensor
            The label tensor.
        w: torch.Tensor
            The weight tensor.
        """
        X, y, w = batch
        if y is not None:
            X, y, w = X[0], y[0], w[0]
        else:
            X = X[0]

        images = []
        for x in X:
            images.append(Image.fromarray(x.astype(np.uint8)))

        self.image_size = images[0].size

        if y is not None:
            masks = y.astype(np.uint8)
        else:
            masks = None

        processed_inputs: Dict[str, torch.Tensor]
        processed_inputs = {}
        for idx, img in enumerate(images):
            if masks is not None:
                inputs = self.model_processor(
                    images=images[idx],
                    segmentation_maps=masks[idx],
                    task_inputs=[self.segmentation_task],
                    return_tensors="pt")
            else:
                inputs = self.model_processor(
                    images=images[idx],
                    task_inputs=[self.segmentation_task],
                    return_tensors="pt")
            # Process and append to the same dictionary
            for k, v in inputs.items():
                v = v.squeeze().to(self.device) if isinstance(
                    v, torch.Tensor) else v[0]

                # If this is the first time we're adding to processed_inputs, initialize it with the correct batch shape
                if k in processed_inputs:
                    processed_inputs[k] = torch.cat(
                        (processed_inputs[k], v.unsqueeze(0)), dim=0)
                else:
                    processed_inputs[k] = v.unsqueeze(0)  # Add batch dimension

        return processed_inputs, y, w

    def fit_generator(self,
                      generator: Iterable[Tuple[Any, Any, Any]],
                      max_checkpoints_to_keep: int = 5,
                      checkpoint_interval: int = 1000,
                      restore: bool = False,
                      variables: Optional[Union[List[torch.nn.Parameter],
                                                torch.nn.ParameterList]] = None,
                      loss: Optional[LossFn] = None,
                      callbacks: Union[Callable, List[Callable]] = [],
                      all_losses: Optional[List[float]] = None) -> float:
        """Train this model on data from a generator.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step, **kwargs) that will be invoked
            after every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.

        Returns
        -------
        The average loss over the most recent checkpoint interval

        Note
        ----
        A HuggingFace model can return embeddings (last hidden state), attentions.
        Support must be added to return the embeddings to the user, so that it can
        be used for other downstream applications.
        """
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        self.model.model.is_training = True
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate,
                              LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                        optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)
        time1 = time.time()

        # Main training loop.
        for batch in generator:
            if restore:
                self.restore()
                restore = False

            inputs, labels, weights = self._prepare_batch(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            optimizer.zero_grad()
            outputs = self.model(**inputs)

            batch_loss = outputs.get("loss")
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += batch_loss

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info('Ending global_step %d: Average loss %g' %
                            (current_step, avg_loss))
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                try:
                    # NOTE In DeepChem > 2.8.0, callback signature is updated to allow
                    # variable arguments.
                    c(self, current_step, iteration_loss=batch_loss)
                except TypeError:
                    # DeepChem <= 2.8.0, the callback should have this signature.
                    c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard('loss', batch_loss,
                                                current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({'train/loss': batch_loss})
                self.wandb_logger.log_data(all_data, step=current_step)

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info('Ending global_step %d: Average loss %g' %
                        (current_step, avg_loss))
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer], uncertainty: bool,
                 other_output_types: Optional[OneOrMany[str]]):
        """Predicts output for data provided by generator.

        This is the private implementation of prediction. Do not
        call it directly. Instead call one of the public prediction methods.
        Please use model.predict() instead.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: list of dc.trans.Transformers
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        uncertainty: bool
            specifies whether this is being called as part of estimating uncertainty.
            If True, it sets the training flag so that dropout will be enabled, and
            returns the values of the uncertainty outputs.
        other_output_types: list, optional
            Provides a list of other output_types (strings) to predict from model.

        Returns
        -------
            A NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs

        Note
        ----
        A HuggingFace model does not output uncertainity. The argument is here
        since it is also present in TorchModel. Similarly, other variables like
        other_output_types are also not used. Instead, a HuggingFace model outputs
        loss, logits, hidden state and attentions.
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
        self.model.model.is_training = False

        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            output_values = []
            # Invoke the model.
            with torch.no_grad():
                output = self.model(**inputs)
                output_values.append(output)

            # Post process the output
            for idx, i in enumerate(output_values):
                if self.segmentation_task == "semantic":
                    output_values[
                        idx] = self.model_processor.post_process_semantic_segmentation(
                            output_values[idx],
                            target_sizes=[self.image_size[::-1]])[0]

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]

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

        # Concatenate arrays to create the final results
        if results is not None:
            final_results = results
        if variances is not None:
            final_variances = variances

        if uncertainty and variances is not None:
            return zip(final_results, final_variances)

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)
