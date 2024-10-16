from PIL import Image
import torch
import numpy as np

import time
import logging
from collections.abc import Sequence as SequenceCollection
from typing import Union, Dict, Tuple, Iterable, List, Optional, Callable, Any

import deepchem as dc
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.utils.typing import LossFn
from deepchem.models.torch_models import HuggingFaceModel

from transformers import AutoProcessor, AutoModelForUniversalSegmentation, OneFormerConfig, OneFormerForUniversalSegmentation


class OneFormer(HuggingFaceModel):

    def __init__(self,
                 model_path: str = 'shi-labs/oneformer_coco_swin_large',
                 segmentation_task: str = "semantic",
                 id2label: Dict = {
                     0: "unlabelled",
                     1: "cell"
                 },
                 torch_dtype: torch.dtype = torch.float32,
                 **kwargs):

        self.model_path = model_path
        self.segmentation_task = segmentation_task
        self.task = "universal_segmentation"
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.torch_dtype = torch_dtype

        if model_path:
            self.model_config = OneFormerConfig().from_pretrained(
                model_path, is_training=True, torch_dtype=torch_dtype)
            self.model_processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.model_config = OneFormerConfig(torch_dtype=torch_dtype)
            self.model_processor = AutoProcessor()

        self.model_config.id2label = self.id2label
        self.model_config.label2id = self.label2id

        self.model = AutoModelForUniversalSegmentation.from_config(
            self.model_config)

        self.model_processor.image_processor.num_text = self.model.config.num_queries - self.model.config.text_encoder_n_ctx

        super().__init__(model=self.model,
                         task=self.task,
                         tokenizer=None,
                         **kwargs)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        X, y, w = batch
        X, y, w = X[0], y[0], w[0]

        images = []
        for x in X:
            images.append(Image.fromarray(x.astype(np.uint8)))

        self.image_size = images[0].size

        if y is not None:
            masks = y.astype(np.uint8)
        else:
            masks = None

        processed_inputs = []
        for idx, img in enumerate(images):
            if masks is not None:
                inputs = self.model_processor(
                    images=images[idx],
                    segmentation_maps=masks[idx],
                    task_inputs=[self.segmentation_task],
                    return_tensors="pt").to(self.device)
            else:
                inputs = self.model_processor(
                    images=images[idx],
                    task_inputs=[self.segmentation_task],
                    return_tensors="pt").to(self.device)
            inputs = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v[0]
                for k, v in inputs.items()
            }
            processed_inputs.append(inputs)

        return inputs, y, w

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
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            inputs = {k: v.to(device) for k, v in inputs.items()}

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

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]]):
        """Predicts output for data provided by generator.

        This is the private implementation of prediction. Do not
        call it directly. Instead call one of the public prediction methods.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: list of dc.trans.Transformers
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.

        Returns
        -------
            A NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs

        """
        results: Optional[List[List[np.ndarray]]] = None
        self._ensure_built()
        self.model.eval()

        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            # Invoke the model.
            output_values = self.model(**inputs)
            for i in output_values:
                if self.segmentation_task == "semantic":
                    output_values[
                        i] = self.processor.post_process_semantic_segmentation(
                            output_values[i],
                            target_sizes=[self.image_size[::-1]])[0]

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]

            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)


model = OneFormer(model_path='shi-labs/oneformer_ade20k_swin_tiny',
                  segmentation_task="instance",
                  id2label={
                      0: "unlabelled",
                      1: "cell"
                  },
                  torch_dtype=torch.float16)

inputs = np.random.rand(10, 224, 224, 3)
labels = np.random.rand(10, 224, 224)

# make deepchem dataset
dataset = dc.data.ImageDataset(inputs, labels)

model.fit(dataset, nb_epoch=1)
