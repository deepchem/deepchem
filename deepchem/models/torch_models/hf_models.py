import logging
import time
from collections.abc import Sequence as SequenceCollection
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional,
                    Tuple, Union, Dict)

import numpy as np
import torch
from transformers.models.auto import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models import TorchModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import LossFn, OneOrMany
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto import AutoModel, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import transformers
    from transformers.modeling_utils import PreTrainedModel


class HuggingFaceModel(TorchModel):
    """Wrap HuggingFace transformers for use in DeepChem.
    
    This allows using pre-trained transformers (like RoBERTa, BERT) for
    molecular tasks via DeepChem's training API. Useful for comparing
    transformer-based approaches with other DeepChem models.
    
    Handles tasks like masked language modeling (pretraining), regression,
    and classification for molecular property prediction.
    
    Note: Tokenization happens on-the-fly from SMILES strings.
    """
    
    def __init__(
            self,
            model: 'PreTrainedModel',
            tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
            task: Optional[str] = None,
            config: Optional[Dict] = None,
            **kwargs):
        """
        Args:
            model: HuggingFace model (e.g., RobertaForMaskedLM)
            tokenizer: Corresponding tokenizer
            task: 'mlm', 'regression', 'classification', or None
            config: Optional model config for loading checkpoints
        """
        self.task = task
        self.tokenizer = tokenizer
        
        # Setup data collator for MLM if needed
        if self.task == 'mlm':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer)
        else:
            self.data_collator = None
            
        self.config = config or {}
        
        # HuggingFace models compute loss internally
        super().__init__(model=model, loss=None, **kwargs)

    def load_from_pretrained(
            self,
            model_dir: Optional[str] = None,
            from_hf_checkpoint: bool = False):
        """Load from checkpoint.
        
        If from_hf_checkpoint=True, loads directly from HuggingFace hub.
        Otherwise loads from DeepChem checkpoint.
        
        For finetuning, we drop classifier weights since they likely
        don't match the new task.
        """
        if model_dir is None:
            model_dir = self.model_dir

        if from_hf_checkpoint:
            # Load directly from HuggingFace
            if self.task == 'mlm':
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_dir, **self.config)
            elif self.task in ['regression', 'classification']:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, **self.config)
            else:
                self.model = AutoModel.from_pretrained(model_dir, **self.config)
        else:
            # Load from DeepChem checkpoint
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if not checkpoints:
                raise ValueError('No checkpoint found')
                
            checkpoint = checkpoints[0]
            data = torch.load(checkpoint, map_location=self.device)
            
            # Clean up state dict for loading
            state_dict = data['model_state_dict']
            
            # Handle DDP wrapper if present
            state_dict = {k.replace('module.', ''): v 
                         for k, v in state_dict.items()}
            
            # Drop classifier weights for finetuning
            keys_to_drop = [
                'classifier.out_proj.weight', 'classifier.out_proj.bias',
                'classifier.dense.weight', 'classifier.dense.bias'
            ]
            for key in keys_to_drop:
                if key in state_dict:
                    del state_dict[key]
            
            self.model.load_state_dict(state_dict, strict=False)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Convert batch to HuggingFace format."""
        smiles_batch, y, w = batch
        
        # Tokenize SMILES
        tokens = self.tokenizer(
            smiles_batch[0].tolist(),
            padding=True,
            return_tensors="pt"
        )

        if self.task == 'mlm':
            # Mask tokens for MLM
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens['input_ids'])
            
            inputs = {
                'input_ids': inputs.to(self.device),
                'labels': labels.to(self.device),
                'attention_mask': tokens['attention_mask'].to(self.device),
            }
            return inputs, None, w
            
        elif self.task in ['regression', 'classification']:
            # Prepare labels for supervised task
            if y is not None:
                y = torch.from_numpy(y[0])
                if self.task == 'regression':
                    y = y.float()
                else:  # classification
                    y = y.long()
                y = y.to(self.device)
            
            # Move everything to device
            for key in tokens:
                tokens[key] = tokens[key].to(self.device)
                
            inputs = {**tokens, 'labels': y}
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
        """Standard training loop - mostly inherited from TorchModel."""
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
            
        self._ensure_built()
        self.model.train()
        
        avg_loss = 0.0
        averaged_batches = 0
        
        # Setup optimizer
        if variables is None:
            optimizer = self._pytorch_optimizer
            lr_schedule = self._lr_schedule
        else:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                optimizer, lr_schedule = self._optimizer_for_vars[var_key]
            else:
                optimizer = self.optimizer._create_pytorch_optimizer(variables)
                if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
                    lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(optimizer)
                else:
                    lr_schedule = None
                self._optimizer_for_vars[var_key] = (optimizer, lr_schedule)

        time1 = time.time()

        for batch in generator:
            if restore:
                self.restore()
                restore = False
                
            inputs, labels, weights = self._prepare_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(**inputs)
            
            batch_loss = outputs.get("loss")
            batch_loss.backward()
            optimizer.step()
            
            if lr_schedule is not None:
                lr_schedule.step()
                
            self._global_step += 1
            current_step = self._global_step

            avg_loss += batch_loss.item()
            averaged_batches += 1

            # Logging and checkpointing
            if current_step % self.log_frequency == 0:
                avg_loss_current = avg_loss / averaged_batches
                logger.info(f'Step {current_step}: loss {avg_loss_current:.4f}')
                
                if all_losses is not None:
                    all_losses.append(avg_loss_current)
                    
                if self.tensorboard:
                    self._log_scalar_to_tensorboard('loss', batch_loss, current_step)
                    
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == 0:
                self.save_checkpoint(max_checkpoints_to_keep)

            for callback in callbacks:
                callback(self, current_step)

        # Final averaging
        if averaged_batches > 0:
            final_loss = avg_loss / averaged_batches
            logger.info(f'Final loss: {final_loss:.4f}')
            if all_losses is not None:
                all_losses.append(final_loss)
                
        if checkpoint_interval > 0:
            self.save_checkpoint(max_checkpoints_to_keep)

        logger.info(f"Training took {time.time() - time1:.1f}s")
        return final_loss if averaged_batches > 0 else 0.0

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer], uncertainty: bool,
                 other_output_types: Optional[OneOrMany[str]]):
        """Get predictions from model."""
        results = None
        self._ensure_built()
        self.model.eval()
        
        for batch in generator:
            inputs, _, _ = self._prepare_batch((batch[0], None, None))

            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Extract logits (main predictions)
            logits = outputs.get('logits')
            if isinstance(logits, torch.Tensor):
                logits = [logits]
                
            batch_results = [t.detach().cpu().numpy() for t in logits]
            
            if results is None:
                results = [[] for _ in range(len(batch_results))]
                
            for i, arr in enumerate(batch_results):
                results[i].append(arr)

        # Combine batches
        if results:
            final = [np.concatenate(r, axis=0) for r in results]
            
            # Handle single output case
            if len(final) == 1:
                return final[0]
            return final
            
        return None

    def fill_mask(self,
                  inputs: Union[str, List[str]],
                  top_k: int = 5) -> Union[List[Dict], List[List[Dict]]]:
        """Fill in masked tokens.
        
        Like HuggingFace's fill-mask pipeline, but integrated here.
        
        Args:
            inputs: Text with mask tokens
            top_k: How many predictions to return per mask
            
        Returns:
            List of predictions for each input
        """
        self._ensure_built()
        self.model.eval()

        # Handle single input
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]

        all_results = []

        for text in inputs:
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Find mask position
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = torch.where(
                encoded["input_ids"][0] == mask_token_id
            )[0]
            
            if len(mask_positions) != 1:
                raise ValueError(
                    f"Need exactly 1 mask token, found {len(mask_positions)}"
                )
            
            mask_pos = mask_positions[0].item()

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits[0, mask_pos, :]
                probs = torch.softmax(logits, dim=-1)

            # Top-k predictions
            top_probs, top_ids = torch.topk(probs, top_k)
            
            # Build results
            predictions = []
            for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
                token_str = self.tokenizer.decode([token_id])
                
                # Create filled sequence
                filled_tokens = encoded["input_ids"][0].clone()
                filled_tokens[mask_pos] = token_id
                filled_seq = self.tokenizer.decode(filled_tokens, skip_special_tokens=True)
                
                predictions.append({
                    "sequence": filled_seq,
                    "score": prob,
                    "token": token_id,
                    "token_str": token_str,
                })

            all_results.append(predictions)

        return all_results[0] if single_input else all_results