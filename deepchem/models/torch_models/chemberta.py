from typing import Dict, Any, Tuple
from deepchem.models.torch_models.hf_models import HuggingFaceModel

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
)
from transformers import RobertaTokenizerFast, RobertaTokenizer
from transformers.modeling_utils import PreTrainedModel

try:
    import torch
    has_torch = True
except:
    has_torch = False


class Chemberta(HuggingFaceModel):
    """Chemberta Model"""

    def __init__(
        self,
        task: str,
        tokenizer_path: str = "seyonec/PubChem10M_SMILES_BPE_60k",
        n_tasks: int = 1,
        config: Dict[Any, Any] = {},
        **kwargs,
    ):
        self.n_tasks = n_tasks

        # ✅ Robust tokenizer loader (transformers 5.x compatible)
        def load_tokenizer(path):
            try:
                return RobertaTokenizerFast.from_pretrained(path)
            except Exception:
                return RobertaTokenizer.from_pretrained(path)

        tokenizer = load_tokenizer(tokenizer_path)

        model: PreTrainedModel
        chemberta_config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            **config,
        )

        if task == "mlm":
            model = RobertaForMaskedLM(chemberta_config)

        elif task == "mtr":
            chemberta_config.problem_type = "regression"
            chemberta_config.num_labels = n_tasks
            model = RobertaForSequenceClassification(chemberta_config)

        elif task == "regression":
            chemberta_config.problem_type = "regression"
            chemberta_config.num_labels = n_tasks
            model = RobertaForSequenceClassification(chemberta_config)

        elif task == "classification":
            if n_tasks == 1:
                chemberta_config.problem_type = "single_label_classification"
            else:
                chemberta_config.problem_type = "multi_label_classification"
                chemberta_config.num_labels = n_tasks
            model = RobertaForSequenceClassification(chemberta_config)

        else:
            raise ValueError("invalid task specification")

        super(Chemberta, self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        smiles_batch, y, w = batch

        tokens = self.tokenizer(
            smiles_batch[0].tolist(),
            padding=True,
            return_tensors="pt",
        )

        if self.task == "mlm":
            inputs, labels = self.data_collator.torch_mask_tokens(
                tokens["input_ids"]
            )
            inputs = {
                "input_ids": inputs.to(self.device),
                "labels": labels.to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device),
            }
            return inputs, None, w

        elif self.task in ["regression", "classification", "mtr"]:
            if y is not None:
                y = torch.from_numpy(y[0])

                if self.task in ["regression", "mtr"]:
                    y = y.float().to(self.device)

                elif self.task == "classification":
                    if self.n_tasks == 1:
                        y = y.long().to(self.device)
                    else:
                        y = y.float().to(self.device)

            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, "labels": y}
            return inputs, y, w