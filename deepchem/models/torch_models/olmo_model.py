from transformers import AutoModelForCausalLM, AutoTokenizer
from deepchem.models.torch_models.hf_models import HuggingFaceModel


class OLMoModel(HuggingFaceModel):

    def __init__(self, model_name="allenai/OLMo-1B"):

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        )

       

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            task=None
        )

       