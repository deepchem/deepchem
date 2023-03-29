from deepchem.feat.vocabulary_builders.vocabulary_builder import VocabularyBuilder
from tokenizers import Tokenizer
from typing import List


class HuggingFaceVocabularyBuilder(VocabularyBuilder):

    def __init__(self, model, trainer):
        # This tokenizer is from transformers
        self.model = model
        self.trainer = trainer
        self.tokenizer = Tokenizer(model)

    # superclass accepts a DeepChem dataset while huggingface vocabulary builders
    # reads data from file
    def build(self, paths: List[str]):  # type: ignore

        self.tokenizer.train(paths, self.trainer)

    @classmethod
    def load(cls, fname: str):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=fname)
        return tokenizer

    def save(self, fname: str):
        """
        Parameters
        ----------
        fname: str
            A json file path
        """
        self.tokenizer.save(fname)
