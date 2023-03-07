from deepchem.feat.vocabulary_builders.vocabulary_builder import VocabularyBuilder
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import tokenizers
    import transformers


class HuggingFaceVocabularyBuilder(VocabularyBuilder):
    """Wrapper for building vocabulary from HuggingFace algorithms

    The wrapper class can be used to build huggingface vocabulary building algorithms
    with DeepChem.

    Parameters
    ----------
    model: tokenizers.models.Model
        The core tokenization algorithm to learn the vocabulary.
    trainers: tokenizers.trainers.Trainer
        The trainer capable of training the model

    Example
    -------
    >>> from tokenizers import models, trainers
    >>> from tokenizers.pre_tokenizers import Whitespace
    >>> model = models.BPE(unk_token="[UNK]")
    >>> special_tokens = ["[UNK]"]
    >>> trainer = trainers.BpeTrainer(vocab_size=20, special_tokens=special_tokens)
    >>> vb = HuggingFaceVocabularyBuilder(model=model, trainer=trainer)
    """

    def __init__(self, model: 'tokenizers.models.Model',
                 trainer: 'tokenizers.trainers.Trainer'):
        from tokenizers import Tokenizer
        self.model = model
        self.trainer = trainer
        self.tokenizer = Tokenizer(model)

    # superclass accepts a DeepChem dataset while huggingface vocabulary builders
    # reads data from file
    def build(self, paths: List[str]):  # type: ignore
        """Trains the algorithm on the vocabulary

        Parameters
        ----------
        paths: List[str]
            A list of file paths for training
        """
        self.tokenizer.train(paths, self.trainer)

    @classmethod
    def load(
        cls, fname: str
    ) -> 'transformers.tokenization_utils_fast.PreTrainedTokenizerFast':
        """Loads vocabulary from a vocabulary file

        Parameters
        ----------
        fname: str
            Filename to load vocabulary from
        """
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=fname)
        return tokenizer

    def save(self, fname: str) -> None:
        """Saves vocabulary to a file

        Parameters
        ----------
        fname: str
            A json file path
        """
        self.tokenizer.save(fname)
