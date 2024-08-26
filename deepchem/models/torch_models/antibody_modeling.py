from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from deepchem.models.torch_models import HuggingFaceModel
from typing import Union


class DeepAbLLM(HuggingFaceModel):
    """Flexible Antibody Language Model for Re-Design of Ab Residues.

    This wrapper class is designed to inherit from the HuggingFaceModel
    object from the DeepChem repositor. The class is designed to leverage
    large language model's (LLM) learned sequence dependencies to assist in
    designing anitbody sequences. Currently supports a variety of
    BERT based models:

    [1] ProtBERT
    [2] IgBERT

    Attributes
    ----------
    task: str
        The task the HuggingFaceModel is performing. Default: 'mlm'.
    model_path: str
        The huggingface model path of the pLM.
    n_tasks: int
        Number of tasks for a given model. Default: 1

    Methods
    -------
    __init__(task, model_path, n_tasks)
        Initialize an DeepAbLLM with specified information.
    _mask_seq_pos(position, idx)
        Mask a sequence at a particular index.
    redesign_residue(n=1.0)
        Change the photo's gamma exposure.
    optimize_residue(n=1.0)
        Change the photo's gamma exposure.
    optimize_sequence(n=1.0)
        Change the photo's gamma exposure.

    Notes
    -----
    Model currently supports:
    1. Light or Heavy Chain Re-Design at Arbitrary Point

    (WIP)
    2. Agnostic to Light or Heavy Chain (Depending on if the specified
                                         model correctly accounts for this)

    (Planned)
    3. Model consumes both epitope and receptor information to influence logits
    4. Conditional Generation (Auto-Regressive/Iterative Unmasking)

    References
    ----------
    .. [1] Elnaggar, Ahmed, et al. "Prottrans: Toward understanding the language
           of life through self-supervised learning." IEEE transactions on pattern
           analysis and machine intelligence 44.10 (2021): 7112-7127.
    .. [2] Kenlay, H., Dreyer, F. A., Kovaltsuk, A., Miketa, D., Pires, D.,
           & Deane, C. M. (2024). Large scale paired antibody language models.
           arXiv [q-Bio.BM]. Retrieved from http://arxiv.org/abs/2403.17889

    Example Usage:
    --------------
    >>> # Optimize Sequence
    >>> from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
    >>> model_path = 'Rostlab/prot_bert'
    >>> anti_llm = DeepAbLLM(task='mlm', model_path=model_path, n_tasks=1)
    >>> anti_llm.model.to('cuda')  # Move to GPU for faster inference
    >>> optimized_sequences = anti_llm.optimize_sequence('GSELTQDPAVSVALGQTVRITCQGDSLRNYYASWYQQKPRQAPVLVFYGKNNRPSGIPDRFSGSSSGNTASLTISGAQAEDEADYYCNSRDSSSNHLVFGGGTKLTVLSQ', rounds=1)
    >>> optimized_sequences

    >>>>    [(0,
            'Q',
            'QSETQDPAVSVALGQTVRITCQGDSLRNYYASWYQQKPRQAPVLVFYGKNNRPSGIPDRFSGSSSGNTASLTISGAQAEDEADYYCNSRDSSSNHLVFGGGTKLTVLSQ',
            0.7314766049385071) ...
            ]
    """

    def __init__(self,
                 task: str = "mlm",
                 model_path: str = 'Rostlab/prot_bert',
                 n_tasks: int = 1,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        task: str
            The task defines the type of learning task in the model. The supported tasks are
            - `mlm` - masked language modeling commonly used in pretraining
            - `classification` - use it for classification tasks
        model_path: str
            Path to the HuggingFace model
            - 'Rostlab/prot_bert' - Pretrained on Uniref100 dataset
            - `Rostlab/prot_bert_bfd` - Pretrained on BFD dataset
        n_tasks: int
            Number of prediction targets for a multitask learning model
        """
        self.n_tasks: int = n_tasks
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, do_lower_casse=False)
        config: AutoConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_path,
            vocab_size=tokenizer.vocab_size)
        model: Union[AutoModel, AutoModelForMaskedLM]
        if task == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        super().__init__(model=model, task=task, tokenizer=tokenizer, **kwargs)

    def _mask_seq_pos(
        self,
        sequence: str,
        idx: int,
    ):
        '''Given an arbitrary antibody sequence with and a seqeunce index,
        convert the residue at that index into the mask token.

        Parameters
        ----------
        sequence: str
            The sequence to be masked at a sepcific residue
        idx: int
            The 0-indexed integer index in which into mask the sequence

        Returns
        -------
        masked_sequence: str
            A nearly identical sequence to the input sequence with the token
            at position idx+1 being the mask token.
        '''
        assert isinstance(idx,
                          int), f"Index must be an int, got type {type(idx)}"
        assert abs(idx) < len(sequence), f"Zero-indexed idx needs to \
            be less than {len(sequence)-1} for sequence of length {len(sequence)}."

        cleaned_sequence = sequence.replace(
            ' ', '')  # Get ride of extraneous spaces if any
        temp_sequence = list(cleaned_sequence)  # Turn the sequence into a list
        temp_sequence[idx] = '*'  # Mask the sequence at idx
        masked_sequence = ' '.join(temp_sequence)  # Convert list -> seq
        masked_sequence = masked_sequence.replace('*',
                                                  self.tokenizer.mask_token)
        return masked_sequence

    def redesign_residue(self,
                         sequence: str,
                         residue_index: int,
                         top_k: int = 10,
                         verbose: bool = False):
        '''Given a sequence and a residue index, mask and subsequently
        unmask that position, returning the proposed residues and their
        respective scores.

        Parameters
        ----------
        sequence: str
            The antibody sequence to redesign.
        residue_index: int
            The residue index to mask and unmask.
        top_k: int
            The top_k logits to return. Defaults to 10.
        verbose: bool
            If verbose, prints the original sequence and the resiude at residue_index
            before designing. Useful for running scripts on clusters.

        Returns
        -------
        sequence_tuples: List[tuple]
            Returns a list of tuples containing the
            (replacement token, full sequence, score) for each unmasked token.

        '''
        masked_sequence = self._mask_seq_pos(sequence, residue_index)
        results = self.fill_mask(masked_sequence,
                                 top_k=top_k)  # List of dictionaries

        if verbose:
            print(
                f"Original Residue at Position {residue_index}: {sequence[residue_index]}\n"
            )
        sequence_tuples = [[(result.get('token_str',
                                        ''), result.get('sequence',
                                                        '').replace(' ', ''),
                             result.get('score', None))
                            for result in results
                            if isinstance(result, dict)]]
        return sequence_tuples

    def optimize_residue_pos(self,
                             sequence: str,
                             residue_index: int,
                             verbose: bool = False,
                             **kwargs):
        '''This is a function to return the optimized residues, as defined
        as the proposed residues that are above a given threshold in probability
        using the masking and unmasking approach. Defualt behaviour returns sequences
        with higher scores than the original sequence.

        Parameters
        ----------
        sequence: str
            Antibody sequence to be optimized at particular index.
        residue_index: int
            Index to optimize input antibody sequence.
        verbose: bool

        Optional:
            top_k: int
                Top K logits to be returned by the redesign_residue method
            threshold: float
                Threshold for probability score

        Returns
        -------
        optimized_sequences: List[tuple]
            Returns list of tuples (token, sequence, score) with higher scores
            than the original and the sequence threshold specified.
        '''
        threshold = kwargs.get('threshold', 0.0)
        assert (threshold >= 0) and (
            threshold <=
            1), "Threshold on probability scores should be between 0,1."
        # Sorted List (by score) of redesigned residues
        redesigned_residues = self.redesign_residue(
            sequence, residue_index,
            kwargs['top_k']) if 'top_k' in kwargs else self.redesign_residue(
                sequence, residue_index)

        original_token_str = sequence[residue_index]
        optimized_sequences = []
        for (token_string, full_sequence, score) in redesigned_residues:
            # Check that the token_string is not the same as the original
            if token_string != original_token_str:
                # If it is above a certain probability threshold, append it
                if score > threshold:
                    optimized_sequences += [(token_string, full_sequence, score)
                                           ]
            else:
                # Once the original token is reached break the loop
                break

        return optimized_sequences

    def optimize_sequence(self, sequence: str, **kwargs):
        '''Applies the optimize_residue_pos function to all sequence positions.

        Parameters
        ----------
        sequence: str
            Antibody sequence to be optimized

        Optional:
            top_k: int
                Top K logits to be returned by the redesign_residue method
            threshold: float
                Threshold for probability score

        Returns
        -------
        optimized_sequences: List[tuple]
            Returns list of tuples (index, token, sequence, score)
            that have higher scores than the original and are higher than the
            sequence threshold specified.

        '''
        optimized_sequences = []
        for i in tqdm(range(len(sequence)),
                      desc='Redesigning each residue position'):
            optimized_sequences += [
                (i,) + x
                for x in self.optimize_residue_pos(sequence, i, **kwargs)
            ]
        return optimized_sequences
