from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from deepchem.models.torch_models import HuggingFaceModel
from typing import Union, Dict, Any


class DeepAbLLM(HuggingFaceModel):
    """Flexible Antibody Language Model for Re-Design of Ab Residues.

    DeepAbLLM is a wrapper class that leverage large language model's (LLMs) learned sequence
    co-dependencies to aid in the (re)-designing anitbody sequences. It supports the instantiation
    of an arbitrary HuggingFace transformer-style model trained on Antibody sequences for the antibody
    sequence redesign, extending an approach introduced in Hie et al's 2023 Nature Biotech paper [1].

    This means the functionality of DeepAbLLM is model architecture-agnostic.

    Currently supports a variety of HuggingFace Protein and Antibody Specific Language Models, including:

    [2] ProtTrans Models (ProtBERT, ProtT5, etc.)
    [3] AbLang
    [4] IgBERT/IgT5
    [5] ESM1b
    [6] ESM1v
    [7] ESM-2

    The model uses single amino acid tokenization to create input tokens for the models from the
    antibody sequences. While most protein models expect spaces in the protein sequences:
        "T H I S I S A P R O T E I N S E Q U E N C E"
    the ESM class of models does not and expects strings of the following format:
        "THISISAPROTEINSEQUENCE"
    Both tokenization schemes are supported by DeepAbLLM by setting the is_esm_variant flag to the
    appropriate value.

    The model supports general pretraining via masked language modeling, domain-adaptive pretraining [8]
    - an additional pretraining step applied to general purpose protein language models for antibody sequences,
    and the finetuning of pre-trained models for regression/classification. To pretrain via masked language
    modeling task, use task = `mlm`, 'regression', or `classification` during initialization.

    Attributes
    ----------
    task: str
        The task the HuggingFaceModel is performing. Default: 'mlm'.
    model_path: str
        The huggingface model path of the pLM.
    n_tasks: int
        Number of tasks for a given model. Default: 1
    is_esm_variant: bool
        Boolean flag to indicate the tokenization scheme.

    Methods
    -------
    __init__(task, model_path, n_tasks)
        Initialize an DeepAbLLM with specified information.
    _mask_seq_pos(position, idx)
        Mask a sequence at a particular index.
    redesign_residue(sequence, residue_index: int, top_k, verbose)
        Mask and unmaks a single residue of a sequence at given index, returning
        top-k plausible amino acid substitutions.
    _optimize_residue_pos(sequence, residue_index, verbose, threshold)
        "Optimizes" a residue position by redesigning it and returning
        only the tokens that rank higher than the original token.
    redesign_sequence(sequence)
        Returns optimized sequences over each residue position with better
        scores than the original sequence.


    Notes
    -----
    (Currently Implements):
    1. Light or Heavy Chain Re-Design at Arbitrary Point
    2. Agnostic to Light or Heavy Chain (Depending on if the specified
                                         model correctly accounts for this)
    (WIP)
    3. Model consumes both epitope and receptor information to influence logits
    (Planned)
    4. Conditional Generation (Auto-Regressive/Iterative Unmasking)

    References
    ----------
    .. [1] Hie, B.L., Shanker, V.R., Xu, D. et al. Efficient evolution of human antibodies
           from general protein language models. Nat Biotechnol 42, 275–283 (2024).
           https://doi.org/10.1038/s41587-023-01763-2

    .. [2] Elnaggar, Ahmed, et al. "Prottrans: Toward understanding the language
           of life through self-supervised learning." IEEE transactions on pattern
           analysis and machine intelligence 44.10 (2021): 7112-7127.

    .. [3] Tobias H Olsen, Iain H Moal, Charlotte M Deane, AbLang: an antibody language
           model for completing antibody sequences, Bioinformatics Advances, Volume 2,
           Issue 1, 2022, vbac046, https://doi.org/10.1093/bioadv/vbac046

    .. [4] Kenlay, H., Dreyer, F. A., Kovaltsuk, A., Miketa, D., Pires, D.,
           & Deane, C. M. (2024). Large scale paired antibody language models.
           arXiv [q-Bio.BM]. Retrieved from http://arxiv.org/abs/2403.17889

    .. [5] Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, Guo D, Ott M, Zitnick CL,
           Ma J, Fergus R. Biological structure and function emerge from scaling
           unsupervised learning to 250 million protein sequences. Proc Natl Acad Sci USA.
           2021 Apr 13;118(15):e2016239118. doi: 10.1073/pnas.2016239118. PMID: 33876751

    .. [6] Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021).
           Language models enable zero-shot prediction of the effects of mutations
           on protein function. bioRxiv. doi:10.1101/2021.07.09.450648

    .. [7] Zeming Lin et al., Evolutionary-scale prediction of atomic-level
           protein structure with a language model.Science379,1123-1130(2023).
           DOI:10.1126/science.ade2574

    .. [8] Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I.,
           Downey, D., & Smith, N. A. (2020). Don’t Stop Pretraining: Adapt
           Language Models to Domains and Tasks. arXiv [Cs.CL].
           Retrieved from http://arxiv.org/abs/2004.10964


    Example Usage:
    --------------
    >>> # Optimize Sequence
    >>> from deepchem.models.torch_models.antibody_modeling import DeepAbLLM
    >>> model_path = 'Rostlab/prot_bert'
    >>> anti_llm = DeepAbLLM(task='mlm', model_path=model_path, n_tasks=1, is_esm_variant=False, device='cpu')
    >>> optimized_sequences = anti_llm.redesign_sequence('GSELTQDPAVSVALGQTVRITCQGDSLRNYYASWYQQKPRQAPVLVFYGKNNRPSGIPDRFSGSSSGNTASLTISGAQAEDEADYYCNSRDSSSNHLVFGGGTKLTVLSQ')
    >>> # Expected Output
    >>> # optimized_sequences[0]
    >>> # (0,'Q','QSETQDPAVSVALGQTVRITCQGDSLRNYYASWYQQKPRQAPVLVFYGKNNRPSGIPDRFSGSSSGNTASLTISGAQAEDEADYYCNSRDSSSNHLVFGGGTKLTVLSQ',0.7314766049385071)
    """

    def __init__(self,
                 task: str = "mlm",
                 model_path: str = 'Rostlab/prot_bert',
                 n_tasks: int = 1,
                 is_esm_variant: bool = False,
                 config: Dict[Any, Any] = {},
                 **kwargs) -> None:
        """
        Parameters
        ----------
        task: str
            The task defines the type of learning task in the model. The supported tasks are
            - `mlm` - masked language modeling commonly used in pretraining
            - `classification` - use it for classification tasks
        model_path: str
            Path to the HuggingFace model; HF Model Hub or local
            - ex: 'Rostlab/prot_bert'
        n_tasks: int
            Number of prediction targets for a multitask learning model
        is_esm_variant: bool
            Flag for proper tokenization (S E Q U E N C E vs SEQUENCE).
        config: dict
            Dictionary of HuggingFace AutoConfig hyper-parameters to update the
            default pretrained model.
        """
        self.n_tasks: int = n_tasks
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_path, do_lower_case=False)
        model_config: AutoConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_path,
            vocab_size=tokenizer.vocab_size)
        model_config.update(config)
        self.is_esm_variant: bool = is_esm_variant
        model: Union[AutoModel, AutoModelForMaskedLM]
        if task == "mlm":
            model = AutoModelForMaskedLM.from_config(model_config)
        else:
            model = AutoModel.from_config(model_config)
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
        if not self.is_esm_variant:
            masked_sequence = ' '.join(
                temp_sequence
            )  # ProtBERT based models expect 'A M I N O A C I D'
        else:
            masked_sequence = ''.join(
                temp_sequence)  # ESM models expect 'AMINOACID'
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
            If verbose, prints the original sequence and the residue at residue_index
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

        sequence_tuples = [(result.get('token_str',
                                       ''), result.get('sequence',
                                                       '').replace(' ', ''),
                            result.get('score', None))
                           for result in results
                           if isinstance(result, dict)]
        if verbose:
            print(
                f"Original Residue at Position {residue_index}: {sequence[residue_index]}\n"
            )
            for i, st in enumerate(sequence_tuples):
                print(f"Redesigned residue {i+1}: {st[0]}, score: {st[-1]}")

        return sequence_tuples

    def _optimize_residue_pos(self,
                              sequence: str,
                              residue_index: int,
                              verbose: bool = False,
                              threshold: float = 0.0,
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

    def redesign_sequence(self, sequence: str, **kwargs):
        '''Applies the _optimize_residue_pos function to all sequence positions.

        Parameters
        ----------
        sequence: str
            Antibody sequence to be optimized

        Optional:
            top_k: int
                Top K logits to be returned by the redesign_residue method
            threshold: float
                Threshold for probability score
            verbose: bool
                Flag to print original and redesigned tokens to the stdout.

        Returns
        -------
        redesigned_sequences: List[tuple]
            Returns list of tuples (index, token, sequence, score)
            that have higher scores than the original and are higher than the
            sequence threshold specified.

        '''
        redesigned_sequences = []
        for i in tqdm(range(len(sequence)),
                      desc='Redesigning each residue position'):
            redesigned_sequences += [
                (i,) + x
                for x in self._optimize_residue_pos(sequence, i, **kwargs)
            ]
        return redesigned_sequences
