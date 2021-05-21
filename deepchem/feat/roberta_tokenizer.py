from deepchem.feat.base_classes import MolecularFeaturizer
from transformers import RobertaTokenizerFast
from deepchem.utils.typing import RDKitMol


class RobertaFeaturizer(RobertaTokenizerFast, MolecularFeaturizer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return

  def _featurize(self, mol: RDKitMol, **kwargs):
    """Calculate encoding using HuggingFace's RobertaTokenizerFast

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object

        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`. The length is 881.

        """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ImportError("This class requires RDKit to be installed.")
    smiles_string = Chem.MolToSmiles(mol)
    # the encoding is natively a dictionary with keys 'input_ids' and 'attention_mask'
    # -> make this a list of two arrays to allow np to handle it
    encoding = list(self(smiles_string, **kwargs).values())
    return encoding

  def __call__(self, *args, **kwargs):
    return super().__call__(*args, **kwargs)
