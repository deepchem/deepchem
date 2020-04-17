from deepchem.feat.base_classes import MolecularFeaturizer


class RawFeaturizer(MolecularFeaturizer):
  """Encodes a molecule as a SMILES string or RDKit mol.

  This featurizer can be useful when you're trying to transform a large
  collection of RDKit mol objects as Smiles strings, or alternatively as a
  "no-op" featurizer in your molecular pipeline.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, smiles=False):
    """Initialize this featurizer.

    Parameters
    ----------
    smiles: bool, optional (default False)
      If True, encode this molecule as a SMILES string. Else as a RDKit mol.
    """
    try:
      from rdkit import Chem
    except ModuleNotFoundError:
      raise ValueError("This class requires RDKit to be installed.")
    self.smiles = smiles

  def _featurize(self, mol):
    """Calculate either smiles string or pass through raw molecule. 

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    Smiles string or raw molecule.
    """
    from rdkit import Chem
    if self.smiles:
      return Chem.MolToSmiles(mol)
    else:
      return mol

class RawReactionFeaturizer(Featurizer):
  """Featurize SMARTS as RDKit Reaction objects.

  This featurizer uses RDKit's `rdkit.Chem.rdChemReactions.ReactionFromSmarts` to parse in input SMARTS strings.
  """

  def __init__(self, smarts=True):
    """
    Parameters
    ----------
    smarts: bool, optional
      If True, process smarts into rdkit Reaction objects. Else don't process.
    """
    self.smarts = smarts 

  def _featurize(self, mol):
    """
    mol: string
      The SMARTS string to process.
    """
    from rdkit.Chem import rdChemReactions
    if self.smarts:
      smarts = mol 
      # Sometimes smarts have extraneous information at end of
      # form " |f:0" that causes parsing to fail. Not sure what
      # this information is, but just ignoring for now.
      smarts = smarts.split(" ")[0]
      rxn = rdChemReactions.ReactionFromSmarts(smarts)
      return rxn
    else:
      return mol
