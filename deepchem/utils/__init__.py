"""
Miscellaneous utility functions.
"""


class ScaffoldGenerator(object):
  """
  Generate molecular scaffolds.

  Parameters
  ----------
  include_chirality : : bool, optional (default False)
      Include chirality in scaffolds.
  """

  def __init__(self, include_chirality=False):
    self.include_chirality = include_chirality

  def get_scaffold(self, mol):
    """
    Get Murcko scaffolds for molecules.

    Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Parameters
    ----------
    mols : array_like
        Molecules.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=self.include_chirality)
