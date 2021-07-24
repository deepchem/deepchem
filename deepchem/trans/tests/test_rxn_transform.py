import unittest
import numpy as np

from deepchem.trans.transformers import RxnSplitTransformer

reactions: np.ndarray = np.array(
    [
        "CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1>C1CCOC1.[Cl-]>CC(C)CC(=O)c1ccc(O)nc1",
        "CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO>>CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(NC=O)cc3)cc21"
    ],
    dtype=object)


class TestRxnSplitTransformer(unittest.TestCase):
  """Tests the Reaction split transformer for the source/target splitting and
    for the reagent mixing operation.
    """

  def test_split(self):
    """
        Tests the source/target split from an input reaction SMILES.
        """
    trans = RxnSplitTransformer(sep_reagent=True)
    split_reactions = trans.transform_array(X=reactions, y=[], w=[], ids=[])
    assert split_reactions[0].shape == (3, 2)
    # Should check for equality of split_reactions and sep_split!

  # def test_mixing(self):
  #    """
  #    Tests the reagent - reactant mixing option.
  #    """
  # WIP
  #    pass
