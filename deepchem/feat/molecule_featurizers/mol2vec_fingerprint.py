from os import path
from typing import Optional

import numpy as np

from deepchem.utils import download_url, get_data_dir, untargz_file
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer

DEFAULT_PRETRAINED_MODEL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz'


class Mol2VecFingerprint(MolecularFeaturizer):
  """Mol2Vec fingerprints.

  This class convert molecules to vector representations by using Mol2Vec.
  Mol2Vec is an unsupervised machine learning approach to learn vector representations
  of molecular substructures and the algorithm is based on Word2Vec, which is
  one of the most popular technique to learn word embeddings using neural network in NLP.
  Please see the details from [1]_.

  The Mol2Vec requires the pretrained model, so we use the model which is put on the mol2vec
  github repository [2]_. The default model was trained on 20 million compounds downloaded
  from ZINC using the following paramters.

  - radius 1
  - UNK to replace all identifiers that appear less than 4 times
  - skip-gram and window size of 10
  - embeddings size 300

  References
  ----------
  .. [1] Jaeger, Sabrina, Simone Fulle, and Samo Turk. "Mol2vec: unsupervised machine learning
     approach with chemical intuition." Journal of chemical information and modeling 58.1 (2018): 27-35.
  .. [2] https://github.com/samoturk/mol2vec/

  Note
  ----
  This class requires mol2vec to be installed.
  """

  def __init__(self,
               pretrain_model_path: Optional[str] = None,
               radius: int = 1,
               unseen: str = 'UNK'):
    """
    Parameters
    ----------
    pretrain_file: str, optional
      The path for pretrained model. If this value is None, we use the model which is put on
      github repository (https://github.com/samoturk/mol2vec/tree/master/examples/models).
      The model is trained on 20 million compounds downloaded from ZINC.
    radius: int, optional (default 1)
      The fingerprint radius. The default value was used to train the model which is put on
      github repository.
    unseen: str, optional (default 'UNK')
      The string to used to replace uncommon words/identifiers while training.
    """
    try:
      from gensim.models import word2vec
      from mol2vec.features import mol2alt_sentence, sentences2vec
    except ModuleNotFoundError:
      raise ImportError("This class requires mol2vec to be installed.")

    self.radius = radius
    self.unseen = unseen
    self.sentences2vec = sentences2vec
    self.mol2alt_sentence = mol2alt_sentence
    if pretrain_model_path is None:
      data_dir = get_data_dir()
      pretrain_model_path = path.join(data_dir, 'mol2vec_model_300dim.pkl')
      if not path.exists(pretrain_model_path):
        targz_file = path.join(data_dir, 'mol2vec_model_300dim.tar.gz')
        if not path.exists(targz_file):
          download_url(DEFAULT_PRETRAINED_MODEL_URL, data_dir)
        untargz_file(
            path.join(data_dir, 'mol2vec_model_300dim.tar.gz'), data_dir)
    # load pretrained models
    self.model = word2vec.Word2Vec.load(pretrain_model_path)

  def _featurize(self, mol: RDKitMol) -> np.ndarray:
    """
    Calculate Mordred descriptors.

    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
      RDKit Mol object

    Returns
    -------
    np.ndarray
      1D array of mol2vec fingerprint. The default length is 300.
    """
    sentence = self.mol2alt_sentence(mol, self.radius)
    feature = self.sentences2vec([sentence], self.model, unseen=self.unseen)[0]
    return feature
