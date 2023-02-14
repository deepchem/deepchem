from os import path
from typing import Optional

import numpy as np
from rdkit.Chem import AllChem
from deepchem.utils import download_url, get_data_dir, untargz_file
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer

DEFAULT_PRETRAINED_MODEL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz'


def _mol2alt_sentence(mol, radius):
    """Same as mol2sentence() except it only returns the alternating sentence
    Calc6ulates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined

    """
    # Copied from https://github.com/samoturk/mol2vec/blob/850d944d5f48a58e26ed0264332b5741f72555aa/mol2vec/features.py#L129-L168
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(
        mol, radius,
        bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][
                radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


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

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> smiles = ['CCC']
    >>> featurizer = dc.feat.Mol2VecFingerprint()
    >>> features = featurizer.featurize(smiles)
    >>> type(features)
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (300,)

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
        except ModuleNotFoundError:
            raise ImportError("This class requires mol2vec to be installed.")

        self.radius = radius
        self.unseen = unseen
        self.mol2alt_sentence = _mol2alt_sentence
        if pretrain_model_path is None:
            data_dir = get_data_dir()
            pretrain_model_path = path.join(data_dir,
                                            'mol2vec_model_300dim.pkl')
            if not path.exists(pretrain_model_path):
                targz_file = path.join(data_dir, 'mol2vec_model_300dim.tar.gz')
                if not path.exists(targz_file):
                    download_url(DEFAULT_PRETRAINED_MODEL_URL, data_dir)
                untargz_file(path.join(data_dir, 'mol2vec_model_300dim.tar.gz'),
                             data_dir)
        # load pretrained models
        self.model = word2vec.Word2Vec.load(pretrain_model_path)

    def sentences2vec(self, sentences: list, model, unseen=None) -> np.ndarray:
        """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
        sum of vectors for individual words.

        Parameters
        ----------
        sentences : list, array
            List with sentences
        model : word2vec.Word2Vec
            Gensim word2vec model
        unseen : None, str
            Keyword for unseen words. If None, those words are skipped.
            https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032
        Returns
        -------
        np.array

        """
        keys = set(model.wv.key_to_index.keys())
        vec = []
        if unseen:
            unseen_vec = model.wv.get_vector(unseen)

        for sentence in sentences:
            if unseen:
                vec.append(
                    sum([
                        model.wv.get_vector(y) if y in set(sentence) &
                        keys else unseen_vec for y in sentence
                    ]))
            else:
                vec.append(
                    sum([
                        model.wv.get_vector(y)
                        for y in sentence
                        if y in set(sentence) & keys
                    ]))
        return np.array(vec)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate Mordred descriptors.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            1D array of mol2vec fingerprint. The default length is 300.

        """
        if 'mol' in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        sentence = self.mol2alt_sentence(datapoint, self.radius)
        feature = self.sentences2vec([sentence], self.model,
                                     unseen=self.unseen)[0]
        return feature
