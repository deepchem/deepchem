import json
from typing import Dict
from collections import Counter
from rdkit import Chem
from deepchem.data import Dataset
from deepchem.feat.base_classes import Featurizer
from deepchem.utils.typing import RDKitMol, RDKitAtom, RDKitBond
from deepchem.feat.vocabulary_builders.vocabulary_builder import VocabularyBuilder


class GroverAtomVocabularyBuilder(VocabularyBuilder):
    """Atom Vocabulary Builder for Grover

    This module can be used to generate atom vocabulary from SMILES strings for
    the GROVER pretraining task. For each atom in a molecule, the vocabulary context is the
    node-edge-count of the atom where node is the neighboring atom, edge is the type of bond (single
    bond or double bound) and count is the number of such node-edge pairs for the atom in its
    neighborhood. For example, for the molecule 'CC(=O)C', the context of the first carbon atom is
    C-SINGLE1 because it's neighbor is C atom, the type of bond is SINGLE bond and the count of such
    bonds is 1. The context of the second carbon atom is C-SINGLE2 and O-DOUBLE1 because
    it is connected to two carbon atoms by a single bond and 1 O atom by a double bond.
    The vocabulary of an atom is then computed as the `atom-symbol_contexts` where the contexts
    are sorted in alphabetical order when there are multiple contexts. For example, the
    vocabulary of second C is `C_C-SINGLE2_O-DOUBLE1`. The algorithm enumerates vocabulary of all atoms
    in the dataset and makes a vocabulary to index mapping by sorting the vocabulary
    by frequency and then alphabetically.


    Example
    -------
    >>> import tempfile
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> file = tempfile.NamedTemporaryFile()
    >>> dataset = dc.data.NumpyDataset(X=[['CCC'], ['CC(=O)C']])
    >>> vocab = GroverAtomVocabularyBuilder()
    >>> vocab.build(dataset)
    >>> vocab.stoi
    {'<pad>': 0, '<other>': 1, 'C_C-SINGLE1': 2, 'C_C-SINGLE2_O-DOUBLE1': 3, 'O_C-DOUBLE1': 4}
    >>> vocab.save(file.name)
    >>> loaded_vocab = vocab.load(file.name)
    >>> mol = Chem.MolFromSmiles('CC(=O)C')
    >>> loaded_vocab.encode(mol, mol.GetAtomWithIdx(1))
    >>> av.encode(mol, atom)
    3

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """

    def __init__(self):
        self.specials = ('<pad>', '<other>')
        self.min_freq = 1
        self.max_size = None
        self.itos = list(self.specials)
        self.stoi = self._make_reverse_mapping(self.itos)
        self.pad_index = 0
        self.other_index = 1

    def build(self, dataset: Dataset) -> None:
        """Builds vocabulary

        Parameters
        ----------
        dataset: dc.data.Dataset
            A dataset object with SMILEs strings in X attribute.
        """
        counter: Dict[str, int] = Counter()
        for x, _, _, _ in dataset.itersamples():
            smiles = x[0]
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                v = self.atom_to_vocab(mol, atom)
                counter[v] += 1

        # sort first by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        for word, freq in words_and_frequencies:
            self.itos.append(word)
        self.stoi = self._make_reverse_mapping(self.itos)

    def save(self, fname: str) -> None:
        """Saves a vocabulary in json format

        Parameter
        ---------
        fname: str
            Filename to save vocabulary
        """
        vocab = {'stoi': self.stoi, 'itos': self.itos}
        with open(fname, 'w') as f:
            json.dump(vocab, f)

    @classmethod
    def load(cls, fname: str) -> 'GroverAtomVocabularyBuilder':
        """Loads vocabulary from the specified json file

        Parameters
        ----------
        fname: str
            JSON file containing vocabulary

        Returns
        -------
        vocab: GroverAtomVocabularyBuilder
            A grover atom vocabulary builder which can be used for encoding
        """
        with open(fname, 'r') as f:
            data = json.load(f)
        vocab = cls()
        vocab.stoi, vocab.itos = data['stoi'], data['itos']
        return vocab

    @staticmethod
    def atom_to_vocab(mol: RDKitMol, atom: RDKitAtom) -> str:
        """Convert atom to vocabulary.

        Parameters
        ----------
        mol: RDKitMol
            an molecule object
        atom: RDKitAtom
            the target atom.

        Returns
        -------
        vocab: str
            The generated atom vocabulary with its contexts.

        Example
        -------
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('[C@@H](C)C(=O)O')
        >>> atom_to_vocab(mol, mol.GetAtomWithIdx(0))
        'C_C-SINGLE2'
        >>> atom_to_vocab(mol, mol.GetAtomWithIdx(3))
        'O_C-DOUBLE1'
        """
        nei: Dict[str, int] = Counter()
        for a in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
            nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
        keys = list(nei.keys())
        # sorting the atoms neighbors
        keys.sort()
        output = atom.GetSymbol()
        # concatenating the sorted neighbors
        for key in keys:
            output = "%s_%s%d" % (output, key, nei[key])
        return output

    def _make_reverse_mapping(self, itos):
        return {tok: i for i, tok in enumerate(itos)}

    def encode(self, mol: RDKitMol, atom: RDKitAtom):
        """Encodes an atom in a molecule

        Parameter
        ---------
        mol: RDKitMol
            An RDKitMol object
        atom: RDKitAtom
            An atom in the molecule
        """
        return self.stoi.get(self.atom_to_vocab(mol, atom))


class GroverBondVocabularyBuilder(VocabularyBuilder):
    """Bond Vocabulary Builder for Grover

    This module can be used to generate bond vocabulary from SMILES strings for
    the GROVER pretraining task. For each bond in a molecule, the vocabulary context is the
    node-edge-count of the bond where node is the neighboring atom, edge is the type of bond
    and count is the number of such node-edge pairs for the bond in its one hop
    neighborhood. The algorithm enumerates vocabulary of all bonds
    in the dataset and makes a vocabulary to index mapping by sorting the vocabulary
    by frequency and then alphabetically.


    Example
    -------
    >>> import tempfile
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> file = tempfile.NamedTemporaryFile()
    >>> dataset = dc.data.NumpyDataset(X=[['CCC']])
    >>> vocab = GroverBondVocabularyBuilder()
    >>> vocab.build(dataset)
    >>> vocab.stoi
    {'<pad>': 0, '<other>': 1, '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1': 2}
    >>> vocab.save(file.name)
    >>> loaded_vocab = vocab.load(file.name)
    >>> mol = Chem.MolFromSmiles('CC')
    >>> loaded_vocab.encode(mol, mol.GetBondWithIdx(0))

    Reference
    ---------
    .. Rong, Yu, et al. "Self-supervised graph transformer on large-scale molecular data." Advances in Neural Information Processing Systems 33 (2020): 12559-12571.
    """
    BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']

    def __init__(self):
        self.specials = ('<pad>', '<other>')
        self.min_freq = 1
        self.max_size = None
        self.itos = list(self.specials)
        self.stoi = self._make_reverse_mapping(self.itos)
        self.pad_index = 0
        self.other_index = 1

    def build(self, dataset: Dataset):
        counter: Dict[str, int] = Counter()
        for x, _, _, _ in dataset.itersamples():
            smiles = x[0]
            mol = Chem.MolFromSmiles(smiles)
            for bond in mol.GetBonds():
                v = self.bond_to_vocab(mol, bond)
                counter[v] += 1

        # sort first by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        for word, freq in words_and_frequencies:
            self.itos.append(word)
        self.stoi = self._make_reverse_mapping(self.itos)

    def save(self, fname: str):
        vocab = {'stoi': self.stoi, 'itos': self.itos}
        with open(fname, 'w') as f:
            json.dump(vocab, f)

    @classmethod
    def load(cls, fname: str):
        with open(fname, 'r') as f:
            data = json.load(f)
        vocab = cls()
        vocab.stoi, vocab.itos = data['stoi'], data['itos']
        return vocab

    @staticmethod
    def bond_to_vocab(mol: RDKitMol, bond: RDKitBond):
        """Convert bond to vocabulary.

        The algorithm considers only one-hop neighbor atoms.

        Parameters
        ----------
        mol: Molecule
            the molecular.
        atom: rdchem.Atom
            the target atom.

        Returns
        -------
        vocab: str
            the generated bond vocabulary with its contexts.

        Example
        -------
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('[C@@H](C)C(=O)O')
        >>> bond_to_vocab(mol, mol.GetBondWithIdx(0))
        '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1'
        >>> bond_to_vocab(mol, mol.GetBondWithIdx(2))
        '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2'
        """
        nei: Dict[str, int] = Counter()
        two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
        two_indices = [a.GetIdx() for a in two_neighbors]
        for nei_atom in two_neighbors:
            for a in nei_atom.GetNeighbors():
                a_idx = a.GetIdx()
                if a_idx in two_indices:
                    continue
                tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
                nei[str(nei_atom.GetSymbol()) + '-' +
                    GroverBondVocabularyBuilder._get_bond_feature_name(
                        tmp_bond)] += 1
        keys = list(nei.keys())
        keys.sort()
        output = GroverBondVocabularyBuilder._get_bond_feature_name(bond)
        for k in keys:
            output = "%s_%s%d" % (output, k, nei[k])
        return output

    @staticmethod
    def _get_bond_feature_name(bond: RDKitBond):
        """Return the string format of bond features."""
        ret = []
        for bond_feature in GroverBondVocabularyBuilder.BOND_FEATURES:
            fea = eval(f"bond.Get{bond_feature}")()
            ret.append(str(fea))

        return '(' + '-'.join(ret) + ')'

    def _make_reverse_mapping(self, itos):
        return {tok: i for i, tok in enumerate(itos)}

    def encode(self, mol: RDKitMol, bond: RDKitBond):
        return self.stoi.get(self.bond_to_vocab(mol, bond))


class GroverAtomVocabTokenizer(Featurizer):

    def __init__(self, filename: str):
        self.vocabulary = GroverAtomVocabularyBuilder.load(filename)

    def _featurize(self, datapoint):
        return self.vocabulary.encode(datapoint[0], datapoint[1])


class GroverBondVocabTokenizer(Featurizer):

    def __init__(self, filename: str):
        self.vocabulary = GroverBondVocabularyBuilder.load(filename)

    def _featurize(self, datapoint):
        return self.vocabulary.encode(datapoint[0], datapoint[1])
