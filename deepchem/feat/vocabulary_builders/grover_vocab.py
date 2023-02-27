import json
from typing import Dict
from collections import Counter
from rdkit import Chem
from deepchem.data import Dataset
from deepchem.utils.typing import RDKitMol, RDKitAtom, RDKitBond
from deepchem.feat.base_classes import Featurizer
from deepchem.feat.vocabulary_builders.vocabulary_builder import VocabularyBuilder


class GroverAtomVocabBuilder(VocabularyBuilder):
    other_index = 1

    def __init__(self):
        self.specials = ('<pad>', '<other>')
        self.min_freq = 1
        self.max_size = None
        self.itos = list(self.specials)
        self.stoi = self._make_reverse_mapping(self.itos)
        self.pad_index = 0

    def build(self, dataset: Dataset):
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
        return vocab.stoi, vocab.itos

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


class GroverBondVocabBuilder(VocabularyBuilder):

    BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']
    other_index = 1

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
        return vocab.stoi, vocab.itos

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
                    GroverBondVocabBuilder._get_bond_feature_name(
                        tmp_bond)] += 1
        keys = list(nei.keys())
        keys.sort()
        output = GroverBondVocabBuilder._get_bond_feature_name(bond)
        for k in keys:
            output = "%s_%s%d" % (output, k, nei[k])
        return output

    @staticmethod
    def _get_bond_feature_name(bond: RDKitBond):
        """Return the string format of bond features."""
        ret = []
        for bond_feature in GroverBondVocabBuilder.BOND_FEATURES:
            fea = eval(f"bond.Get{bond_feature}")()
            ret.append(str(fea))

        return '(' + '-'.join(ret) + ')'

    def _make_reverse_mapping(self, itos):
        return {tok: i for i, tok in enumerate(itos)}


class GroverAtomTokenizer(Featurizer):

    def __init__(self, filename: str):
        self.stoi, _ = GroverAtomVocabBuilder.load(filename)

    def _featurize(self, datapoint):
        # returns an encoding
        return self.stoi.get(
            GroverAtomVocabBuilder.atom_to_vocab(datapoint[0], datapoint[1]),
            GroverAtomVocabBuilder.other_index)


class GroverBondTokenizer(Featurizer):

    def __init__(self, filename: str):
        self.stoi, _ = GroverBondVocabBuilder.load(filename)

    def _featurize(self, datapoint):
        # returns an encoding
        return self.stoi.get(
            GroverBondVocabBuilder.bond_to_vocab(datapoint[0], datapoint[1]),
            GroverBondVocabBuilder.other_index)
