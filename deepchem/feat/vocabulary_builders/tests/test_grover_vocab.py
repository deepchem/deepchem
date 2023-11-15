import tempfile
import os
import pandas as pd
from rdkit import Chem
import deepchem as dc


def testGroverAtomVocabularyBuilder():
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverAtomVocabularyBuilder
    file = tempfile.NamedTemporaryFile()
    dataset = dc.data.NumpyDataset(X=[['CC(=O)C'], ['CCC']])
    vocab = GroverAtomVocabularyBuilder()
    vocab.build(dataset)
    assert vocab.stoi == {
        '<pad>': 0,
        '<other>': 1,
        'C_C-SINGLE1': 2,
        'C_C-SINGLE2': 3,
        'C_C-SINGLE2_O-DOUBLE1': 4,
        'O_C-DOUBLE1': 5
    }
    assert vocab.itos == [
        '<pad>', '<other>', 'C_C-SINGLE1', 'C_C-SINGLE2',
        'C_C-SINGLE2_O-DOUBLE1', 'O_C-DOUBLE1'
    ]
    vocab.save(file.name)

    loaded_vocab = GroverAtomVocabularyBuilder.load(file.name)
    mol = Chem.MolFromSmiles('CC(=O)C')
    atom = mol.GetAtomWithIdx(0)
    assert loaded_vocab.atom_to_vocab(mol, atom) == 'C_C-SINGLE1'
    assert loaded_vocab.encode(mol, mol.GetAtomWithIdx(0)) == 2

    # test with max size
    vocab = GroverAtomVocabularyBuilder(max_size=3)
    vocab.build(dataset)
    assert vocab.size == 3
    assert vocab.size == len(vocab.itos)


def test_grover_atom_vocabulary_build_from_csv(tmpdir):
    # test build from csv
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverAtomVocabularyBuilder
    atom_vocab = GroverAtomVocabularyBuilder()
    X = ['CC(=O)C', 'CCC']
    df = pd.DataFrame({'X': X})
    csv_path = os.path.join(tmpdir, 'temp.csv')
    df.to_csv(csv_path)

    atom_vocab.build_from_csv(csv_path, smiles_field='X')
    assert atom_vocab.stoi == {
        '<pad>': 0,
        '<other>': 1,
        'C_C-SINGLE1': 2,
        'C_C-SINGLE2': 3,
        'C_C-SINGLE2_O-DOUBLE1': 4,
        'O_C-DOUBLE1': 5
    }
    assert atom_vocab.itos == [
        '<pad>', '<other>', 'C_C-SINGLE1', 'C_C-SINGLE2',
        'C_C-SINGLE2_O-DOUBLE1', 'O_C-DOUBLE1'
    ]


def testGroverBondVocabularyBuilder():
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverBondVocabularyBuilder
    file = tempfile.NamedTemporaryFile()
    dataset = dc.data.NumpyDataset(X=[['CC(=O)C'], ['CCC']])
    vocab = GroverBondVocabularyBuilder()
    vocab.build(dataset)
    assert vocab.stoi == {
        '<pad>':
            0,
        '<other>':
            1,
        '(SINGLE-STEREONONE-NONE)_C-(DOUBLE-STEREONONE-NONE)1_C-(SINGLE-STEREONONE-NONE)1':
            2,
        '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1':
            3,
        '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2':
            4,
    }
    assert vocab.itos == [
        '<pad>', '<other>',
        '(SINGLE-STEREONONE-NONE)_C-(DOUBLE-STEREONONE-NONE)1_C-(SINGLE-STEREONONE-NONE)1',
        '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1',
        '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2'
    ]
    vocab.save(file.name)

    loaded_vocab = GroverBondVocabularyBuilder.load(file.name)
    mol = Chem.MolFromSmiles('CC(=O)C')
    bond = mol.GetBondWithIdx(0)
    assert loaded_vocab.bond_to_vocab(
        mol, bond
    ) == '(SINGLE-STEREONONE-NONE)_C-(DOUBLE-STEREONONE-NONE)1_C-(SINGLE-STEREONONE-NONE)1'
    assert loaded_vocab.encode(mol, bond) == 2

    # test with max size
    vocab = GroverBondVocabularyBuilder(max_size=3)
    vocab.build(dataset)
    assert vocab.size == 3
    assert vocab.size == len(vocab.itos)


def test_grover_bond_vocabulary_build_from_csv(tmpdir):
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverBondVocabularyBuilder
    bond_vocab = GroverBondVocabularyBuilder()
    X = ['CC(=O)C', 'CCC']
    df = pd.DataFrame({'X': X})
    csv_path = os.path.join(tmpdir, 'temp.csv')
    df.to_csv(csv_path)

    bond_vocab.build_from_csv(csv_path, smiles_field='X')

    assert bond_vocab.stoi == {
        '<pad>':
            0,
        '<other>':
            1,
        '(SINGLE-STEREONONE-NONE)_C-(DOUBLE-STEREONONE-NONE)1_C-(SINGLE-STEREONONE-NONE)1':
            2,
        '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1':
            3,
        '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2':
            4,
    }
    assert bond_vocab.itos == [
        '<pad>', '<other>',
        '(SINGLE-STEREONONE-NONE)_C-(DOUBLE-STEREONONE-NONE)1_C-(SINGLE-STEREONONE-NONE)1',
        '(SINGLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)1',
        '(DOUBLE-STEREONONE-NONE)_C-(SINGLE-STEREONONE-NONE)2'
    ]


def testGroverAtomVocabTokenizer():
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverAtomVocabularyBuilder, GroverAtomVocabTokenizer
    file = tempfile.NamedTemporaryFile()
    dataset = dc.data.NumpyDataset(X=[['CC(=O)C'], ['CCC']])
    vocab = GroverAtomVocabularyBuilder()
    vocab.build(dataset)
    vocab.save(file.name)  # build and save the vocabulary

    # load the vocabulary by passing filename
    atom_tokenizer = GroverAtomVocabTokenizer(file.name)
    mol = Chem.MolFromSmiles('CC(=O)C')
    # test tokenization of a single point
    atom_tokenizer.featurize([(mol, mol.GetAtomWithIdx(0))]) == 2


def testGroverBondVocabTokenizer():
    from deepchem.feat.vocabulary_builders.grover_vocab import GroverBondVocabularyBuilder, GroverBondVocabTokenizer
    file = tempfile.NamedTemporaryFile()
    dataset = dc.data.NumpyDataset(X=[['CC(=O)C'], ['CCC']])
    vocab = GroverBondVocabularyBuilder()
    vocab.build(dataset)
    vocab.save(file.name)  # build and save the vocabulary

    # load the vocabulary by passing the filename
    bond_tokenizer = GroverBondVocabTokenizer(file.name)
    mol = Chem.MolFromSmiles('CC(=O)C')
    # test tokenization of a single point
    bond_tokenizer.featurize([(mol, mol.GetBondWithIdx(0))])[0] == 2
