import pytest
from deepchem.models.brics_genrator import BRICSGenerator
from rdkit import Chem


@pytest.fixture
def brics_generator():
    return BRICSGenerator()


@pytest.fixture
def sample_smiles():
    return ['CC(=O)Oc1ccccc1C(=O)O', 'CC(=O)NC1=CC=C(O)C=C1']


@pytest.fixture
def sample_psmiles():
    return ['*CC(=O)CC*', '*c1ccc(CC*)cc1', '*CC(CC*)CC*']


@pytest.fixture
def sample_psmiles_dendrimers():
    return [
        # Branched aliphatic cores
        '*C(=O)N(CC(=O)*)CC(=O)*',  # 3-point N-centered with amide bonds
        '*OC(=O)c1c(C(=O)O*)cc(C(=O)O*)cc1',  # 3-point aromatic with ester bonds
    ]


def test_brics_decompose(brics_generator, sample_smiles):
    result = brics_generator._BRICS_decompose(sample_smiles)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, str) for x in result)
    assert len(result) == 7


def test_brics_build(brics_generator, sample_smiles):
    decomposed = brics_generator._BRICS_decompose(sample_smiles)
    result = brics_generator._BRICS_build(decomposed)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, str) for x in result)
    assert len(result) == 49


def test_replace_wildcards_with_vatoms(brics_generator):
    input_smiles = ['*CC*', '*CCN*', '[*]CC[*]']
    result = brics_generator.replace_wildcards_with_vatoms(input_smiles)
    assert all('[At]' in x for x in result)
    assert len(result) == len(input_smiles)
    assert result == ['[At]CC[At]', '[At]CCN[At]', '[At]CC[At]']


def test_replace_wildcards_with_vatoms_error(brics_generator):
    with pytest.raises(ValueError):
        brics_generator.replace_wildcards_with_vatoms(['*CC'
                                                      ])  # Only one wildcard


def test_replace_vatoms_with_wildcards(brics_generator):
    input_smiles = ['[At]CC[At]', '[At]CCN[At]']
    result = brics_generator.replace_vatoms_with_wildcards(input_smiles)
    assert all('[*]' in x for x in result)
    assert len(result) == len(input_smiles)
    assert result == ['[*]CC[*]', '[*]CCN[*]']


def test_replace_vatoms_with_wildcards_error(brics_generator):
    with pytest.raises(ValueError):
        brics_generator.replace_vatoms_with_wildcards(
            ['[At]CC'])  # Only one virtual atom


def test_filter_candidates(brics_generator):
    test_mols = ['CC[At]CC[At]', 'CC[At]CC', 'CC[At]CC[At]CC[At]']

    # Test polymer only
    polymer_results = brics_generator.filter_candidates(test_mols,
                                                        is_polymer=True)
    assert all(mol.count('[At]') == 2 for mol in polymer_results)

    # Test polymer and dendrimer
    dendrimer_results = brics_generator.filter_candidates(test_mols,
                                                          is_polymer=True,
                                                          is_dendrimer=True)
    assert all(mol.count('[At]') >= 2 for mol in dendrimer_results)


def test_filter_candidates_error(brics_generator):
    with pytest.raises(ValueError):
        brics_generator.filter_candidates(['CC[At]CC[At]'],
                                          is_polymer=False,
                                          is_dendrimer=True)


def test_sample_smiles(brics_generator, sample_smiles):
    results, count = brics_generator.sample(sample_smiles)
    assert isinstance(results, list)
    assert isinstance(count, int)
    assert count == 49
    assert all(isinstance(x, str) for x in results)


def test_sample_psmiles(brics_generator, sample_psmiles):
    results, count = brics_generator.sample(sample_psmiles, is_polymer=True)
    assert isinstance(results, list)
    assert isinstance(count, int)
    assert count == 2
    assert all(isinstance(x, str) for x in results)


def test_sample_dendrimer(brics_generator, sample_psmiles_dendrimers):
    results, count = brics_generator.sample(sample_psmiles_dendrimers,
                                            is_polymer=True,
                                            is_dendrimer=True)
    assert isinstance(results, list)
    assert isinstance(count, int)
    assert count == len(results)
    assert all(isinstance(x, str) for x in results)
    assert all(x.count('[At]') >= 2 for x in results)
    assert len([x for x in results if x.count('[At]') > 2]) == 8


def test_verbose_output(capsys):
    generator = BRICSGenerator(verbose=True)
    generator.sample(['CC(=O)Oc1ccccc1C(=O)O'])
    captured = capsys.readouterr()
    assert "[+]" in captured.out
