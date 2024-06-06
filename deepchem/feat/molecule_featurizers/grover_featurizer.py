"""
Grover Featurizer.
The adaptation is based on https://github.com/tencent-ailab/grover/blob/0421d97a5e1bd1b59d1923e3afd556afbe4ff782/grover/data/molgraph.py

"""
from typing import Optional, List
import numpy as np
from deepchem.feat.graph_data import GraphData
from deepchem.feat.molecule_featurizers import RDKitDescriptors
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.typing import RDKitMol
from rdkit import Chem

GROVER_RDKIT_PROPS = [
    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
    'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
    'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
    'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
    'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
    'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
    'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
    'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
    'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole',
    'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
    'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
    'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
    'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
    'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
    'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea'
]


class GroverFeaturizer(MolecularFeaturizer):
    """Featurizer for GROVER Model

    The Grover Featurizer is used to compute features suitable for grover model.
    It accepts an rdkit molecule of type `rdkit.Chem.rdchem.Mol` or a SMILES string
    as input and computes the following sets of features:
        1. a molecular graph from the input molecule
        2. functional groups which are used **only** during pretraining
        3. additional features which can **only** be used during finetuning

    Parameters
    ----------
    additional_featurizer: dc.feat.Featurizer
        Given a molecular dataset, it is possible to extract additional molecular features in order
    to train and finetune from the existing pretrained model. The `additional_featurizer` can
    be used to generate additional features for the molecule.

    References
    ---------
    .. [1] Rong, Yu, et al. "Self-supervised graph transformer on large-scale
        molecular data." NeurIPS, 2020

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.feat import GroverFeaturizer
    >>> feat = GroverFeaturizer(features_generator = dc.feat.CircularFingerprint())
    >>> out = feat.featurize('CCC')

    Note
    ----
    This class requires RDKit to be installed.

    """

    def __init__(self,
                 features_generator: Optional[MolecularFeaturizer] = None,
                 bond_drop_rate: float = 0.0):
        self.featurizer = features_generator
        self.functional_group_generator = RDKitDescriptors(
            descriptors=GROVER_RDKIT_PROPS, labels_only=True)
        self.bond_drop_rate = bond_drop_rate

    def _get_atom_features(self, atom, mol):
        from deepchem.feat.molecule_featurizers.dmpnn_featurizer import atom_features
        features = atom_features(atom)
        atom_idx = atom.GetIdx()

        hydrogen_donor = Chem.MolFromSmarts(
            "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
        )

        hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
        hydrogen_acceptor_match = sum(
            mol.GetSubstructMatches(hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(acidic), ())
        basic_match = sum(mol.GetSubstructMatches(basic), ())
        ring_info = mol.GetRingInfo()
        features = features + \
                   one_hot_encode(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6], include_unknown_set=True) + \
                   [atom_idx in hydrogen_acceptor_match] + \
                   [atom_idx in hydrogen_donor_match] + \
                   [atom_idx in acidic_match] + \
                   [atom_idx in basic_match] + \
                   [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                    ring_info.IsAtomInRingOfSize(atom_idx, 4),
                    ring_info.IsAtomInRingOfSize(atom_idx, 5),
                    ring_info.IsAtomInRingOfSize(atom_idx, 6),
                    ring_info.IsAtomInRingOfSize(atom_idx, 7),
                    ring_info.IsAtomInRingOfSize(atom_idx, 8)]
        return features

    def _make_mol_graph(self, mol: RDKitMol) -> GraphData:
        from deepchem.feat.molecule_featurizers.dmpnn_featurizer import bond_features
        smiles = Chem.MolToSmiles(mol)
        f_atoms = []  # mapping from atom index to atom features
        f_bonds = [
        ]  # mapping from bond index to concat(from_atom, bond) features
        edge_index = []

        n_atoms = mol.GetNumAtoms()  # number of atoms
        n_bonds = 0  # number of bonds
        a2b: List[List[int]] = [
        ]  # mapping from atom index to incoming bond indices
        b2a = [
        ]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond

        for _ in range(n_atoms):
            a2b.append([])

        for _, atom in enumerate(mol.GetAtoms()):
            f_atoms.append(self._get_atom_features(atom, mol))

        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                if np.random.binomial(1, self.bond_drop_rate):
                    continue

                f_bond = bond_features(bond)
                # Always treat the bond as directed.
                f_bonds.append(f_atoms[a1] + f_bond)
                f_bonds.append(f_atoms[a2] + f_bond)

                edge_index.extend([[a1, a2], [a2, a1]])

                b1 = n_bonds  # b1: bond id
                b2 = b1 + 1  # b2: reverse bond id
                # add mapping between bond b1 and atom a2 (destination atom)
                a2b[a2].append(b1)  # b1 = a1 --> a2
                # add mapping between bond id and atom id (a1)
                b2a.append(a1)
                # add mapping between bond id and atom a1 (source atom)
                a2b[a1].append(b2)  # b2 = a2 --> a1
                b2a.append(a2)
                # update index on bond and reverse bond mappings
                b2revb.append(b2)
                b2revb.append(b1)
                n_bonds += 2

        molgraph = GraphData(node_features=np.asarray(f_atoms),
                             edge_index=np.asarray(edge_index).T,
                             edge_features=np.asarray(f_bonds),
                             b1=b1,
                             b2=b2,
                             a2b=a2b,
                             b2a=b2a,
                             b2revb=b2revb,
                             n_bonds=n_bonds,
                             n_atoms=n_atoms,
                             smiles=smiles)
        return molgraph

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Featurize a single input molecule.

        Parameters
        ----------
        datapoint: RDKitMol
            Singular Molecular Graph derived from a SMILES string.

        Returns
        -------
        output: MolGraph
            MolGraph generated by Grover

        """
        molgraph = self._make_mol_graph(datapoint)
        setattr(molgraph, 'fg_labels',
                self.functional_group_generator.featurize(datapoint)[0])
        if self.featurizer:
            setattr(molgraph, 'additional_features',
                    self.featurizer.featurize(datapoint)[0])
        return molgraph
