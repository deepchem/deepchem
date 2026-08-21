# deepchem/feat/drug_modality_featurizer.py

from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem.utils.conversion_utils import sanitize_smiles


class DrugModalityFeaturizer:
    def __init__(self, modality_type: str = 'PROTAC'):
        self.modality_type = modality_type
        self._load_validation_rules()

    def _load_validation_rules(self):
        # Define modality-specific validation rules
        self.rules = {
            'PROTAC': {
                'max_atoms': 200,
                'required_substructures': ['[*]C(=O)N[*]', '[*]C#N'],  # example SMARTS patterns
                'allowed_elements': {'C', 'N', 'O', 'S', 'P'}
            },
            'peptide': {
                'max_residues': 30,
                'allowed_amino_acids': ['A', 'R', 'N', 'D', 'C']  # Toy example
            }
        }

    def featurize(self, smiles: str) -> dict:
        """Convert and validate SMILES for specific drug modalities"""
        clean_smiles = sanitize_smiles(smiles)
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string after sanitization.")

        if self.modality_type == 'PROTAC':
            self._validate_protac(mol)
        elif self.modality_type == 'peptide':
            self._validate_peptide(clean_smiles)

        return {
            'sanitized_smiles': clean_smiles,
            'features': self._compute_3d_descriptors(mol)
        }

    def _validate_protac(self, mol):
        rules = self.rules['PROTAC']
        if mol.GetNumAtoms() > rules['max_atoms']:
            raise ValueError("Too many atoms for a PROTAC.")

        if not all(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in rules['required_substructures'])
        ):
            raise ValueError("Missing required PROTAC substructures.")

        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in rules['allowed_elements']:
                raise ValueError(f"Element {atom.GetSymbol()} not allowed in PROTACs.")

    def _validate_peptide(self, smiles):
        aa_seq = smiles.replace('.', '')  # toy model
        rules = self.rules['peptide']
        if len(aa_seq) > rules['max_residues']:
            raise ValueError("Too many residues for a peptide.")
        for aa in aa_seq:
            if aa not in rules['allowed_amino_acids']:
                raise ValueError(f"Amino acid {aa} not allowed in peptides.")

    def _compute_3d_descriptors(self, mol):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
        return [atom.GetAtomicNum() for atom in mol.GetAtoms()]
