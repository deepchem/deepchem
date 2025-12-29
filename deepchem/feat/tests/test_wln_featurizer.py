import unittest
import numpy as np
from deepchem.feat.molecule_featurizers.wln_featurizer import WeisfeilerLehmanScoringModelFeaturizer
from deepchem.feat.molecule_featurizers.wln_featurizer import PaddedGraphFeatures

reactionsmiles1 = [
"[CH2:1]1[O:2][CH2:3][CH2:4][CH2:5]1.[CH:18]([Cl:19])([Cl:20])[Cl:21].[I:6][c:7]1[n:8][cH:9][cH:10][n:11][c:12]1[O:13][CH3:14].[NH2:16][NH2:17].[OH2:15]>>[c:7]1([NH:16][NH2:17])[n:8][cH:9][cH:10][n:11][c:12]1[O:13][CH3:14] 6-7-0.0;16-7-1.0",
"[CH2:15]([CH:16]([CH3:17])[CH3:18])[Mg+:19].[CH2:20]1[O:21][CH2:22][CH2:23][CH2:24]1.[Cl-:14].[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[N:8]([O:9][CH3:10])[CH3:11])[cH:12][cH:13]1>>[OH:1][c:2]1[n:3][cH:4][c:5]([C:6](=[O:7])[CH2:15][CH:16]([CH3:17])[CH3:18])[cH:12][cH:13]1 6-8-0.0;15-6-1.0;15-19-0.0",
"[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14] 12-13-0.0;12-15-1.0",
"[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH2:19])[cH:17][cH:18]3)[cH:11][c:12]12.[CH:25](=[O:26])[OH:27]>>[CH2:1]([CH3:2])[n:3]1[cH:4][c:5]([C:22](=[O:23])[OH:24])[c:6](=[O:21])[c:7]2[cH:8][c:9]([F:20])[c:10](-[c:13]3[cH:14][cH:15][c:16]([NH:19][CH:25]=[O:26])[cH:17][cH:18]3)[cH:11][c:12]12 19-25-1.0;25-27-0.0",
]


reactionsmiles2 = [
    "[F:19][c:20]1[cH:21][cH:22][c:23]([CH2:24][NH:25][C:26]([CH:27]([C:28](=[O:29])[OH:30])[CH3:31])=[O:32])[cH:33][cH:34]1.[NH2:1][CH:2]1[c:3]2[c:4]([cH:15][cH:16][cH:17][cH:18]2)-[c:5]2[c:6]([cH:11][cH:12][cH:13][cH:14]2)[N:7]([CH3:10])[C:8]1=[O:9]>>[NH:1]([CH:2]1[c:3]2[c:4]([cH:15][cH:16][cH:17][cH:18]2)-[c:5]2[c:6]([cH:11][cH:12][cH:13][cH:14]2)[N:7]([CH3:10])[C:8]1=[O:9])[C:28]([CH:27]([C:26]([NH:25][CH2:24][c:23]1[cH:22][cH:21][c:20]([F:19])[cH:34][cH:33]1)=[O:32])[CH3:31])=[O:29] 1-28-1.0;28-30-0.0",
    "[Br:1][c:2]1[c:3]([C:12]([F:13])([F:14])[F:15])[cH:4][c:5]([CH2:8][C:9](=[O:10])[OH:11])[cH:6][cH:7]1.[CH:56]([N:57]([CH2:58][CH3:59])[CH:60]([CH3:61])[CH3:62])([CH3:63])[CH3:64].[NH2:16][c:17]1[cH:18][cH:19][c:20]([N:23]2[CH2:24][CH2:25][N:26]([C:29]([CH3:30])=[O:31])[CH2:27][CH2:28]2)[cH:21][n:22]1.[O:65]=[CH:66][N:67]([CH3:68])[CH3:69].[P-:32]([F:33])([F:34])([F:35])([F:36])([F:37])[F:38].[n:39]1([O:40][C:41]([N:42]([CH3:43])[CH3:44])=[N+:45]([CH3:46])[CH3:47])[c:48]2[n:49][cH:50][cH:51][cH:52][c:53]2[n:54][n:55]1>>[Br:1][c:2]1[c:3]([C:12]([F:13])([F:14])[F:15])[cH:4][c:5]([CH2:8][C:9](=[O:11])[NH:16][c:17]2[cH:18][cH:19][c:20]([N:23]3[CH2:24][CH2:25][N:26]([C:29]([CH3:30])=[O:31])[CH2:27][CH2:28]3)[cH:21][n:22]2)[cH:6][cH:7]1 10-9-0.0;11-9-2.0;16-9-1.0",
    "[BH4-:8].[C:10](#[N:11])[c:12]1[cH:13][cH:14][c:15]([CH2:18][CH2:19][N:20]2[CH2:21][CH2:22][C:23]([OH:26])([CH2:27][NH:28][c:29]3[cH:30][cH:31][c:32]([C:33](=[O:34])[O:35][C:36]([CH3:37])([CH3:38])[CH3:39])[cH:40][cH:41]3)[CH2:24][CH2:25]2)[cH:16][cH:17]1.[CH2:6]=[O:7].[Na+:42].[Na+:9].[O:47]1[CH2:48][CH2:49][CH2:50][CH2:51]1.[OH:43][C:44](=[O:45])[O-:46].[S:1](=[O:2])(=[O:3])([OH:4])[OH:5]>>[C:10](#[N:11])[c:12]1[cH:13][cH:14][c:15]([CH2:18][CH2:19][N:20]2[CH2:21][CH2:22][C:23]([OH:26])([CH2:27][N:28]([c:29]3[cH:30][cH:31][c:32]([C:33](=[O:34])[O:35][C:36]([CH3:37])([CH3:38])[CH3:39])[cH:40][cH:41]3)[CH3:44])[CH2:24][CH2:25]2)[cH:16][cH:17]1 28-44-1.0;43-44-0.0;44-45-0.0;44-46-0.0",
]

invalid_reactionsmiles = [
    "C1CC1.CC>>C1CC1CC",
    "C1CC1.CC>>C1CC1CC"
]

class TestWeisfeilerLehmanScoringModelFeaturizer(unittest.TestCase):
    
    def test_reaction_featurizer(self):
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")
        
        featurizer = WeisfeilerLehmanScoringModelFeaturizer()
        
        reaction = featurizer.featurize(reactionsmiles1)
        assert isinstance(reaction,PaddedGraphFeatures)
        
        reaction2 = featurizer.featurize(reactionsmiles2)         
        assert reaction2.atom_features.shape != reaction.atom_features.shape
        
        max_num_atoms_in_rcnt = max([Chem.MolFromSmiles(i.split(">>")[0]).GetNumAtoms() for i in reactionsmiles1]) + 1 # one is added to avoid getting max atom as zero in case whole reaction list contains invalied smiles
        # test for correct padding
        assert reaction.atom_features.shape[1] == max_num_atoms_in_rcnt
        
        
        assert len(reactionsmiles1) == reaction.atom_features.shape[0]

        assert reaction.atom_features.shape != reaction2.atom_features.shape
        
        #test if all features are not zero
        assert not np.all(reaction.atom_features.shape == 0)
        
    def test_on_invalid_reaction_smiles(self):
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise ImportError("This method requires RDKit to be installed.")
        
        featurizer = WeisfeilerLehmanScoringModelFeaturizer()
        
        badreaction = featurizer.featurize(invalid_reactionsmiles)
        assert isinstance(badreaction,PaddedGraphFeatures)
        # invalid reaction are stored as arrays of zeros
        assert np.all(badreaction.atom_features ==0)
        
        assert badreaction.atom_features.shape[1] == 1 # one is added for invalid reaction to avoid batch issues since features are stored as (batch ,maxatoms,maxatoms,features)
        
        
if __name__ == "__main__":
    unittest.main()