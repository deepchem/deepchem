import os
import math
import pickle
import gzip
import logging
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from deepchem.utils.data_utils import get_data_dir, download_url

logger = logging.getLogger(__name__)

# REFERENCE ----- Official RDKit fragment scores source
SAS_URL = "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"

class SAScorer:
    """
    SAScore Calculator adapted for DeepChem.
    """
    def __init__(self, fscore_path: Optional[str] = None):
        self.fscore_path = fscore_path
        self._fscores = None
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        
        # Load scores immediately upon initialization
        self._load_fragment_scores()

    def _load_fragment_scores(self):
        """Loads scores from path or downloads them if missing."""
        if self.fscore_path is None:
            # Use DeepChem's default data directory
            data_dir = get_data_dir()
            self.fscore_path = os.path.join(data_dir, "fpscores.pkl.gz")
            
        if not os.path.exists(self.fscore_path):
            logger.info("Downloading SAScore fragment scores...")
            download_url(SAS_URL, dest_dir=get_data_dir())

        # Logic from readFragmentScores adapted for the class
        with gzip.open(self.fscore_path, 'rb') as f:
            data = pickle.load(f)
        
        out_dict = {}
        for i in data:
            for j in range(1, len(i)):
                out_dict[i[j]] = float(i[0])
        self._fscores = out_dict

    def calculate_score(self, mol: Chem.Mol) -> float:
        """The math logic from RDKit goes here..."""
        if not mol or mol.GetNumAtoms() == 0:
            return 0.0
        if not mol.GetNumAtoms():
           return None

        if self._fscores is None:
            self._load_fragment_scores()    
        # fragment score
        sfp = self.mfpgen.GetSparseCountFingerprint(mol)

        score1 = 0.
        nf = 0
        nze = sfp.GetNonzeroElements()
        for id, count in nze.items():
            nf += count
            score1 += self._fscores.get(id, -4) * count

        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nBridgeheads, nSpiro = SAScorer.numBridgeheadsAndSpiro(mol, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        numBits = len(nze)
        if nAtoms > numBits:
            score3 = math.log(float(nAtoms) / numBits) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.

        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore


def processMols(mols):
  print('smiles\tName\tsa_score')
  for i, m in enumerate(mols):
    if m is None:
      continue

    s = SAScorer.calculate_score(m)

    smiles = Chem.MolToSmiles(m)
    if s is None:
      print(f"{smiles}\t{m.GetProp('_Name')}\t{s}")
    else:
      print(f"{smiles}\t{m.GetProp('_Name')}\t{s:3f}")


#AS MENTIONED IN THE SAScore's repo:
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#