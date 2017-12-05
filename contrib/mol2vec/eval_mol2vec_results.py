import gensim
from gensim import models
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np

def main() :
    model = models.KeyedVectors.load_word2vec_format("vec.txt")
    embeddings = list()

    # Using canonical smiles for glycine, as in original research paper
    mol = Chem.MolFromSmiles("C(C(=O)O)N")
    try:
        info = {}
        rdMolDescriptors.GetMorganFingerprint(mol, 0, bitInfo=info)
        keys = info.keys()
        keys_list = list(keys)
        totalvec = np.zeros(200)
        for k in keys_list:
            wordvec = model.wv[str(k)]
            totalvec = np.add(totalvec, wordvec)
        embeddings.append(totalvec)
    except Exception as e:
        print(e)
        pass

    print(embeddings[0])


