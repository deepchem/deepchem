import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import gzip
import os

def main() :

    sdf_root_path = "/media/data/pubchem/SDF"

    for path, dirs, filenames in os.walk(sdf_root_path) :
        for filename in filenames:
            filepath = os.path.join(sdf_root_path, filename)

            # This SDF file fails to parse with RDKit on Ubuntu 16.04
            if "Compound_102125001_102150000" in filename:
                continue

            with gzip.open(filepath, 'rb') as myfile:
                suppl = Chem.ForwardSDMolSupplier(myfile)

                for mol in suppl:

                    if not mol:
                        continue

                    try :
                        info = {}
                        rdMolDescriptors.GetMorganFingerprint(mol,1,bitInfo=info)
                        keys = info.keys()
                        keys_list = list(keys)
                        for k in keys_list:
                            print(k,end=' ')
                        print()
                    except Exception:
                        pass

if __name__ == "__main__" :
    main()
