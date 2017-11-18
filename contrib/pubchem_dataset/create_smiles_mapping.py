import pandas as pd
import os
from rdkit import Chem
import time
import gzip

def main() :
    print("Processing PubChem FTP Download")

    sdf_root_path = "/media/data/pubchem/SDF"
    results_path = "/media/data/pubchem/smiles"

    try:
        os.makedirs(results_path)
    except :
        print("Results directory already exists")

    i = 0
    max_smiles_len = 200

    processed_files = os.listdir(results_path)
    processed_files.append("Compound_102125001_102150000_smiles.csv")

    for path, dirs, filenames in os.walk(sdf_root_path) :
        for filename in filenames:

            print("Processing: {0}".format(filename))

            expected_file_name = filename.replace(".sdf.gz", "_smiles.csv")
            new_file_name = os.path.join(results_path,expected_file_name)


            if expected_file_name in processed_files:
                print("Skipping: {0}".format(new_file_name))
                i = i + 1
                continue

            keys = list()
            values = list()

            start = time.time()
            filepath = os.path.join(sdf_root_path,filename)

            with gzip.open(filepath,'rb') as myfile:
                suppl = Chem.ForwardSDMolSupplier(myfile)
                for mol in suppl:
                    if mol is None: continue
                    cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
                    smiles = mol.GetProp("PUBCHEM_OPENEYE_ISO_SMILES")
                    if len(smiles) > max_smiles_len:
                        i = i + 1
                        print("Skipped compound: {0} due to large size".format(cid))
                        continue
                    keys.append(int(cid))
                    values.append(smiles)
                end = time.time()

                print("Processed file number: {0} in {1} seconds".format(i, end - start))
                i = i + 1

                df = pd.DataFrame({"PUBCHEM_CID" : keys, "SMILES" : values},index=keys)
                df.to_csv(new_file_name,index=False)



    # Now parse all results smile files into one big file
    df_list = list()
    processed_files = os.listdir(results_path)
    for filename in processed_files :
        print("Processing: {} for summary CSV".format(filename))
        df = pd.read_csv(os.path.join(results_path,filename))
        df_list.append(df)

    df_full = pd.concat(df_list)


    print("Writing out summary CSV")
    df_full.to_csv("/media/data/pubchem/summary.csv",index=False)

    print("Stored data as CSV")



    print("Done")

if __name__ == '__main__':
    main()
