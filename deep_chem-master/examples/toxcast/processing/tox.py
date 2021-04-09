
#Processing of ToxCast data
#Author - Aneesh Pappu

import pandas as pd
import numpy as np

#Loading dataframes and editing indices
path_to_casn_smiles = "./casn_to_smiles.csv.gz"
path_to_code_casn = "./code_to_casn.csv.gz"
path_to_hitc_code = "./code_to_hitc.csv.gz"

casn_smiles_df = pd.read_csv(path_to_casn_smiles)
code_casn_df = pd.read_csv(path_to_code_casn)
hitc_code_df = pd.read_csv(path_to_hitc_code)

casn_smiles_df = casn_smiles_df[['Substance_CASRN', 'Structure_SMILES']]
code_casn_df = code_casn_df[['casn', 'code']]
hitc_code_df.rename(columns = {'Unnamed: 0': 'code'}, inplace = True)
casn_smiles_df.rename(columns = {'Substance_CASRN': 'casn', 'Structure_SMILES': 'smiles'}, inplace = True)

code_casn_df.set_index('code', inplace = True)
casn_smiles_df.set_index('casn', inplace= True)

#Loop through rows of hitc matrix and replace codes with smiles strings
badCounter = 0 #keep track of rows with no corresponding smiles strings
for index, data in hitc_code_df.iterrows():
    rowList = data.values.tolist()
    code = rowList[0]
    #get corresponding casn
    try:
        casn = code_casn_df.loc[code, 'casn']
    except KeyError:
        badCounter+=1
        pass
    #get corresponding smiles
    try:
        smiles = casn_smiles_df.loc[casn, 'smiles']
    except KeyError:
        badCounter+=1
        pass
    #write to cell
    hitc_code_df.loc[index, 'code'] = smiles

#Tidy up and write to csv
hitc_code_df.rename(columns = {'code': 'smiles'}, inplace = True)
hitc_code_df.dropna(subset = ['smiles'], inplace = True)
hitc_code_df.reset_index(inplace = True, drop = True)
hitc_code_df.to_csv("./reprocessed_tox_cast.csv", index=False)
