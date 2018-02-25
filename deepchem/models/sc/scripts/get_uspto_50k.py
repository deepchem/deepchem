from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
import time
from tqdm import tqdm 
import os

project_root = os.path.dirname(os.path.dirname(__file__))

def get_data_df(fpath):
    return pd.read_csv(fpath)

def dump_data(data):
    classes = sorted(np.unique(data['class']))
    for class_ in classes:
        indeces = data.loc[data['class'] == class_].index
        N = len(indeces)
        print('{} rows with class value {}'.format(N, class_))
        with open(os.path.join(project_root, 'data', 'uspto50k_class{}.txt'.format(class_)), 'w') as fid:
            for i in indeces:
                if data['keep'][i]:
                    fid.write('{}\n'.format(data['rxn_smiles'][i]))
    
if __name__ == '__main__':
    data = get_data_df(os.path.join(project_root, 'data', 'data_processed.csv'))
    dump_data(data)
    
