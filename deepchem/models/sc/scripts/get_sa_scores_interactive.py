import os, sys
import rdkit.Chem as Chem 
import rdkit.Chem.AllChem as AllChem
import random 
import numpy as np

from utils.SA_Score import sascorer
def get_score_from_smi(smi):
    try:
        return (smi, sascorer.calculateScore(Chem.MolFromSmiles(smi)))
    except Exception:
        return (smi, 0.)

prompt = raw_input('enter a tag for this session: ')
interactive_path = 'sascore.interactive.%s' % (prompt.strip())
fid = open(interactive_path, 'a')

while True:
    try:
        prompt = raw_input('\nEnter SMILES (or quit): ')
        if prompt.strip() == 'quit':
            break
        if str('>>') in prompt: # reaction
            reactants = prompt.strip().split('>>')[0].split('.')
            reactants_smi = []
            reactants_score = 0.
            for reactant in reactants:
                (smi, cur_score) = get_score_from_smi(reactant)
                reactants_smi.append(smi)
                reactants_score = max(reactants_score, cur_score)
            products = prompt.strip().split('>>')[1].split('.')
            products_smi = []
            products_score = 0.
            for product in products:
                (smi, cur_score) = get_score_from_smi(product)
                products_smi.append(smi)
                products_score = max(products_score, cur_score)
            smi = '{}>>{}'.format('.'.join(reactants_smi), '.'.join(products_smi))
            fid.write('%s %s %.4f %.4f %.4f\n' % (prompt.strip(), smi, reactants_score, products_score, products_score-reactants_score))
        else: # single or list of mols
            reactants = prompt.strip().split('.')
            reactants_smi = []
            reactants_score = 0.
            for reactant in reactants:
                (smi, cur_score) = get_score_from_smi(reactant)
                reactants_smi.append(smi)
                reactants_score = max(reactants_score, cur_score)
            fid.write('%s %s %.4f\n' % (prompt.strip(), '.'.join(reactants_smi), reactants_score))

    except KeyboardInterrupt:
        print('Breaking out of prompt')
        fid.close()
        raise KeyboardInterrupt
    except Exception as e:
        print(e)
        fid.write('%s\n' % prompt.strip())
        continue
