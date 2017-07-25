# -*- coding: utf-8 -*-
"""
Join sweetfda and aacttox data
@author Caleb Geniesse
"""
import pandas as pd


##############################################################################
### save dataset
##############################################################################
### load datasets
# load sweetfda
sweetfda_fn = 'sweetfda/sweetfda_approved_processed.csv'
sweetfda_df = pd.read_csv(sweetfda_fn, index_col=False, na_filter=False,
                          delimiter=',', lineterminator='\n')

# load aact
aact_fn = 'aacttox/aacttox_phase_multiclass.csv'
aact_df = pd.read_csv(aact_fn, index_col=False, na_filter=False,
                      delimiter=',', lineterminator='\n')


### fixup smiles for matching
def convert_smiles(s):
  """
  convert smiles to a common format
  """
  #return smiles
  s = list(s)
  s = [_.lower() for _ in s]
  s = [_.replace('=', '') for _ in s]
  return pd.Series(s)


# map original smiles to converted smiles
sweetfda_smiles = list(sweetfda_df['smiles'])
sweetfda_df['smiles'] = convert_smiles(sweetfda_df['smiles'])
sweetfda_smiles_map = dict(zip(list(sweetfda_df['smiles']), sweetfda_smiles))

aact_smiles = list(aact_df['smiles'])
aact_df['smiles'] = convert_smiles(aact_df['smiles'])
aact_smiles_map = dict(zip(list(aact_df['smiles']), aact_smiles))



### join dataframes, index on smiles
sweetfda_df.set_index('smiles', inplace=True)
aact_df.set_index('smiles', inplace=True)
df_join = sweetfda_df.join(aact_df, how='outer')

# map original smiles back
index_smiles = list(df_join.index)
for idx, smiles in enumerate(index_smiles):
  if smiles in aact_smiles_map:
    index_smiles[idx] = aact_smiles_map[smiles]
  elif smiles in sweetfda_smiles_map:
    index_smiles[idx] = sweetfda_smiles_map[smiles]
df_join.index = pd.Series(index_smiles)


### fill all nan with 0
df_join.fillna('0', inplace=True)


### construct datasets
datasets = [[], [], [], [], [], []]
for smiles in df_join.index:

  def check_dtype(d):
    """
    Convert to str(int()))
    """
    if isinstance(d, pd.Series):
      d = list(set(d))[0]
    try:
      d = str(float(d))
      return '' if str(d).lower() == 'nan' else str(int(float(d)))
    except (TypeError, ValueError):
      return str(d)

  fda = cto = check_dtype(df_join.FDA_APPROVED[smiles])
  ct_tox = check_dtype(df_join.CT_TOX[smiles])
  ct_tox_phase = check_dtype(df_join.CT_TOX_PHASE[smiles])
  fda_tox = str(fda) if fda == ct_tox else ''

  if not len(fda):
    fda = cto = str(1 - int(ct_tox))
  if not len(ct_tox):
    ct_tox = str(1 - int(fda))
  if not len(ct_tox_phase):
    ct_tox_phase = '' if int(ct_tox) else '0'

  ct_tox_phases = ['0', '0', '0', '0']
  if ct_tox_phase.isdigit() and int(ct_tox_phase) > 0:
    for phase, _ in enumerate(ct_tox_phases, start=1):
      if phase >= int(ct_tox_phase):
        ct_tox_phases[phase-1] = str(ct_tox)

  print('\t'.join(["==>", fda, ct_tox, cto, ct_tox_phase,
                   '|'.join(ct_tox_phases), smiles]))

  # store in new datasets
  datasets[0].append([smiles, fda, ct_tox])
  datasets[1].append([smiles, fda, ct_tox, ct_tox_phase])
  datasets[2].append([smiles, cto])
  datasets[3].append([smiles, cto, fda_tox])
  datasets[4].append([smiles, fda, ct_tox] + ct_tox_phases)


### save datasets
fout = "clintox.csv"
cols = ['smiles', 'FDA_APPROVED', 'CT_TOX']
pd.DataFrame(datasets[0], columns=cols).to_csv(fout, index=False)


#fout = "aacttox_sweetfda_phase_multiclass.csv"
#cols = ['smiles', 'FDA_APPROVED', 'CT_TOX','CT_TOX_PHASE']
#pd.DataFrame(datasets[1], columns=cols).to_csv(fout, index=False)

#fout = "aacttox_sweetfda_cto_singletask.csv"
#columns=['smiles', 'CLINICAL_TRIAL_OUTCOME']
#pd.DataFrame(datasets[2], columns=cols).to_csv(fout, index=False)

#fout = "aacttox_sweetfda_cto_fdatox.csv"
#columns = ['smiles', 'CLINICAL_TRIAL_OUTCOME', 'FDA_APPROVED_TOX']
#pd.DataFrame(datasets[3], columns=cols).to_csv(fout, index=False)

#fout = "aacttox_sweetfda_phase_multitask.csv"
#columns=['smiles', 'FDA_APPROVED', 'CT_TOX',
#     'CT_TOX_PHASE_1', 'CT_TOX_PHASE_2',
#     'CT_TOX_PHASE_3', 'CT_TOX_PHASE_4']
#pd.DataFrame(datasets[4], columns=cols).to_csv(fout, index=False)

