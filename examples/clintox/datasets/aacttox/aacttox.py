# -*- coding: utf-8 -*-
"""
Processing of AACT data
@author Caleb Geniesse
"""
import re
import subprocess as sp

import numpy as np
import pandas as pd

##############################################################################
### some options
##############################################################################
skip_other_names = True


##############################################################################
### AACT database extraction
##############################################################################
### Loading dataframes and editing indices
aact_dir = "./AACT201603_pipe_delimited"
aact_files = ['./AACT201603_pipe_delimited/clinical_study_noclob.txt',
              './AACT201603_pipe_delimited/interventions.txt',
              './AACT201603_pipe_delimited/intervention_other_names.txt']
aact_fields = [['NCT_ID', 'PHASE', 'OVERALL_STATUS', 'WHY_STOPPED'],
               ['NCT_ID', 'INTERVENTION_TYPE', 'INTERVENTION_NAME',
                'DESCRIPTION'],
               ['NCT_ID', 'OTHER_NAME']]
aact_query_map = {'OVERALL_STATUS': {'any': ['Terminated', 'Suspended',
                                             'Withdrawn']},
                  'WHY_STOPPED': {'any': ['adverse', 'toxic', 'death'],
                                  'not any': ['no safety', 'no risk',
                                              'never opened', 'recruit',
                                              'not due to safety',
                                              'Replaced by another study',
                                              'Not enough patients',
                                              'PI moved', 'death of the PI',
                                              'PI accepted position',
                                              'death of a co-investigator',
                                              'placebo arm had more adverse',
                                              'deaths not attributed to treat',
                                              'not safety']},
                  'INTERVENTION_TYPE': {'any': ['Drug']},
                  'INTERVENTION_NAME': {'not any': [',', 'saline', 'placebo',
                                                    'sugar pill', 'ethanol']},
                  'DESCRIPTION': {'not any': []}}


### Load dataframes
aact_dfs = []
for idx, (fn, fields) in enumerate(zip(aact_files, aact_fields)):

  if skip_other_names:
    if 'other_names' in fn.lower():
      continue

  # make sure \n removed from file (tr -d '\n' < fin.txt > fout.txt)
  aact_df = pd.read_csv(fn, usecols=fields, index_col=False, na_filter=False,
                        delimiter='|', lineterminator='\r')
  aact_dfs.append(aact_df.set_index('NCT_ID'))


# Join tables, save copy
aact_df = aact_dfs[0].join(aact_dfs[1:], how='outer')
#aact_df.to_csv("./aacttox_extracted_raw.csv", sep='|', index=True)


### Clean up dataframes
size_0 = len(aact_df)
for field, query_map in aact_query_map.iteritems():
  for query_type, kws in query_map.iteritems():

    # skip empty kws
    if not (kws and len(kws)):
      continue

    # define mask, based on query type
    mask = sum([aact_df[field].str.contains(_, case=False) for _ in kws])
    if "all" in query_type:
      mask = (mask == len(kws))
    if "any" in query_type:
      mask = (mask >= 1)
    if "not" in query_type:
      mask = (~mask)

    # apply mask
    aact_df = aact_df[mask]


# save copy of filtered tables
#aact_df.to_csv("./aacttox_extracted_raw_filtered.csv", sep='|', index=True)


### extract drug names
drug_set = []
drug_info_set = []
for data_idx, (nct_id, data) in enumerate(aact_df.iterrows()):

  def force_unique_list(d):
    """
    Extract unique list from data (i.e. str, pandas.Series, etc)
    """
    try:
      if np.isnan(float(d)):
        d = []
    except (TypeError, ValueError):
      pass
    if isinstance(d, str):
      d = [d]
    return list(set(d))


  # parse drug names
  data_row = force_unique_list(aact_df['INTERVENTION_NAME'][nct_id])
  other_row = []
  if skip_other_names is False:
    other_row = force_unique_list(aact_df['OTHER_NAME'][nct_id])
  drug_names = data_row + other_row

  # parse tox/phase info
  try:
    ct_tox = force_unique_list(aact_df['OVERALL_STATUS'][nct_id])[0]
    ct_tox = ['terminated', 'suspended', 'withdrawn'].count(ct_tox.lower())
  except IndexError:
    ct_tox = 0

  try:
    ct_phase = force_unique_list(aact_df['PHASE'][nct_id])[0]
    ct_phase = re.findall(r'\d+', ct_phase)[0]
  except IndexError:
    ct_phase = ''
    # how do we differentiate b/t phase 0 vs. no tox?
    if ct_tox == 0:
      ct_phase = '0'

  # remove placebo rows (hacky..)
  placebo_kws = ['placebo', 'positive control']
  data_desc = force_unique_list(aact_df['DESCRIPTION'][nct_id])
  for other_idx, other_drug in enumerate(other_row):
    for drug_name, drug_desc in zip(data_row, data_desc):
      if not any(_ in drug_name.lower() for _ in placebo_kws):
        continue
      if other_drug.lower() not in drug_desc.lower():
        continue
      other_row[other_idx] = None
      print other_idx, other_drug, drug_name, drug_desc

  # add unique, lower case drug names to list
  for drug_name in drug_names:

    # skip bad names
    if not (drug_name and len(drug_name)):
      continue

    # standardize to lower case
    drug_name = drug_name.lower()

    # fixup drug names
    drug_name = drug_name.replace('® (trade name)', '')
    drug_name = drug_name.replace('.', '')
    if '(' in drug_name:
      drug_name = drug_name.split('(')[0].strip()

    # add to list, if not already
    if drug_name in drug_set:
      continue
    drug_set.append(drug_name)

    # define drug info object for each drug
    drug_info = {'name': drug_name,
                 'NCT_ID': nct_id,
                 'CT_TOX': str(ct_tox),
                 'CT_TOX_PHASE': str(ct_phase)}

    # store drug info objects
    drug_info_set.append(drug_info)



##############################################################################
### dataset generation
##############################################################################
# check for smiles on disk
try:
  with open('smiles_cache.csv', 'r') as fid_cache:
    smiles_cache = [_.strip().split(',') for _ in fid_cache.xreadlines()]
    smiles_cache = dict([tuple(_) for _ in smiles_cache])
except IOError:
  smiles_cache = {}


# helper function for fetching smiles
def fetch_smiles(name, props='IsomericSMILES', fmt='TXT', cache=True):
  """
  Fetch smiles for given compound name
  """
  url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'
  url += '/name/{}/property/{}/{}'.format(name, props, fmt)
  url = url.replace(' ', '%20')
  curl = "curl {}".format(url)
  try:
    result, _ = sp.Popen(curl, shell=True,
                         stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    if "Status:" in result:
      print "[-]\t\t[err]\t{}".format(result).replace('\n', '|')
      print "  \t\t[url]\t{}".format(url)
      return None
  except sp.CalledProcessError, sp_err:
    print "[-]\t\t[err]\t{}".format(sp_err).replace('\n', '|')
    print "  \t\t[url]\t{}".format(url)
    return None

  # only keep first smiles for now
  result = result.split('\n')[0]
  if len(result) < 2:
    print "[-]\t\t[err]\t{}".format(result)
    return None

  # save results to disk
  if cache:
    with open('smiles_cache.csv', 'a') as fid:
      fid.write('{},{}\n'.format(name, result))

  return result

### compute other props and write file
datasets = [[], [], [], [], []]
for drug_idx, drug_info in enumerate(drug_info_set):

  # fetch smiles
  drug_name = drug_info['name']
  nct_id = drug_info['NCT_ID']
  ct_tox = drug_info['CT_TOX']
  ct_tox_phase = drug_info['CT_TOX_PHASE']
  fda = str(1 - int(ct_tox))

  ct_tox_phases = ['0', '0', '0', '0']
  if ct_tox_phase.isdigit() and int(ct_tox_phase) > 0:
    for phase, _ in enumerate(ct_tox_phases, start=1):
      if phase >= int(ct_tox_phase):
        ct_tox_phases[phase-1] = str(ct_tox)


  # check for smiles on disk, fetch if not found
  if drug_name in smiles_cache:
    smiles = smiles_cache[drug_name]
  else:
    smiles = fetch_smiles(drug_name)

  if smiles is None:
    continue

  # hardcode ftt for now
  if smiles in [_[0] for _ in datasets[0]]:
    print("[~]\t\t[dup]\t{:8}\t{:20}\t{}\tPhase {}\t{}"
          .format(drug_idx, drug_name, nct_id, ct_tox_phase, smiles))
    continue

  # display progress
  print("[+]\t\t[new]\t{:8}\t{:20}\t{}\tPhase {}\t{}"
        .format(drug_idx, drug_name, nct_id, ct_tox_phase, smiles))

  # store in new datasets
  datasets[0].append([smiles, ct_tox])
  datasets[1].append([smiles, ct_tox, ct_tox_phase])

  datasets[2].append([drug_name, smiles, ct_tox, ct_tox_phase])
  datasets[3].append([smiles, ct_tox] + ct_tox_phases)
  datasets[4].append([drug_name, smiles, ct_tox] + ct_tox_phases)

# display results
print "\nFound {} drugs that match specifications.".format(len(datasets[0]))



##############################################################################
### save dataset
##############################################################################
### save dataset

fout = "aacttox.csv"
cols = ['smiles', 'CT_TOX']
pd.DataFrame(datasets[0], columns=cols).to_csv(fout, index=False)

fout = "aacttox_phase_multiclass.csv"
cols = ['smiles', 'CT_TOX', 'CT_TOX_PHASE']
pd.DataFrame(datasets[1], columns=cols).to_csv(fout, index=False)



#fout = "aacttox_phase_multiclass+name.csv"
#cols = ['name', 'smiles', 'CT_TOX','CT_TOX_PHASE']
#pd.DataFrame(datasets[2], columns=cols).to_csv(fout, index=False)

#fout = "aacttox_phase_multitask.csv"
#cols = ['smiles', 'CT_TOX',
#        'CT_TOX_PHASE_1', 'CT_TOX_PHASE_2',
#        'CT_TOX_PHASE_3', 'CT_TOX_PHASE_4']
#pd.DataFrame(datasets[3], columns=cols).to_csv(fout, index=False)

#fout = "aacttox_phase_multitask+name.csv"
#cols = ['name', 'smiles', 'CT_TOX',
#        'CT_TOX_PHASE_1', 'CT_TOX_PHASE_2',
#        'CT_TOX_PHASE_3', 'CT_TOX_PHASE_4']
#pd.DataFrame(datasets[4], columns=cols).to_csv(fout, index=False)

