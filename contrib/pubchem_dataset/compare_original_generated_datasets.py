import gzip
import deepchem
import os
import pandas as pd

###
# This script prepares two reference assay datasets for further inspection (e.g., with diff)
###

data_dir = deepchem.utils.get_data_dir()
f_orig = gzip.open(os.path.join(data_dir,"orig/pcba_128.csv.gz"))
f_new = gzip.open(os.path.join(data_dir,"new/pcba_128.csv.gz"))

df_orig = pd.read_csv(f_orig)
df_orig = df_orig.sort_values("mol_id")
df_new = pd.read_csv(f_new)

df_orig = df_orig.fillna(0)
df_new = df_new.fillna(0)

print("{} molecules in original dataset".format(len(df_orig["mol_id"].values)))
print("{} molecules in new dataset: ".format(len(df_new["mol_id"].values)))


set_diff_orig_new = set(df_orig["mol_id"].values).difference(set(df_new["mol_id"].values))
set_diff_new_orig = set(df_new["mol_id"].values).difference(set(df_orig["mol_id"].values))

print("{} molecules in original dataset but not in new: ".format(len(set_diff_orig_new)))
print("{} molecules in new dataset but not in old: ".format(len(set_diff_new_orig)))

# Compare all assay results for their intersection

intersection_keys = list(set(df_orig["mol_id"]).intersection(set(df_new["mol_id"])))
print("{} molecules in new dataset and in old: ".format(len(intersection_keys)))
print("{}, fraction of molecules in the new dataset and also in old".format(len(intersection_keys) / len(df_new)))

df_new = df_new[df_new["mol_id"].isin(intersection_keys)]
df_orig = df_orig[df_orig["mol_id"].isin(intersection_keys)]

print(len(df_new) == len(df_orig))
if(len(df_new) != len(df_orig)) :
  raise Exception

df_orig_compare = df_orig.sort_values("mol_id")
df_new_compare = df_new.sort_values("mol_id")
df_new_compare = df_new_compare[df_orig_compare.columns]

# Drop smiles for compare

df_orig_compare = df_orig_compare.drop("smiles",axis=1)
df_new_compare = df_new_compare.drop("smiles",axis=1)

df_orig_compare = df_orig_compare.set_index("mol_id")
df_new_compare = df_new_compare.set_index("mol_id")

df_orig_compare = df_orig_compare.sort_index()
df_new_compare = df_new_compare.sort_index()

df_new_compare.to_csv("new.csv")
df_orig_compare.to_csv("orig.csv")

f_orig.close()
f_new.close()