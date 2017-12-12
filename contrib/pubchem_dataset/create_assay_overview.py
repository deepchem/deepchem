import pandas as pd
import os
import pickle
import array
from bisect import bisect_left
import gzip
import shutil
import deepchem
import requests
import argparse

data_dir = deepchem.utils.get_data_dir()
sdf_dir = os.path.join(data_dir, "Data")

class PCBADatsetBuilder:

  def __init__(self):

    self.pcba_128_assay_list = "PCBA-1030,PCBA-1379,PCBA-1452,PCBA-1454,PCBA-1457,PCBA-1458,PCBA-1460,PCBA-1461,PCBA-1468,PCBA-1469,PCBA-1471,PCBA-1479,PCBA-1631,PCBA-1634,PCBA-1688,PCBA-1721,PCBA-2100,PCBA-2101,PCBA-2147,PCBA-2242,PCBA-2326,PCBA-2451,PCBA-2517,PCBA-2528,PCBA-2546,PCBA-2549,PCBA-2551,PCBA-2662,PCBA-2675,PCBA-2676,PCBA-411,PCBA-463254,PCBA-485281,PCBA-485290,PCBA-485294,PCBA-485297,PCBA-485313,PCBA-485314,PCBA-485341,PCBA-485349,PCBA-485353,PCBA-485360,PCBA-485364,PCBA-485367,PCBA-492947,PCBA-493208,PCBA-504327,PCBA-504332,PCBA-504333,PCBA-504339,PCBA-504444,PCBA-504466,PCBA-504467,PCBA-504706,PCBA-504842,PCBA-504845,PCBA-504847,PCBA-504891,PCBA-540276,PCBA-540317,PCBA-588342,PCBA-588453,PCBA-588456,PCBA-588579,PCBA-588590,PCBA-588591,PCBA-588795,PCBA-588855,PCBA-602179,PCBA-602233,PCBA-602310,PCBA-602313,PCBA-602332,PCBA-624170,PCBA-624171,PCBA-624173,PCBA-624202,PCBA-624246,PCBA-624287,PCBA-624288,PCBA-624291,PCBA-624296,PCBA-624297,PCBA-624417,PCBA-651635,PCBA-651644,PCBA-651768,PCBA-651965,PCBA-652025,PCBA-652104,PCBA-652105,PCBA-652106,PCBA-686970,PCBA-686978,PCBA-686979,PCBA-720504,PCBA-720532,PCBA-720542,PCBA-720551,PCBA-720553,PCBA-720579,PCBA-720580,PCBA-720707,PCBA-720708,PCBA-720709,PCBA-720711,PCBA-743255,PCBA-743266,PCBA-875,PCBA-881,PCBA-883,PCBA-884,PCBA-885,PCBA-887,PCBA-891,PCBA-899,PCBA-902,PCBA-903,PCBA-904,PCBA-912,PCBA-914,PCBA-915,PCBA-924,PCBA-925,PCBA-926,PCBA-927,PCBA-938,PCBA-995".split(',')
    self.pcba_146_additional_assays = "PCBA-686971,PCBA-504834,PCBA-588856,PCBA-720533,PCBA-1865,PCBA-651820,PCBA-923,PCBA-493014,PCBA-504648,PCBA-624418,PCBA-1159614,PCBA-2289,PCBA-1159524,PCBA-1463,PCBA-504832,PCBA-540256,PCBA-485298,PCBA-2685".split(
      ',')
    self.pcba_146_assay_list = self.pcba_128_assay_list
    self.pcba_146_assay_list.extend(self.pcba_146_additional_assays)

  def create_cid_list(self,assays_to_parse):
    """Find the union of all compounds tested across one or more assays
    """

    min_assay_size = 10000
    assay_paths = list()
    cid_set = set()

    for path, dirs, filenames in os.walk(sdf_dir):
      for dir in dirs:
        # Each directory holds a range of assay results
        joined_path = os.path.join(sdf_dir,dir)
        for path, dirs, filenames in os.walk(joined_path):
          for filename in filenames:
            assay_name = "PCBA-" + filename.replace(".csv","")
            if assay_name not in assays_to_parse:
                continue
            file_path = os.path.join(joined_path,filename)
            df = pd.read_csv(file_path,usecols=["PUBCHEM_CID","PUBCHEM_ACTIVITY_OUTCOME"])
            df = df.dropna()
            df["PUBCHEM_CID"] = df["PUBCHEM_CID"].astype(int)
            if df["PUBCHEM_CID"].count() < min_assay_size:
              continue

            df = df.set_index("PUBCHEM_CID")
            assay_paths.append(file_path)
            cid_set = cid_set.union(set(df.index.values))
            print("Parsed: {0}".format(assay_name))

    return assay_paths, cid_set

  def create_overview_146(self) :
    assay_list = self.pcba_146_assay_list
    self.create_assay_file(assays_to_parse=assay_list,file_name="pcba_146.csv")

  def create_overview_128(self):
    assay_list = self.pcba_128_assay_list
    self.create_assay_file(assays_to_parse=assay_list,file_name="pcba_128.csv")

  def create_overview_for_gene(self,gene_symbol):
    assays_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/genesymbol/{0}/aids/TXT".format(gene_symbol)
    r = requests.get(assays_url)
    assays_to_parse = ["PCBA-" + str(x) for x in r.text.split('\n') if len(x) > 0]
    file_name = "pcba_{0}.csv".format(gene_symbol)
    self.create_assay_file(assays_to_parse=assays_to_parse,file_name=file_name)

  def create_assay_file(self,assays_to_parse,file_name):

    assay_paths, cid_set = self.create_cid_list(assays_to_parse)
    cid_ref_list = list(cid_set)
    cid_ref_list.sort()

    print("CID length is: {0}".format(len(cid_ref_list)))
    path_final = os.path.join(data_dir,file_name)
    assay_results = list()
    assay_names = list()
    cid_len = len(cid_ref_list)

    for assay_path in assay_paths :

      filename = os.path.basename(assay_path)
      assay_name = "PCBA-" + filename.replace(".csv","")
      print("Looking at: {0}".format(assay_name))

      df = pd.read_csv(assay_path,usecols=["PUBCHEM_CID","PUBCHEM_ACTIVITY_OUTCOME"])
      df = df.dropna(subset=["PUBCHEM_CID","PUBCHEM_ACTIVITY_OUTCOME"])
      if len(df.index) == 0:
          continue
      df["IS_ACTIVE"] = df["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"
      df = df.rename(columns={'IS_ACTIVE': assay_name})
      df["PUBCHEM_CID"] = df["PUBCHEM_CID"].astype(int)
      df[assay_name] = df[assay_name].astype(int)
      df = df.set_index("PUBCHEM_CID")
      df = df[~df.index.duplicated(keep='first')]

      assay_results_array = array.array('i',(-1 for i in range(0,cid_len)))
      print(assay_path)
      for i in range(0,cid_len) :
          cid = cid_ref_list[i]
          if cid in df.index:
              val = df.get_value(cid, assay_name)
          else:
              # Just write NA
              val = -1

          assay_results_array[i] = val

      assay_names.append(assay_name)
      assay_results.append(assay_results_array)
      print("Parsed: {0}".format(assay_name))

    # Now, write out the results csv, going line by line through all molecule results
    assay_results_len = len(assay_results)

    print("Reading in smiles info")
    with open(os.path.join(data_dir,"pubchemsmiles_tuple.pickle"),"rb") as f:
      keys, values = pickle.load(f)

    header_line = list()
    header_line.append("mol_id")
    header_line.append(",smiles")
    for assay_name in assay_names:
      header_line.append(",")
      header_line.append(assay_name)
    header_line_txt = "".join(header_line)

    f_final = open(path_final, "w+")
    f_final.write(header_line_txt + "\n")

    for i in range(0,cid_len) :
      cid = cid_ref_list[i]

      # printing the mol_id
      line_for_comp = "CID"+str(cid)

      # printing the SMILES
      bisect_pos = bisect_left(keys, cid, 0)
      cid_pos = bisect_pos if bisect_pos != len(keys) and keys[bisect_pos] == cid else -1

      if cid_pos == -1:
          continue

      line_for_comp += ","+str(values[cid_pos])
      for j in range(0,assay_results_len) :
          val = assay_results[j][i]
          if val == -1:
              line_for_comp += ","
          else:
              line_for_comp += "," + str(val)
      f_final.write(line_for_comp + "\n")

    f_final.close()

    # Now gzip it
    with open(path_final, 'rb') as f_in:
      with gzip.open(path_final+".gz", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

parser = argparse.ArgumentParser(
    description='Deepchem dataset builder for PCBA datasets')
parser.add_argument(
    '-d',
    action='store',
    dest='dataset_name',
    default="",
    help='Choice of dataset: pcba_128, pcba_146')
parser.add_argument(
    '-g',
    action='store',
    dest='gene_arg',
    default=None,
    help='Name of gene to create a dataset for')

args = parser.parse_args()
pcba_builder = PCBADatsetBuilder()
if args.dataset_name == "pcba_128":
  pcba_builder.create_overview_128()
elif args.dataset_name == "pcba_146":
  pcba_builder.create_overview_146()
elif args.gene_arg is not None:
  pcba_builder.create_overview_for_gene(args.gene_arg)
else :
  parser.print_help()





