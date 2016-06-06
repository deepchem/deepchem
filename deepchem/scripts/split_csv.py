"""
Splits large CSVs into multiple shards.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import gzip
import pandas as pd

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--csv-file", required=1,
    help="Name of input CSV file.")
  parser.add_argument(
    "--shard-size", required=1, type=int,
    help="Number of shards to split file into.")
  parser.add_argument(
    "--out", required=1,
    help="Root name of output CSV shards.")
  parser.add_argument(
    "--gzip-output", action="store_true",
    help="Gzip the output.")
  return parser.parse_args(input_args)

def shard_csv(input_file, shard_size, out_name, gzip_output):
  """Shard the csv file into multiple shards."""
  compression = "gzip" if gzip_output else None
  file_obj = None
  try:
    if input_file.endswith(".gz"):
      file_obj = gzip.open(input_file)
    else:
      file_obj = open(input_file)
    for shard_num, df_shard in enumerate(
        pd.read_csv(input_file, index_col=0, chunksize=shard_size)):
      suffix = "czv.gz" if gzip_output else "csv"
      output_name = "%s_%d.%s" % (out_name, shard_num, suffix)
      print("Writing output to %s" % output_name)
      df_shard.to_csv(output_name, compression=compression)
  finally:
    if file_obj is not None:
      file_obj.close()

def main():
  args = parse_args()
  input_file = args.csv_file
  shard_size = args.shard_size
  out_name = args.out
  gzip_output = args.gzip_output
  shard_csv(input_file, shard_size, out_name, gzip_output)

if __name__ == '__main__':
  main()
