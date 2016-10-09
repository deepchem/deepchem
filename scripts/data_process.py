"""
Manipulate CSV data files.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import csv
import argparse

def parse_args(input_args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--descriptor", required=1,
    help="Name of input descriptor CSV file.")
  parser.add_argument(
    "--activities", required=1,
    help="Name of input activities CSV file.")
  parser.add_argument(
    "--output", required=1,
    help="Name of output CSV file.")
  return parser.parse_args(input_args)


def remove_trailing_comma(infile, outfile):
  with open(infile, 'r') as infile, open(outfile, 'w') as outfile:
    csv_writer = csv.writer(outfile)
    for row in csv.reader(infile):
      csv_writer.writerow(row[:-1])  # removes the last comma in each row


# TODO (Bowen): make this function less memory intensive
def combine_descriptor_activities(descriptor_file, activities_file, outfile):
  # set 1st column as the column index of dataframe
  descriptor_df = pd.read_csv(descriptor_file,
                              index_col=0)
  activities_df = pd.read_csv(activities_file, index_col=0)

  # merge descriptor and activities dataframe into output dataframe based on
  # the molecule name, which is the index for both dataframes (but named
  # differently). Default merge is inner merge
  combined_df = pd.merge(descriptor_df, activities_df,
                         left_index=True, right_index=True)
  # need to manually set dataframe indexname after merge based on index
  combined_df.index.name = "Molecule"
  combined_df.to_csv(outfile)


def main():
  args = parse_args()
  descriptor_file = args.descriptor
  activities_file = args.activities
  outfile = args.output
  combine_descriptor_activities(descriptor_file, activities_file, outfile)


if __name__ == '__main__':
  main()
