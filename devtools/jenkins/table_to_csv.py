"""
Utility script to convert the benchmark markdown table into a CSV
"""
import sys


def classification_table_to_csv(lines):
  output = []
  headers = [
      "split", "dataset", "model", "Train score/ROC-AUC", "Valid score/ROC-AUC"
  ]
  output.append(",".join(headers))
  for line in lines:
    vars = [x.strip() for x in line.split('|')]
    if len(vars) == 0:
      continue
    if len(vars) == 1 and vars[0] == "":
      continue
    if len(vars) == 1:
      split = vars[0]
      continue
    if vars[1] == "Dataset":
      continue
    if vars[1].startswith("-----"):
      continue
    my_dataset, model, train, test = vars[1:-1]
    if my_dataset != "":
      dataset = my_dataset
    output.append(",".join([split, dataset, model, train, test]))
  for l in output:
    print(l)


def regression_table_to_csv(lines):
  output = []

  for line in lines:
    vars = [x.strip() for x in line.split('|')]
    if len(vars) == 0:
      continue
    if len(vars) == 1 and vars[0] == "":
      continue
    if len(vars) == 1:
      continue
    if vars[1] == "Dataset":
      continue
    if vars[1].startswith("-----"):
      continue
    my_dataset, model, split, train, test = vars[1:-1]
    if my_dataset != "":
      dataset = my_dataset
    if model == "MT-NN regression":
      model = "NN regression"
    split = "%s splitting" % split
    output.append(",".join([split, dataset, model, train, test]))
  for l in output:
    print(l)


def split_classification_regression(lines):
  for i in range(len(lines)):
    if lines[i].startswith("* Regression"):
      split_index = i
      break
  return lines[:split_index], lines[split_index:]


def create_csv(f1):
  lines = [x.strip() for x in open(f1).readlines()]
  classification, regression = split_classification_regression(lines)
  classification_table_to_csv(classification)
  regression_table_to_csv(regression)


if __name__ == "__main__":
  create_csv(sys.argv[1])
