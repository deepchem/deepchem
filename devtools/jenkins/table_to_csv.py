import sys


def table_to_csv(lines):
  output = []
  headers = ["split", "dataset", "model", "Train score/ROC-AUC", "Valid score/ROC-AUC"]
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


def table_to_json(f):
  lines = [x.strip() for x in open(f).readlines()]
  table_to_csv(lines)


if __name__ == "__main__":
  table_to_json(sys.argv[1])
