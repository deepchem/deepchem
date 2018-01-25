import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import time
plt.switch_backend('agg')

TODO = {
    ('tox21', 'random'): [
        'weave', 'graphconv', 'tf', 'tf_robust', 'irv', 'xgb', 'logreg',
        'textcnn'
    ],
    ('clintox', 'random'): [
        'weave', 'graphconv', 'tf', 'tf_robust', 'irv', 'xgb', 'logreg',
        'textcnn'
    ],
    ('sider', 'random'): [
        'weave', 'graphconv', 'tf', 'tf_robust', 'irv', 'xgb', 'logreg',
        'textcnn'
    ],
    ('bbbp', 'scaffold'):
    ['weave', 'graphconv', 'tf', 'irv', 'xgb', 'logreg', 'textcnn'],
    ('bace_c', 'scaffold'):
    ['weave', 'graphconv', 'tf', 'irv', 'xgb', 'logreg', 'textcnn'],
    ('hiv', 'scaffold'):
    ['weave', 'graphconv', 'tf', 'irv', 'xgb', 'logreg', 'textcnn'],
    ('muv', 'random'): ['graphconv', 'tf', 'tf_robust', 'irv', 'xgb', 'logreg'],
    ('delaney', 'random'): [
        'weave_regression', 'graphconvreg', 'tf_regression', 'xgb_regression',
        'krr', 'textcnn_regression', 'dag_regression', 'mpnn'
    ],
    ('sampl', 'random'): [
        'weave_regression', 'graphconvreg', 'tf_regression', 'xgb_regression',
        'krr', 'textcnn_regression', 'dag_regression', 'mpnn'
    ],
    ('lipo', 'random'): [
        'weave_regression', 'graphconvreg', 'tf_regression', 'xgb_regression',
        'krr', 'textcnn_regression', 'dag_regression', 'mpnn'
    ],
    ('qm7', 'stratified'): [
        'dtnn', 'graphconvreg', 'tf_regression_ft', 'krr_ft'
    ],
    ('qm8', 'random'): [
        'dtnn', 'graphconvreg', 'weave_regression', 'textcnn_regression',
        'mpnn', 'tf_regression', 'tf_regression_ft'
    ],
}

ORDER = [
    'logreg', 'rf', 'rf_regression', 'xgb', 'xgb_regression', 'kernelsvm',
    'krr', 'krr_ft', 'tf', 'tf_regression', 'tf_regression_ft', 'tf_robust',
    'irv', 'textcnn', 'textcnn_regression', 'graphconv', 'graphconvreg', 'dag',
    'dag_regression', 'ani', 'weave', 'weave_regression', 'dtnn', 'mpnn'
]

COLOR = {
    'logreg': '#3F3F3F',
    'rf': '#67AD4F',
    'rf_regression': '#67AD4F',
    'xgb': '#0E766C',
    'xgb_regression': '#0E766C',
    'kernelsvm': '#FC926B',
    'krr': '#FC926B',
    'krr_ft': '#5A372A',
    'tf': '#2B6596',
    'tf_regression': '#2B6596',
    'tf_regression_ft': '#162939',
    'tf_robust': '#775183',
    'irv': '#D9D9D9',
    'graphconv': '#A4D192',
    'graphconvreg': '#A4D192',
    'dag': '#D06329',
    'dag_regression': '#D06329',
    'ani': '#D9D9D9',
    'weave': '#8196AE',
    'weave_regression': '#8196AE',
    'textcnn': '#811B18',
    'textcnn_regression': '#811B18',
    'dtnn': '#D06329',
    'mpnn': '#7B0A48'
}

TODO_list = set()
for key in TODO.keys():
  for val in TODO[key]:
    TODO_list.add((key[0], key[1], val))


def read_results(path):
  Results = set()
  with open(path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
      Results.add((line[0], line[1], line[3]))
  return Results


def run_benchmark(path, deepchem_dir):
  finished = read_results(path)
  os.chdir(deepchem_dir)
  os.chdir('./examples')
  while len(TODO_list - finished) > 0:
    todo = TODO_list - finished
    for p in todo:
      os.system('python benchmark.py --seed 123 -d ' + p[0] + ' -s ' + p[1] +
                ' -m ' + p[2])


def plot(dataset, split, path, out_path):
  if dataset in [
      'bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'pcba_146',
      'pcba_2475', 'sider', 'tox21', 'toxcast'
  ]:
    mode = 'classification'
  else:
    mode = 'regression'
  data = {}
  with open(path, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
      if line[0] == dataset and line[1] == split:
        data[line[3]] = line[8]
  labels = []
  values = []
  colors = []
  for model in ORDER:
    if model in data.keys():
      labels.append(model)
      colors.append(COLOR[model])
      values.append(float(data[model]))
  y_pos = np.arange(len(labels))
  plt.rcdefaults()
  fig, ax = plt.subplots()

  ax.barh(y_pos, values, align='center', color='green')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(labels)
  ax.invert_yaxis()
  if mode == 'regression':
    ax.set_xlabel('R square')
    ax.set_xlim(left=0., right=1.)
  else:
    ax.set_xlabel('ROC-AUC')
    ax.set_xlim(left=0.4, right=1.)
  t = time.localtime(time.time())
  ax.set_title("Performance on %s (%s split), %i-%i-%i" %
               (dataset, split, t.tm_year, t.tm_mon, t.tm_mday))
  plt.tight_layout()
  for i in range(len(colors)):
    ax.get_children()[i].set_color(colors[i])
    ax.text(
        values[i] - 0.1, y_pos[i] + 0.1, str("%.3f" % values[i]), color='white')
  fig.savefig(os.path.join(out_path, dataset + '_' + split + '.png'))
  #plt.show()


if __name__ == '__main__':
  current_dir = os.path.dirname(os.path.realpath(__file__))
  DEEPCHEM_DIR = os.path.split(os.path.split(current_dir)[0])[0]
  FILE = os.path.join(os.path.join(DEEPCHEM_DIR, 'examples'), 'results.csv')
  #run_benchmark(FILE, DEEPCHEM_DIR)
  save_dir = os.path.join(DEEPCHEM_DIR, 'datasets/MolNet_pic')
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  for pair in TODO.keys():
    plot(pair[0], pair[1], FILE, save_dir)
  os.system(
      'aws s3 sync ' + save_dir + ' s3://deepchem.io/trained_models/MolNet_pic')
