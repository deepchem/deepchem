import os
from deepchem.molnet import load_Platinum_Adsorption


def test_Platinum_Adsorption_loader():

  current_dir = os.path.dirname(os.path.abspath(__file__))
  kwargs = {"log_every_n": 100}
  tasks, datasets, transformers = load_Platinum_Adsorption(
      reload=False, data_dir=current_dir, save_dir=current_dir, **kwargs)
  assert tasks[0] == "Formation Energy"
  assert datasets[0].X[0]['X_Sites'].shape[1] == 3
  assert datasets[0].X[0]['X_NSs'].shape[3] == 19
  assert datasets[0].X[0]['X_NSs'].shape[2] == 6
  assert datasets[0].X[0]['X_NSs'].shape[1] == datasets[0].X[0][
      'X_Sites'].shape[0]

  if os.path.exists(os.path.join(current_dir, 'Platinum_Adsorption.json')):
    os.remove(os.path.join(current_dir, 'Platinum_Adsorption.json'))

  if os.path.exists(os.path.join(current_dir, 'input.in')):
    os.remove(os.path.join(current_dir, 'input.in'))

  if os.path.exists(os.path.join(current_dir, 'platinum_adsorption.tar.gz')):
    os.remove(os.path.join(current_dir, 'platinum_adsorption.tar.gz'))


