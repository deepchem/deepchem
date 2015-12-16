from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#from deep_chem.models.deep import SingleTaskDNN
#from deep_chem.models.deep import MultiTaskDNN
from deep_chem.models.deep3D import DockingDNN

def model_builder(model_type, task_types, model_params):
  if model_type == "singletask_deep_network":
    model = SingleTaskDNN(task_types, model_params)
  elif model_type == "multitask_deep_network":
    model = MultiTaskDNN(task_types, model_params)
  elif model_type== "3D_cnn":
    model = DockingDNN(task_types, model_params)
  else:
    model = SklearnModel(task_types, model_params)
  return(model)
