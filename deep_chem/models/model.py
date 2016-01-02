"""
Factory function to construct models.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deep_chem.models.deep import SingleTaskDNN
#from deep_chem.models.deep import MultiTaskDNN
#from deep_chem.models.deep3d import DockingDNN
#from deep_chem.models.standard import SklearnModel

#def model_builder(model_type, task_types, model_params,
#                  initialize_raw_model=True):
#  """
#  Factory function to construct model.
#  """
#  if model_type == "singletask_deep_network":
#    model = SingleTaskDNN(task_types, model_params,
#                          initialize_raw_model)
#  elif model_type == "multitask_deep_network":
#    model = MultiTaskDNN(task_types, model_params,
#                         initialize_raw_model)
#  elif model_type == "convolutional_3D_regressor":
#    model = DockingDNN(task_types, model_params,
#                       initialize_raw_model)
#  else:
#    model = SklearnModel(task_types, model_params)
#  return model
