#from deep_chem.models.deep import SingleTaskDNN
#from deep_chem.models.deep import MultiTaskDNN
from deep_chem.models.deep3D import DockingDNN

def model_builder(model_type, task_type, training_params):
  if model_type == "singletask_deep_network":
    model = SingleTaskDNN(task_types, training_params)
  elif model_type == "multitask_deep_network":
    model = MultiTaskDNN(task_types, training_params)
  elif model_type== "3D_cnn":
    model = DockingDNN(task_types, training_params)
  else:
    #model = sklean_models(train_dict, model)
    raise ValueError("Model type not recognized.")
  return(model)
