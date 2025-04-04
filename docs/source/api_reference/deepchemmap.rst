DeepChemMap 
===========

The Map class provides static methods to load DeepChem models using JSON-based configuration.
It supports loading parameter dictionaries (load_param_dict), full model instances from directory configs (load_from_config), and pretrained models with weights (load_from_pretrained). 
It uses internal dictionaries to map model, optimizer, and loss names to their respective classes. 
This utility ensures flexible and consistent model restoration, with error handling for missing files or unknown model types. 
It's ideal for reproducible experiments and deploying saved models in a modular, extensible way.


.. autoclass:: deepchem.deepchemmmap.Map
  :members:




