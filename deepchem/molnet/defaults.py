"""
Featurizers, transformers, and splitters for MolNet.
"""

import os
import importlib
import inspect
import logging
import json
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_defaults(inspect_modules: bool = False) -> Dict[str, List[str]]:
  """Get featurizers, transformers, and splitters.

  This function returns a dictionary with keys 'featurizer', 'transformer',
  and 'splitter'. Each value is a list of names of classes in that
  category. All MolNet ``load_x`` functions should specify which
  featurizers, transformers, and splitters the dataset supports and
  provide sensible defaults.

  Parameters
  ----------
  inspect_modules : bool (default False)
    Inspect dc.feat, dc.trans, and dc.splits modules to get class names.

  Returns
  -------
  defaults : dict
    Contains names of all available featurizers, transformers, and splitters.

  """

  if not inspect_modules:
    path = os.path.dirname(os.path.abspath(__file__))
    defaults = json.load(open(os.path.join(path, "defaults.json")))
  else:
    module = importlib.import_module("deepchem.feat", package="deepchem")
    featurizers = [x[0] for x in inspect.getmembers(module, inspect.isclass)]

    module = importlib.import_module("deepchem.trans", package="deepchem")
    transformers = [x[0] for x in inspect.getmembers(module, inspect.isclass)]

    module = importlib.import_module("deepchem.splits", package="deepchem")
    splitters = [x[0] for x in inspect.getmembers(module, inspect.isclass)]

    defaults = {
        'featurizer': featurizers,
        'transformer': transformers,
        'splitter': splitters
    }

  return defaults
