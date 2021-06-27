"""
Featurizers, transformers, and splitters for MolNet.
"""

import importlib
import inspect
import logging
from typing import Dict, Any

from deepchem.feat.base_classes import Featurizer
from deepchem.trans.transformers import Transformer
from deepchem.splits.splitters import Splitter

logger = logging.getLogger(__name__)


def get_defaults(module_name: str = None) -> Dict[str, Any]:
  """Get featurizers, transformers, and splitters.

  This function returns a dictionary with class names as keys and classes
  as values. All MolNet ``load_x`` functions should specify which
  featurizers, transformers, and splitters the dataset supports and
  provide sensible defaults.

  Parameters
  ----------
  module_name : {"feat", "trans", "splits"}
    Default classes from deepchem.`module_name` will be returned.

  Returns
  -------
  defaults : Dict[str, Any]
    Keys are class names and values are class constructors.

  Examples
  --------
  >> splitter = get_defaults('splits')['RandomSplitter']()
  >> transformer = get_defaults('trans')['BalancingTransformer'](dataset, {"transform_X": True})
  >> featurizer = get_defaults('feat')["CoulombMatrix"](max_atoms=5)

  """

  if module_name not in ["feat", "trans", "splits"]:
    raise ValueError(
        "Input argument must be either 'feat', 'trans', or 'splits'.")

  if module_name == "feat":
    sc: Any = Featurizer
  elif module_name == "trans":
    sc = Transformer
  elif module_name == "splits":
    sc = Splitter

  module_name = "deepchem." + module_name

  module = importlib.import_module(module_name, package="deepchem")

  defaults = {
      x[0]: x[1]
      for x in inspect.getmembers(module, inspect.isclass)
      if issubclass(x[1], sc)
  }

  return defaults
