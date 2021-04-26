from typing import Tuple, Dict, Any


class RetrosynthesisEngine:
  """Retrosynthesis engine

  This holds an interface to the AiZynthFinder retrosynthesis library

  Note
  ----
  This class requires AiZynthFinder to be installed.
  """
  def __init__(self,
               expansion_policy_paths: Tuple[str, str],
               stock_path: str,
               filter_policy_path: str = None) -> None:
    """Initialize the engine

    Parameters
    ----------
    expansion_policy_paths: tuple of str,
      the path to the expansion policy model and the corresponding template file
    stock_path: str
      the path to the file with the stock molecules
    filter_policy_path: str, optional
      the path to the filter policy model

    Raises
    ------
    ImportError
      if AiZynthFinder is not installed
    """
    try:
      from aizynthfinder.aizynthfinder import AiZynthFinder
    except ModuleNotFoundError:
      raise ImportError("This class requires AiZynthFinder to be installed.")

    conf = {
        "policy": {
            "files": {
                "deepchem_policy": expansion_policy_paths
            }
        },
        "stock": {
            "files": {
                "deepchem_stock": stock_path
            }
        }
    }
    if filter_policy_path:
      conf["filter"] = {"files": {"deepchem_policy": filter_policy_path}}
    self._finder = AiZynthFinder(configdict=conf)
    self._finder.expansion_policy.select("deepchem_policy")
    self._finder.stock.select("deepchem_stock")
    if filter_policy_path:
      self._finder.expansion_policy.select("deepchem_policy")

  def find_routes(self, target: str) -> Dict[str, Any]:
    """ Find routes for a given target molecule

    The method will return a dictionary with tree items
    1. Statistics on the search
    2. The top-ranked routes as list of dictionaries
    3. The top-ranked routes as a list of PIL images

    Parameters
    ----------
    target: str
      the SMILES string of the target molecule

    Returns
    -------
      the routes
    """
    self._finder.target_smiles = target
    self._finder.tree_search()
    self._finder.build_routes()
    return {
        "stats": self._finder.extract_statistics(),
        "dicts": self._finder.routes.dicts,
        "images": self._finder.routes.images
    }
